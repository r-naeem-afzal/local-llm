"""Making sure the model a stage needs is resident, and that loading it cannot overfill VRAM.

Naming a different model on a request makes the server load it just in time. That is
convenient and, on this machine, dangerous: LM Studio loads what is asked for and leaves
what is already there, so a request that names a second model **adds** it rather than
swapping. On a 16 GB card that holds one 14B at a time, the result is two models sharing
memory that fits one.

That failure has now happened twice, in two different ways, and both were invisible while
they were happening:

* a router sent ranking to a small model while a 14B was resident, which loaded both and
  turned a 12 second extraction into 115 seconds; and
* a single model was loaded **twice** — once by just-in-time loading and once explicitly —
  producing instances `qwen/qwen2.5-coder-14b` and `qwen/qwen2.5-coder-14b:2`. Nothing
  noticed, because both report the same *model key* and only the *instance identifier*
  differs, so any check that deduplicates by key sees exactly one model.

So the rule this module enforces is: **swap deliberately, between stages, or not at all**,
and never load anything without first establishing that it fits.

## The two-part safeguard, and why one part is not enough

`VramBudget` estimates before loading, so an impossible load is refused rather than
attempted. But the estimate is a rule of thumb — the KV cache depends on the context the
server chooses, which is not knowable in advance — so estimation alone would be false
confidence. `ModelLoader` therefore also **verifies afterwards**: it checks that exactly
one instance is resident and that the card still has breathing room, and unloads if not.

Estimating catches the obviously-impossible cheaply. Verifying catches everything else,
including whatever the estimate got wrong. Neither alone is sufficient.

## Why overfilling matters more than it sounds

Asking for more VRAM than the card has does not fail. Windows places the overflow in
shared system memory and the model keeps working, roughly fifteen times slower, with no
error anywhere — `nvidia-smi` reports the same memory used and the same 100% utilisation
either way. A guard that runs *before* the load is the only cheap way to avoid a state
that is expensive to even detect.

## When it is worth swapping at all

A research run extracts from dozens of pages, and extraction measurably yields more
verified claims on the coder model than on the reasoning one at similar latency — so one
load repaid across forty calls is obviously worth it. A single ad-hoc call is not.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any


class VramBudget:
    """Answers "will this model fit?" before anything tries to load it.

    Separate from `ModelLoader` because it is a different question. The loader knows how
    to drive the CLI; this knows what the card can hold. Splitting them means the budget
    can be consulted by anything that wants to decide *whether* to swap — the router, the
    dashboard — without that caller also acquiring the ability to load models.
    """

    # Memory that must stay free after a model is loaded. Not slack for its own sake: the
    # desktop compositor, the browser and the editor all allocate on the same card, and
    # their use moves by hundreds of megabytes as windows open and close. A load that fits
    # with nothing to spare works until the moment something else asks for memory, and
    # then the driver starts placing allocations in system RAM — which does not fail, it
    # just runs about fifteen times slower, silently.
    # Calibrated down from 700 because 700 refused the model the pipeline actually uses.
    # Measured: qwen2.5-coder-14b sits at 14,193 MiB resident on a card with 14,765 MiB
    # free once the desktop's 1,538 MiB is accounted for — so it runs, at 69 tok/s, with
    # about 570 MiB to spare. A 700 MiB requirement would have blocked the known-good
    # configuration, which is a worse failure than the one being prevented: a guard that
    # refuses everything is indistinguishable from a broken pipeline.
    #
    # 300 MiB is what is left over as genuine margin. The load is not being trusted to this
    # number alone — `ModelLoader` verifies the card's state after loading, so the estimate
    # only has to be good enough to reject the impossible.
    _HEADROOM_MIB = 300

    # What a model costs beyond its own weights, as a multiple of the weights.
    #
    # Measured rather than derived, because the exact figure needs the KV-cache geometry
    # (layers, key/value heads, head dimension) and the context length the server picks,
    # neither of which the CLI exposes before loading. One calibration point:
    #
    #     qwen2.5-coder-14b   weights 8,571 MiB   resident total 14,193 MiB   = 1.66x
    #
    # Used only to refuse loads that cannot possibly fit, and deliberately a little
    # pessimistic: refusing a load that would have just fitted costs one stage a slower
    # model, while allowing one that does not fit costs the whole run a fifteenfold
    # slowdown that nothing reports.
    _OVERHEAD_FACTOR = 1.66

    def __init__(self, gpu_probe: Any) -> None:
        # Takes the probe rather than calling nvidia-smi itself, so this shares whatever
        # caching and failure handling the monitor already has, and so a test can hand it
        # a fake card.
        self._gpu_probe = gpu_probe

    def _gpu(self) -> Any | None:
        """The card's current state, or None when it cannot be read.

        `GpuInfo.available` is False when pynvml is missing or the driver call failed. A
        card that cannot be read is treated as absent rather than as full, because every
        caller here interprets None as "do not block on these grounds".
        """
        try:
            info = self._gpu_probe.read()
        except Exception:
            return None
        return info if getattr(info, "available", False) else None

    def total_mib(self) -> int | None:
        """The card's total VRAM, or None when it cannot be read.

        Needed as a ceiling on estimates. Without it, "how much would be free if
        everything were unloaded" can exceed the physical card — which is not an
        arithmetic slip but the signature of the very problem being guarded against: it
        happens exactly when the resident models already total more than the card holds.
        """
        gpu = self._gpu()
        return None if gpu is None else int(gpu.total_mib)

    def free_mib(self) -> int | None:
        """VRAM the card reports as unused, or None when the card cannot be read.

        None is distinct from zero and must stay that way: zero means "no room", while
        None means "unknown", and the safe response to unknown is to carry on rather than
        to refuse every load because a monitoring tool is missing.
        """
        gpu = self._gpu()
        return None if gpu is None else int(gpu.free_mib)

    def required_mib(self, weights_mib: int) -> int:
        """What a model of this weight actually occupies once loaded.

            8,571 MiB of weights  ->  14,227 MiB expected resident
                                         (measured: 14,193 MiB — the factor is calibrated
                                          on exactly this model, so treat it as a floor
                                          for anything with different attention geometry)

        The gap is the KV cache and the compute buffers, which together are larger than
        people expect — on a 14B at a 30k context the cache alone is about 4.7 GiB, more
        than half the size of the weights again.
        """
        return int(weights_mib * self._OVERHEAD_FACTOR)

    def fits(self, weights_mib: int, *, already_free_mib: int | None = None) -> bool:
        """Whether loading this model leaves the card with room to spare.

            free 14,765 MiB, weights 8,571  ->  needs 14,227 + 300  ->  True  (just)
            free 14,765 MiB, weights 11,548 ->  needs 19,169 + 300  ->  False

        The first line is worth reading twice: the model this pipeline uses most fits with
        about 600 MiB to spare on an otherwise empty card. That is not slack in the
        arithmetic, it is the honest statement that this machine is one browser window away
        from trouble with a 14B — which is worth knowing rather than rounding away.

        Returns True when the card cannot be read at all. Refusing every load because
        `nvidia-smi` is missing would break a working setup to protect against a problem
        that might not exist.
        """
        free = already_free_mib if already_free_mib is not None else self.free_mib()
        if free is None:
            return True
        return self.required_mib(weights_mib) + self._HEADROOM_MIB <= free

    def has_headroom(self) -> bool:
        """Whether the card currently has enough free memory to be healthy.

        Used *after* a load, as the check the estimate cannot provide. If this is false
        with one model resident, the load overfilled the card whatever the arithmetic said
        beforehand.
        """
        free = self.free_mib()
        return True if free is None else free >= self._HEADROOM_MIB


class ModelLoader:
    """Ensures a named model is the one and only resident model, using the `lms` CLI.

    Takes the runner, registry and budget rather than reaching for them, so a test can
    drive this with fakes and so it shares the registry's short cache instead of spawning
    its own subprocesses.
    """

    # How long to wait for a load before giving up. A cold 14B takes about 18 seconds
    # here; a slow disk or a model being read for the first time can take longer, and
    # failing early would turn a slow load into a spurious error.
    _LOAD_TIMEOUT_S = 300

    # How long to wait for VRAM to actually be released after an unload. The unload
    # command returns as soon as the request is *accepted*, not when the memory is free,
    # and issuing the next load into memory that has not been released yet re-creates the
    # overlap this class exists to prevent.
    _UNLOAD_SETTLE_S = 15

    # How long the card is given to reach a healthy amount of free memory after a load.
    # A swap has two models resident for a moment, and the released memory comes back a
    # beat later, so anything shorter reports overflow that is not there.
    _HEADROOM_SETTLE_S = 10

    # Options every load carries.
    #
    # `--gpu max` asks for every layer on the card. On its own it has not been shown to
    # change throughput, but it is what is wanted whenever the model fits, and the
    # alternative is letting the server decide silently.
    #
    # `--parallel 1` because there is one pipeline and one card. Each extra prediction
    # slot wants its own KV cache, so `--parallel 4` multiplies the memory pressure this
    # whole class exists to keep under control.
    _LOAD_OPTIONS = ("--gpu", "max", "--parallel", "1")

    def __init__(self, runner: Any, registry: Any, budget: VramBudget | None = None) -> None:
        self._runner = runner
        self._registry = registry
        # Optional so existing callers that build a loader with two arguments keep
        # working; without it the class behaves as it did before, doing the swapping but
        # none of the checking.
        self._budget = budget
        # Why the last `ensure` failed, in words a person can act on. Kept because
        # returning a bare False tells a caller that something went wrong and nothing
        # about what, and "the model does not fit" and "the CLI is missing" want
        # completely different responses.
        self.last_error: str = ""

    # ── inspection ──

    def resident(self) -> list[Any]:
        """Every model instance currently in VRAM, duplicates included."""
        try:
            return list(self._registry.loaded())
        except Exception:
            # Unknown is treated as "nothing resident", which at worst causes one
            # redundant load attempt. Guessing the other way would skip a needed load.
            return []

    def resident_keys(self) -> set[str]:
        """Distinct model keys in VRAM.

        Deliberately *not* the residency test to use before loading. Two instances of one
        model collapse to a single key here, which is exactly the blindness that let a
        duplicate load go unnoticed — use `duplicates()` for that question.
        """
        return {model.key for model in self.resident()}

    def duplicates(self) -> list[Any]:
        """Instances that are a second (or third) copy of a model already loaded.

            [coder(id="qwen/coder-14b"), coder(id="qwen/coder-14b:2")]
              ->  [coder(id="qwen/coder-14b:2")]

        The first instance of each key is kept and the rest are reported, because the
        extra copies are pure waste: they serve no request the first cannot, and each one
        takes its full share of weights and KV cache.
        """
        seen: set[str] = set()
        extras = []
        for model in self.resident():
            if model.key in seen:
                extras.append(model)
            else:
                seen.add(model.key)
        return extras

    def unload_duplicates(self) -> int:
        """Unload every duplicate instance, returning how many were removed.

        Worth doing even when nothing is about to be loaded, because a duplicate arrives
        without anything asking for it: just-in-time loading creates one whenever a
        request names a model whose loaded instance carries a different identifier.
        """
        extras = self.duplicates()
        for model in extras:
            # By identifier, not by key — unloading by key would be ambiguous between the
            # copy being removed and the one being kept.
            self._run(["unload", model.identifier])
        return len(extras)

    # ── loading ──

    def ensure(self, model_key: str, *, evict_others: bool = True) -> bool:
        """Make `model_key` the resident model. Returns whether it is, as far as we know.

        The sequence, and why each step is there:

        1. **Clear duplicates** even when the right model is already loaded — a second
           copy of the right model is as damaging as the wrong model.
        2. **Refuse an impossible load** before touching anything, so a model too large
           for the card fails fast and leaves the working one alone. Without this the
           unload happens first and the run is left with *no* model.
        3. **Unload everything and wait for the memory to actually come back**, rather
           than assuming the command's return means the card is free.
        4. **Load**, then confirm by asking what is resident rather than trusting the exit
           code — `lms load` has been observed to accept options it then ignores.
        5. **Verify the card still has headroom**, which is the check the estimate cannot
           make and the one that catches whatever the estimate got wrong.

        Returns False rather than raising when anything fails. A caller that cannot get
        its preferred model should carry on with whatever is loaded — a slower or slightly
        worse model is a far better outcome than aborting a run that has already paid for
        searching and ranking.
        """
        self.last_error = ""

        if self.unload_duplicates():
            self._settle()

        weights_mib = self._weights_mib(model_key)
        if self._budget is not None and weights_mib:
            # Measured against an *empty* card rather than current free memory: everything
            # else is about to be unloaded, so the question is whether this model fits on
            # its own, not whether it fits alongside what happens to be loaded now.
            free_when_empty = self._free_if_emptied()
            if not self._budget.fits(weights_mib, already_free_mib=free_when_empty):
                self.last_error = (
                    f"{model_key} needs about "
                    f"{self._budget.required_mib(weights_mib)} MiB but only "
                    f"{free_when_empty} MiB would be free with nothing else loaded"
                )
                return False

        # "The right model is loaded" is not the same as "the card is in a good state".
        # Returning early on the first condition alone was a real bug: with gpt-oss-20b
        # and the coder both resident, asking for the coder returned True and left 199 MiB
        # free, because the coder *was* there — so the guard reported success on exactly
        # the overfilled card it exists to prevent. When eviction is wanted, the model must
        # be resident **and alone**.
        others = [key for key in self.resident_keys() if key != model_key]
        if model_key in self.resident_keys() and not (evict_others and others):
            return True

        if evict_others:
            # Ignore the result: if nothing was loaded this is a no-op, and if it fails
            # the load below surfaces the real problem.
            self._run(["unload", "--all"])
            self._settle()

        if not self._run(["load", model_key, *self._LOAD_OPTIONS, "-y"]):
            self.last_error = f"lms load {model_key} failed"
            return False

        if not self._await_resident(model_key):
            self.last_error = f"{model_key} did not become resident after loading"
            return False

        # A load can succeed and still leave the card in the state this class exists to
        # avoid, if the estimate was wrong or something else took memory meanwhile.
        # Unloading is the right response: running on an overfilled card is slower than
        # running on the CPU would be, and unlike a failure it reports nothing.
        if self._budget is not None and not self._await_headroom():
            free = self._budget.free_mib()
            self.last_error = (
                f"{model_key} loaded but left only {free} MiB free; unloaded it rather "
                f"than run in a state where allocations spill to system memory"
            )
            self._run(["unload", "--all"])
            return False

        return True

    def _await_headroom(self) -> bool:
        """Whether the card has room to spare, allowing time for memory to settle.

        Checking once was wrong, and produced a false alarm on the first real run: the
        previous model's memory had not finished being released when the new one finished
        loading, so a momentary reading of 452 MiB free was taken as proof of an overfilled
        card. The loader then unloaded a model that was in fact fine — the very next
        reading was 14,694 MiB free.

        Both models being briefly resident is normal during a swap and says nothing about
        the end state, so the check waits for the reading to become healthy rather than
        judging the first sample. It gives up only if the card is still short after several
        seconds, which is what a genuine overflow looks like: persistent, not transient.
        """
        for _ in range(int(self._HEADROOM_SETTLE_S * 2)):
            if self._budget is None or self._budget.has_headroom():
                return True
            time.sleep(0.5)
        return False

    # ── internals ──

    def _weights_mib(self, model_key: str) -> int:
        """The on-disk size of a model in MiB, or 0 when it cannot be determined.

            {"key": "qwen/qwen3.5-9b", "size_mib": 6706}  ->  6706

        Zero means "unknown", and every caller treats unknown as "do not refuse on these
        grounds" — a missing size must not become a reason to block a load that would have
        worked.
        """
        try:
            for entry in self._registry.installed():
                if entry.get("key") == model_key:
                    return int(entry.get("size_mib") or 0)
        except Exception:
            pass
        return 0

    def _free_if_emptied(self) -> int | None:
        """How much VRAM would be free if every loaded model were unloaded.

            free now 600 MiB, resident models 8,571 + 6,706  ->  15,877 MiB

        Computed rather than measured, because measuring it would mean unloading first and
        the point of the check is to decide *whether* to unload. The models' own weights
        are a floor rather than the true figure — their KV caches come back too — so this
        under-estimates the free memory, which errs toward refusing a marginal load.
        """
        if self._budget is None:
            return None
        free = self._budget.free_mib()
        if free is None:
            return None
        recovered = free + sum(model.size_mib for model in self.resident())

        # Capped at the physical card. Two models whose weights total more than the card
        # produced an estimate of 20,332 MiB free on a 16,303 MiB card — which is not a
        # slip in the arithmetic but the fingerprint of the overflow being guarded
        # against, since it can only happen when what is loaded already exceeds what fits.
        # Left uncapped, the guard would read that as abundant room and wave through the
        # next load.
        total = self._budget.total_mib()
        return recovered if total is None else min(recovered, total)

    def _settle(self) -> None:
        """Wait until the card's free memory stops rising, or the timeout expires.

        `lms unload` returns when the request is accepted, not when the driver has
        released the memory. Polling until the figure stops moving is more reliable than
        any fixed sleep: the release is usually under a second and occasionally several,
        so a fixed sleep is either wasteful or wrong.
        """
        if self._budget is None:
            time.sleep(2.0)
            return
        previous = -1
        deadline = time.monotonic() + self._UNLOAD_SETTLE_S
        while time.monotonic() < deadline:
            time.sleep(0.5)
            free = self._budget.free_mib()
            if free is None:
                return
            if free <= previous:
                # Stopped rising, so the release is done.
                return
            previous = free

    def _await_resident(self, model_key: str) -> bool:
        """Poll until the model shows up as loaded, rather than trusting the exit code."""
        for _ in range(20):
            if model_key in self.resident_keys():
                return True
            time.sleep(1.0)
        return False

    def _run(self, args: list[str]) -> bool:
        """Run one `lms` command, returning whether it succeeded."""
        executable = getattr(self._runner, "_executable", None)
        if executable is None:
            return False
        try:
            path = executable()
        except Exception:
            return False
        if not path or not path.exists():
            return False

        try:
            completed = subprocess.run(
                [str(path), *args],
                capture_output=True, text=True,
                timeout=self._LOAD_TIMEOUT_S, check=False,
                # The CLI prints progress spinners containing bytes the Windows console
                # codepage cannot decode, which raised UnicodeDecodeError inside
                # subprocess's reader thread and made a successful load look like a crash.
                encoding="utf-8", errors="replace",
            )
            return completed.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
