"""Making sure the model a stage needs is resident before the stage starts.

Naming a different model on a request makes the server load it just in time. That is
convenient and, on this machine, unreliable: alternating two 14B models during a benchmark
lost **11 of 12 runs** to `HTTP 500 Internal Server Error`. The card sits at 97.8% VRAM
with roughly 357 MiB free, so evicting one 14B to load another intermittently fails
outright rather than taking the expected ~18 seconds. Loading each model explicitly first
gave 6 of 6 successes for both.

So the rule this class exists to enforce is: **swap deliberately, between stages, or not at
all.** Never let a swap happen as a side effect of a request in the middle of a batch.

When it is worth swapping at all is a question of amortisation. A research run extracts
from dozens of pages, and extraction measurably yields 41% more verified claims on the
coder model than on the reasoning one at identical latency — so one 18-second load repaid
across forty calls is obviously worth it. A single ad-hoc call is not.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any


class ModelLoader:
    """Ensures a named model is the resident one, using the `lms` CLI.

    Takes the runner and registry rather than reaching for them, so a test can drive this
    with fakes and so it shares the registry's short cache instead of spawning its own
    subprocesses.
    """

    # How long to wait for a load before giving up. A cold 14B takes about 18 seconds
    # here; a slow disk or a model being read for the first time can take longer, and
    # failing early would turn a slow load into a spurious error.
    _LOAD_TIMEOUT_S = 180

    def __init__(self, runner: Any, registry: Any) -> None:
        self._runner = runner
        self._registry = registry

    def resident_keys(self) -> set[str]:
        try:
            return {model.key for model in self._registry.loaded()}
        except Exception:
            # Unknown is treated as "not resident", which at worst causes one redundant
            # load attempt. Guessing the other way would skip a load that was needed.
            return set()

    def ensure(self, model_key: str, *, evict_others: bool = True) -> bool:
        """Make `model_key` resident. Returns whether it is, as far as we can tell.

        `evict_others` unloads everything else first. That is the default because the
        failure being avoided is precisely two 14B models contending for a card that fits
        one: asking the server to load the second while the first is still resident is
        what produced the 500s. Unloading first turns an overlapping swap into two
        sequential operations.

        Returns False rather than raising when the CLI is unavailable or the load fails.
        A caller that cannot get its preferred model should carry on with whatever is
        loaded — a slower or slightly worse model is a far better outcome than aborting
        the run.
        """
        if model_key in self.resident_keys():
            return True

        if evict_others:
            # Ignore the result: if nothing was loaded this is a no-op, and if it fails
            # the load below will surface the real problem.
            self._run(["unload", "--all"])
            # A brief settle. The unload returns as soon as the request is accepted, and
            # issuing the load before the memory is actually released re-creates the
            # overlap this method exists to prevent.
            time.sleep(2.0)

        if not self._run(["load", model_key, "-y"]):
            return False

        # Confirm rather than trust the exit code. `lms load` has been observed to accept
        # options it then ignores — `--context-length` is silently dropped — so the only
        # reliable evidence that a model is resident is asking what is resident.
        for _ in range(10):
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
            )
            return completed.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
