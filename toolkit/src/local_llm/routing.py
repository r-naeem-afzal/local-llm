"""Choosing which local model runs a given task.

Every call used to go to whatever `LOCAL_LLM_MODEL` named. That is fine with one model
installed and wrong with several, because the models differ in ways that matter to
specific tasks rather than in general quality.

The distinction that does the most work here is **reasoning versus not**. Qwen3 is a
hybrid-reasoning model: it thinks before answering, and that thinking is billed against
the same output budget as the answer. Measured on this machine:

* 203 output tokens spent merely to say "OK";
* a ranking call over 31 results exhausted a 2,048-token budget on reasoning alone and
  returned truncated JSON.

For a task whose output shape is fixed by a schema — extract these fields, rate these
indices — that deliberation buys nothing and costs both budget and latency. A
non-reasoning model of the same size answers immediately. Conversely, deciding *what to
search for* is exactly where deliberation helps.

## The constraint that shapes all of this

One 14B model occupies about 8.4 GiB of weights plus roughly 7 GiB of KV cache at 32K
context, which is 97% of a 16 GB card. **Only one fits.** Switching models means evicting
the resident one and paying an ~18 second load.

So routing here is deliberately *coarse*. Picking the best model per individual call would
thrash: a pipeline alternating between two 14B models every few seconds would spend more
time loading than generating. `ModelRouter` therefore answers "which model should this
kind of work use", and callers are expected to group work by model — which the research
pipeline does naturally, since each stage is a batch of one kind of call.

`prefer_loaded` exists for the same reason: when the difference is marginal, staying with
the model already in VRAM beats an 18-second swap to a slightly better one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import Settings


class ModelRole(str, Enum):
    """What a model is *for*, rather than what it is called.

    Roles rather than names because the installed set changes — the user adds models as
    they find them — so anything that hard-codes `qwen/qwen3-14b` is wrong the moment a
    better model arrives. Routing decisions are expressed in roles and resolved against
    whatever is actually installed.
    """

    #: Deliberation helps: planning, open-ended questions, weighing alternatives.
    REASONING = "reasoning"
    #: Fixed output shape: schema-constrained extraction, classification, rewriting.
    #: Reasoning is pure overhead here.
    STRUCTURED = "structured"
    #: Cheap, high-volume gating where a large model's latency dominates.
    TRIAGE = "triage"
    #: Not a chat model at all — vector similarity, for deduplication and search.
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class ModelProfile:
    """One installed model, classified.

    Frozen because it describes a fact about what is on disk, not a setting to adjust.
    """

    key: str
    roles: tuple[ModelRole, ...]
    size_mib: int
    params: str
    is_reasoning: bool

    @property
    def is_large(self) -> bool:
        """Whether loading this evicts whatever else is resident.

        The threshold is about the 16 GB card this runs on: anything over ~5 GiB cannot
        share VRAM with another model of similar size once the KV cache is counted. A 4B
        at 2.4 GiB is small enough that it *might* coexist; a 14B at 8.4 GiB never is.
        """
        return self.size_mib > 5000


class ModelClassifier:
    """Works out what an installed model is good for, from its metadata.

    Deliberately based on architecture and name fragments rather than a hard-coded list of
    known models. The user adds models regularly, and a lookup table would silently treat
    every new arrival as unusable — the failure would look like "routing ignores my new
    model" rather than like a missing table entry.
    """

    # Families that think before answering. Qwen3 is the one measured here; the others are
    # listed because they behave the same way and would otherwise be misrouted to
    # schema-constrained work where their reasoning is wasted budget.
    _REASONING_MARKERS = ("qwen3", "deepseek-r1", "qwq", "reasoning", "thinking")

    # Families trained for code and structured output. They are non-reasoning, which is
    # exactly what makes them good at filling a fixed schema quickly.
    _STRUCTURED_MARKERS = ("coder", "codestral", "starcoder", "deepseek-coder")

    _EMBEDDING_MARKERS = ("embed", "bge", "gte", "nomic")

    def classify(self, entry: dict[str, Any]) -> ModelProfile:
        key = str(entry.get("key", "")).lower()
        architecture = str(entry.get("architecture", "")).lower()
        kind = str(entry.get("type", "llm")).lower()
        size_mib = int(entry.get("size_mib") or 0)
        params = str(entry.get("params", ""))
        haystack = f"{key} {architecture}"

        if kind == "embedding" or any(m in haystack for m in self._EMBEDDING_MARKERS):
            return ModelProfile(
                key=str(entry.get("key", "")), roles=(ModelRole.EMBEDDING,),
                size_mib=size_mib, params=params, is_reasoning=False,
            )

        is_reasoning = any(marker in haystack for marker in self._REASONING_MARKERS)
        is_structured = any(marker in haystack for marker in self._STRUCTURED_MARKERS)

        roles: list[ModelRole] = []
        if is_structured:
            # A coder model leads on structured work and can still reason adequately, so
            # it is listed as a fallback for reasoning rather than excluded from it.
            roles = [ModelRole.STRUCTURED, ModelRole.REASONING]
        elif is_reasoning:
            roles = [ModelRole.REASONING, ModelRole.STRUCTURED]
        else:
            # An unrecognised instruct model. Usable for both, preferred for neither —
            # which is the honest position for a model nothing is known about.
            roles = [ModelRole.STRUCTURED, ModelRole.REASONING]

        # Small models take the triage role as well, because that role is about cost and
        # latency rather than capability: rating twenty snippets does not need 14B.
        if size_mib and size_mib <= 5000:
            roles.append(ModelRole.TRIAGE)

        return ModelProfile(
            key=str(entry.get("key", "")), roles=tuple(roles), size_mib=size_mib,
            params=params, is_reasoning=is_reasoning,
        )


class ModelRouter:
    """Picks a model for a kind of work, from what is actually installed.

    Takes the registry rather than a model list so it reflects reality at call time: a
    model downloaded five minutes ago is available without restarting anything.
    """

    # Which roles suit which tool, best first. A tool absent from this map falls through
    # to the configured default, which is the right behaviour for anything unclassified —
    # better a known-working model than a guess.
    _TASK_ROLES: dict[str, tuple[ModelRole, ...]] = {
        # Deciding what to search for benefits from weighing alternatives, and the output
        # is small, so the reasoning cost is affordable.
        "plan_queries": (ModelRole.REASONING, ModelRole.STRUCTURED),
        # Rating results is high-volume and schema-constrained. It is the single worst
        # place for a reasoning model: the ranking failure that sent the pipeline back to
        # unranked search order was reasoning exhausting the output budget.
        "rank_results": (ModelRole.TRIAGE, ModelRole.STRUCTURED, ModelRole.REASONING),
        # Extraction fills a fixed schema from a supplied document. Deliberation adds
        # latency to the pipeline's dominant cost and changes nothing about the answer.
        "extract_claims": (ModelRole.STRUCTURED, ModelRole.REASONING),
    }

    def __init__(self, settings: Settings, registry: Any,
                 classifier: ModelClassifier | None = None) -> None:
        self._settings = settings
        self._registry = registry
        self._classifier = classifier or ModelClassifier()

    def profiles(self) -> list[ModelProfile]:
        """Every installed model, classified. Empty if the CLI is unavailable."""
        try:
            return [self._classifier.classify(e) for e in self._registry.installed()]
        except Exception:
            # Routing must never be the reason a call fails. With no inventory the caller
            # falls back to the configured model, which is exactly the old behaviour.
            return []

    def _loaded_keys(self) -> set[str]:
        try:
            return {m.key for m in self._registry.loaded()}
        except Exception:
            return set()

    def choose(self, task: str, *, prefer_loaded: bool = True) -> str:
        """The model key to use for `task`, falling back to the configured default.

            choose("extract_claims")  ->  "qwen/qwen2.5-coder-14b"   (structured work)
            choose("plan_queries")    ->  "qwen/qwen3-14b"           (deliberation helps)
            choose("anything_else")   ->  settings.model

        With `prefer_loaded`, a resident model that can do the job wins over a marginally
        better one that is not loaded. On a card that holds one 14B at a time, swapping
        costs an ~18 second load — more than the task usually takes — so the "better"
        choice is frequently the slower one overall.
        """
        roles = self._TASK_ROLES.get(task)
        if not roles:
            return self._settings.model

        profiles = self.profiles()
        if not profiles:
            return self._settings.model

        loaded = self._loaded_keys() if prefer_loaded else set()

        # Walk roles best-first. Within a role, a resident model wins; otherwise take the
        # best-ranked candidate.
        for role in roles:
            candidates = self._rank_for_role(
                [p for p in profiles if role in p.roles], role
            )
            if not candidates:
                continue
            resident = [p for p in candidates if p.key in loaded]
            if resident:
                return resident[0].key
            # Nothing resident for this role. Only accept a swap for the *first* choice
            # role; for later, weaker roles a swap is not worth an eviction, so keep
            # looking and let the default catch it.
            if role is roles[0]:
                return candidates[0].key

        return self._settings.model

    @staticmethod
    def _rank_for_role(candidates: list[ModelProfile], role: ModelRole) -> list[ModelProfile]:
        """Order candidates best-first *for this role*.

        Taking whichever model the registry happened to list first was a real bug: with a
        4B, a 14B coder and a 14B reasoning model installed, all three carry the
        `structured` role, and the registry listed the 4B first — so extraction, the task
        the benchmark says should use the 14B coder, was routed to a 4B instead.

        Two rules, and they pull in opposite directions depending on the role:

        * **Triage wants the smallest.** The whole point of that role is that rating
          twenty snippets does not need a 14B, and the small model's lower latency is the
          benefit being sought.
        * **Every other role wants the largest**, and prefers a model whose *primary* role
          is the one being asked for — a coder model leads on structured work, so it
          should outrank a reasoning model that merely lists structured as a fallback.

            structured, [4B qwen3, 14B coder, 14B qwen3]  ->  [14B coder, 14B qwen3, 4B qwen3]
            triage,     [4B qwen3, 14B coder, 14B qwen3]  ->  [4B qwen3, ...]
        """
        if role is ModelRole.TRIAGE:
            return sorted(candidates, key=lambda p: p.size_mib)
        # `roles[0] == role` is the primary-role test; False sorts before True, so it is
        # negated to put primary matches first.
        return sorted(candidates, key=lambda p: (p.roles[0] is not role, -p.size_mib))

    def explain(self) -> dict[str, Any]:
        """What the router would do right now — for the dashboard and for debugging.

        Worth exposing because a routing decision is invisible in its effects: a call
        simply runs, and nothing on screen says why that model and not another.
        """
        return {
            "default": self._settings.model,
            "loaded": sorted(self._loaded_keys()),
            "installed": [
                {
                    "key": p.key,
                    "params": p.params,
                    "size_mib": p.size_mib,
                    "reasoning": p.is_reasoning,
                    "roles": [r.value for r in p.roles],
                }
                for p in self.profiles()
            ],
            "routes": {task: self.choose(task) for task in self._TASK_ROLES},
        }
