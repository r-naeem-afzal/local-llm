"""Smoke test the toolkit against whatever local model is currently loaded.

Run from the package root, with the local server up and a model loaded:

    lms server start            # run twice if the daemon reports a timeout
    lms ps                      # confirm something is loaded
    python scripts/smoke_test.py

Exercises, in order: the configured database, a plain completion, a typed
structured-output extraction against a real page, and the observability that all of them
should have recorded. Exits non-zero if any stage fails, so it doubles as a post-change
check.

One deliberate choice worth noting: the reasoning stage uses a **large** token budget.
An earlier version of this test passed with the 2048-token default and hid a real bug —
a reasoning model spends that budget on thinking before it answers, so a tight budget
produces an empty answer that looks like a model failure. Testing near the limits rather
than in the comfortable middle is what surfaced it.
"""

from __future__ import annotations

import asyncio
import sys
import time

import local_llm
from local_llm import LocalLLMError, Toolkit

TEST_URL = "https://en.wikipedia.org/wiki/Payoneer"
TEST_QUESTION = "How does Payoneer let freelancers in Pakistan receive USD payments?"


class SmokeTest:
    """Runs each stage in order, reporting as it goes.

    A class so the wired-up `Toolkit` is built once and shared by every stage, which is
    also the point being tested: the composition root should be the only thing that knows
    how the pieces fit together.
    """

    def __init__(self, toolkit: Toolkit | None = None) -> None:
        self._toolkit = toolkit or Toolkit()
        self._failures: list[str] = []

    @property
    def settings(self):
        return self._toolkit.settings

    def report_environment(self) -> None:
        settings = self.settings
        database = settings.database
        print(f"local_llm {local_llm.__version__}")
        print(f"  endpoint : {settings.url}")
        print(f"  model    : {settings.model}")
        print(f"  engine   : {database.engine}")
        # For SQLite the interesting fact is the file path; for a server engine it is
        # where we are connecting, so each prints what is actually useful.
        if self._toolkit.backend.name == "sqlite":
            print(f"  database : {settings.db_path}")
        else:
            print(f"  database : {database.user}@{database.host}:{database.port}/{database.name}")
        print()

    async def run(self) -> int:
        self.report_environment()
        await self.stage_plain()
        page = await self.stage_fetch()
        if page is not None:
            await self.stage_extract(page)
        self.stage_store()

        if self._failures:
            print("\nFAILED:")
            for failure in self._failures:
                print(f"  - {failure}")
            # A non-zero exit is what makes this usable as a gate in a script, rather
            # than something whose output a human has to read and interpret.
            return 1
        print("\nAll stages passed.")
        return 0

    async def stage_plain(self) -> None:
        """Cheapest possible call: proves the server is reachable and answering."""
        started = time.monotonic()
        try:
            text = await self._toolkit.client.complete(
                [{"role": "user", "content": "Reply with three words about SQLite."}],
                tool="smoke_plain",
                # Generous on purpose: if the loaded model is a reasoning model, a small
                # budget is consumed by thinking and the answer comes back empty.
                max_tokens=1500,
            )
            print(f"[plain]   {time.monotonic() - started:5.1f}s  {text[:60]!r}")
        except LocalLLMError as exc:
            self._failures.append(f"plain completion: {exc}")
            print(f"[plain]   FAILED: {exc}")

    async def stage_fetch(self):
        """Fetch a real page, and report which extractor handled it.

        The extractor name matters: `trafilatura` means the article body was isolated,
        while `regex` means the fallback ran and the model will also be reading
        navigation and footer text.
        """
        try:
            page = await self._toolkit.page_fetcher.fetch(TEST_URL)
            print(f"[fetch]   extractor={page.extractor} chars={len(page.text)} "
                  f"truncated={page.truncated}")
            return page
        except Exception as exc:
            self._failures.append(f"page fetch: {exc}")
            print(f"[fetch]   FAILED: {exc}")
            return None

    async def stage_extract(self, page) -> None:
        """The real work: structured output validated against a pydantic schema."""
        started = time.monotonic()
        try:
            result = await self._toolkit.claim_extractor.extract(TEST_URL, TEST_QUESTION)
            print(f"[extract] {time.monotonic() - started:5.1f}s  "
                  f"quality={result.source_quality} claims={len(result.claims)}")
            for claim in result.claims:
                print(f"          - [{claim.importance}] {claim.claim[:76]}")

            # Asserting the *type*, not just that something came back. This is what
            # proves schema-constrained generation and validation actually ran: a dict
            # here would mean the reply was accepted unvalidated.
            if result.claims and not isinstance(result.claims[0], local_llm.Claim):
                self._failures.append("claims[0] is not a Claim instance")
        except LocalLLMError as exc:
            self._failures.append(f"claim extraction: {exc}")
            print(f"[extract] FAILED: {exc}")

    def stage_store(self) -> None:
        """Confirm the calls above were actually recorded, with usage and timing."""
        try:
            totals = self._toolkit.repository.stats()["totals"]
            size = self._toolkit.repository.size_bytes()
            print(f"\n[store]   calls={totals.get('calls')} errors={totals.get('errors')} "
                  f"tokens={totals.get('tokens_in')}/{totals.get('tokens_out')} "
                  f"size={size}B")
            for record in self._toolkit.repository.list_calls(limit=3):
                print(f"          {record.tool:<16} {record.status:<7} "
                      f"{record.duration_ms}ms {record.tokens_in}/{record.tokens_out}")

            if not totals.get("calls"):
                self._failures.append("no calls recorded — observability is not working")
        except Exception as exc:
            self._failures.append(f"store readback: {exc}")
            print(f"[store]   FAILED: {exc}")


if __name__ == "__main__":
    sys.exit(asyncio.run(SmokeTest().run()))
