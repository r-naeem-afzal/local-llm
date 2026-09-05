"""Run the research pipeline from the command line.

    python scripts/research.py "Which merchant of record onboards Pakistan-based sellers?"
    python scripts/research.py "…" --pages 12        # read more sources, costs more time

Everything runs on the local model, so a run costs GPU time and electricity rather than
plan usage. Expect roughly fifteen seconds per page read, two pages at a time.

Progress is printed as it happens rather than at the end. A run is minutes long, and a
silent process is indistinguishable from a hung one — the same reason the dashboard has a
live panel at all.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from local_llm import Toolkit
from local_llm.pipeline import ResearchReport, build_pipeline


class ResearchCommand:
    """Parses the arguments, runs the pipeline, and reports where the output went.

    A class for the same reason the other scripts here are: the `Toolkit` is built once
    and shared, and the composition root stays the only thing that knows how the pieces
    fit together.
    """

    def __init__(self, toolkit: Toolkit | None = None) -> None:
        self._toolkit = toolkit or Toolkit()
        self._pipeline = build_pipeline(self._toolkit)

    async def run(self, question: str, pages: int, per_query: int) -> ResearchReport:
        # `flush=True` on every line: stdout is block-buffered when redirected to a file,
        # so without it a run piped to a log shows nothing for two minutes and then
        # everything at once, which defeats the point of progress output.
        try:
            return await self._run_inner(question, pages, per_query)
        finally:
            # In a finally so an interrupted or failed run still releases the browser.
            await self._pipeline.aclose()

    async def _run_inner(self, question: str, pages: int, per_query: int) -> ResearchReport:
        report = await self._pipeline.run(
            question,
            max_pages=pages,
            per_query=per_query,
            on_progress=lambda line: print(line, flush=True),
        )
        return report

    def summarise(self, report: ResearchReport) -> None:
        path = self._pipeline.save(report)
        verified = len(report.verified_claims)
        print()
        print(f"  queries      {len(report.queries)}")
        print(f"  pages found  {report.searched}")
        print(f"  pages read   {report.read} ({report.cached} cached)")
        print(f"  claims       {len(report.claims)} ({verified} with a verified quote)")
        print(f"  duration     {report.duration_s:.0f}s")
        if report.failures:
            print(f"  failures     {len(report.failures)}")
        print()
        print(f"  report -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Research a question on local models.")
    parser.add_argument("question", help="The research question, in quotes.")
    parser.add_argument(
        "--pages", type=int, default=8,
        help="Maximum pages to read. Each is ~15s of GPU time, so this is the cost dial.",
    )
    parser.add_argument(
        "--per-query", type=int, default=8,
        help="Search results to request per query, before ranking and deduplication.",
    )
    args = parser.parse_args()

    command = ResearchCommand()
    report = asyncio.run(command.run(args.question, args.pages, args.per_query))
    command.summarise(report)
    # Non-zero when nothing could be established, so a scripted run can tell a thin
    # result from a failed one.
    return 0 if report.claims else 1


if __name__ == "__main__":
    sys.exit(main())
