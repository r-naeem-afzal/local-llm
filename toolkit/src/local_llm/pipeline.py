"""The credit-free research pipeline: question in, cited markdown report out.

This is what the rest of the toolkit was built for. A thorough research run through Claude
spends roughly 55 agents, and the great majority of them do nothing but fetch a page and
pull claims out of it against a fixed schema — mechanical work a local 14B model does
well. So the whole loop runs here, on the GPU, and the metered plan is spent only on
deciding what to do with the result.

    scope  →  search  →  dedupe  →  rank  →  extract  →  verify  →  synthesise

Each stage is a class, and `ResearchPipeline` only sequences them. That split is what lets
a stage be swapped — a better ranker, a different verifier — without touching the others,
and it is why the pipeline itself reads as a list of steps rather than as an algorithm.

## Where the cost actually goes, and what this does about it

Naively, "research" means fetching everything found and asking the model about all of it.
Measured on this machine, one claim extraction is 12-15 seconds of GPU time, so the number
of pages extracted *is* the runtime. Four things keep that bounded, in the order they
matter:

1. **Rank before fetching.** One ranking call triages twenty results; extracting twenty
   pages would be twenty calls of fifteen seconds. The gate costs about 3% of what it
   saves, which is why it comes first.
2. **Deduplicate by canonical URL.** The same article arrives from several queries with
   different tracking suffixes. Without this it is read repeatedly, and worse, the report
   counts one source as several agreeing ones.
3. **Cache extractions across runs.** Research is iterative — the second run of a refined
   question re-encounters most of the same pages. A cached extraction is free.
4. **Verify quotes with string matching, not a model.** Checking that a quoted sentence
   actually appears in the page it is attributed to costs nothing and catches the failure
   that matters most, which is a model inventing supporting evidence.

## What this deliberately does not do

It does not judge. The report presents claims, their sources, and whether each quote
checked out; it does not decide what is true. That is the judgement half of "local models
read, Claude judges", and moving it here would produce a confident local summary that
nobody should trust — the same boundary `mcp_server.py` draws around its tools.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .client import CompletionClient
from .config import Settings
from .extract import ClaimExtractor, PageFetcher, ResultRanker, SourceExtraction
from .search import SearchResult, SearchService, UrlCanonicaliser

# ─────────────────────────── schemas ───────────────────────────


class SearchPlan(BaseModel):
    """The queries a question gets broken into before anything is searched."""

    queries: list[str] = Field(
        default_factory=list,
        # Capped because each query costs a search and adds pages to rank and read. Five
        # angles on a question is thorough; twenty is a way to spend an afternoon of GPU
        # time restating the same question.
        max_length=5,
        description="Distinct web search queries covering different angles of the question.",
    )


@dataclass
class VerifiedClaim:
    """A claim plus the result of checking it against the page it came from."""

    claim: str
    quote: str
    importance: str
    url: str
    source_quality: str
    publish_date: str
    # True when the quote was found verbatim in the fetched page text. The single most
    # useful signal in the whole report, and it costs nothing to produce.
    quote_verified: bool
    extractor: str
    truncated: bool


@dataclass
class ResearchReport:
    """Everything one run produced, in a form both a file and a caller can use."""

    question: str
    started_ts: str
    duration_s: float
    queries: list[str] = field(default_factory=list)
    searched: int = 0
    ranked: int = 0
    read: int = 0
    cached: int = 0
    claims: list[VerifiedClaim] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    markdown: str = ""

    @property
    def verified_claims(self) -> list[VerifiedClaim]:
        return [claim for claim in self.claims if claim.quote_verified]


# ─────────────────────────── stages ───────────────────────────


class QueryPlanner:
    """Turns one research question into several search queries.

    Worth a model call — about three seconds — because search engines answer keywords, not
    questions. Asking "how do Pakistani freelancers receive USD" verbatim returns a thin,
    single-angle result set; asking it as four differently-framed queries is what surfaces
    the primary sources. The alternative, handing the raw question to the engine, is the
    single biggest determinant of a shallow report.
    """

    SYSTEM = (
        "You turn a research question into web search queries. Produce 3-5 short keyword "
        "queries that attack the question from different angles: the direct question, the "
        "official or primary source, a comparison or alternative, and a recent-experience "
        "angle. Use search keywords, not sentences. Never repeat a query."
    )

    def __init__(self, client: CompletionClient) -> None:
        self._client = client

    async def plan(self, question: str) -> list[str]:
        """Return the queries to run, always including the question itself.

            "Which merchant of record works for a Pakistan-based solo dev?"
              ->  ["merchant of record Pakistan solo developer",
                   "Paddle supported countries Pakistan",
                   "Lemon Squeezy vs Paddle payout Payoneer", …]

        The original question is appended as a query of its own regardless of what the
        model produces. That is the guard against a planning step that returns something
        useless: the run then degrades to a plain search rather than searching for
        nothing, which a failed plan would otherwise cause.
        """
        try:
            plan = await self._client.complete(
                [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": f"Research question: {question}"},
                ],
                schema=SearchPlan,
                tool="plan_queries",
                meta={"question": question},
            )
            queries = [q.strip() for q in plan.queries if q.strip()]
        except Exception:
            # A failed plan must not end the run. Searching the question directly is a
            # worse plan, not no plan.
            queries = []

        if question not in queries:
            queries.append(question)
        # Order preserved, duplicates dropped — the model does sometimes restate a query
        # with different capitalisation, and each duplicate is a wasted search.
        seen: set[str] = set()
        unique = []
        for query in queries:
            key = query.lower()
            if key not in seen:
                seen.add(key)
                unique.append(query)
        return unique


class ExtractionCache:
    """Remembers what was extracted from each page, so a re-run is nearly free.

    Research is iterative: the second run of a sharpened question re-encounters most of
    the same pages, and each one costs 12-15 seconds of GPU time to read again for an
    identical answer. Keyed by canonical URL so tracking parameters do not produce misses.

    Stored as one JSON file per page under the data directory rather than in the call
    database, for the same reason live progress is: this is derived data that can be
    rebuilt by re-reading the page, so it does not belong in the durable history and
    should be deletable by removing a folder.
    """

    def __init__(self, directory: Path, canonicaliser: UrlCanonicaliser) -> None:
        self._directory = directory
        self._canonical = canonicaliser
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        """One file per page, named by a filesystem-safe digest of the canonical URL.

            "https://example.com/a?id=7"  ->  <cache>/https___example.com_a_id_7.json

        Hashing would be shorter, but a readable name means the cache can be inspected and
        a single stale entry deleted by hand — which is what you actually want at 2am when
        one page is producing nonsense.
        """
        canonical = self._canonical.canonical(url)
        safe = re.sub(r"[^A-Za-z0-9.]+", "_", canonical)[:120]
        return self._directory / f"{safe}.json"

    def get(self, url: str) -> SourceExtraction | None:
        path = self._path(url)
        try:
            return SourceExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            # Missing, unreadable, or written by an older schema. All three mean the same
            # thing to the caller — read the page again — so none is worth distinguishing.
            return None

    def put(self, url: str, extraction: SourceExtraction) -> None:
        try:
            self._path(url).write_text(extraction.model_dump_json(indent=1), encoding="utf-8")
        except OSError:
            # A cache that cannot be written is a slower pipeline, not a broken one.
            pass


class QuoteVerifier:
    """Checks that a claim's quote actually appears in the page it is attributed to.

    The most valuable check in the pipeline, and it uses no model at all. A language model
    asked for supporting quotes will, under pressure, produce a plausible sentence that
    the document does not contain — and a fabricated quote is far more damaging than a
    missing one, because it reads as evidence. String matching catches exactly that, in
    microseconds, with no possibility of the checker itself hallucinating.

    Matching is deliberately loose about whitespace and quotation marks. Extractors
    normalise both, so a strict comparison would reject honest quotes over a curly
    apostrophe — failing the good case far more often than it catches the bad one.
    """

    # Below this, a "quote" is too short to be evidence of anything and would match by
    # accident in any long document.
    _MIN_QUOTE_CHARS = 12

    # An elision marker: the model joining two separated spans of the source. Measured on a
    # real run, this is the single commonest reason an honest quote failed verification —
    # most extracted quotes contained one, so the whole report came back "unverified" while
    # every fragment was in fact present in the page.
    _ELISION = re.compile(r"\s*(?:\[\s*\.\.\.\s*\]|\[…\]|\.\.\.|…)\s*")

    def verify(self, quote: str, page_text: str) -> bool:
        """Is this quote supported by the page, allowing for elision?

            "Payoneer  supports  USD"          in "…payoneer supports usd…"      ->  True
            "Payoneer is a bridge [...] not a bank"  where both halves appear    ->  True
            "Payoneer is the best"             in a page that never says it      ->  False

        Split on elision markers and require *every* fragment to appear. That is stricter
        than it sounds: a fabricated span cannot pass by hiding between two real ones,
        because each part is checked on its own. What it stops rewarding is the model's
        habit of joining two genuine sentences with an ellipsis, which a literal
        comparison rejects even though nothing was invented.

        Fragments too short to be evidence are dropped rather than failed — a stray "and"
        left beside an ellipsis should not decide the outcome either way — but a quote
        with no usable fragment left is not verified.
        """
        haystack = self._normalise(page_text)
        fragments = [
            normalised
            for fragment in self._ELISION.split(quote)
            if len(normalised := self._normalise(fragment)) >= self._MIN_QUOTE_CHARS
        ]
        if not fragments:
            return False
        return all(fragment in haystack for fragment in fragments)

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, collapse whitespace, and flatten the quote characters.

            "The “best” way\n to  pay"  ->  "the \"best\" way to pay"

        Curly quotes and non-breaking spaces are introduced by publishers' content systems
        and removed inconsistently by extractors, so comparing them literally rejects
        genuine matches for reasons that have nothing to do with the claim.
        """
        flattened = (
            text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace(" ", " ").replace("–", "-").replace("—", "-")
        )
        return re.sub(r"\s+", " ", flattened).strip().lower()


class ReportWriter:
    """Renders a finished run as cited markdown.

    Deterministic string building rather than a model call, on purpose. Synthesis is
    exactly the step where a local model starts smoothing over disagreements between
    sources and producing a confident paragraph that no single source supports. The report
    therefore *presents* the evidence — grouped, cited, and flagged — and leaves the
    judgement to the reader, which is the same boundary the MCP tools draw.
    """

    def write(self, report: ResearchReport) -> str:
        lines: list[str] = [
            f"# {report.question}",
            "",
            f"Researched {report.started_ts} in {report.duration_s:.0f}s on local models — "
            f"no metered plan usage.",
            "",
            "## How this was produced",
            "",
            f"- Queries run: {len(report.queries)}",
            f"- Results found: {report.searched}",
            f"- Results ranked worth reading: {report.ranked}",
            f"- Pages read: {report.read} ({report.cached} from cache)",
            f"- Claims extracted: {len(report.claims)} "
            f"({len(report.verified_claims)} with a quote verified against the page)",
            "",
        ]

        if report.queries:
            lines += ["Search queries:", ""]
            lines += [f"{index}. `{query}`" for index, query in enumerate(report.queries, 1)]
            lines.append("")

        lines += self._claims_section(report)
        lines += self._sources_section(report)

        if report.failures:
            # Failures are printed rather than swallowed. A report built from four pages
            # when eight were selected is a different report, and the reader has to be
            # able to see that rather than infer it from a thin result.
            lines += ["## Pages that could not be read", ""]
            lines += [f"- {failure}" for failure in report.failures]
            lines.append("")

        lines += [
            "---",
            "",
            "Quotes marked **unverified** were not found verbatim in the fetched page. "
            "That usually means the extractor paraphrased, but it can mean the model "
            "invented the quote — treat those claims as unsupported until checked by hand.",
            "",
        ]
        return "\n".join(lines)

    def _claims_section(self, report: ResearchReport) -> list[str]:
        if not report.claims:
            return ["## Findings", "", "No claims were extracted.", ""]

        lines = ["## Findings", ""]
        # Central claims first: they answer the question, and a reader who stops after the
        # first screen should have read the load-bearing evidence rather than the asides.
        order = {"central": 0, "supporting": 1, "tangential": 2}
        for claim in sorted(report.claims, key=lambda c: (order.get(c.importance, 3), c.url)):
            mark = "" if claim.quote_verified else " **unverified**"
            lines += [
                f"### {claim.claim}",
                "",
                f"> {claim.quote}",
                "",
                f"— [{self._host(claim.url)}]({claim.url}) · {claim.importance} · "
                f"{claim.source_quality}"
                + (f" · {claim.publish_date}" if claim.publish_date else "")
                + mark,
                "",
            ]
        return lines

    def _sources_section(self, report: ResearchReport) -> list[str]:
        urls: list[str] = []
        for claim in report.claims:
            if claim.url not in urls:
                urls.append(claim.url)
        if not urls:
            return []
        lines = ["## Sources", ""]
        lines += [f"{index}. <{url}>" for index, url in enumerate(urls, 1)]
        lines.append("")
        return lines

    @staticmethod
    def _host(url: str) -> str:
        """`https://www.paddle.com/billing/pakistan` -> `paddle.com`, for a readable cite."""
        match = re.match(r"https?://(?:www\.)?([^/]+)", url)
        return match.group(1) if match else url


# ─────────────────────────── the pipeline ───────────────────────────


class ResearchPipeline:
    """Sequences the stages. Holds no research logic of its own.

    Every collaborator arrives through the constructor, so a test can drive the whole
    pipeline with a fake search service and a fake model and never touch the network — and
    so the ranker or the verifier can be replaced without this class changing.
    """

    def __init__(
        self,
        settings: Settings,
        search: SearchService,
        planner: QueryPlanner,
        ranker: ResultRanker,
        extractor: ClaimExtractor,
        fetcher: PageFetcher,
        cache: ExtractionCache,
        verifier: QuoteVerifier | None = None,
        writer: ReportWriter | None = None,
        canonicaliser: UrlCanonicaliser | None = None,
    ) -> None:
        self._settings = settings
        self._search = search
        self._planner = planner
        self._ranker = ranker
        self._extractor = extractor
        # Injected rather than reached for through the extractor. The verifier needs the
        # page text, and taking it from `extractor._fetcher` would couple this class to
        # another's private field — the one thing the constructor-injection style here
        # exists to prevent.
        self._fetcher = fetcher
        self._cache = cache
        self._verifier = verifier or QuoteVerifier()
        self._writer = writer or ReportWriter()
        self._canonical = canonicaliser or UrlCanonicaliser()

    async def aclose(self) -> None:
        """Release resources the search providers hold open.

        Specifically the browser provider's Chromium, which is a child process: a run that
        ends without this leaves it resident, and a few runs leave a pile of them. Called
        by the CLI in a `finally`, so an interrupted or failed run cleans up too — the
        cases where a leak is most likely are exactly the ones that skip a happy-path
        cleanup.
        """
        closer = getattr(self._search, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:
                pass

    async def run(self, question: str, max_pages: int = 8,
                  per_query: int = 8,
                  on_progress: Callable[[str], None] | None = None) -> ResearchReport:
        """Research `question` end to end and return the report.

        `max_pages` is the real cost dial. Each page is one model call of roughly 12-15
        seconds, so eight pages is about two minutes of GPU time. It is a hard cap rather
        than a target: the ranker may select fewer, and that is a good outcome, not a
        shortfall to be padded out.

        `on_progress` is called with one line per stage. The pipeline is minutes long and
        silent by default, and a run that prints nothing is indistinguishable from a hung
        one — the same reason `/live` exists.
        """
        started = time.perf_counter()
        report = ResearchReport(
            question=question,
            started_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_s=0.0,
        )
        say = on_progress or (lambda _message: None)

        # ── scope ──
        say("planning queries…")
        report.queries = await self._planner.plan(question)
        say(f"  {len(report.queries)} queries: " + "; ".join(report.queries))

        # ── search ──
        results = await self._gather_results(report, per_query, say)
        if not results:
            report.failures.append("no search results for any query")
            return self._finish(report, started)

        # ── rank ──
        selected = await self._select(question, results, max_pages, say)
        report.ranked = len(selected)
        if not selected:
            report.failures.append("ranking selected no results worth reading")
            return self._finish(report, started)

        # ── read and verify ──
        say(f"reading {len(selected)} pages (concurrency {self._settings.concurrency})…")
        await self._read_all(question, selected, report, say)

        return self._finish(report, started)

    async def _gather_results(self, report: ResearchReport, per_query: int,
                              say: Callable[[str], None]) -> list[SearchResult]:
        """Run every query and pool the hits, deduplicated across queries.

        Sequential rather than concurrent on purpose. These are third-party endpoints and
        the free one is scraped; firing five simultaneous requests at it is how an IP
        starts getting rate-limited, which would cost far more than the seconds saved.
        """
        pooled: list[SearchResult] = []
        for query in report.queries:
            try:
                hits = await self._search.search(query, limit=per_query)
            except Exception as exc:
                report.failures.append(f"search failed for {query!r}: {exc}")
                continue
            pooled.extend(hits)
            say(f"  {len(hits)} results for {query!r}")

        deduped = self._canonical.deduplicate(pooled)
        report.searched = len(deduped)
        say(f"  {len(pooled)} results, {len(deduped)} distinct pages")
        return deduped

    async def _select(self, question: str, results: list[SearchResult],
                      limit: int, say: Callable[[str], None]) -> list[SearchResult]:
        """Ask the model which results are worth the expensive step.

        The cheap gate: one call triages the whole pool, where reading them all would be
        one call per page. If the ranker fails, fall back to the search engines' own order
        — which is a real ranking, just a worse one — rather than abandoning the run.
        """
        say(f"ranking {len(results)} results…")
        by_url = {r.url: r for r in results}
        try:
            ranking = await self._ranker.rank(
                question, [r.model_dump() for r in results]
            )
        except Exception as exc:
            say(f"  ranking failed ({exc}); falling back to search order")
            return results[:limit]

        # High relevance first, then medium. Low is dropped entirely: the ranker is
        # explicitly told to mark SEO spam and content farms low, and spending fifteen
        # seconds of GPU time reading a content farm is the exact waste this gate exists
        # to prevent.
        chosen: list[SearchResult] = []
        for wanted in ("high", "medium"):
            for row in ranking.results:
                if row.relevance != wanted:
                    continue
                original = by_url.get(row.url)
                if original is not None and original not in chosen:
                    chosen.append(original)
                if len(chosen) >= limit:
                    return chosen
        say(f"  {len(chosen)} results rated high or medium")
        return chosen

    async def _read_all(self, question: str, selected: list[SearchResult],
                        report: ResearchReport, say: Callable[[str], None]) -> None:
        """Extract claims from every selected page, two at a time.

        The semaphore is the GPU. There is one card and one resident model, so a third
        concurrent request does not go faster — it queues behind the other two while
        consuming context. `settings.concurrency` defaults to 2 for exactly this reason,
        and is honoured here rather than at the model client, so the limit stays visible
        at the place that decides how much work to create.
        """
        gate = asyncio.Semaphore(self._settings.concurrency)

        async def read(result: SearchResult) -> None:
            async with gate:
                await self._read_one(question, result, report, say)

        await asyncio.gather(*(read(result) for result in selected))

    async def _read_one(self, question: str, result: SearchResult,
                        report: ResearchReport, say: Callable[[str], None]) -> None:
        cached = self._cache.get(result.url)
        if cached is not None:
            report.cached += 1
            say(f"  cached: {result.url[:80]}")
            # Still fetch the page, even though the extraction is cached. The expensive
            # part of reading a page is the 12-15 second model call, not the one-second
            # HTTP request — and skipping the fetch meant every cached claim arrived with
            # no text to check its quote against, so it was recorded as unverified. A
            # second run of the same question therefore *lost* verification it had earned
            # the first time, which made the cache look like a quality regression.
            self._collect(cached, report, await self._page_text(result.url))
            return

        try:
            extraction = await self._extractor.extract(result.url, question)
        except Exception as exc:
            # One unreadable page — a paywall, a timeout, a 403 — must not end a run that
            # has already paid for a plan, five searches and a ranking.
            report.failures.append(f"{result.url}: {type(exc).__name__}: {exc}")
            say(f"  failed: {result.url[:70]} ({type(exc).__name__})")
            return

        report.read += 1
        say(f"  read {result.url[:70]} — {len(extraction.claims)} claims")
        self._cache.put(result.url, extraction)

        # Re-fetch the page text for verification only if there are claims to check. The
        # fetcher caches nothing, so this is a second HTTP request — cheap next to the
        # model call, and the alternative is threading page text through the extractor's
        # return type purely for this one check.
        page_text = await self._page_text(result.url) if extraction.claims else None
        self._collect(extraction, report, page_text)

    async def _page_text(self, url: str) -> str | None:
        """Fetch a page purely so its quotes can be checked.

        A second HTTP request for a page that was just read, which is a deliberate trade:
        threading the page text back through the extractor's return type would widen that
        type for one consumer's benefit, and the fetch is roughly a second against a model
        call of fifteen.

        Returns None on any failure, because verification is a bonus check rather than a
        requirement — losing it downgrades claims to "unverified", which is the honest
        outcome, not a reason to discard a page that was read successfully.
        """
        try:
            return (await self._fetcher.fetch(url)).text
        except Exception:
            return None

    def _collect(self, extraction: SourceExtraction, report: ResearchReport,
                 page_text: str | None) -> None:
        """Turn one page's extraction into verified claims on the report.

        When `page_text` is None — a cached extraction, or a re-fetch that failed — the
        quote cannot be checked, and it is recorded as unverified rather than assumed good.
        Erring that way is the whole point: an unverified claim shown as verified is the
        one outcome this check exists to prevent.
        """
        for claim in extraction.claims:
            report.claims.append(
                VerifiedClaim(
                    claim=claim.claim,
                    quote=claim.quote,
                    importance=claim.importance,
                    url=extraction.url,
                    source_quality=extraction.source_quality,
                    publish_date=extraction.publish_date,
                    quote_verified=(
                        self._verifier.verify(claim.quote, page_text)
                        if page_text is not None
                        else False
                    ),
                    extractor=extraction.extractor,
                    truncated=extraction.truncated,
                )
            )

    def _finish(self, report: ResearchReport, started: float) -> ResearchReport:
        report.duration_s = time.perf_counter() - started
        report.markdown = self._writer.write(report)
        return report

    def save(self, report: ResearchReport, directory: Path | None = None) -> Path:
        """Write the report to disk and return where it went.

        Named by timestamp and a slug of the question, so a directory of runs is browsable
        and two runs of the same question do not overwrite each other — an earlier run is
        often the thing you want to compare against.
        """
        target = directory or (self._settings.data_dir / "research")
        target.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", report.question.lower()).strip("-")[:60]
        stamp = report.started_ts.replace(":", "").replace("-", "")[:15]
        path = target / f"{stamp}-{slug}.md"
        path.write_text(report.markdown, encoding="utf-8")
        # The structured form goes beside it, because the markdown is for reading and the
        # JSON is for a later run that wants to compare or re-verify without re-parsing
        # prose.
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "question": report.question,
                    "started_ts": report.started_ts,
                    "duration_s": report.duration_s,
                    "queries": report.queries,
                    "claims": [vars(c) for c in report.claims],
                    "failures": report.failures,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return path


def build_pipeline(toolkit: Any) -> ResearchPipeline:
    """Assemble the pipeline from an existing `Toolkit`.

    A function rather than another `cached_property` on `Toolkit`, because the pipeline is
    the one component that is genuinely optional: it needs the search extra, it is not
    used by the dashboard or the MCP server, and building it eagerly would pull search
    dependencies into every process that merely wants to check whether the model server is
    up.
    """
    return ResearchPipeline(
        settings=toolkit.settings,
        search=toolkit.search,
        planner=QueryPlanner(toolkit.client),
        ranker=toolkit.result_ranker,
        extractor=toolkit.claim_extractor,
        fetcher=toolkit.page_fetcher,
        cache=ExtractionCache(
            toolkit.settings.data_dir / "extraction-cache", toolkit.url_canonicaliser
        ),
        canonicaliser=toolkit.url_canonicaliser,
    )
