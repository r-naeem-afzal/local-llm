"""Web search behind one interface, with interchangeable providers.

The research pipeline needs to find sources before it can read them, and every option for
doing that has a different failure mode: an API needs a key and has a quota, a self-hosted
instance needs to be running, and a free scraped endpoint breaks whenever the site changes
its markup. Picking one and hard-coding it would mean the whole pipeline stops the day
that one choice fails.

So this is a small hierarchy with a chooser in front, mirroring `extract.py`:

    SearchProvider (abstract)
      ├── BraveSearchProvider       paid API, best quality, needs a key
      ├── SearxngSearchProvider     self-hosted metasearch, needs a running instance
      └── DuckDuckGoProvider        free, no key — the one that always works
    SearchService                   picks a provider, falls back, records the search

Adding a provider means adding a class and listing it, not editing a chain of `if`s.

## Why searches are recorded like model calls

A search is not a model call — no GPU, no tokens — but it is a step in a pipeline whose
whole point is being auditable. Recording it in the same table means one timeline shows
"searched, ranked, fetched, extracted" in order, instead of the searches being invisible
and the ranking appearing to come from nowhere. The rows carry `model=""` so they cannot
be mistaken for inference, and the raw hits go in the payload so a bad extraction can be
traced back to the result that produced it.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

from .config import Settings
from .store import CallRepository


class SearchResult(BaseModel):
    """One hit, in the shape the ranker and the pipeline expect."""

    url: str
    title: str
    snippet: str = ""
    # Which provider produced it, kept because result quality differs sharply between
    # them and a disappointing run should be traceable to its source.
    provider: str = ""


class SearchError(RuntimeError):
    """Every provider failed. Raised rather than returning an empty list, because
    "nothing was found" and "nothing could be asked" demand different responses: the
    first is a real answer, the second means the run should stop."""


# ─────────────────────────── URL canonicalisation ───────────────────────────


class UrlCanonicaliser:
    """Reduces a URL to a comparable form, so the same page is not read twice.

    Its own class because deduplication is where a research run quietly wastes most of its
    budget. The same article arrives from three providers with three different tracking
    suffixes; without this, each one is fetched, extracted and synthesised separately, and
    the report then counts one source as three pieces of corroborating evidence. That is
    worse than the wasted time — it manufactures false agreement.
    """

    # Parameters that never change which document is returned. Stripped so the same page
    # with different campaign tags compares equal.
    _TRACKING_PREFIXES = ("utm_", "ga_", "mc_")
    _TRACKING_EXACT = frozenset({
        "fbclid", "gclid", "msclkid", "igshid", "ref", "ref_src", "source",
        "spm", "cmpid", "campaign_id", "_hsenc", "_hsmi", "mkt_tok",
    })

    def canonical(self, url: str) -> str:
        """Normalise for comparison only — never for fetching.

            "https://WWW.Example.com/a/?utm_source=x&id=7#top"
              ->  "https://example.com/a?id=7"

        The fragment goes because it selects a position within a document, not a different
        document. The trailing slash goes because `/a` and `/a/` are the same page in
        practice. `www.` goes because almost every site serves both. Query parameters that
        are *not* tracking are kept, since `?id=7` genuinely selects a different page and
        dropping it would merge unrelated documents into one.
        """
        try:
            parsed = urlparse(url.strip())
        except ValueError:
            # Unparseable: return it unchanged so it is still compared as itself rather
            # than colliding with every other unparseable URL under a shared empty string.
            return url

        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]

        kept = [
            (key, value)
            for key, value in parse_qs(parsed.query, keep_blank_values=True).items()
            for value in value
            if not self._is_tracking(key)
        ]
        query = "&".join(f"{key}={value}" for key, value in sorted(kept))

        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            path = ""

        return urlunparse((parsed.scheme.lower(), host, path, "", query, ""))

    def _is_tracking(self, key: str) -> bool:
        lowered = key.lower()
        return lowered in self._TRACKING_EXACT or lowered.startswith(self._TRACKING_PREFIXES)

    def deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Keep the first occurrence of each distinct page, preserving order.

            [a?utm_source=x, a, b]  ->  [a?utm_source=x, b]

        First rather than best, because results arrive in relevance order — the earliest
        appearance is the highest-ranked one, and the later duplicate carries no extra
        information beyond the fact that it was found twice.
        """
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for result in results:
            key = self.canonical(result.url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
        return unique


# ─────────────────────────── providers ───────────────────────────


class SearchProvider(ABC):
    """One way of asking the web a question.

    `available()` is separate from `search()` on purpose. Whether a provider *can* run is
    a question about configuration — is there a key, is there a URL — and it must be
    answerable without making a network request, or the chooser would have to try each
    provider in turn and wait for a timeout to discover that one was never configured.
    """

    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool:
        """Whether this provider is configured enough to be worth trying."""

    @abstractmethod
    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """Run the search. Raises rather than returning [] when it cannot run."""


class BraveSearchProvider(SearchProvider):
    """Brave's Search API. The best quality here, and the only one that costs money.

    First choice when a key is present: it is a real index with a documented response
    shape, so it neither breaks on a markup change nor depends on a server staying up.
    """

    name = "brave"
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def available(self) -> bool:
        return bool(self._settings.brave_api_key)

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """Ask Brave, and map its response onto SearchResult.

            {"web": {"results": [{"title": "Payoneer", "url": "https://…",
                                  "description": "<strong>Payoneer</strong> lets…"}]}}
              ->  [SearchResult(title="Payoneer", url="https://…",
                                snippet="Payoneer lets…")]

        Note the description contains HTML — Brave marks the matched terms with `<strong>`
        — so it is stripped. Left in, those tags reach the ranking model as literal text
        and it starts treating markup as content.
        """
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._settings.brave_api_key,
        }
        # `count` is capped at 20 by the API; asking for more is rejected rather than
        # silently truncated, so it is clamped here instead of failing the whole search.
        params = {"q": query, "count": min(limit, 20)}

        async with httpx.AsyncClient(timeout=self._settings.fetch_timeout_s) as http:
            response = await http.get(self._ENDPOINT, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()

        results = (payload.get("web") or {}).get("results") or []
        return [
            SearchResult(
                url=item.get("url", ""),
                title=_strip_tags(item.get("title", "")),
                snippet=_strip_tags(item.get("description", "")),
                provider=self.name,
            )
            for item in results
            if item.get("url")
        ]


class SearxngSearchProvider(SearchProvider):
    """A self-hosted SearXNG instance, which aggregates other engines.

    Second choice: free and unlimited because it is your own server, but only as available
    as that server. It returns JSON, so unlike the scraped endpoint below it does not
    break when someone redesigns a results page.
    """

    name = "searxng"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def available(self) -> bool:
        return bool(self._settings.searxng_url)

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        base = self._settings.searxng_url.rstrip("/")
        params = {"q": query, "format": "json"}

        async with httpx.AsyncClient(timeout=self._settings.fetch_timeout_s) as http:
            response = await http.get(f"{base}/search", params=params)
            response.raise_for_status()
            payload = response.json()

        # `content` is SearXNG's name for what everything else calls a snippet. Renamed
        # here so the rest of the package never has to know which provider it came from.
        return [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                provider=self.name,
            )
            for item in (payload.get("results") or [])[:limit]
            if item.get("url")
        ]


class DuckDuckGoBackend(ABC):
    """One way of reaching DuckDuckGo. Two exist because both are fragile in
    different ways, and the pipeline needs at least one of them to work."""

    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def fetch(self, query: str, limit: int) -> list[SearchResult]: ...


class DdgsPackageBackend(DuckDuckGoBackend):
    """The maintained `ddgs` library, when it is installed.

    Preferred over scraping because it is somebody else's job to keep it working: when
    DuckDuckGo changes something, the fix arrives as a version bump rather than as a
    debugging session here.

    It is synchronous, so it runs in a worker thread — see `fetch`.
    """

    name = "ddgs"

    def __init__(self) -> None:
        try:
            from ddgs import DDGS

            self._ddgs: Any | None = DDGS
        except ImportError:
            try:
                # The package was renamed; older environments still have the old name.
                # Both are tried so an existing install keeps working after an upgrade.
                from duckduckgo_search import DDGS  # type: ignore[no-redef]

                self._ddgs = DDGS
            except ImportError:
                self._ddgs = None

    def available(self) -> bool:
        return self._ddgs is not None

    async def fetch(self, query: str, limit: int) -> list[SearchResult]:
        """Run the blocking library call off the event loop.

        `asyncio.to_thread` matters here rather than being a formality: the pipeline runs
        several stages concurrently, and a synchronous network call made directly would
        block the whole event loop for its duration — freezing the live-progress reporting
        that the dashboard is reading at the same time.
        """
        import asyncio

        def run() -> list[dict[str, Any]]:
            with self._ddgs() as client:
                return list(client.text(query, max_results=limit))

        rows = await asyncio.to_thread(run)
        # The library's keys have changed across versions, so each field is looked up
        # under both spellings. A rename would otherwise produce results with empty
        # titles, which looks like DuckDuckGo returning junk rather than a version skew.
        return [
            SearchResult(
                url=row.get("href") or row.get("url") or "",
                title=row.get("title") or "",
                snippet=row.get("body") or row.get("description") or "",
                provider="duckduckgo",
            )
            for row in rows
            if row.get("href") or row.get("url")
        ]


class DdgsHtmlBackend(DuckDuckGoBackend):
    """The no-JavaScript HTML endpoint, parsed directly. Always available.

    Worse than the library and kept anyway, because it is the only path that needs no key,
    no server and no dependency. When it breaks the pipeline has nothing left, so it is
    the last line rather than the first.
    """

    name = "ddg-html"
    _ENDPOINT = "https://html.duckduckgo.com/html/"

    # Result anchors carry `class="result__a"`; snippets carry `class="result__snippet"`.
    # Written to tolerate extra attributes and either quote style, because those change
    # without the structure changing.
    _RESULT = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    _SNIPPET = re.compile(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<text>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    def available(self) -> bool:
        return True

    async def fetch(self, query: str, limit: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
            response = await http.post(
                self._ENDPOINT,
                data={"q": query},
                headers={
                    # A plain desktop user-agent. The endpoint answers a bot challenge
                    # instead of results without one.
                    "user-agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) local-llm-toolkit/1.0"
                    ),
                    "accept": "text/html,application/xhtml+xml",
                    # Sent because the form posts from this page; some deployments reject
                    # a POST whose origin does not match.
                    "referer": "https://duckduckgo.com/",
                },
            )
            response.raise_for_status()
            html = response.text

        titles = list(self._RESULT.finditer(html))
        snippets = [_strip_tags(m.group("text")) for m in self._SNIPPET.finditer(html)]

        results: list[SearchResult] = []
        for index, match in enumerate(titles[:limit]):
            results.append(
                SearchResult(
                    url=self._real_url(match.group("href")),
                    title=_strip_tags(match.group("title")),
                    # Positional pairing, because the snippet is a sibling element with no
                    # id linking it to its result. Guarded with a length check: a results
                    # page with a missing snippet would otherwise shift every later
                    # snippet onto the wrong result, which is worse than having none.
                    snippet=snippets[index] if index < len(snippets) else "",
                    provider="duckduckgo",
                )
            )
        return results

    @staticmethod
    def _real_url(href: str) -> str:
        """Unwrap DuckDuckGo's redirector.

            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=…"
              ->  "https://example.com/a"

        Necessary rather than cosmetic: left wrapped, every result would be fetched
        *through* DuckDuckGo, deduplication would compare redirector URLs instead of
        pages, and the report would cite duckduckgo.com as the source of every claim.
        """
        if "uddg=" not in href:
            return href
        try:
            query = urlparse(href if href.startswith("http") else f"https:{href}").query
            target = parse_qs(query).get("uddg", [""])[0]
            return unquote(target) or href
        except (ValueError, IndexError):
            return href


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo through whichever backend is working, library first.

    The chain exists because this is the fallback everything else depends on. If the
    library is absent the scraper covers it; if the scraper's markup assumptions break the
    library covers it. Both failing at once is possible, and that is exactly when
    `SearchService` should say so loudly rather than return an empty list.
    """

    name = "duckduckgo"

    def __init__(self, backends: list[DuckDuckGoBackend] | None = None) -> None:
        self._backends = backends or [DdgsPackageBackend(), DdgsHtmlBackend()]

    def available(self) -> bool:
        return any(backend.available() for backend in self._backends)

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        errors: list[str] = []
        for backend in self._backends:
            if not backend.available():
                continue
            try:
                results = await backend.fetch(query, limit)
            except Exception as exc:
                # Caught broadly on purpose: a third-party library and a scraper can raise
                # almost anything, and the point of a chain is that one backend's failure
                # is the next one's cue rather than the end of the run.
                errors.append(f"{backend.name}: {type(exc).__name__}: {exc}")
                continue
            if results:
                return results
            errors.append(f"{backend.name}: no results")
        raise SearchError("; ".join(errors) or "no DuckDuckGo backend available")


# ─────────────────────────── the service ───────────────────────────


class SearchService:
    """Chooses a provider, falls back when one fails, and records what was asked.

    The ordering policy lives here and nowhere else: best quality first, free-and-always-
    available last. `settings.search_provider` pins one by name for when you want to know
    which you are testing; `"auto"` walks the list.
    """

    def __init__(self, settings: Settings, providers: list[SearchProvider] | None = None,
                 repository: CallRepository | None = None,
                 canonicaliser: UrlCanonicaliser | None = None) -> None:
        self._settings = settings
        # Quality order. Brave is a real index; SearXNG aggregates several; DuckDuckGo
        # scraped is the floor. A run should get the best available rather than the
        # first one someone happened to configure.
        self._providers = providers or [
            BraveSearchProvider(settings),
            SearxngSearchProvider(settings),
            DuckDuckGoProvider(),
        ]
        self._repository = repository
        self._canonical = canonicaliser or UrlCanonicaliser()

    def chosen(self) -> list[SearchProvider]:
        """The providers to try, in order, given the configuration.

            search_provider="auto",  no keys  ->  [DuckDuckGoProvider]
            search_provider="brave", no key   ->  []   (and the search then fails loudly)

        Returning an empty list rather than silently falling back on a pinned provider is
        deliberate: someone who named a provider wants that one, and quietly using another
        would make a quota problem or a bad key look like a quality problem.
        """
        wanted = (self._settings.search_provider or "auto").lower()
        if wanted != "auto":
            return [p for p in self._providers if p.name == wanted and p.available()]
        return [p for p in self._providers if p.available()]

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search, deduplicate, and record the attempt.

        Returns results from the first provider that produces any. The failures leading up
        to it are recorded in the call row's metadata rather than raised, so a run that
        silently degraded from Brave to DuckDuckGo is visible afterwards — otherwise the
        only symptom is a report built on weaker sources with no explanation.
        """
        providers = self.chosen()
        if not providers:
            raise SearchError(
                f"no search provider available (search_provider="
                f"{self._settings.search_provider!r}); set LOCAL_LLM_BRAVE_API_KEY, "
                f"LOCAL_LLM_SEARXNG_URL, or install ddgs"
            )

        started = time.perf_counter()
        failures: list[str] = []

        for provider in providers:
            try:
                raw = await provider.search(query, limit)
            except Exception as exc:
                failures.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue

            results = self._canonical.deduplicate(raw)[:limit]
            if not results:
                failures.append(f"{provider.name}: no results")
                continue

            self._record(query, provider.name, results, failures, started)
            return results

        self._record(query, "none", [], failures, started, failed=True)
        raise SearchError("; ".join(failures))

    def _record(self, query: str, provider: str, results: list[SearchResult],
                failures: list[str], started: float, failed: bool = False) -> None:
        """Write the search into the same history table as model calls.

        Wrapped so a storage problem cannot fail a search that already succeeded — the
        record is for observability, and observability must never be able to break the
        thing it observes.

        `model=""` because no model ran. That is what keeps searches out of the
        tokens-per-model statistics while still placing them on the one timeline.
        """
        if self._repository is None:
            return
        try:
            now = datetime.now(timezone.utc).isoformat()
            call_id = self._repository.next_call_id("search-")
            self._repository.start_call(
                ts=now,
                tool=f"search:{provider}",
                model="",
                meta={"query": query, "provider": provider, "fallbacks": failures},
                prompt=[{"role": "query", "content": query}],
                call_id=call_id,
            )
            self._repository.finish_call(
                call_id,
                status="error" if failed else "ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                error="; ".join(failures) if failed else None,
                # The hits go in the payload so a questionable claim can be traced back to
                # the result that led to the page it came from.
                response=json.dumps([r.model_dump() for r in results], indent=1),
                tool=f"search:{provider}",
                model="",
                ts=now,
                meta={"query": query, "provider": provider, "results": len(results)},
            )
        except Exception:
            pass


def _strip_tags(text: str) -> str:
    """Remove markup and decode the few entities that appear in search snippets.

        "<strong>Payoneer</strong> &amp; friends"  ->  "Payoneer & friends"

    Providers mark matched terms with tags inside titles and descriptions. Passed through,
    those tags reach the ranking model as literal text and it begins scoring markup as
    though it were content.
    """
    without_tags = re.sub(r"<[^>]+>", "", text)
    for entity, char in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
        ("&#39;", "'"), ("&#x27;", "'"), ("&nbsp;", " "),
    ):
        without_tags = without_tags.replace(entity, char)
    return re.sub(r"\s+", " ", without_tags).strip()
