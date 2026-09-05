"""A search provider that drives a real browser.

Every other provider here needs something: Brave needs an API key, SearXNG needs an
instance to point at, and the `ddgs` package needs DuckDuckGo to keep serving an endpoint
it does not document. This one needs only Chromium, which is already installed for the
dashboard's UI verification.

## Why it earns its place

It exists because of a measured ceiling rather than a preference. A research run asking
which merchant-of-record platform onboards a Pakistan-based individual came back with
claims about *payment rails* instead — Raast, Payoneer, SadaPay. The extraction was fine;
the search results never contained the answer. Only 6 of 43 results were rated worth
reading, and the rest were freelancer-payment listicles.

A pipeline's ceiling is set by its search results, not by how well it reads them. So the
useful lever is another way of asking the web, one that fails for different reasons than
the others: a rate limit that stops the HTTP client does not stop a browser, and a page
that requires JavaScript to render its results is invisible to a scraper but ordinary to
Chromium.

## What it costs, and why it is last in the chain

Launching a browser is expensive next to an HTTP request — roughly a second of startup
before any query runs, against milliseconds for an API call. The browser is therefore
started once and reused across queries within a run, and this provider sits **after** the
cheaper ones so it is reached only when they are unavailable or have failed.

## The honest caveats

Scraping a result page depends on that page's markup, so this breaks when the layout
changes — the same fragility as the HTML fallback it sits beside, just harder to detect
because a browser renders *something* either way. The selectors below are therefore
written to fail loudly (zero results, which the service treats as a failure and falls
through) rather than to guess.

Search engines also actively discourage automation. This runs at human-ish rates with a
single browser and no parallel queries, which is the difference between using a tool and
abusing a service. Do not raise the concurrency here to make a run faster.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote_plus

from .search import SearchProvider, SearchResult


class BrowserSearchProvider(SearchProvider):
    """Runs a query in headless Chromium and reads the results off the page.

    Playwright is imported lazily inside `available()` so the toolkit still installs and
    runs without it — the browser extra is large, and a machine that only wants the API
    should not be made to download Chromium.
    """

    name = "browser"  # Bing via headless Chromium

    # Bing, deliberately — **not** DuckDuckGo.
    #
    # The first version of this class pointed at DuckDuckGo's HTML endpoint, which was a
    # design error worth recording. The problem this provider exists to solve is that
    # DuckDuckGo returns weak results for niche commercial questions: a run asking which
    # merchant-of-record platform onboards Pakistani individuals came back with freelancer
    # payment listicles, and only 6 of 43 results were worth reading. DuckDuckGo was not
    # failing; it was answering badly.
    #
    # A fallback that queries the same index cannot fix that. It buys redundancy against a
    # rate limit and nothing else — the same pool, reached a second way. Bing is a
    # genuinely separate crawl and ranking, so it can surface documents the other never
    # had, which is the only thing that raises the pipeline's ceiling.
    #
    # `format=rss` is avoided on purpose: it returns fewer results and drops the snippet,
    # and the snippet is what the ranking stage reads to triage before fetching.
    _SEARCH_URL = "https://www.bing.com/search?q={query}&count=20&setlang=en"

    # A real desktop user agent. The default Playwright agent announces "HeadlessChrome",
    # which is refused or degraded by several engines; this is the difference between
    # getting results and getting a challenge page.
    _USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    _NAV_TIMEOUT_MS = 25_000

    def __init__(self) -> None:
        self._playwright: Any | None = None
        self._browser: Any | None = None
        # Serialises queries. Two searches sharing one browser would interleave page
        # navigations, and beyond that, issuing parallel automated queries at a search
        # engine is the behaviour that gets an address blocked.
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        """Whether Playwright and a browser binary are present.

        Only checks that the module imports. Whether Chromium is actually downloaded is
        not knowable without launching it, and launching costs a second — too expensive
        for a question asked before every search. A missing browser surfaces as a launch
        failure, which the service records as a provider failure and falls through.
        """
        try:
            import playwright.async_api  # noqa: F401

            return True
        except ImportError:
            return False

    async def _ensure_browser(self) -> Any:
        """Start Chromium once and keep it, since startup dominates a short query."""
        if self._browser is not None:
            return self._browser

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            # Disable the automation banner and the features that most obviously mark a
            # browser as scripted. This is about getting ordinary results rather than a
            # degraded page, not about concealing anything from the operator.
            args=["--disable-blink-features=AutomationControlled"],
        )
        return self._browser

    async def close(self) -> None:
        """Shut the browser down. Safe to call twice, and safe if it never started."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """Run one query and return what the page shows.

        Raises rather than returning an empty list when it genuinely cannot run, because
        the service distinguishes "this provider failed, try the next" from "this provider
        worked and the web has nothing" — and only an exception carries the first meaning.
        """
        async with self._lock:
            browser = await self._ensure_browser()
            context = await browser.new_context(
                user_agent=self._USER_AGENT,
                locale="en-US",
                viewport={"width": 1366, "height": 900},
            )
            try:
                page = await context.new_page()
                # `domcontentloaded` rather than `networkidle`: a results page keeps
                # loading trackers and images long after the results themselves are in the
                # DOM, so waiting for the network to go quiet would add seconds per query
                # for nothing.
                await page.goto(
                    self._SEARCH_URL.format(query=quote_plus(query)),
                    wait_until="domcontentloaded",
                    timeout=self._NAV_TIMEOUT_MS,
                )
                return await self._read_results(page, limit)
            finally:
                # The context is closed per query but the browser is kept. A context is
                # cheap and disposing it discards cookies between queries, which stops one
                # search's session state influencing the next.
                await context.close()

    async def _read_results(self, page: Any, limit: int) -> list[SearchResult]:
        """Pull title, url and snippet out of the rendered page.

        Extraction happens inside one `page.evaluate` rather than through per-element
        Playwright locators. Each locator call is a round trip to the browser, so reading
        thirty results three fields at a time would be ninety round trips; this is one.

            <div class="result">
              <a class="result__a" href="https://x">Title</a>
              <a class="result__snippet">Summary…</a>
            </div>
              ->  {"title": "Title", "url": "https://x", "snippet": "Summary…"}
        """
        rows = await page.evaluate(
            """(limit) => {
                const out = [];
                // Bing wraps each organic result in <li class="b_algo">. Ads and the
                // "people also ask" panels use different classes, so selecting this one
                // filters them out rather than needing to recognise and skip them.
                const nodes = document.querySelectorAll('li.b_algo');
                for (const node of nodes) {
                    const link = node.querySelector('h2 a');
                    if (!link) continue;
                    // The caption element holds the snippet; its inner <p> is the text,
                    // but on some layouts the caption itself carries it directly.
                    const caption = node.querySelector('.b_caption p, .b_caption, .b_snippet');
                    out.push({
                        title: (link.textContent || '').trim(),
                        url: link.getAttribute('href') || '',
                        snippet: (caption?.textContent || '').trim(),
                    });
                    if (out.length >= limit) break;
                }
                return out;
            }""",
            limit,
        )

        results: list[SearchResult] = []
        for row in rows:
            url = self._clean_url(str(row.get("url", "")))
            if not url.startswith("http"):
                # Relative links and javascript: handlers are navigation chrome, not
                # results. Skipping keeps a malformed row from reaching the fetcher.
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=str(row.get("title", "")).strip(),
                    snippet=str(row.get("snippet", "")).strip(),
                    # Tagged so a merged pool says which index found each document.
                    # Without it these rows come back with an empty provider and the
                    # question "did adding Bing actually help?" cannot be answered.
                    provider=self.name,
                )
            )
        return results

    @staticmethod
    def _clean_url(href: str) -> str:
        """Unwrap a tracking redirect if the result link is wrapped in one.

        Bing usually returns the destination directly, but click-tracking links do appear,
        with the real target base64-encoded in a `u` parameter. Left wrapped, every result
        would share the bing.com host — so canonical-URL deduplication would collapse them
        all into one, and the fetcher would follow a redirect per page.

            https://www.bing.com/ck/a?...&u=a1aHR0cHM6Ly9wYWRkbGUuY29t...
              ->  https://paddle.com
        """
        import base64
        import binascii
        from urllib.parse import parse_qs, urlparse

        if href.startswith("//"):
            href = f"https:{href}"
        if "bing.com/ck/a" not in href:
            return href

        encoded = parse_qs(urlparse(href).query).get("u", [""])[0]
        # The value carries a two-character prefix ("a1") before the base64 payload.
        if not encoded.startswith("a1"):
            return href
        payload = encoded[2:]
        try:
            # Base64 without padding is common here; restore it before decoding, since
            # `b64decode` raises on a length that is not a multiple of four.
            padded = payload + "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode("utf-8", "replace")
        except (binascii.Error, ValueError):
            # An unwrappable link is returned as-is rather than dropped: a bing.com URL is
            # a poor result but still a real one, and losing it silently would be worse.
            return href
        return decoded if decoded.startswith("http") else href
