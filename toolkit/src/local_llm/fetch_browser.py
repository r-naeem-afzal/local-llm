"""Fetching a page through a real browser, for content JavaScript builds after load.

Its own module, mirroring `search_browser.py`, for one practical reason: Playwright is a
heavy optional dependency and a Chromium launch costs roughly a second. Nothing that only
wants an HTTP fetch should import this, and `extract.py` therefore does not — the browser
source is injected into `PageFetcher` by the composition root or not at all.

## Why this exists

Measured 2026-09-08. Paddle's supported-countries page builds its country table in
JavaScript. Fetched over HTTP and passed through trafilatura it yields 2,715 characters
containing zero occurrences of "Pakistan" or "PK" — and yields them *successfully*, at a
length that looks entirely normal for a documentation page. Rendered in a browser the same
URL yields 5,287 characters including the whole list.

So the failure being defended against is not an error. It is a page that reports success
while silently omitting the only part anyone wanted, after which a language model is asked
a question the text cannot answer and produces a confident answer anyway — in both
directions on different runs, as happened here.

## The lifecycle problem, and why the browser is kept open

Launching Chromium takes about a second and starting it per page would dominate the cost of
fetching several. So one browser is launched on first use and reused, which makes this
class stateful and means it must be closed. `aclose()` exists for that, and the research
pipeline already has an `aclose()` chain for the browser search provider — this hooks into
the same discipline. Leaking a Chromium process per run is the failure if it is skipped:
they do not exit on their own and each holds a few hundred megabytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .extract import ExtractorChain, Page, PageSource

# Tags whose text is a heading for whatever follows it. `dt`, `th`, `caption`, `summary`
# and `legend` are included alongside h1-h6 because real documentation labels its lists
# with them at least as often — Lemon Squeezy's payout list is introduced by a paragraph
# heading, and pricing tables label their columns with `th`.
HEADING_TAGS = frozenset(
    {"h1", "h2", "h3", "h4", "h5", "h6", "dt", "th", "caption", "summary", "legend"}
)

# JavaScript that walks the rendered document and returns ordered blocks of text with the
# tag that produced each. Run in the page rather than reconstructed from `inner_text`,
# because the browser is the only thing that actually knows the document structure.
#
# The reason this exists: `inner_text` flattens a list into one bare line per item, so a
# line-based guess at "which of these lines is a heading" cannot tell a country name from
# a section title. Asked which heading "Pakistan" sat under, that guess answered "Oman" —
# the country immediately above it. Tags remove the guesswork entirely.
#
# Only leaf elements emit text, so a paragraph inside a div is not reported twice.
_BLOCKS_SCRIPT = """
() => {
  const out = [];
  const headingTags = new Set(
    ['h1','h2','h3','h4','h5','h6','dt','th','caption','summary','legend']);
  const walk = (el) => {
    for (const child of el.children) {
      const tag = child.tagName.toLowerCase();
      if (tag === 'script' || tag === 'style' || tag === 'noscript') continue;
      const text = (child.innerText || '').trim();
      if (headingTags.has(tag)) {
        if (text) out.push({tag: tag, text: text});
      } else if (child.children.length === 0) {
        if (text) out.push({tag: tag, text: text});
      } else {
        walk(child);
      }
    }
  };
  walk(document.body);
  return out;
}
"""


@dataclass(frozen=True)
class Block:
    """One run of text from the rendered page, with the tag that produced it.

    Frozen because it describes what a document contained at a moment in time. The `tag`
    is the whole value here: it is what makes heading attribution exact rather than a
    guess about line lengths.
    """

    tag: str
    text: str

    @property
    def is_heading(self) -> bool:
        return self.tag in HEADING_TAGS


class BrowserPageSource(PageSource):
    """Renders a page in headless Chromium and returns its visible text.

    Deliberately returns `inner_text("body")` rather than the rendered HTML put through
    the article extractors. Two reasons, and the second is the important one:

    * `inner_text` is what a *reader* would see, with scripts, styles and hidden elements
      already excluded by the browser itself — the job the extractors exist to approximate,
      done by the engine that actually knows the layout.
    * It preserves the line structure of lists and tables. That matters because the facts
      this class was built to retrieve are country lists and pricing tables, and
      `ListMembershipChecker` attributes a term to the nearest preceding heading by
      looking at lines. Flattening the table back into prose would destroy exactly the
      structure that makes the answer checkable.
    """

    name = "browser"

    def __init__(self, settings: Settings, chain: ExtractorChain | None = None) -> None:
        self._settings = settings
        # Kept for the fallback path below, where a page yields no text at all through the
        # DOM and the raw HTML is worth a try.
        self._chain = chain or ExtractorChain()
        self._playwright: Any | None = None
        self._browser: Any | None = None

    def available(self) -> bool:
        """Whether Playwright is importable. Checked without launching anything."""
        try:
            import playwright.async_api  # noqa: F401
        except ImportError:
            return False
        return True

    async def _page(self) -> Any:
        """Return a new tab, launching the browser on first use."""
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
        return await self._browser.new_page()

    async def fetch(self, url: str) -> Page:
        """Render `url` and return its visible text.

            "https://developer.paddle.com/…/supported-countries-locales/"
              ->  Page(text="…| PK | Pakistan | …", source="browser", extractor="dom")

        Two wait strategies are tried in order, and the fallback is not optional. Waiting
        for `networkidle` is what guarantees late-arriving table data has rendered, but a
        page with a long-poll, an analytics beacon or a video never reaches network idle
        and the wait times out — on Paddle's developer docs it does exactly that. Falling
        back to `domcontentloaded` plus a fixed settle delay gets the content that a
        stricter wait would have thrown away.
        """
        if url.split(":", 1)[0] not in ("http", "https"):
            # The same safety boundary the HTTP source enforces. A browser will happily
            # open file:// and read the local disk, so this matters more here, not less.
            raise ValueError(f"refusing non-http(s) URL: {url.split(':', 1)[0]}")

        page = await self._page()
        try:
            text = ""
            for wait_until in ("networkidle", "domcontentloaded"):
                try:
                    await page.goto(
                        url,
                        wait_until=wait_until,
                        timeout=int(self._settings.browser_page_timeout_s * 1000),
                    )
                    # A settle delay even after the wait resolves. Frameworks commonly
                    # render a table one tick after the document is ready, and without
                    # this the fetch can win the race and capture the empty shell.
                    await page.wait_for_timeout(
                        int(self._settings.browser_settle_s * 1000)
                    )
                    text = await page.inner_text("body")
                    if text.strip():
                        break
                except Exception:
                    # Try the next, more forgiving strategy. Both failing raises below.
                    continue

            if not text.strip():
                # Nothing visible. Fall back to the raw HTML through the normal extractor
                # chain — better than returning an empty page and letting a caller read
                # that as "the page says nothing".
                html = await page.content()
                extracted, extractor = self._chain.extract(html)
                text = extracted
            else:
                extractor = "dom"

            limit = self._settings.max_page_chars
            return Page(
                url=url,
                text=text[:limit],
                truncated=len(text) > limit,
                extractor=extractor,
                source=self.name,
            )
        finally:
            # Close the tab, not the browser. The browser is reused across fetches; a tab
            # left open holds its whole DOM in memory, so a run of twenty pages would
            # accumulate twenty live documents.
            try:
                await page.close()
            except Exception:
                pass

    async def fetch_blocks(self, url: str) -> list[Block]:
        """Render `url` and return its text as tagged blocks, in document order.

            [Block("h2", "Supported countries"),
             Block("p",  "Bank payouts supported in the following countries:"),
             Block("li", "Oman"), Block("li", "Pakistan"), …]

        Used where *which heading a term sits under* is the answer rather than a detail —
        see `membership.py`. Returns an empty list rather than raising, because the caller
        always has a text-based fallback and an exception here would lose a usable result.
        """
        if url.split(":", 1)[0] not in ("http", "https"):
            raise ValueError(f"refusing non-http(s) URL: {url.split(':', 1)[0]}")

        page = await self._page()
        try:
            for wait_until in ("networkidle", "domcontentloaded"):
                try:
                    await page.goto(
                        url,
                        wait_until=wait_until,
                        timeout=int(self._settings.browser_page_timeout_s * 1000),
                    )
                    await page.wait_for_timeout(
                        int(self._settings.browser_settle_s * 1000)
                    )
                    raw = await page.evaluate(_BLOCKS_SCRIPT)
                    if raw:
                        return [
                            Block(tag=str(item.get("tag", "")), text=str(item.get("text", "")))
                            for item in raw
                        ]
                except Exception:
                    continue
            return []
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def aclose(self) -> None:
        """Shut the browser down. Must be called, or a Chromium process is left running.

        Tolerant of being called twice and of a browser that never launched, because it
        runs from a `finally` block in the pipeline — where raising would mask whatever
        real error caused the cleanup.
        """
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
