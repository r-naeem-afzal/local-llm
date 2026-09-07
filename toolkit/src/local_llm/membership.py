"""Answering "is X in this list?" as a fact, not as a model's opinion.

This module exists because of a specific, expensive failure. Asked five times whether
Paddle and Lemon Squeezy accept sellers based in Pakistan, the research pipeline found the
right primary sources both times and returned **zero verified claims out of ten**,
including these two side by side, each cited to a primary source:

    "Paddle does not support sellers based in Pakistan."
    "Paddle supports sellers based in Pakistan."

The question was a set-membership question, and set membership has an exact answer that a
string comparison gives for free. Routing it through a 14B model converted a certainty into
a probability, and the probability came out wrong.

## The two things that make this harder than `"Pakistan" in text`

**Heading attribution.** The same page often carries several lists, and which one a term
sits in decides the answer. Lemon Squeezy's supported-countries page has both
"Bank payouts supported in the following countries:" and "Unsupported countries for
purchases" — Pakistan appears exactly once, and reading it against the wrong heading
inverts the conclusion. So a membership answer that does not say *which list* is not an
answer.

**Negative lists.** Paddle's page says it works with businesses "anywhere in the world with
the exception of the unsupported countries listed below". There, membership means
*unsupported* and absence means *supported*. A claim-extractor will never produce
"Pakistan is supported" from that page, because the supporting evidence is the absence of a
word — which is not quotable. This module reports presence and absence as equally
first-class results so the caller can apply the page's own logic.

## Known limitation, stated rather than discovered later

Heading attribution is exact about *which element* is a heading and only approximate about
*which* heading applies. On Lemon Squeezy's page the term is correctly attributed to the
real `h2` "Supported countries" at a distance of 87 blocks, when the precise label is the
paragraph "Bank payouts supported in the following countries:" much closer above it — that
paragraph is not recovered because it is nested inside a container the block walker
descends past. The answer is true and usefully specific, just coarser than the page allows.

This is why `heading_distance` is reported on every occurrence. A distance of 2 means the
attribution is almost certainly right; 87 means a human should look at the page before
resting a decision on which list it was. The design goal was never to remove judgement, it
was to stop a model inventing the answer.

## What is deliberately not here

No model, no scoring, no inference about what the list means. This reports where a term
appears, under which heading, with surrounding text, and whether the page was rendered
properly. Deciding that "listed under Unsupported" means "we cannot sell there" is the
caller's judgement — the same boundary the MCP tools draw, for the same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .extract import Page, PageFetcher


@dataclass
class Occurrence:
    """One appearance of the term, with the heading it sits under."""

    # The nearest preceding line that looks like a heading. This is the field that decides
    # the answer, so it is reported rather than merely used.
    heading: str
    line: str
    context: str
    # How many lines back the heading was found. A large number means the attribution is
    # weak — the term may not belong to that heading at all — so the caller can see how
    # much to trust it instead of being handed a bare string.
    heading_distance: int


@dataclass
class MembershipResult:
    """Where a term appears on a page, and whether the page could be read at all."""

    term: str
    url: str
    found: bool
    occurrences: list[Occurrence] = field(default_factory=list)
    # Which source produced the text — "http" or "browser".
    source: str = ""
    page_chars: int = 0
    # Headings that were seen on the page even though the term was not under them. Useful
    # when the answer is "absent": it shows *which* lists were actually examined, so
    # "not in the unsupported list" can be distinguished from "no such list was found".
    headings_seen: list[str] = field(default_factory=list)
    # Set when the page could not be read properly. The single most important field here.
    error: str = ""

    @property
    def trustworthy(self) -> bool:
        """Whether a `found=False` from this result means anything.

        An absence is only evidence if the page was genuinely retrieved and rendered. This
        is the distinction the whole module exists to preserve: the original bug was a page
        that returned 2,715 characters with no country table, from which "Pakistan is not
        listed" was indistinguishable from "the list never loaded".
        """
        return not self.error and self.page_chars > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "url": self.url,
            "found": self.found,
            "trustworthy": self.trustworthy,
            "source": self.source,
            "page_chars": self.page_chars,
            "headings_seen": self.headings_seen,
            "error": self.error,
            "occurrences": [
                {
                    "heading": o.heading,
                    "line": o.line,
                    "context": o.context,
                    "heading_distance": o.heading_distance,
                }
                for o in self.occurrences
            ],
        }


class HeadingIndex:
    """Decides which lines of a page are headings, and what each other line sits under.

    Its own class because this is the one genuinely fiddly rule here, and separating it
    means it can be exercised on a list of strings with no page, no network and no browser.

    The rule is a heuristic and is written down as such. A heading is a line that:

    * is short — long lines are prose, and prose is not a heading;
    * either ends with a colon, or contains no sentence-ending punctuation at all;
    * is not itself a list item (no leading bullet or dash).

    Worked example, from Lemon Squeezy's supported-countries page:

        "Supported countries"                              -> heading (short, no period)
        "Bank payouts supported in the following countries:"-> heading (ends with colon)
        "  - Pakistan"                                      -> item, attributed to it
        "We currently allow purchases from all countries…"  -> prose, not a heading

    This is a heuristic and will occasionally pick the wrong line, which is exactly why
    `Occurrence` reports the heading it chose and how far back it was: a wrong attribution
    should be visible to a reader rather than silently changing the answer.
    """

    _MAX_HEADING_CHARS = 90
    _LIST_ITEM = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
    _SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

    def is_heading(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > self._MAX_HEADING_CHARS:
            return False
        if self._LIST_ITEM.match(line):
            return False
        if stripped.endswith(":"):
            return True
        return not self._SENTENCE_END.search(stripped)

    def heading_for(self, lines: list[str], index: int) -> tuple[str, int]:
        """The nearest heading at or before `index`.

            lines = ["Unsupported Countries", "Afghanistan", "Cuba"], index = 2
              ->  ("Unsupported Countries", 2)

        Returns an empty heading and -1 when the term appears before any heading, which is
        itself informative: a term with no heading above it is not in a labelled list.
        """
        for back in range(index, -1, -1):
            if back != index and self.is_heading(lines[back]):
                return lines[back].strip(), index - back
        return "", -1


class ListMembershipChecker:
    """Reports whether a term appears on a page, and under which heading.

    Takes its fetcher through the constructor like everything else here, so a test can
    hand it a fake page and never touch the network.
    """

    # How much text either side of the match to keep. Enough to see the row a term sits in
    # without pasting a whole table into a report.
    _CONTEXT_CHARS = 120

    def __init__(self, fetcher: PageFetcher, headings: HeadingIndex | None = None,
                 browser: Any | None = None, reattribute_above: int = 20) -> None:
        self._fetcher = fetcher
        self._headings = headings or HeadingIndex()
        # The browser source directly, not through the fetcher. The fetcher escalates on
        # *absence*; this class also needs to escalate on a weak heading attribution, which
        # is a different trigger and not the fetcher's concern.
        self._browser = browser
        self._reattribute_above = reattribute_above

    async def check(self, url: str, term: str) -> MembershipResult:
        """Look for `term` on `url`, escalating to a browser if it is not found.

            check("https://docs.lemonsqueezy.com/…/supported-countries", "Pakistan")
              ->  found=True, heading="Bank payouts supported in the following countries:"

            check("https://www.paddle.com/…/which-countries-are-supported", "Pakistan")
              ->  found=False, trustworthy=True,
                  headings_seen=[…, "Unsupported Countries", …]

        The term is passed to the fetcher as `expect`, which is the whole reason a
        `found=False` here can be believed: absence triggers a browser re-fetch, so the
        answer is "absent from the rendered page" rather than "absent from whatever HTML
        happened to arrive". Without that escalation this method would have confidently
        reported Pakistan missing from a page whose country table had not loaded.
        """
        try:
            page = await self._fetcher.fetch(url, expect=term)
        except Exception as exc:
            # Reported, never raised. A membership check that throws would be handled by
            # a caller as "unknown", but a caller that forgets to handle it would read the
            # exception as absence — the exact conflation this class exists to prevent.
            return MembershipResult(
                term=term, url=url, found=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        result = self._locate(page, term)

        # A second look when the heading attribution is weak. See
        # `membership_reattribute_above_lines`: HTTP-sourced text often arrives with a
        # whole list flattened onto one line, which leaves the nearest "heading" dozens of
        # lines away and coarse. Re-reading through the browser restores one item per line.
        # Only worth doing when the term was actually found — an absence does not have an
        # attribution to improve — and only from the http path, since the browser is
        # already the better source.
        if result.found and page.source != "browser" and self._weakly_attributed(result):
            better = await self._reattribute(url, term)
            if better is not None:
                return better

        return result

    # Tags that can carry a heading-like label when the page did not use a real heading
    # element. Deliberately excludes every list and table-cell tag: an item inside a list
    # is a member of that list, never the label for it.
    _PROSE_TAGS = frozenset({"p", "div", "span", "strong", "b", "em", "label"})

    def _is_label_paragraph(self, block: Any) -> bool:
        """Whether a non-heading block is nonetheless labelling what follows it.

            Block("p",  "Bank payouts supported in the following countries:")  ->  True
            Block("li", "Oman")                                                ->  False
            Block("p",  "We allow purchases from all countries except…")       ->  False

        Needed because Lemon Squeezy introduces the list that decides this whole question
        with a paragraph ending in a colon rather than an `h2`, so restricting attribution
        to real heading tags would miss the single most important label on the page.

        The trailing colon is required, and that is what keeps this safe. An earlier
        version accepted any short line without sentence punctuation, which matched every
        country name in the list — so "Pakistan" was attributed to "Oman", the item
        immediately above it, and the answer named the wrong list entirely.
        """
        if block.tag not in self._PROSE_TAGS:
            return False
        text = block.text.strip()
        return bool(text) and text.endswith(":") and len(text) <= 140

    def _weakly_attributed(self, result: MembershipResult) -> bool:
        """Whether the closest heading is too far away to be believed.

            occurrences with heading_distance [86]  ->  True   (flattened list)
            occurrences with heading_distance [2]   ->  False  (item under its heading)
        """
        distances = [
            o.heading_distance for o in result.occurrences if o.heading_distance >= 0
        ]
        if not distances:
            return True
        return min(distances) > self._reattribute_above

    async def _reattribute(self, url: str, term: str) -> MembershipResult | None:
        """Re-read through the browser as tagged blocks, so the heading is exact.

        Asks for `fetch_blocks` rather than `fetch`, and that distinction is the whole
        point of this method. Text-based attribution guesses which lines are headings from
        their length and punctuation, and on a rendered country list *every* item looks
        like a heading — asked which heading "Pakistan" sat under, the text heuristic
        answered "Oman", the country immediately above it. Tagged blocks carry the real
        element name, so a `li` can never be mistaken for an `h2`.

        Returns None on any failure, so a weak attribution is still returned rather than
        being replaced by nothing — a coarse heading beats no answer.
        """
        if self._browser is None or not hasattr(self._browser, "fetch_blocks"):
            return None
        try:
            blocks = await self._browser.fetch_blocks(url)
        except Exception:
            return None
        if not blocks:
            return None
        return self._locate_blocks(blocks, url, term)

    def _locate_blocks(self, blocks: list[Any], url: str, term: str) -> MembershipResult:
        """Find `term` among tagged blocks, attributing it to the last real heading.

            [Block("p", "Bank payouts supported in the following countries:"),
             Block("li", "Oman"), Block("li", "Pakistan")]
              ->  heading "Bank payouts supported in the following countries:", distance 2

        "Last real heading" is a single backwards scan rather than a guess, because
        `Block.is_heading` comes from the element's tag. Note that a heading-*shaped*
        paragraph — Lemon Squeezy introduces its payout list with a `p` ending in a colon,
        not an `h2` — is still caught, because the text heuristic is applied to non-heading
        blocks as a second chance. Without that, the most important label on the page would
        be invisible simply for being marked up as a paragraph.
        """
        needle = term.lower()
        result = MembershipResult(
            term=term, url=url, found=False, source="browser",
            page_chars=sum(len(b.text) for b in blocks),
            headings_seen=[b.text for b in blocks if b.is_heading],
        )

        for index, block in enumerate(blocks):
            if needle not in block.text.lower():
                continue
            heading, distance = "", -1
            for back in range(index - 1, -1, -1):
                candidate = blocks[back]
                if candidate.is_heading or self._is_label_paragraph(candidate):
                    heading, distance = candidate.text.strip(), index - back
                    break
            result.occurrences.append(
                Occurrence(
                    heading=heading,
                    line=block.text.strip(),
                    context=block.text.strip()[: self._CONTEXT_CHARS * 2],
                    heading_distance=distance,
                )
            )

        result.found = bool(result.occurrences)
        if not result.trustworthy:
            result.error = result.error or "page produced no text; absence proves nothing"
        return result

    def _locate(self, page: Page, term: str) -> MembershipResult:
        """Find every occurrence of `term` in an already-fetched page."""
        lines = page.text.split("\n")
        needle = term.lower()

        result = MembershipResult(
            term=term,
            url=page.url,
            found=False,
            source=page.source,
            page_chars=len(page.text),
            headings_seen=[
                line.strip() for line in lines if self._headings.is_heading(line)
            ],
        )

        for index, line in enumerate(lines):
            if needle not in line.lower():
                continue
            heading, distance = self._headings.heading_for(lines, index)
            position = line.lower().index(needle)
            start = max(0, position - self._CONTEXT_CHARS)
            result.occurrences.append(
                Occurrence(
                    heading=heading,
                    line=line.strip(),
                    context=line[start:position + len(term) + self._CONTEXT_CHARS].strip(),
                    heading_distance=distance,
                )
            )

        result.found = bool(result.occurrences)
        if not result.trustworthy:
            # An empty page is not an answer. Said explicitly so a caller cannot read
            # `found=False` off a page that never loaded.
            result.error = result.error or "page produced no text; absence proves nothing"
        return result
