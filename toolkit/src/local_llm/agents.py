"""Which Claude Code agents are running *right now*, read from the local transcripts.

This exists to answer one question before a fan-out is launched rather than after it:
what is running, how long has it been going, and did it fail? The `/usage` panel next
door is retrospective — it sums the trailing 5-hour window — so on its own it can only
tell you what a batch of agents cost once the batch is over, which is too late to be a
decision aid.

## How agent activity is actually observable — and how it is not

The obvious mechanism does not work here. Claude Code marks subagent messages with
`isSidechain: true` in some versions, and an earlier design assumed that. **Measured on
this machine on 2026-09-05 (Claude Code 2.1.260, VS Code entrypoint): there are zero
sidechain records in every transcript of every project, including sessions that provably
ran subagents.** Subagents get no transcript of their own either. So a reader built on
`isSidechain` would have reported "no agents ever" and looked like it was working.

What *is* written to the parent session's transcript is the tool call itself:

    # an `assistant` record — the agent was launched
    {"type":"assistant","timestamp":"2026-09-04T20:48:50.858Z","sessionId":"ded195c6-…",
     "message":{"content":[{"type":"tool_use","id":"toolu_01Kbx…","name":"Agent",
        "input":{"subagent_type":"Explore","description":"Map transcript parsing",
                 "model":"opus","prompt":"…","run_in_background":false}}]}}

    # a later `user` record — the agent returned
    {"type":"user","timestamp":"2026-09-04T20:50:51.467Z",
     "message":{"content":[{"type":"tool_result","tool_use_id":"toolu_01Kbx…",
        "content":"…the agent's report…","is_error":false}]}}

That pairing is the whole mechanism, and it is better than the sidechain flag would have
been in three ways: the agent has a stable identity (`tool_use_id`), a name and type
worth displaying (`description`, `subagent_type`), and an unambiguous end — the arrival of
the `tool_result` — rather than an end inferred from having gone quiet.

    running   = a tool_use whose tool_use_id has no matching tool_result yet
    finished  = the tool_result arrived, is_error false
    errored   = the tool_result arrived with is_error true

## Background agents end somewhere else entirely

That rule holds only for an agent the parent waits for. An agent launched with
`run_in_background: true` gets a `tool_result` **within a second or two**, and it is not
the report — it is a launch acknowledgement saying the agent has started. Measured: two
background agents that ran for four and two minutes were both marked finished 1.6 and 2.2
seconds after launch. Taking that result at face value breaks the single number this
module exists to provide, since `running` reads 0 while agents are in fact running.

Their real ending arrives later, as a task notification recorded as an `attachment` whose
`prompt` holds an XML block:

    <task-notification>
      <tool-use-id>toolu_011hr…</tool-use-id>
      <status>completed</status>
      <summary>Agent "Review new dashboard TypeScript" finished</summary>
      <usage><subagent_tokens>55343</subagent_tokens><tool_uses>7</tool_uses>
             <duration_ms>113246</duration_ms></usage>
    </task-notification>

So a background invocation ignores its own `tool_result` and is closed by the notification
that carries its `tool-use-id`.

## What an agent cost: measured for background agents, estimated for the rest

That notification is also the answer to the cost question. `subagent_tokens` is a real
measured figure, and `tokens_measured` on the row says when it is being used.

A foreground agent produces no such notification, and its tokens appear in no local
record at all — the parent transcript's `usage` blocks cover only the parent's own
messages, and a subagent writes no transcript of its own. For those rows the number is
estimated from the characters going in and coming back, divided by a chars-per-token
ratio.

**Read the estimate as a floor.** An agent that reads twenty files spends most of its
budget on content appearing in neither the prompt nor the report. It is shown anyway
because a floor you can watch climb beats a blank column when a rolling allowance is the
thing being protected — but every display of it must be labelled, or it will eventually be
trusted as a bill.

## Classes

    AgentInvocation       one Agent tool call — the raw fact, running or finished
    AgentTranscriptScanner  pulls invocations out of one transcript, caching by file
    ActiveAgent           one invocation shaped for display, with elapsed time and status
    AgentActivity         the envelope the API returns
    AgentActivityReader   the service: scan the recent transcripts, decide what is live

Split this way because the scanner is the part with fiddly rules worth testing on a list
of fake records, and the reader is the part that touches the filesystem and the clock.
Handing the scanner a list keeps its tests free of both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .monitor import TranscriptLocator, parse_iso_timestamp

# The tool name Claude Code uses to launch a subagent. A constant because it appears in
# the scanner and in its documentation, and a typo would produce an empty panel with no
# error anywhere — the most expensive kind of bug in a monitoring tool.
AGENT_TOOL_NAMES = ("Agent", "Task")

# Marker identifying a task-notification record — the block that reports a *background*
# agent's real ending. Checked as a plain substring before any parsing, because it appears
# in a handful of lines out of thousands and the cheap test keeps the scanner off the
# regex engine for everything else.
NOTIFICATION_MARKER = "<task-notification>"

# One compiled pattern per field, rather than parsing the block as XML. Deliberate: this
# text is a message the harness wrote for a human to read, not a document with a schema.
# An XML parser would raise on the first unescaped `&` or `<` inside a summary — which is
# ordinary in an agent description — and lose a completion that a substring search finds
# without complaint.
_NOTIFICATION_FIELDS = {
    name: re.compile(rf"<{name}>(.*?)</{name}>", re.DOTALL)
    for name in ("tool-use-id", "status", "subagent_tokens", "duration_ms")
}


@dataclass
class AgentInvocation:
    """One `Agent` tool call, as found in a transcript. The raw fact, not the display.

    Mutable on purpose, unlike `UsageRecord` next door: an invocation is created when its
    `tool_use` is seen and *completed later* when the matching `tool_result` arrives, very
    often on a later poll of a file that has grown in between. A frozen record would mean
    rebuilding it to record the ending.
    """

    key: str
    """The `tool_use_id`. Globally unique and stable, so it is the identity used
    everywhere — including by the dashboard to tell a new agent from a known one."""

    project: str
    session: str
    subagent_type: str
    description: str
    model: str
    background: bool
    started: datetime
    finished: datetime | None = None
    is_error: bool = False
    prompt_chars: int = 0
    result_chars: int = 0

    # Set only from a task notification, which only background agents produce. When it is
    # present it is a real measurement and beats the character-count estimate; when it is
    # None the estimate is all there is. Kept as `None` rather than 0 precisely so the two
    # cases stay distinguishable — an agent that genuinely spent nothing and an agent
    # whose spend is unknown must not display the same.
    reported_tokens: int | None = None

    @property
    def running(self) -> bool:
        return self.finished is None

    def duration_ms(self, now: datetime) -> int:
        """Elapsed time, measured to the finish if it finished and to `now` if not.

            started 20:48:50, finished 20:50:51           ->  120_609
            started 20:48:50, still running, now 20:49:20 ->   30_000

        `now` is passed in rather than read from the clock so the whole snapshot shares
        one instant. Reading the clock per row would let two rows disagree about how much
        time has passed, which shows up as a row whose age briefly runs backwards.
        """
        end = self.finished or now
        return max(0, int((end - self.started).total_seconds() * 1000))


@dataclass
class ActiveAgent:
    """One invocation shaped for the dashboard.

    Separate from `AgentInvocation` because this is where the clock and the display
    vocabulary get involved. The scanner should not have to know that the dashboard's
    status badge understands the words "running", "ok" and "error"; keeping the two apart
    means a change to the panel's vocabulary never reaches into the parsing.
    """

    key: str
    project: str
    session: str
    subagent_type: str
    description: str
    model: str
    background: bool
    # "running" | "ok" | "error" — chosen to match the status vocabulary the dashboard's
    # existing StatusBadge already colours, so no UI change is needed to render them.
    status: str
    started_ts: str
    finished_ts: str | None
    elapsed_ms: int
    # Tokens the agent spent. Measured when `tokens_measured` is true (a background
    # agent's task notification reported it); otherwise a floor estimated from the prompt
    # in and the report out, since an agent's own reading is invisible.
    tokens: int
    tokens_measured: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "project": self.project,
            "session": self.session,
            "subagent_type": self.subagent_type,
            "description": self.description,
            "model": self.model,
            "background": self.background,
            "status": self.status,
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "elapsed_ms": self.elapsed_ms,
            "tokens": self.tokens,
            "tokens_measured": self.tokens_measured,
        }


@dataclass
class AgentActivity:
    """The envelope the `/agents` endpoint returns.

    Shaped like `ClaudeUsage` next door — counts, a list, and an empty-string `error`
    sentinel — so the dashboard's existing panel conventions apply unchanged. In
    particular `error` is `""` rather than `None` because the panels render a non-empty
    error string as the whole panel body, and a `None` would print "null".
    """

    window_s: float
    agents: list[ActiveAgent]
    running: int = 0
    finished: int = 0
    errored: int = 0
    # True when at least one row on screen is showing an estimate rather than a measured
    # figure, so the panel knows whether to print the caveat. Computed from the rows
    # rather than hard-coded in the UI, so a batch of purely background agents — whose
    # tokens *are* measured — does not carry a warning that does not apply to it.
    tokens_estimated: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_s": self.window_s,
            "agents": [agent.as_dict() for agent in self.agents],
            "running": self.running,
            "finished": self.finished,
            "errored": self.errored,
            "tokens_estimated": self.tokens_estimated,
            "error": self.error,
        }


class AgentTranscriptScanner:
    """Pulls `Agent` tool calls out of one transcript, caching what it has already read.

    Kept apart from `monitor.TranscriptParser` even though both read the same files,
    because they want different lines: the usage parser wants `assistant` records with a
    `usage` block, and this wants `tool_use` and `tool_result` blocks inside the message
    content of *both* `assistant` and `user` records. Merging them into one parser would
    produce a class with two unrelated outputs and a caller that always discards one.

    The cost of the split is that a transcript is read twice. That is acceptable, and
    smaller than it looks: this scanner reads only the bytes appended since its last look,
    and the operating system serves the second read from its page cache — the file was
    just read, so it is still in memory rather than being fetched from disk again.
    """

    def __init__(self) -> None:
        # Per-file state: (mtime, byte offset already consumed, invocations by key).
        #
        # Byte-offset tailing rather than re-reading, because a transcript is already
        # close to 1 MB after an hour and the dashboard polls every couple of seconds.
        # Re-parsing
        # from the start each time would make the monitoring more expensive than the work
        # being monitored — the exact failure this toolkit was built to avoid.
        #
        # The invocations dict is tiny (one small object per agent launched), so unlike a
        # cache of parsed messages it can be kept for the life of the process.
        self._cache: dict[str, tuple[float, int, dict[str, AgentInvocation]]] = {}

    def scan(self, path: Path, project: str) -> dict[str, AgentInvocation]:
        """Every agent invocation this transcript has recorded, keyed by tool_use_id."""
        stat = path.stat()
        cached = self._cache.get(str(path))

        if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            # Untouched since the last look: the answer cannot have changed.
            return cached[2]

        if cached is not None and stat.st_size >= cached[1]:
            offset, invocations = cached[1], cached[2]
        else:
            # No cache, or the file is *smaller* than what we already consumed — it was
            # truncated, rotated or replaced. Seeking to the old offset would then land
            # past the end (returning nothing, so the panel would freeze) or mid-record in
            # unrelated content. Start over instead; correctness beats the saved read.
            offset, invocations = 0, {}

        consumed = self._read_from(path, offset, project, invocations)
        self._cache[str(path)] = (stat.st_mtime, consumed, invocations)
        return invocations

    def _read_from(self, path: Path, offset: int, project: str,
                   invocations: dict[str, AgentInvocation]) -> int:
        """Read appended lines into `invocations`; return the byte offset now consumed.

        Opened in binary mode and decoded per line, because the offset must be a count of
        *bytes*. In text mode Python's `tell()` returns an opaque cookie rather than a
        byte position, so it cannot be compared against `st_size` to decide whether the
        file has grown.
        """
        consumed = offset
        with path.open("rb") as handle:
            handle.seek(offset)
            for raw_line in handle:
                # A transcript is being appended to while we read it, so the last line is
                # routinely half-written — no trailing newline yet. Consuming it would
                # advance the offset past a record we never parsed, and that record would
                # be lost permanently once it was completed. So: stop here and leave the
                # offset before it. The next poll sees the whole line.
                if not raw_line.endswith(b"\n"):
                    break
                consumed += len(raw_line)
                self._consume_line(raw_line.decode("utf-8", errors="replace"),
                                   project, invocations)
        return consumed

    def _consume_line(self, line: str, project: str,
                      invocations: dict[str, AgentInvocation]) -> None:
        """Fold one transcript line into the invocation map, if it concerns an agent."""
        line = line.strip()
        if not line:
            return
        # Cheap substring test before any JSON work: notification records are a handful of
        # lines out of thousands, and this is what a background agent's ending looks like.
        has_notification = NOTIFICATION_MARKER in line

        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            # Not a complete record. Skipping is correct — a transcript also carries
            # bookkeeping record types we have no interest in, and neither case is an
            # error worth reporting from a monitoring path.
            return

        if not isinstance(raw, dict):
            # Valid JSON but not an object. `.get` on a list or a number raises
            # AttributeError, which the decode guard above does not catch, so a single odd
            # line would abort the rest of the file.
            return

        if has_notification:
            self._apply_notification(raw, invocations)
            # Deliberately not returning: a record could in principle carry both, and
            # `_apply_notification` is a no-op for anything it does not recognise.

        if raw.get("type") not in ("assistant", "user"):
            return
        content = (raw.get("message") or {}).get("content")
        if not isinstance(content, list):
            # A plain string content (common for user messages) can hold no tool blocks.
            return

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                self._start(raw, block, project, invocations)
            elif block.get("type") == "tool_result":
                self._finish(raw, block, invocations)

    def _start(self, raw: dict[str, Any], block: dict[str, Any], project: str,
               invocations: dict[str, AgentInvocation]) -> None:
        """Record an agent launch.

            {"type":"tool_use","id":"toolu_01Kbx…","name":"Agent",
             "input":{"subagent_type":"Explore","description":"Map dashboard data",
                      "model":"opus","prompt":"…"}}
              ->  AgentInvocation(key="toolu_01Kbx…", subagent_type="Explore", …)
        """
        if block.get("name") not in AGENT_TOOL_NAMES:
            return
        key = block.get("id")
        timestamp = parse_iso_timestamp(raw.get("timestamp"))
        if not isinstance(key, str) or timestamp is None:
            # Without an id there is nothing to match a result against, and without a
            # timestamp there is no elapsed time to show. Either way the row would be
            # useless, so it is dropped rather than displayed as a blank.
            return

        payload = block.get("input")
        payload = payload if isinstance(payload, dict) else {}
        prompt = payload.get("prompt")

        invocations[key] = AgentInvocation(
            key=key,
            project=project,
            session=str(raw.get("sessionId") or ""),
            subagent_type=str(payload.get("subagent_type") or "general-purpose"),
            # The description is the human-readable label the launcher wrote, which is far
            # more use on screen than the agent type alone — three "Explore" rows are
            # indistinguishable, "Map dashboard data" is not.
            description=str(payload.get("description") or ""),
            # Empty means the agent inherited the parent's model. Left empty rather than
            # guessed, because a guessed model shown as fact is worse than a blank.
            model=str(payload.get("model") or ""),
            background=bool(payload.get("run_in_background")),
            started=timestamp,
            prompt_chars=len(prompt) if isinstance(prompt, str) else 0,
        )

    def _finish(self, raw: dict[str, Any], block: dict[str, Any],
                invocations: dict[str, AgentInvocation]) -> None:
        """Close out an invocation when its result arrives.

        Results for every other tool flow through the same records, so the `tool_use_id`
        lookup is also the filter: an id we never recorded as an agent launch is some
        other tool's result and is ignored.
        """
        key = block.get("tool_use_id")
        if not isinstance(key, str):
            return
        invocation = invocations.get(key)
        if invocation is None:
            return

        if invocation.background:
            # A background agent's `tool_result` arrives within a second or two of launch
            # and is only an acknowledgement that the agent has started — not its report.
            # Measured: two background agents that ran for four and two minutes were both
            # acknowledged inside 2.2 seconds. Treating that as the ending made `running`
            # read 0 while agents were in fact running, which is the one number this whole
            # module exists to get right. Its real ending arrives later as a task
            # notification; see `_apply_notification`.
            return

        invocation.finished = parse_iso_timestamp(raw.get("timestamp")) or invocation.started
        invocation.is_error = bool(block.get("is_error"))
        invocation.result_chars = self._content_chars(block.get("content"))

    def _apply_notification(self, raw: dict[str, Any],
                            invocations: dict[str, AgentInvocation]) -> None:
        """Close a background invocation from its task notification.

            <task-notification><tool-use-id>toolu_011hr…</tool-use-id>
              <status>completed</status>
              <usage><subagent_tokens>55343</subagent_tokens>…</usage>
            </task-notification>
              ->  invocations["toolu_011hr…"].finished = <this record's timestamp>
                  …                        .reported_tokens = 55343

        This is also the only place a *measured* token count enters the module.

        A notification can legitimately arrive more than once for the same agent — the
        harness says so, because a background agent can be resumed and will notify again
        when it next stops. Re-applying is therefore correct rather than something to
        guard against: the latest notification holds the latest figures.
        """
        text = self._notification_text(raw)
        if text is None:
            return

        fields = {
            name: match.group(1).strip()
            for name, pattern in _NOTIFICATION_FIELDS.items()
            if (match := pattern.search(text))
        }

        key = fields.get("tool-use-id")
        invocation = invocations.get(key) if key else None
        if invocation is None:
            # A notification for an agent whose launch is outside the scanned range, or
            # for something that is not an agent at all. Nothing to attach it to.
            return

        invocation.finished = parse_iso_timestamp(raw.get("timestamp")) or invocation.started
        # Anything other than an explicit "completed" is treated as a failure. Erring this
        # way round matters: a status word we have not seen before showing up as a green
        # "ok" would quietly hide a broken agent, while showing it as an error is
        # self-correcting — it gets noticed and looked at.
        invocation.is_error = fields.get("status") != "completed"

        tokens = fields.get("subagent_tokens")
        if tokens is not None:
            try:
                invocation.reported_tokens = int(tokens)
            except ValueError:
                # Keep the estimate rather than crashing on an unexpected format. A wrong
                # cost display is bad; a monitoring endpoint that 500s is worse.
                pass

    @staticmethod
    def _notification_text(raw: dict[str, Any]) -> str | None:
        """Find the notification text, whichever record type is carrying it.

        The same block shows up in more than one record shape — an `attachment` record
        stores it under `attachment.prompt`, and a `queue-operation` record under
        `content`. Both are checked because relying on one shape would mean a Claude Code
        change that moved it silently stopped closing background agents, and the symptom —
        agents that never finish — looks like a bug in the grace-window logic instead.
        """
        attachment = raw.get("attachment")
        if isinstance(attachment, dict):
            prompt = attachment.get("prompt")
            if isinstance(prompt, str) and NOTIFICATION_MARKER in prompt:
                return prompt

        content = raw.get("content")
        if isinstance(content, str) and NOTIFICATION_MARKER in content:
            return content

        return None

    @staticmethod
    def _content_chars(content: Any) -> int:
        """Size of a tool result, whichever of its two shapes it arrived in.

            "the report text"                                  ->  15
            [{"type":"text","text":"the report text"}]          ->  15

        Both occur — a result is a plain string when it is simple text and a list of
        blocks when it is structured. Handling only the string form would silently score
        every structured result as zero, which is exactly the kind of quiet wrong number
        this module is supposed to avoid producing.
        """
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        total += len(text)
                elif isinstance(block, str):
                    total += len(block)
            return total
        return 0


class AgentActivityReader:
    """The service: what agents are running or have just finished, across all projects.

    Takes its collaborators through the constructor like everything else here, so a test
    can point it at a fixture directory of transcripts and a fake clock without patching
    anything global.
    """

    def __init__(self, settings: Settings, locator: TranscriptLocator | None = None,
                 scanner: AgentTranscriptScanner | None = None) -> None:
        self._settings = settings
        self._locator = locator or TranscriptLocator(settings)
        self._scanner = scanner or AgentTranscriptScanner()

    def read(self, window_s: float | None = None,
             project: str | None = None) -> AgentActivity:
        """Agents worth showing right now.

        `window_s` is how long a *finished* agent stays on screen. Finished agents are
        kept briefly rather than removed the instant they complete, for two reasons: a
        result that flickers into and out of existence between two polls is invisible, and
        the dashboard detects a completion by watching a row change status — which it
        cannot do if the row simply disappears.
        """
        now = datetime.now(timezone.utc)
        grace = float(window_s if window_s is not None else self._settings.agent_window_s)
        activity = AgentActivity(window_s=grace, agents=[])

        root = self._locator.projects_dir()
        if not root.exists():
            activity.error = f"no Claude transcripts at {root}"
            return activity

        # Scan back far enough to catch a session that has been quiet for a while but
        # still holds a running agent. A long-running agent can leave its transcript
        # untouched for minutes at a time — the parent writes nothing while it waits — so
        # a window as tight as the display grace period would lose exactly the agents most
        # worth watching.
        scan_since = now - timedelta(hours=self._settings.agent_scan_hours)

        invocations: list[AgentInvocation] = []
        for project_name, transcript in self._locator.recent_transcripts(scan_since, project):
            try:
                found = self._scanner.scan(transcript, project_name)
            except OSError:
                # The session ended and its file went away mid-scan. One missing
                # transcript must not blank the panel for every other project.
                continue
            invocations.extend(found.values())

        for invocation in invocations:
            agent = self._present(invocation, now, grace)
            if agent is None:
                continue
            activity.agents.append(agent)
            if not agent.tokens_measured:
                # One estimated row is enough to warrant the caveat on the panel, since a
                # reader adding the column up would otherwise be adding measured and
                # estimated figures together without being told.
                activity.tokens_estimated = True
            if agent.status == "running":
                activity.running += 1
            elif agent.status == "error":
                activity.errored += 1
            else:
                activity.finished += 1

        # Newest first, matching every other list on the dashboard. Sorting by start time
        # rather than status keeps a row in place when it finishes, so a completing agent
        # is seen changing state instead of jumping to a different part of the list.
        activity.agents.sort(key=lambda agent: agent.started_ts, reverse=True)
        return activity

    def _present(self, invocation: AgentInvocation, now: datetime,
                 grace: float) -> ActiveAgent | None:
        """Shape one invocation for display, or None if it should not be shown.

        Two exclusions, each defending against a specific wrong display:

        * A finished agent older than the grace period — otherwise the panel would grow
          into a history list, which `/calls` already is.
        * A "running" agent older than `agent_stale_after_s` — because a session that was
          killed mid-agent leaves a `tool_use` with no result *forever*. Without this the
          panel would show a phantom agent running for days, and the one number the panel
          exists to make trustworthy is "how many are running now".
        """
        if invocation.running:
            age_s = (now - invocation.started).total_seconds()
            if age_s > self._settings.agent_stale_after_s:
                return None
            status = "running"
        else:
            assert invocation.finished is not None  # narrows the type; `running` proved it
            if (now - invocation.finished).total_seconds() > grace:
                return None
            status = "error" if invocation.is_error else "ok"

        return ActiveAgent(
            key=invocation.key,
            project=invocation.project,
            session=invocation.session,
            subagent_type=invocation.subagent_type,
            description=invocation.description,
            model=invocation.model,
            background=invocation.background,
            status=status,
            started_ts=invocation.started.isoformat(),
            finished_ts=invocation.finished.isoformat() if invocation.finished else None,
            elapsed_ms=invocation.duration_ms(now),
            tokens=self._tokens(invocation),
            tokens_measured=invocation.reported_tokens is not None,
        )

    def _tokens(self, invocation: AgentInvocation) -> int:
        """What the invocation cost — measured if it was reported, estimated otherwise.

            reported_tokens=55343                               ->  55,343  (measured)
            prompt 2,267 chars + result 14,493, ratio 4.0       ->   4,190  (estimated)

        The estimate is a floor, not an approximation, and the gap is large. Because
        background agents report a real figure, the two can be compared on the same agent:
        one reported **55,343 tokens** where its prompt plus report came to about 16,700
        characters, which this method would have estimated at roughly 4,200. So the
        estimate understates by something like **thirteen times**, because almost all of
        an agent's spend is the files it read and the context it carried, none of which
        appears at either end.

        That is why `tokens_measured` travels alongside the number rather than the two
        being summed into one column. It is also a practical argument for launching agents
        in the background where there is a choice: the cost then stops being guesswork.
        """
        if invocation.reported_tokens is not None:
            return invocation.reported_tokens
        chars = invocation.prompt_chars + invocation.result_chars
        return int(chars / self._settings.agent_chars_per_token)
