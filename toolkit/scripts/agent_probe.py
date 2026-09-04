"""Print the live Claude agent view from the command line.

    python scripts/agent_probe.py            # one reading
    python scripts/agent_probe.py --watch    # refresh until interrupted

Why this exists rather than just opening the dashboard: the panel it feeds is only
interesting while agents are running, so without a command-line probe the only way to
check the reader is to launch agents, race to the browser, and hope. This prints the same
data with no API and no UI in the way, which makes the Python half debuggable on its own.

It also shows the awkward asymmetry the feature had to be built around: a token figure
prefixed with `~` is an estimate, because a *foreground* agent's usage is recorded in no
local file. Background agents report theirs, and understate by nothing at all. See
`local_llm/agents.py`.
"""

from __future__ import annotations

import sys
import time

from local_llm import Toolkit
from local_llm.agents import ActiveAgent

# Long enough to see a row change state, short enough to feel live. Matches the
# dashboard's own poll interval so both show the same thing at the same rate.
WATCH_INTERVAL_S = 2.0


class AgentProbe:
    """Formats one reading of the agent activity reader as a table.

    A class, like `SmokeTest` next door, so the `Toolkit` is built once and its scanner
    keeps its byte offsets between refreshes — which is also the behaviour being checked.
    A fresh reader per refresh would re-read every transcript from the start and hide
    whether the incremental tailing works at all.
    """

    def __init__(self, toolkit: Toolkit | None = None) -> None:
        self._toolkit = toolkit or Toolkit()

    def print_once(self) -> int:
        """Print one reading. Returns the number of agents currently running."""
        activity = self._toolkit.agent_reader.read()

        if activity.error:
            print(f"  {activity.error}")
            return 0

        # ASCII separators only. The Windows console defaults to code page 850/437, where
        # printing a middle dot raises UnicodeEncodeError or prints a replacement glyph —
        # so a decorative character would make the probe fail on the machine it is for.
        header = (
            f"running {activity.running} | finished {activity.finished} "
            f"| errored {activity.errored} | window {activity.window_s:.0f}s"
        )
        print(header)
        if not activity.agents:
            print("  no Claude agents active")
            return 0

        if activity.tokens_estimated:
            print("  ~ marks an estimate from prompt + result size; it understates ~13x")

        print(f"  {'status':8} {'type':18} {'elapsed':>9} {'tokens':>10}  description")
        for agent in activity.agents:
            print(f"  {self._format_row(agent)}")
        return activity.running

    @staticmethod
    def _format_row(agent: ActiveAgent) -> str:
        """One agent as a fixed-width line.

            ActiveAgent(status="running", subagent_type="Explore", elapsed_ms=72_400,
                        tokens=4296, tokens_measured=False, description="Map dashboard")
              ->  "running  Explore              1m 12s    ~4,296  Map dashboard"
        """
        seconds, milliseconds = divmod(agent.elapsed_ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        elapsed = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}.{milliseconds // 100}s"
        # The tilde is load-bearing, not decoration: a measured 55,343 and an estimated
        # 4,190 are the same kind of column but not the same kind of number.
        tokens = f"{'' if agent.tokens_measured else '~'}{agent.tokens:,}"
        return (
            f"{agent.status:8} {agent.subagent_type[:18]:18} {elapsed:>9} "
            f"{tokens:>10}  {agent.description[:50]}"
        )

    def watch(self) -> None:
        """Refresh until interrupted.

        KeyboardInterrupt is caught and swallowed rather than allowed to print a traceback:
        Ctrl-C is how this program is *meant* to end, and a stack trace would suggest
        something went wrong.
        """
        try:
            while True:
                print()
                self.print_once()
                time.sleep(WATCH_INTERVAL_S)
        except KeyboardInterrupt:
            print("\nstopped")


def main() -> int:
    probe = AgentProbe()
    if "--watch" in sys.argv:
        probe.watch()
        return 0
    probe.print_once()
    # Always zero. This is a probe, not a test: "no agents running" is a perfectly correct
    # reading, so a non-zero exit would make a normal answer look like a failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
