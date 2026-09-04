"""FastAPI layer — the read-only HTTP surface the Next.js dashboard talks to.

Why an API at all, rather than the dashboard reading the SQLite file directly: the
database lives on the machine running the models, a browser cannot open a local file, and
putting the queries here means the aggregation and retention logic exists once instead of
being duplicated in TypeScript.

The routes are grouped into router classes — `SystemRoutes`, `HistoryRoutes`,
`MaintenanceRoutes` — each taking only the collaborators it actually needs. FastAPI wants
functions as handlers, so each class registers bound methods on a router in
`register()`. That keeps the dependency-injection shape while giving the framework what
it expects, and it means a route can never quietly reach for a service it was not given.

Everything is read-only apart from the two explicit maintenance endpoints. A monitoring
surface that could start or cancel model work would need authentication and an audit
trail; keeping it observational means it can serve on localhost with no auth and still be
safe.

Run it with:

    uvicorn local_llm.api:app --port 7878
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .container import Toolkit
from .monitor import USAGE_WINDOW_HOURS, ClaudeUsageReader, SystemMonitor
from .store import CallRepository, LiveProgressStore, RetentionService

# ─────────────────────────── routers ───────────────────────────


class SystemRoutes:
    """Health, machine state and Claude plan usage."""

    def __init__(self, monitor: SystemMonitor, usage_reader: ClaudeUsageReader,
                 repository: CallRepository, model_name: str, endpoint: str) -> None:
        self._monitor = monitor
        self._usage_reader = usage_reader
        self._repository = repository
        self._model_name = model_name
        self._endpoint = endpoint

    def register(self, router: APIRouter) -> None:
        router.add_api_route("/health", self.health, methods=["GET"])
        router.add_api_route("/system", self.system, methods=["GET"])
        router.add_api_route("/usage", self.usage, methods=["GET"])

    def health(self) -> dict[str, Any]:
        """Cheap liveness check that deliberately does *not* touch the GPU or the CLI.

        Kept separate from `/system` so a caller can ask "is the API up?" without paying
        for a subprocess spawn and an NVML round-trip. If health were expensive, polling
        it frequently would itself become a load source — which is exactly what a health
        check must never be.
        """
        return {
            "ok": True,
            "version": "0.1.0",
            "model": self._model_name,
            "endpoint": self._endpoint,
            "db_bytes": self._repository.size_bytes(),
        }

    def system(self) -> dict[str, Any]:
        """GPU, host RAM, loaded and installed models, and model-server reachability."""
        return self._monitor.snapshot()

    def usage(
        self,
        hours: int = Query(default=USAGE_WINDOW_HOURS, ge=1, le=168),
        project: str | None = None,
    ) -> dict[str, Any]:
        """Claude plan usage over a trailing rolling window, split main loop vs subagent.

        The default window is 5 hours because that is the period the plan allowance resets
        on, so it is the only window that answers "how much have I got left".

        The response keeps the four token categories separate rather than presenting one
        total, because they are billed very differently and the mix is extreme in
        practice: one measured window held 73.5M tokens of which 72.2M were cache *reads*
        and only 854 were fresh input. A single figure would overstate real spend by
        roughly fifty times and be useless for deciding whether to keep working.

        `hours` is capped at 168 (one week) because the parse cost grows with the number
        of transcripts in range, and an unbounded window would let one request read every
        session file on the machine.
        """
        return self._usage_reader.read(window_hours=hours, project=project).as_dict()


class HistoryRoutes:
    """Call history, payloads on demand, live progress, aggregates, and the SSE stream."""

    # How often the change-notification stream checks for something new. 1 s is fast
    # enough to feel live against generations that take 10-30 s, while keeping the
    # fingerprint query — one indexed row plus a directory listing — negligible.
    _POLL_INTERVAL_S = 1.0

    def __init__(self, repository: CallRepository, live_store: LiveProgressStore) -> None:
        self._repository = repository
        self._live = live_store

    def register(self, router: APIRouter) -> None:
        router.add_api_route("/calls", self.calls, methods=["GET"])
        router.add_api_route("/calls/{call_id}/payload", self.payload, methods=["GET"])
        router.add_api_route("/live", self.live, methods=["GET"])
        router.add_api_route("/stats", self.stats, methods=["GET"])
        router.add_api_route("/events", self.events, methods=["GET"])

    def calls(
        self,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        tool: str | None = None,
        status: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        """A page of call history — metadata only, never the prompts.

        This is the reason the store splits metadata from payloads. A row here is roughly
        200 bytes, so a 100-row page is about 20 KB. Were prompts included, a page of
        claim-extraction calls would carry ~55 KB *each* — 5.5 MB for one screen of
        history, nearly all of it never read.

        `limit` is capped at 1000 so a malformed or hostile query cannot ask the server to
        serialise the entire history into one response and exhaust memory.
        """
        records = self._repository.list_calls(
            limit=limit, offset=offset, tool=tool, status=status, search=search
        )
        return {
            "calls": [record.as_dict() for record in records],
            # Returned so the client can decide whether to offer a "next page" control
            # without making a speculative extra request that usually comes back empty.
            "has_more": len(records) == limit,
            "limit": limit,
            "offset": offset,
        }

    def payload(self, call_id: str) -> dict[str, Any]:
        """The full prompt, response and reasoning for one call, fetched on demand.

        404 rather than an empty body when it is missing, because absence is a real and
        expected state the UI should explain: retention prunes payloads after
        `payload_days` while keeping the metadata row forever. A silent empty response
        would look like a bug instead of a pruned record.
        """
        found = self._repository.get_payload(call_id)
        if found is None:
            raise HTTPException(
                status_code=404,
                detail="no payload for this call — it may have been pruned by retention",
            )
        return found.as_dict()

    def live(self) -> dict[str, Any]:
        """Calls generating right now, with a tail of the text produced so far.

        This is what makes a busy model distinguishable from an idle one. Without it a
        12-second extraction and a hung request look identical from outside, since the
        history row does not appear until the call finishes.

        ## Why running calls are merged in from the database

        A live-progress file only appears once the model emits its first token. That left
        a real blind spot, found by testing rather than by reasoning: if the requested
        model is not already resident it must be loaded first, taking about 18 seconds on
        this machine, and for that whole window the call exists as a `running` row with no
        progress file. `/live` therefore reported nothing while the machine was busy —
        precisely the failure this endpoint exists to prevent.

        So running rows without a progress file are included and marked `starting`. The
        dashboard can then say "loading model" instead of showing an idle system, and a
        genuinely hung request becomes visible as a call stuck in `starting` rather than
        vanishing until it times out.
        """
        entries = self._live.read_all()
        streaming = {entry.get("id") for entry in entries}
        entries.extend(self._starting_calls(streaming))

        # Newest first, matching how the history list is ordered. A table that reordered
        # itself between views would be disorienting.
        entries.sort(key=lambda entry: entry.get("ts") or "", reverse=True)
        return {"live": entries, "count": len(entries)}

    def _starting_calls(self, already_streaming: set[str | None]) -> list[dict[str, Any]]:
        """Running rows that have not produced a token yet — usually a model load.

        Only the newest few are considered. Concurrency is capped at 2, so a long list
        here would mean rows stranded as `running` by a crashed process, and replaying all
        of those as though they were in flight would show activity that is not happening.
        """
        try:
            return [
                {
                    "id": record.id,
                    "ts": record.ts,
                    "tool": record.tool,
                    "model": record.model,
                    # A distinct status, not "running", so the UI can label this phase
                    # honestly: the request is accepted but no tokens exist yet.
                    "status": "starting",
                    "chunks": 0,
                    "chars": 0,
                    "reasoning_chars": 0,
                    "elapsed_ms": None,
                    "tail": "",
                }
                for record in self._repository.list_calls(limit=20, status="running")
                if record.id not in already_streaming
            ]
        except Exception:
            # The progress files are the primary source. If this database read fails,
            # return what we already have rather than losing the live view entirely.
            return []

    def stats(self) -> dict[str, Any]:
        """Lifetime aggregates: totals, per-tool counts and per-model counts.

        Cheap even over a long history because it aggregates the small metadata table and
        never touches payloads.
        """
        return {**self._repository.stats(), "db_bytes": self._repository.size_bytes()}

    async def events(self) -> StreamingResponse:
        """Server-Sent Events stream that tells the dashboard when to refetch.

        SSE ("Server-Sent Events") is a one-way stream of text over an ordinary HTTP
        response that stays open: the server writes `data: ...` lines whenever it has
        something to say, and the browser's built-in `EventSource` fires an event for
        each. Chosen over WebSockets because the traffic is entirely server-to-client —
        the dashboard never sends anything back — and SSE needs no protocol upgrade, no
        extra dependency, and reconnects on its own if the connection drops.

        Deliberately a *notification* stream, not a data stream: it says "something
        changed", and the client then calls the normal endpoints. That keeps one
        definition of each payload shape instead of a second copy embedded in the event,
        and means a client that missed events while backgrounded simply refetches current
        state rather than replaying a backlog.
        """
        return StreamingResponse(
            self._event_stream(),
            media_type="text/event-stream",
            headers={
                # Without no-cache an intermediary may buffer the stream and deliver
                # everything at the end, which defeats the point entirely.
                "Cache-Control": "no-cache",
                # nginx-specific and harmless elsewhere: disables response buffering, the
                # most common reason a working SSE endpoint appears dead behind a proxy.
                "X-Accel-Buffering": "no",
            },
        )

    async def _event_stream(self):
        """Yield an event whenever the fingerprint changes, and a keepalive otherwise.

        An async generator: a function that produces values over time with `yield`, which
        the framework consumes one at a time and writes to the open response. That is what
        lets one request keep delivering for as long as the dashboard is open.
        """
        last = ""
        # Tells the browser how long to wait before reconnecting after a dropped
        # connection. Sent once, up front, because after a drop there is no opportunity
        # to send it.
        yield "retry: 2000\n\n"

        while True:
            current = self._fingerprint()
            if current != last:
                last = current
                yield f"data: {json.dumps({'changed': True, 'fingerprint': current})}\n\n"
            else:
                # A comment line (starting with ':') is ignored by EventSource but is
                # still traffic on the socket. Needed because proxies and some browsers
                # close a connection that has been silent too long, and without it an
                # idle dashboard would silently stop receiving updates.
                yield ": keepalive\n\n"

            # Polling here rather than in the browser means one query per second in
            # total, however many tabs are open.
            await asyncio.sleep(self._POLL_INTERVAL_S)

    def _fingerprint(self) -> str:
        """A short string that changes whenever there is something new to look at.

        Built from the number of in-flight calls, the id and status of the newest history
        row, and the database size. Those cover the three things a viewer cares about: a
        call started, a call finished, or a call is still producing output.

            no activity        ->  "0|abc-3|ok|32768"
            a call starts      ->  "1|abc-3|ok|32768"     (live count changed)
            that call finishes ->  "0|abc-4|ok|36864"     (new newest row)

        Comparing a fingerprint is far cheaper than diffing the payloads themselves, which
        is what makes a one-second poll affordable.
        """
        try:
            newest = self._repository.list_calls(limit=1)
            newest_id = newest[0].id if newest else "-"
            newest_status = newest[0].status if newest else "-"
            live_count = len(self._live.read_all())
            return f"{live_count}|{newest_id}|{newest_status}|{self._repository.size_bytes()}"
        except Exception:
            # A fingerprint that cannot be computed must not kill the stream the client
            # depends on. A constant simply means "nothing changed" this tick.
            return "error"


class MaintenanceRoutes:
    """The only endpoints that change anything."""

    def __init__(self, retention: RetentionService, repository: CallRepository) -> None:
        self._retention = retention
        self._repository = repository

    def register(self, router: APIRouter) -> None:
        # POST rather than GET because these delete data: browsers and proxies pre-fetch
        # and cache GETs, so exposing them as GETs risks them firing with nobody asking.
        router.add_api_route("/prune", self.prune, methods=["POST"])
        router.add_api_route("/vacuum", self.vacuum, methods=["POST"])

    def prune(self, payload_days: int | None = None,
              max_payloads: int | None = None) -> dict[str, Any]:
        """Drop stored prompts and responses past the retention limits, keeping metadata.

        Reports the freed rows but not a smaller file — SQLite keeps freed pages for
        reuse. Call `/vacuum` to actually shrink the file on disk.
        """
        result = self._retention.prune(payload_days=payload_days, max_payloads=max_payloads)
        return {**result, "db_bytes": self._repository.size_bytes()}

    def vacuum(self) -> dict[str, Any]:
        """Rebuild the database file so freed pages return to the filesystem.

        Separate from `/prune` because VACUUM rewrites the whole file and needs temporary
        space roughly equal to the database size, so it should be an explicit choice
        rather than something that happens implicitly on every prune.
        """
        return {"db_bytes": self._retention.vacuum()}


# ─────────────────────────── the application ───────────────────────────


class DashboardApi:
    """Builds the FastAPI application from an injected `Toolkit`.

    A class rather than module-level route functions so the app can be constructed
    against a different Toolkit — pointed at a temporary database and a fake model server
    — without patching anything. `app` at the bottom of this module is simply the default
    instance that `uvicorn local_llm.api:app` imports.
    """

    # Restricted to localhost origins on purpose. `allow_origins=["*"]` would let any web
    # page the user happens to visit read their local call history, including every full
    # prompt ever stored.
    _ALLOWED_ORIGINS = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:7878",
    )

    def __init__(self, toolkit: Toolkit | None = None) -> None:
        self._toolkit = toolkit or Toolkit()

    def build(self) -> FastAPI:
        app = FastAPI(
            title="local-llm",
            description="Observability for local model calls and Claude plan usage.",
            version="0.1.0",
        )

        # The dashboard is served by Next.js on its own port (3000 in development), so
        # every request from it is cross-origin and the browser discards the response
        # unless the server explicitly permits it. Without this the dashboard shows empty
        # panels and a CORS error in the console rather than any useful failure.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(self._ALLOWED_ORIGINS),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

        router = APIRouter()
        for routes in self._routers():
            routes.register(router)
        app.include_router(router)
        return app

    def _routers(self) -> list[Any]:
        toolkit = self._toolkit
        return [
            SystemRoutes(
                toolkit.monitor,
                toolkit.usage_reader,
                toolkit.repository,
                toolkit.settings.model,
                toolkit.settings.url,
            ),
            HistoryRoutes(toolkit.repository, toolkit.live_store),
            MaintenanceRoutes(toolkit.retention, toolkit.repository),
        ]


# The module-level application uvicorn imports. Built once, at import time, from a
# default Toolkit — which is lazy, so this does not touch the disk or the GPU until the
# first request that needs them.
app = DashboardApi().build()
