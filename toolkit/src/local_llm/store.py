"""Persistence for model call records, split into single-responsibility classes.

The earlier version was a module of functions sharing three globals — a counter, a lock
and a thread-local connection — with SQLite syntax written directly into every query.
That made the pieces impossible to use independently, and made a second database engine
impossible to add without editing every query.

Now:

* `ConnectionProvider` — hands out connections, one per thread, and replaces dead ones.
* `SchemaMigrator` — creates tables and adds columns introduced in later versions.
* `SqlCallRepository` — reads and writes durable call records, in no particular dialect.
* `FileLiveProgressStore` — the ephemeral in-flight progress, on disk, not in the DB.
* `RetentionService` — how long payloads live, and reclaiming their space.

`CallRepository` and `LiveProgressStore` are abstract, so callers depend on the
capability rather than on SQLite or the filesystem. All engine-specific SQL lives in
`database.py`, which is why this module never mentions `pragma`, `on conflict` or `%s`.

## Why the storage is shaped this way

1. The prompt is stored **once**, in a side table. The original JSONL log wrote the full
   page text twice per call — start and end — about 55 KB per call.
2. Listing history touches indexed metadata only, roughly 200 bytes per row, so the
   history view stays fast and the browser is never sent every prompt it has ever seen.
   Payloads are fetched one row at a time, on demand.
3. Payloads are prunable independently of metadata, so long-term statistics survive while
   the bulky text ages out.

Live progress is deliberately not in the database: it is overwritten several times a
second and would be pure write amplification. One file per in-flight call, overwritten in
place and deleted on completion, bounds its size by concurrency rather than by history.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any, Iterator

from .database import DatabaseBackend

# ─────────────────────────── records ───────────────────────────


@dataclass(frozen=True)
class CallRecord:
    """One row of call history, as the rest of the package sees it.

    A frozen dataclass rather than a raw dict so a mistyped field name is an
    AttributeError at the point of the mistake, instead of a silent `None` that surfaces
    somewhere far away. Frozen because a history record describes something that already
    happened and must not be edited in place.
    """

    id: str
    ts: str
    tool: str | None
    model: str | None
    status: str | None
    duration_ms: int | None
    tokens_in: int | None
    tokens_out: int | None
    error: str | None
    stage: str | None
    meta: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for JSON serialisation by the API layer."""
        return {
            "id": self.id, "ts": self.ts, "tool": self.tool, "model": self.model,
            "status": self.status, "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "error": self.error, "stage": self.stage, "meta": self.meta,
        }


@dataclass(frozen=True)
class CallPayload:
    """The bulky text for one call, fetched only when someone opens that row."""

    prompt: list[dict[str, str]] | None
    response: str | None
    reasoning: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "response": self.response, "reasoning": self.reasoning}


# ─────────────────────────── connections ───────────────────────────


class ConnectionProvider:
    """Supplies one connection per thread, creating and replacing them as needed.

    A connection cannot be shared across threads. sqlite3 raises "SQLite objects created
    in a thread can only be used in that same thread", and PyMySQL connections are
    likewise not thread-safe. The API server handles requests on a thread pool, so
    without per-thread connections the dashboard would fail intermittently under any
    concurrency — the worst kind of bug, because a single-threaded test never sees it.

    `threading.local()` is the mechanism: an object whose attributes are private to
    whichever thread touches them, so `self._local.conn` holds a different value in each.
    """

    def __init__(self, backend: DatabaseBackend, migrator: SchemaMigrator) -> None:
        self._backend = backend
        self._migrator = migrator
        self._local = threading.local()
        # Serialises writes across threads. Both engines can handle concurrent writers by
        # rejecting one, but taking a lock in-process turns that contention into a short
        # wait instead of an error every caller would have to retry.
        self._write_lock = threading.Lock()

    @property
    def backend(self) -> DatabaseBackend:
        return self._backend

    def connection(self) -> Any:
        conn = getattr(self._local, "conn", None)
        # `is_alive` is why this is not simply "create once and keep". A MySQL server
        # closes idle connections on its own, so a cached one can be dead through no
        # fault of ours; the backend either revives it or we open a fresh one. For SQLite
        # this is always true and costs nothing.
        if conn is not None and self._backend.is_alive(conn):
            return conn

        conn = self._backend.connect()
        self._migrator.apply(conn)
        self._local.conn = conn
        return conn

    @contextmanager
    def write(self) -> Iterator[Any]:
        """Take the write lock and run inside a transaction.

        Transactions are committed and rolled back explicitly rather than through
        `with connection:`, because the two drivers disagree about what that means:
        sqlite3's connection context manager commits on success, while PyMySQL's has
        varied across versions in both what it yields and what it commits. Being explicit
        removes the ambiguity, and guarantees a half-finished multi-statement write is
        never visible to a reader.
        """
        conn = self.connection()
        with self._write_lock:
            # Opens a transaction where the backend needs one opened explicitly. MySQL
            # runs in autocommit mode — so that a read never parks an open transaction and
            # blocks maintenance — which means a multi-statement write has to say so.
            self._backend.begin_transaction(conn)
            try:
                yield conn
            except Exception:
                self._backend.rollback(conn)
                raise
            else:
                self._backend.commit(conn)


class SchemaMigrator:
    """Owns creating the schema and upgrading it between versions.

    Separate from the repository because "what shape is the database" and "how do I read
    and write records" change for different reasons and at different times. Keeping the
    DDL out means every repository method is a query, with no setup mixed in.

    The statements themselves come from the backend, since column types are not portable:
    `text` is a perfectly good primary key in SQLite and is illegal as one in MySQL.
    """

    # Columns added after the first release, as (table, column). Applied one at a time to
    # databases that predate them.
    _ADDED_COLUMNS = (("payloads", "reasoning"),)

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend

    def apply(self, connection: Any) -> None:
        self._create_schema(connection)
        self._add_missing_columns(connection)
        self._backend.commit(connection)

    def _create_schema(self, connection: Any) -> None:
        for statement in self._backend.schema_statements():
            try:
                self._backend.execute(connection, statement)
            except Exception as exc:
                # MySQL has no `create index if not exists`, so the second startup against
                # an existing database raises "Duplicate key name". The index existing is
                # the desired state, so that specific case is ignored — without this the
                # process would fail on every start after the first. Any other error is
                # re-raised, because a genuinely broken schema must not be hidden.
                if self._backend.tolerates_duplicate_index_error() and self._is_duplicate(exc):
                    continue
                raise

    @staticmethod
    def _is_duplicate(exc: Exception) -> bool:
        text = str(exc).lower()
        return "duplicate" in text or "already exists" in text

    def _add_missing_columns(self, connection: Any) -> None:
        """Add columns introduced after a given database was first created.

        Necessary because `create table if not exists` does *nothing at all* to a table
        that already exists — it will never add a new column. Without this step an older
        database keeps working right up until the first write touches the new column, then
        fails with "no such column" at runtime, during a real call.
        """
        for table, column in self._ADDED_COLUMNS:
            existing = self._backend.existing_columns(connection, table)
            if column not in existing:
                self._backend.add_column(connection, table, column)


# ─────────────────────────── abstractions ───────────────────────────


class CallRepository(ABC):
    """The persistence capability the client and API depend on.

    An abstract base class so callers depend on this interface rather than on any engine.
    That is what allows a test to pass an in-memory fake, and what allowed MySQL to be
    added without a single call site changing.
    """

    @abstractmethod
    def start_call(self, *, ts: str, tool: str, model: str, meta: dict,
                   prompt: list[dict[str, str]], call_id: str | None = None) -> str: ...

    @abstractmethod
    def finish_call(self, call_id: str, *, status: str, duration_ms: int | None = None,
                    usage: dict | None = None, error: str | None = None,
                    response: str | None = None, reasoning: str | None = None,
                    stage: str | None = None, ts: str | None = None,
                    tool: str | None = None, model: str | None = None,
                    meta: dict | None = None) -> None: ...

    @abstractmethod
    def list_calls(self, *, limit: int = 200, offset: int = 0, tool: str | None = None,
                   status: str | None = None, search: str | None = None) -> list[CallRecord]: ...

    @abstractmethod
    def get_payload(self, call_id: str) -> CallPayload | None: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...

    @abstractmethod
    def size_bytes(self) -> int: ...


class LiveProgressStore(ABC):
    """In-flight progress for calls that are still generating.

    A separate interface from `CallRepository` on purpose — interface segregation. The two
    have opposite lifetimes: history is durable and append-only, progress is overwritten
    several times a second and deleted the moment a call ends. A caller that only reports
    progress should not be handed the ability to query history.
    """

    @abstractmethod
    def write(self, entry: dict[str, Any]) -> None: ...

    @abstractmethod
    def clear(self, call_id: str) -> None: ...

    @abstractmethod
    def read_all(self) -> list[dict[str, Any]]: ...


# ─────────────────────────── implementations ───────────────────────────


class SqlCallRepository(CallRepository):
    """Durable call history in any SQL engine the backend supports.

    Every write is wrapped so a storage failure cannot propagate. That is a deliberate
    policy decision, not laziness: this class exists to *observe* model calls, and an
    observability layer able to abort the work it observes is worse than no observability.
    A full disk, or an unreachable MySQL server, should cost you the log line — not the
    extraction that was running.
    """

    def __init__(self, connections: ConnectionProvider) -> None:
        self._connections = connections
        self._backend = connections.backend
        # Call ids must be unique across processes, because the pipeline, the API and a
        # CLI invocation can all write to the same database at once. The process id makes
        # them unique between processes and the counter unique within one.
        #   pid 26620, third call  ->  "26620-3"
        self._counter = count(1)
        self._pid = os.getpid()

    def next_call_id(self, prefix: str = "") -> str:
        return f"{self._pid}-{prefix}{next(self._counter)}"

    # ── writes ──

    def start_call(self, *, ts: str, tool: str, model: str, meta: dict,
                   prompt: list[dict[str, str]], call_id: str | None = None) -> str:
        call_id = call_id or self.next_call_id()
        try:
            with self._connections.write() as conn:
                self._backend.execute(
                    conn, self._backend.replace_call_start(),
                    (call_id, ts, tool, model, json.dumps(meta) if meta else None),
                )
                self._backend.execute(
                    conn, self._backend.replace_payload_prompt(),
                    (call_id, json.dumps(prompt)),
                )
        except Exception:
            pass
        return call_id

    def finish_call(self, call_id: str, *, status: str, duration_ms: int | None = None,
                    usage: dict | None = None, error: str | None = None,
                    response: str | None = None, reasoning: str | None = None,
                    stage: str | None = None, ts: str | None = None,
                    tool: str | None = None, model: str | None = None,
                    meta: dict | None = None) -> None:
        ts = ts or datetime.now(timezone.utc).isoformat()
        try:
            with self._connections.write() as conn:
                self._backend.execute(
                    conn, self._backend.upsert_call(),
                    (
                        call_id, ts, tool, model, status, duration_ms,
                        # The server reports usage in OpenAI's naming, which differs from
                        # our column names:
                        #   {"prompt_tokens": 6156, "completion_tokens": 589}
                        #     ->  tokens_in = 6156, tokens_out = 589
                        (usage or {}).get("prompt_tokens"),
                        (usage or {}).get("completion_tokens"),
                        error, stage, json.dumps(meta) if meta else None,
                    ),
                )
                if response is not None or reasoning is not None:
                    # Reasoning gets its own column instead of being appended to the
                    # response, so the dashboard can hide it by default — it is routinely
                    # several times longer than the answer and rarely what you want to
                    # read — while retention still prunes both together.
                    self._backend.execute(
                        conn, self._backend.upsert_payload(),
                        (call_id, response, reasoning),
                    )
        except Exception:
            pass

    # ── reads ──

    def list_calls(self, *, limit: int = 200, offset: int = 0, tool: str | None = None,
                   status: str | None = None, search: str | None = None) -> list[CallRecord]:
        """A page of history, newest first, metadata only.

        Filters are assembled into a WHERE clause with placeholders rather than by string
        interpolation. Every user-supplied value travels as a bound parameter, so a search
        for `%'; drop table calls; --` is matched as literal text instead of executed —
        the difference between a filter and an SQL injection.
        """
        clauses: list[str] = []
        args: list[Any] = []
        if tool:
            clauses.append("tool = ?")
            args.append(tool)
        if status:
            clauses.append("status = ?")
            args.append(status)
        if search:
            clauses.append("(tool like ? or model like ? or meta_json like ? or error like ?)")
            args.extend([f"%{search}%"] * 4)
        where = f"where {' and '.join(clauses)}" if clauses else ""

        rows = self._backend.query(
            self._connections.connection(),
            f"""select id, ts, tool, model, status, duration_ms, tokens_in, tokens_out,
                       error, stage, meta_json
                from calls {where} order by ts desc limit ? offset ?""",
            (*args, limit, offset),
        )
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: dict[str, Any]) -> CallRecord:
        """Turn a database row into a `CallRecord`, decoding the JSON metadata column.

            {"id": "26620-1", "tool": "annotate_code", "meta_json": '{"file":"config.py"}'}
              ->  CallRecord(id="26620-1", …, meta={"file": "config.py"})
        """
        meta_json = row.get("meta_json")
        return CallRecord(
            id=row["id"], ts=row["ts"], tool=row["tool"], model=row["model"],
            status=row["status"], duration_ms=row["duration_ms"],
            tokens_in=row["tokens_in"], tokens_out=row["tokens_out"],
            error=row["error"], stage=row["stage"],
            meta=json.loads(meta_json) if meta_json else None,
        )

    def get_payload(self, call_id: str) -> CallPayload | None:
        rows = self._backend.query(
            self._connections.connection(),
            "select prompt, response, reasoning from payloads where call_id = ?",
            (call_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return CallPayload(
            # The prompt is stored as a JSON array of chat messages, so it is decoded back
            # into a list rather than handed over as a string.
            #   '[{"role":"user","content":"hi"}]'  ->  [{"role":"user","content":"hi"}]
            prompt=json.loads(row["prompt"]) if row["prompt"] else None,
            response=row["response"],
            # Null for calls made before reasoning was captured, and for models that do
            # not produce it at all.
            reasoning=row["reasoning"],
        )

    def stats(self) -> dict[str, Any]:
        """Lifetime aggregates. Cheap because it only touches the small metadata table.

        `coalesce(sum(x), 0)` matters: SQL `sum` over zero rows returns NULL, not 0, so a
        fresh database would otherwise hand the dashboard `None` where it expects a number
        and the display would break on first run.
        """
        conn = self._connections.connection()
        totals_rows = self._backend.query(
            conn,
            """select count(*) as calls,
                      coalesce(sum(case when status = 'error' then 1 else 0 end), 0) as errors,
                      coalesce(sum(tokens_in), 0)   as tokens_in,
                      coalesce(sum(tokens_out), 0)  as tokens_out,
                      coalesce(sum(duration_ms), 0) as ms
               from calls""",
        )
        by_tool = self._backend.query(
            conn, "select tool, count(*) as n from calls group by tool order by n desc"
        )
        by_model = self._backend.query(
            conn,
            """select model,
                      count(*) as calls,
                      coalesce(sum(case when status = 'error' then 1 else 0 end), 0) as errors,
                      coalesce(sum(tokens_in), 0)   as tokens_in,
                      coalesce(sum(tokens_out), 0)  as tokens_out,
                      coalesce(sum(duration_ms), 0) as ms,
                      max(ts) as last_ts
               from calls group by model""",
        )
        return {
            "totals": totals_rows[0] if totals_rows else {},
            "by_tool": by_tool,
            "by_model": by_model,
        }

    def size_bytes(self) -> int:
        try:
            return self._backend.size_bytes(self._connections.connection())
        except Exception:
            return 0


class FileLiveProgressStore(LiveProgressStore):
    """In-flight progress as one small JSON file per call, overwritten in place.

    Files rather than database rows because this data is rewritten every 0.7 s during a
    generation. In SQL that would mean thousands of writes per call competing for the same
    write lock the durable history needs. On disk each call touches its own file, and the
    total footprint is bounded by how many calls run at once — two, given the concurrency
    cap — rather than growing with history.

    It also stays local when the database is remote: with MySQL configured, progress
    frames would otherwise become network round-trips several times a second per call.
    """

    def __init__(self, live_dir: Path) -> None:
        self._live_dir = live_dir

    def _path(self, call_id: str) -> Path:
        return self._live_dir / f"{call_id}.json"

    def write(self, entry: dict[str, Any]) -> None:
        """Replace this call's progress file atomically.

        Written to a temporary file and then renamed over the real one, rather than
        written in place. The reason is a visible bug this caused, not tidiness.

        `write_text` opens the file with mode "w", which **truncates it immediately** and
        only then writes the new bytes. During that gap the file exists but is empty or
        half-written. Progress is rewritten roughly every 0.7 s and the dashboard polls
        every 0.9 s, so the two collide often. `read_all` then fails to parse the file and
        skips the call — and `/live` merges in running database rows that have no progress
        file, so the very same call immediately reappears labelled `starting`. The row
        flickers between "running, 4,102 chars" and "loading model into VRAM…" several
        times a minute, which reads as a machine thrashing rather than one working.

        `os.replace` is atomic on both Windows and POSIX: a reader sees either the whole
        old file or the whole new one, never a partial. The temporary file carries the
        call id so two concurrent calls cannot overwrite each other's staging file.

            .local-llm-data/live/26432-3.json.tmp   ->  .local-llm-data/live/26432-3.json
        """
        path = self._path(entry["id"])
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(entry), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            # Progress reporting is strictly best-effort; losing a frame must never
            # interrupt the generation it is describing.
            try:
                temporary.unlink()
            except OSError:
                # Nothing to clean up, or it cannot be removed. Either way the next frame
                # overwrites it, so this must not raise on the generation's path.
                pass

    def clear(self, call_id: str) -> None:
        try:
            self._path(call_id).unlink()
        except OSError:
            # Already gone, which is the desired end state anyway.
            pass

    # How many times to re-attempt one progress file before giving up on it, and how long
    # to wait between attempts. Both tiny, because the thing being waited out is a rename
    # that takes microseconds.
    _READ_ATTEMPTS = 3
    _READ_BACKOFF_S = 0.002

    def read_all(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            paths = list(self._live_dir.glob("*.json"))
        except OSError:
            return []

        for path in paths:
            entry = self._read_one(path)
            if entry is not None:
                entries.append(entry)
        return entries

    def _read_one(self, path: Path) -> dict[str, Any] | None:
        """Read one progress file, retrying briefly through a concurrent replacement.

        The retry is Windows-specific in origin and worth spelling out. `os.replace`
        cannot swap a file that another handle has open, so a reader and a writer meeting
        on the same file produce a `PermissionError` on whichever arrives second — the
        reader sees "access denied" for a file that exists and is perfectly valid, and a
        `FileNotFoundError` is possible in the same window.

        Giving up on that first failure is what made the dashboard blink: the call
        vanished from `/live` for that poll, and because `/live` merges in running database
        rows that have no progress file, the same call immediately reappeared labelled
        `starting`. So the row did not merely disappear, it oscillated between "running,
        4,102 chars" and "loading model into VRAM…".

        Three attempts two milliseconds apart is more than enough for a rename, and costs
        nothing in the normal case where the first attempt succeeds.
        """
        for attempt in range(self._READ_ATTEMPTS):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                if attempt + 1 < self._READ_ATTEMPTS:
                    time.sleep(self._READ_BACKOFF_S)
        # Genuinely unreadable — the call most likely finished and its file was removed
        # between the glob and the read. Dropping it is right; it is no longer live.
        return None


class RetentionService:
    """Enforces how long bulky payloads are kept, and reclaims their space.

    Its own class because retention is a *policy*, changed for entirely different reasons
    than the queries in the repository. Keeping it separate means the repository holds no
    opinion about how long data lives.
    """

    def __init__(self, connections: ConnectionProvider, repository: CallRepository,
                 payload_days: int, max_payloads: int) -> None:
        self._connections = connections
        self._backend = connections.backend
        self._repository = repository
        self._payload_days = payload_days
        self._max_payloads = max_payloads

    def prune(self, *, payload_days: int | None = None,
              max_payloads: int | None = None) -> dict[str, int]:
        """Delete stored prompts and responses past either limit, keeping metadata rows.

        Two independent limits, because either alone leaves a hole: an age limit lets a
        burst of activity fill the disk inside the retention window, and a count limit
        alone keeps ancient payloads alive forever on a quiet system.

        Metadata rows are never deleted, so lifetime statistics stay correct long after
        the text they came from has gone.
        """
        payload_days = self._payload_days if payload_days is None else payload_days
        max_payloads = self._max_payloads if max_payloads is None else max_payloads
        cutoff = (datetime.now(timezone.utc) - timedelta(days=payload_days)).isoformat()

        with self._connections.write() as conn:
            by_age = self._backend.execute(
                conn,
                "delete from payloads where call_id in (select id from calls where ts < ?)",
                (cutoff,),
            )
            by_count = self._backend.execute(
                conn, self._backend.delete_payloads_beyond_count(), (max_payloads,)
            )

        self._backend.checkpoint(self._connections.connection())
        # rowcount is -1 when a statement affected nothing, so clamp to 0 rather than
        # reporting a negative number of deleted rows to the dashboard.
        return {"pruned_by_age": max(by_age, 0), "pruned_by_count": max(by_count, 0)}

    def vacuum(self) -> int:
        """Reclaim the space freed by pruning.

        Deleting rows does not shrink storage on either engine: SQLite keeps freed pages
        in its file and InnoDB keeps them inside the tablespace. The backend knows which
        statement actually releases them — VACUUM or OPTIMIZE TABLE.
        """
        self._backend.reclaim(self._connections.connection())
        return self._repository.size_bytes()
