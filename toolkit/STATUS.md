# Local LLM Toolkit — status and resume notes

Updated 2026-09-05 (was paused 2026-09-04). This file is the handoff: what exists, what is proven, what is
broken, and what to do next.

## Goal

Delegate mechanical work to a local model instead of spending Claude Team Pro plan
usage on it, with an interface that shows exactly what the local models are doing —
and eventually ship it as a reusable, `.env`-configured package.

Constraints this design has to respect:

- Claude effort stays at `high` at most; no subagent fan-out for cost reasons.
- Plan usage resets on a 5-hour rolling window, so consumption must be visible.
- 16 GB VRAM total. One 14B model at a time, plus a small model at most.
- Model downloads are the user's job via the Bionic GUI, not the CLI.
- Python for logic, React + TypeScript (Next.js) for the interface.

## Environment as it stands

| Thing | State |
|---|---|
| Bionic (LM Studio renamed) | installed at `%LOCALAPPDATA%\Programs\Bionic`, home is `~/.lmstudio` |
| Local server | ON, `http://localhost:1234/v1` (`lms server start` if not) |
| `lms` CLI | `~/.lmstudio/bin/lms.exe`, on PATH |
| Qwen3 14B Q4_K_M | installed, loads in ~18s, 32K context (its maximum), ~49 tok/s |
| `nomic-embed-text-v1.5` | installed |
| GPU | RTX 5070 Ti, 16303 MiB. **With Qwen3 14B resident at 32K, VRAM sits at ~94%.** |
| Python | 3.14.7, pip available, no `uv` |
| Node | 24.19.0, npm 11.17.0 |

Models still wanted (user installs these via the GUI):

- **Qwen2.5-Coder 14B Instruct Q4_K_M** (~9 GB) — the coding lane.
- **Qwen3 4B Q4_K_M** (~2.6 GB) — cheap triage. Note it will *not* fit beside the
  14B at 32K context; either lower the 14B's context or rely on TTL eviction.

## What works (Python package, verified live)

Installed editable: `pip install -e .` from `local-llm/toolkit/`.
Extras installed so far: `trafilatura`, `psutil`, `nvidia-ml-py`, `fastapi`, `uvicorn`,
`PyMySQL`.

The package is **classes, following SOLID**, with collaborators injected through
constructors. `Toolkit` in `container.py` is the composition root and the only place that
names concrete implementations. Comments follow the house standard in the project
`CLAUDE.md`: the why, the failure mode a guard defends against, jargon glossed plainly,
and a worked example at every data transformation.

| Module | Status | Classes |
|---|---|---|
| `config.py` | OK | `Settings`, `DatabaseConfig` — data only |
| `database.py` | OK | `DatabaseBackend` -> `SqliteBackend`, `MySqlBackend` |
| `store.py` | OK | `ConnectionProvider`, `SchemaMigrator`, `SqlCallRepository`, `FileLiveProgressStore`, `RetentionService` |
| `client.py` | OK | `StreamAccumulator`, `ProgressReporter`, `ThinkingStripper`, `JsonResponseParser`, `LocalLLMClient` |
| `extract.py` | OK | `TrafilaturaExtractor`, `RegexExtractor`, `ExtractorChain`, `PageFetcher`, `PromptLibrary`, `ClaimExtractor`, `ResultRanker` |
| `monitor.py` | OK | `GpuProbe`, `HostProbe`, `LmsCommandRunner`, `ModelRegistry`, `ModelServerProbe`, `TranscriptParser`, `ClaudeUsageReader`, `SystemMonitor` |
| `api.py` | OK | `SystemRoutes`, `HistoryRoutes`, `MaintenanceRoutes`, `DashboardApi` |
| `container.py` | OK | `Toolkit` |
| `scripts/smoke_test.py` | OK | `SmokeTest` — run this first on resume |

Usage is now via the composition root:

```python
from local_llm import Toolkit
toolkit = Toolkit()
extraction = await toolkit.claim_extractor.extract(url, question)
```

## Database: choose the engine in one config block

Everything about the database lives in `DatabaseConfig`. Nothing else in the package
mentions it, so switching engines is configuration, not code.

```dotenv
# SQLite — the default, works with nothing set
LOCAL_LLM_DATABASE__ENGINE=sqlite
LOCAL_LLM_DATABASE__NAME=C:/data/calls.db

# MySQL / MariaDB
LOCAL_LLM_DATABASE__ENGINE=mysql
LOCAL_LLM_DATABASE__NAME=local_llm
LOCAL_LLM_DATABASE__HOST=127.0.0.1
LOCAL_LLM_DATABASE__PORT=3306
LOCAL_LLM_DATABASE__USER=llm
LOCAL_LLM_DATABASE__PASSWORD=secret
```

`pip install -e ".[mysql]"` for the driver (PyMySQL — pure Python, so no compiler needed).

**MySQL is verified against a live MySQL 8.0.46** (already running here; credentials are
in `.env`). Confirmed: schema creation, idempotent re-apply, insert, upsert without row
duplication, the coalesce upsert preserving reasoning, utf8mb4 emoji/CJK round-trip,
stats, size, count- and age-based retention, `OPTIMIZE TABLE`, a real model call end to
end, and all six API endpoints.

Two bugs that only the live server exposed, both now fixed:

- **`autocommit=False` made a read park an open transaction**, so `OPTIMIZE TABLE` hung
  178s behind an idle connection holding a 498s transaction — and the dashboard would
  have shown a permanently stale snapshot. Now autocommit on, with explicit `begin()`
  for multi-statement writes via `backend.begin_transaction`.
- **MySQL returns `Decimal` for `SUM()`**, which `json.dumps` refuses, so `/stats` was
  200 on SQLite and **500 on MySQL**. Normalised in the MySQL backend's `query`.

The dialect differences handled, each a runtime failure rather than a syntax error:

| Concern | SQLite | MySQL |
|---|---|---|
| Parameters | `?` | `%s` |
| Upsert | `on conflict … do update set x = excluded.x` | `on duplicate key update x = values(x)` |
| Retention by count | `not in (select … limit ?)` | **rejected** — LIMIT in an IN subquery needs a derived-table wrapper |
| Size / reclaim | `stat()` / `VACUUM` | `information_schema` / `OPTIMIZE TABLE` |
| Idle connections | never expire | closed past `wait_timeout` — needs `ping(reconnect=True)` |
| Primary key type | `text` | `text` is illegal; `varchar(191)` with utf8mb4 |
| Index DDL | `create index if not exists` | no such form; "Duplicate key name" must be ignored |
| Transactions | implicit, commit on success | autocommit **on** required, else reads block maintenance |
| Aggregate types | `int` | `Decimal` — not JSON-serialisable |

Measured on the live model:

- Plain completion: 3.3s — small asks are effectively free.
- Claim extraction with trafilatura: **13.6s**, page fits in 18,738 chars untruncated.
  With the regex fallback: 17.5s and truncated at the 24,000 char cap.
- `claims[0]` really is a `Claim` instance, so schema validation is doing its job.
- All API endpoints answer 200 after the rewrite, verified on a clean process.

### Storage question, answered

The original JSONL log wrote the full page prompt **twice per call** (~55 KB/call) and
was parsed in full on every dashboard load. The SQLite store keeps metadata at about
**200 bytes per call** and stores each prompt once in a side table that can be pruned
independently. Same six calls: 115 KB of JSONL versus a 32 KB database with payloads
pruned. Live progress is not persisted at all — it goes to one overwritten file per
in-flight call under `.local-llm-data/live/`.

## What is broken or incomplete

1. ~~`pyproject.toml` declares a console script that does not exist~~ — **fixed.** The
   `[project.scripts]` block is removed; better to ship no entry point than a broken one.
2. ~~The old JavaScript dashboard is broken~~ — **replaced.** The Next.js dashboard
   at `../dashboard` is built and verified, and `Start-Local-LLM-Dashboard.cmd` is
   rewritten to launch the model server, the API and the UI together.
3. ~~The JS files are superseded but kept until the Python MCP server is proven~~ —
   **done.** `mcp_server.py` was written and verified over real stdio first (handshake,
   tool listing, all four tools), then the four `.mjs` files were deleted and `.mcp.json`
   repointed at `python -m local_llm.mcp_server`.

   Note the SDK moved: `mcp` is 2.1.1, where `FastMCP` became `MCPServer`
   (`from mcp.server.mcpserver import MCPServer`). v1 examples will not import.

## Next steps, in order

1. ~~`monitor.py`~~ — **done and verified live.** See the correction below: NVML
   cannot do per-process VRAM either.
2. ~~`api.py`~~ — **done and verified live.** `uvicorn local_llm.api:app --port 7878`.
3. ~~Next.js + React + TypeScript dashboard~~ — **done.** At `../dashboard`. Type-checks
   clean, builds to 111 KB first-load JS, verified against the live API with real model
   activity. See its README for how it updates and why the panels are ordered as they are.
4. `search.py` — pluggable providers: Brave API, SearXNG, DuckDuckGo. User chose
   "pluggable, all three".
5. `pipeline.py` — the credit-free research orchestrator: scope → search → dedup →
   extract → adversarial verify → synthesize, entirely on local models, writing a
   cited markdown report. Cap concurrency at 2; one GPU.
6. ~~`mcp_server.py`~~ — **done and verified.** Four tools: `local_extract_claims`,
   `local_rank_results`, `local_complete`, `local_status`.
7. **A live view of currently-active Claude agents in the dashboard.** The existing
   `/usage` panel is retrospective — it sums the 5-hour window from transcripts. It does
   not show what is running *now*, which is the stated condition for using subagent
   fan-out at all. Implementable from the same transcripts: a sidechain with a message in
   the last N seconds is an active agent.

### Claude plan usage is measurable locally

`~/.claude/projects/<project>/<session>.jsonl` records every assistant message with a
`usage` block, and `isSidechain` marks messages produced by a subagent rather than the
main loop. That is what makes "show me the agents and what they cost" implementable:
sum `input_tokens` / `output_tokens` / cache fields over a 5-hour window, split by
`isSidechain`. Parsed transcripts should be cached by mtime and size — the current
session file is already over 1 MB.

## Fixed: the reasoning channel was being discarded

`client.py` read only `delta.content`. Bionic/LM Studio streams a reasoning model's
thinking in a **separate `reasoning_content` field**, and `enable_thinking: false` is
accepted and **ignored** (28 reasoning tokens out of a 30-token budget). So every
Qwen3 call threw away the thinking, and any call whose budget was consumed by it
failed with the unhelpful `local model returned empty content`.

The `_THINK_RE` strip for inline `<think>` tags was therefore dead code on this
runtime. It is kept for llama.cpp / vLLM / Ollama, which do inline it.

Now: reasoning is accumulated separately, stored in a new `payloads.reasoning` column
(with a `_migrate` step, since `create table if not exists` never adds a column to an
existing database), and shown live as a distinct `thinking` status with a character
count — verified at `status: thinking, chunks: 244, reasoning_chars: 1197`. Previously
that window showed `chars: 0` and was indistinguishable from a stalled call.

Budget implications, measured: Qwen3 14B spends **203 output tokens and ~1,000
characters of reasoning to say "OK"**. Use a Coder (non-reasoning) model for extraction
and ranking, or set `max_tokens` well clear of the expected answer length.

Note the original smoke test passed throughout, because its 2048-token default left
room to answer after thinking. Test near the limits, not in the comfortable middle.

## Corrections to earlier notes (measured 2026-09-05)

**Per-process VRAM is not obtainable on this machine at all.** The earlier note said
NVML via `pynvml` was "the route that can attribute memory to a process". It is not.
`nvmlDeviceGetComputeRunningProcesses` returns every process with
`usedGpuMemory = None` on this consumer GeForce card, because under Windows' WDDM
driver model the OS owns video memory allocation, not the NVIDIA driver. This is a
platform limitation with no workaround, not a missing call or a permissions issue.

The replacement is better anyway: `lms ps --json` reports each resident model's
`sizeBytes` directly, so memory is attributed to a *named model* instead of a process
id. `monitor.py` does that and exposes the gap between the two numbers.

**Most of the used VRAM is not the model.** Measured with Qwen2.5-Coder 14B resident:
16,195 of 16,303 MiB used (99.3%), of which the model accounts for only 8,571 MiB. The
other ~7.4 GB is the desktop compositor, the browser, and the KV cache for the 30,464
loaded context. This is the concrete reason a 9 GB model does not leave 7 GB free, and
why the second 14B will not fit.

**The `lms` CLI needs `server start` called twice from cold.** The first invocation
returns "Timed out waiting for LM Studio daemon to start" even with Bionic already
running; the second succeeds immediately. Budget for a retry rather than treating the
first failure as the server being unavailable.

**Two more models are installed** beyond what the earlier note recorded as "wanted":
Qwen2.5-Coder 14B (8,571 MiB) and Qwen3 4B (2,381 MiB). Both are on disk; only one 14B
is ever resident.

**Claude usage is dominated by cache reads.** Over one 5-hour window: 73.5M total
tokens, of which 72.2M were cache *reads* and only 854 were fresh input. Any usage
display that sums tokens without separating cache reads from fresh input will overstate
real spend by roughly fifty times. `monitor.py` keeps the four categories apart for
exactly this reason.

## Resume checklist

```powershell
lms server start          # if the server is down; run TWICE from cold
lms ps                    # confirm Qwen3 14B is loaded
cd C:\Work\Personal\Ideas\local-llm\toolkit
python scripts\smoke_test.py
```

If the smoke test passes, pick up at step 1 above.
