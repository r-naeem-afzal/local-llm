# local-llm

Delegate the mechanical share of the work to a local model on your own GPU, with full
visibility into what it did — and see that alongside what the metered Claude plan cost in
the same window. The local half of that picture is complete. The Claude half is a floor,
because a foreground subagent's usage is billed to the window but written to no local
record; see [What is not built yet](#what-is-not-built-yet).

The premise is a cost one. A thorough research run through Claude spends roughly 55 agents,
most of them doing nothing but fetching a page and pulling claims out of it. That is
mechanical extraction against a fixed schema, which a local 14B model does well. So the
split is **local models read, Claude judges** — and the reason the observability matters is
that without it, "the work moved off the plan" is a claim rather than something you can
check.

## Layout

| Path | What it is |
| --- | --- |
| `toolkit/` | The Python package: model client, storage, extraction, search, the research pipeline, model routing, monitoring, the API, and an MCP server |
| `dashboard/` | Next.js + React + TypeScript UI, a pure client of the API |
| `Start-Dashboard.cmd` | Starts the model server, the API and the UI together (Windows) |
| `CLAUDE.md` | The code standards this repo is written to |

The two halves are one repo rather than two because the dashboard is a client of the
toolkit's HTTP API. Split apart, they would version independently and drift out of sync
with no compiler to catch it.

## Quick start

```bash
# 1. the toolkit
cd toolkit
pip install -e ".[extraction,api,monitor,mcp]"
cp .env.example .env          # then edit: endpoint, model, database
python scripts/smoke_test.py  # verifies everything against the loaded model

# 2. the API
uvicorn local_llm.api:app --port 7878

# 3. the dashboard
cd ../dashboard
npm install
npm run dev                   # http://localhost:3000
```

On Windows, `Start-Dashboard.cmd` does all three.

## Using it as a library

```python
import asyncio
from local_llm import Toolkit

toolkit = Toolkit()

extraction = asyncio.run(
    toolkit.claim_extractor.extract(
        "https://example.com/article",
        "What does this say about pricing?",
    )
)
for claim in extraction.claims:
    print(claim.importance, claim.claim)
```

`Toolkit` is the composition root — the only place that knows history is stored in SQL and
live progress in files. Everything else takes its collaborators through its constructor, so
any piece can be used alone or driven by a fake in a test.

## Model server

Points at any OpenAI-compatible endpoint: LM Studio (shipped as **Bionic**, which is LM
Studio renamed — same `~/.lmstudio` home and `lms` CLI), llama.cpp, vLLM, or Ollama's
compatibility endpoint.

Use `127.0.0.1`, not `localhost`. On Windows `localhost` resolves to IPv6 `::1` first and
LM Studio binds only IPv4, so every request waits for that attempt to time out — measured
at 2,017 ms against 7 ms, on every call.

```dotenv
LOCAL_LLM_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_MODEL=qwen/qwen3-14b
```

## Database

Every database detail lives in one config block; nothing else in the package mentions it.

```dotenv
# SQLite — the default, needs nothing set
LOCAL_LLM_DATABASE__ENGINE=sqlite

# MySQL / MariaDB — pip install -e ".[mysql]"
LOCAL_LLM_DATABASE__ENGINE=mysql
LOCAL_LLM_DATABASE__NAME=local_llm
LOCAL_LLM_DATABASE__HOST=127.0.0.1
LOCAL_LLM_DATABASE__USER=llm
LOCAL_LLM_DATABASE__PASSWORD=secret
```

Both engines are verified against real servers. The differences that needed handling go
well beyond placeholders — MySQL rejects `LIMIT` inside an `IN` subquery, `text` cannot be
a primary key, idle connections are closed by the server, and `SUM()` returns `Decimal`
rather than `int`. See `toolkit/STATUS.md`.

## MCP server

Lets Claude call the local model directly, so the reading costs no plan usage:

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "python",
      "args": ["-m", "local_llm.mcp_server"]
    }
  }
}
```

Four tools: `local_extract_claims`, `local_rank_results`, `local_complete`,
`local_status`. Calls through MCP land in the same database and appear in the same
dashboard as any other.

The tool set is deliberately narrow, and the boundary was found by measurement rather than
taste: each tool is a job whose answer is **contained in the text the model is handed**.
Asked instead to explain undocumented intent — why a particular design choice was made —
a local model produces confident invention. That shape of task is not offered.

## Claude agent monitoring

The dashboard's first panel shows which Claude Code agents are running right now — type,
description, elapsed time, tokens — and raises a toast when one starts, finishes or fails.
`GET /agents` serves it; `python scripts/agent_probe.py --watch` shows the same thing in a
terminal.

It reads the `Agent` tool calls in Claude Code's own transcripts, because the obvious
mechanism does not exist: `isSidechain` is never set on this version, and subagents write
no transcript of their own. One consequence is worth knowing before trusting any number
here. A **background** agent (`run_in_background: true`) reports its real token usage when
it completes, so those figures are measured. A **foreground** agent's usage is recorded
nowhere, so its figure is estimated from prompt and result size — which understates by
roughly thirteen times. Estimated rows are marked with `~` and never summed with measured
ones.

## Research pipeline

`python scripts/research.py "your question"` runs the whole loop locally and writes a
cited markdown report. Measured on a real run:

    queries 6 -> 43 distinct pages -> ranked -> 5 read (1 cached)
    20 claims, 19 with a quote verified against the page, 149s, no plan usage

    scope -> search -> dedupe -> rank -> extract -> verify -> report

Four things keep the runtime bounded, in the order they matter: ranking before fetching
(one call triages 43 results; extracting them would be 43 calls of ~15s), deduplicating by
canonical URL, caching extractions across runs, and verifying quotes by string matching
rather than by asking a model.

It deliberately does **not** judge. The report presents claims, their sources, and whether
each quote checked out. Deciding what is true is the half that stays with Claude — moving
it here would produce a confident local summary nobody should trust.

## Model routing

With several models installed, each task goes to the one that suits it rather than to a
single configured default:

| Task | Role wanted | Why |
| --- | --- | --- |
| `plan_queries` | reasoning | deliberation helps choose search angles; output is small |
| `rank_results` | triage | high-volume and schema-bound; a 14B is overkill |
| `extract_claims` | structured | fills a fixed schema, so reasoning is pure overhead |

Models are classified by architecture and name fragments, not a hard-coded list, so a
model downloaded a minute ago is routed to without restarting anything.

The constraint behind it: one 14B occupies ~8.4 GiB of weights plus ~7 GiB of KV cache at
32K context — 97% of a 16 GB card — so **only one fits**, and switching costs an ~18s
load. Routing is therefore coarse and `prefer_loaded` keeps the resident model when the
difference is marginal.

## What is not built yet

Semantic deduplication of claims. `nomic-embed-text-v1.5` is installed but unused, so
dedup is by URL only — the same fact from five sites counts as five claims, and the report
reads as consensus when it is one fact echoed.

Recency and source quality are extracted and displayed but never weighted, so a 2019 forum
post ranks level with a 2026 primary source. There is no contradiction detection, and no
second pass to re-query when an angle comes back thin.

Search has no fallback configured: Brave needs a key and SearXNG a URL, so every query goes
to DuckDuckGo with nothing behind it.

Also unbuildable rather than merely pending: any accounting of foreground subagent spend.
The usage panel's `subagent_messages` figure is derived from `isSidechain` and is therefore
structurally always zero — it means "not visible", never "not incurred".

`toolkit/STATUS.md` is the current handoff, and records what has been measured rather than
assumed.
