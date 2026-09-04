# local-llm

Delegate the mechanical share of the work to a local model on your own GPU, with full
visibility into what it did — and see that alongside what the metered Claude plan cost in
the same window.

The premise is a cost one. A thorough research run through Claude spends roughly 55 agents,
most of them doing nothing but fetching a page and pulling claims out of it. That is
mechanical extraction against a fixed schema, which a local 14B model does well. So the
split is **local models read, Claude judges** — and the reason the observability matters is
that without it, "the work moved off the plan" is a claim rather than something you can
check.

## Layout

| Path | What it is |
| --- | --- |
| `toolkit/` | The Python package: model client, storage, extraction, monitoring, the API, and an MCP server |
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

```dotenv
LOCAL_LLM_URL=http://localhost:1234/v1
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

## What is not built yet

`search.py` (pluggable Brave / SearXNG / DuckDuckGo) and `pipeline.py` (the credit-free
research orchestrator). `toolkit/STATUS.md` is the current handoff, and records what has
been measured rather than assumed.
