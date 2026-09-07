# local-llm

**Run the mechanical parts of an AI workflow on your own GPU, and see exactly what
happened.**

Agentic workflows spend most of their tokens on work that does not need a frontier model:
fetching a page and pulling structured facts out of it, ranking search results, filling a
fixed schema. A 14B model on a consumer card does that well. This toolkit moves that work
onto local hardware, records every call with its full prompt and response, and shows you
the result live — so "the work moved off the paid API" is something you can check rather
than something you hope.

It also includes a complete research pipeline built on top of that: ask a question, get a
cited markdown report, with every quote verified against the page it came from.

```
scope -> search -> dedupe -> rank -> extract -> verify -> report
```

A real run: 6 queries → 43 distinct pages → ranked → 5 read → **20 claims, 19 with a quote
verified against the source, 149 seconds, no API spend.**

---

## What is in here

| Piece | What it does |
| --- | --- |
| **Model client** | Streaming, schema-constrained output via pydantic, every call recorded |
| **Research pipeline** | Question → searched, ranked, read, quote-verified, cited report |
| **Search** | Pluggable providers: Brave, SearXNG, DuckDuckGo, and a browser-driven one |
| **Extraction** | Article text via trafilatura, with fallbacks; claims against a fixed schema |
| **Model routing** | Sends each task to a suitable local model by role, not to one default |
| **Storage** | Call history in SQLite or MySQL; bulky payloads prunable separately |
| **Monitoring** | GPU, VRAM, host, resident models, and which agents are running now |
| **HTTP API** | FastAPI, read-only, `localhost` only |
| **Dashboard** | Next.js + TypeScript, live view of calls, models and cost |
| **MCP server** | Lets an MCP client (such as Claude Code) call the local model directly |

## Requirements

- **Python 3.11+**
- **An OpenAI-compatible model server.** LM Studio (also shipped as *Bionic*), llama.cpp,
  vLLM, or Ollama's compatibility endpoint. Nothing here is tied to a particular one.
- **A GPU with enough VRAM for your chosen model.** Developed against a 16 GB card running
  a 14B model at Q4; smaller models work fine, and see [Model routing](#model-routing) for
  what 16 GB actually buys you.
- **Node 20+** — only if you want the dashboard. The toolkit works without it.

## Quick start

```bash
git clone https://github.com/r-naeem-afzal/local-llm.git
cd local-llm/toolkit

pip install -e ".[extraction,api,monitor,search,mcp]"
cp .env.example .env          # then set the endpoint and model

python scripts/smoke_test.py  # checks config, database, and a real model call
```

If the smoke test passes, run some research:

```bash
python scripts/research.py "which merchant of record supports sellers in Kenya?"
```

Then, optionally, the dashboard:

```bash
uvicorn local_llm.api:app --port 7878    # terminal 1
cd ../dashboard && npm install && npm run dev   # terminal 2 -> localhost:3000
```

On Windows, `Start-Dashboard.cmd` starts the model server, the API and the UI together.

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
live progress in files. Every other class takes its collaborators through its constructor,
so any piece can be used on its own or driven by a fake in a test.

## Configuration

Everything is read from the environment or a `.env` file, prefixed `LOCAL_LLM_`. See
[`toolkit/.env.example`](toolkit/.env.example) for the full list.

```dotenv
LOCAL_LLM_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_MODEL=qwen/qwen3-14b
```

**Use `127.0.0.1`, not `localhost`.** On Windows, `localhost` resolves to IPv6 `::1` first
and LM Studio binds only IPv4, so every request waits for that attempt to time out —
measured at 2,017 ms against 7 ms, on every call.

### Database

Every database detail lives in one config block; nothing else in the package mentions it.

```dotenv
LOCAL_LLM_DATABASE__ENGINE=sqlite     # the default, needs nothing else set

# MySQL / MariaDB — pip install -e ".[mysql]"
LOCAL_LLM_DATABASE__ENGINE=mysql
LOCAL_LLM_DATABASE__NAME=local_llm
LOCAL_LLM_DATABASE__HOST=127.0.0.1
LOCAL_LLM_DATABASE__USER=llm
LOCAL_LLM_DATABASE__PASSWORD=secret
```

Both engines are tested against real servers. The portability problems went well beyond
placeholder syntax: MySQL rejects `LIMIT` inside an `IN` subquery, `text` cannot be a
primary key, idle connections are closed by the server, and `SUM()` returns `Decimal`
rather than `int` — which made one endpoint return 200 on SQLite and 500 on MySQL with no
other symptom.

### Search providers

```dotenv
LOCAL_LLM_SEARCH_PROVIDER=auto        # auto | brave | searxng | duckduckgo
LOCAL_LLM_BRAVE_API_KEY=...           # optional
LOCAL_LLM_SEARXNG_URL=...             # optional
```

With no key and no SearXNG instance, everything goes to DuckDuckGo. That works, but search
coverage is the pipeline's real ceiling — see [Known limitations](#known-limitations).

## Privacy — what stays on your machine

Worth stating plainly, because parts of this read local files.

- **Every prompt and response is stored locally**, in SQLite by default, under
  `toolkit/.local-llm-data/`. That directory is gitignored. Retention prunes bulky payloads
  on a schedule while keeping the metadata.
- **The agent monitor reads your local Claude Code transcripts** — the JSONL files under
  `~/.claude/projects/` — to work out which agents are running. It reads them; it never
  sends them anywhere.
- **The API binds to localhost and is read-only**, apart from two explicit maintenance
  endpoints. CORS is restricted to localhost origins, deliberately: `allow_origins=["*"]`
  would let any page you visit read your local call history, including every stored prompt.
- **Nothing is transmitted anywhere** except to the model server you configure, the search
  provider you configure, and the pages the fetcher is asked to read.

## Research pipeline

```bash
python scripts/research.py "your question" --pages 8
```

Four things keep the runtime bounded, in the order they matter:

1. **Rank before fetching.** One ranking call triages 43 results; extracting all of them
   would be 43 calls of ~15 s each.
2. **Deduplicate by canonical URL**, so the same page arriving from three queries is read
   once — and, more importantly, is not counted as three agreeing sources.
3. **Cache extractions across runs.** A refined question re-encounters most of the same
   pages.
4. **Verify quotes by string matching, not by asking a model.** Free, and it catches the
   failure that matters most: a model inventing supporting evidence. Measured base rate
   across two different 14B models — about **one quote in five does not appear in the page
   it is attributed to**. The verifier is load-bearing, not a nicety.

It deliberately does **not** decide what is true. The report presents claims, their
sources, and whether each quote checked out. A local model asked to synthesise produces a
confident summary that smooths over disagreement between sources.

## Model routing

With several models installed, each task goes to one that suits it rather than to a single
configured default:

| Task | Role wanted | Why |
| --- | --- | --- |
| `plan_queries` | reasoning | deliberation helps choose search angles; output is small |
| `rank_results` | triage | high-volume and schema-bound; a 14B is overkill |
| `extract_claims` | structured | fills a fixed schema, so reasoning is pure overhead |

Models are classified by architecture and name fragments rather than a hard-coded list, so
a model downloaded a minute ago is routed to without restarting anything.

The constraint behind it: one 14B occupies ~8.4 GiB of weights plus ~7 GiB of KV cache at
32K context — 97% of a 16 GB card — so **only one fits**, and switching costs an ~18 s
load. Routing is therefore coarse, and `prefer_loaded` keeps the resident model when the
difference is marginal.

## MCP server

Lets an MCP client call the local model directly, so bulk reading does not consume paid
tokens:

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

Four tools: `local_extract_claims`, `local_rank_results`, `local_complete`, `local_status`.
Calls made through MCP land in the same database and appear in the same dashboard as any
other.

The tool set is deliberately narrow, and the boundary was found by measurement. Each tool
is a job whose answer is **contained in the text the model is handed**. Asked instead to
explain undocumented intent — why a design choice was made — a local model produces
confident invention, so that shape of task is not offered.

## Agent monitoring

The dashboard's first panel shows which Claude Code agents are running right now, with
elapsed time and token cost, and raises a notification when one starts, finishes or fails.
`GET /agents` serves it, and `python scripts/agent_probe.py --watch` shows the same thing in
a terminal.

One caveat matters before trusting any number there. A **background** agent reports its
real usage on completion, so those figures are measured. A **foreground** agent's usage is
recorded nowhere on the machine, so its figure is estimated from prompt and result size,
which understates by roughly thirteen times. Estimated rows are marked `~` and never summed
with measured ones.

## Known limitations

Stated because they change what the output is worth, not to be modest.

- **Search coverage is the pipeline's ceiling**, not extraction quality. Asked a niche
  commercial question, a run can return claims about the wrong thing simply because the
  answer was never in the result pool. No reading model recovers a question the search
  missed.
- **Deduplication is by URL only.** `nomic-embed-text-v1.5` is installed but unused, so the
  same fact from five sites counts as five claims and the report reads as consensus when it
  is one fact echoed.
- **Recency and source quality are extracted and displayed but never weighted**, so a 2019
  forum post ranks level with a 2026 primary source. There is no contradiction detection.
- **Facts inside JavaScript-rendered tables** need the browser fetch path; the HTTP fetcher
  will report success and silently omit them.
- **Foreground subagent spend cannot be accounted for at all.** The usage panel's
  `subagent_messages` figure derives from a flag that is never set on current Claude Code,
  so it is structurally always zero — it means "not visible", never "not incurred".
- **No test suite yet.** `scripts/smoke_test.py` exercises the real paths against a live
  model and is the current check. See [Contributing](#contributing).

## Project status and platform support

A working personal research toolkit, used daily, not a packaged product. Interfaces may
change without ceremony. Developed and tested on **Windows 11** with an RTX 5070 Ti; the
Python is portable and should work anywhere, but only `Start-Dashboard.cmd` is
Windows-specific and only Windows has been exercised. Reports from other platforms are
welcome and useful.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how the codebase is
organised, what the comment standard asks for, and where help would land best — a test
suite, non-Windows verification, and semantic claim deduplication are the three most
valuable things right now.

Two things to read before a first pull request:

- **[CLAUDE.md](CLAUDE.md)** — the code standards this repository is written to. Unusually
  strict about comments: they must record *why* a choice was made and what breaks without
  it, not restate what the code does.
- **[toolkit/STATUS.md](toolkit/STATUS.md)** — the engineering log. Every non-obvious
  decision here was driven by a measurement, and that file records them, including the ones
  that overturned an earlier assumption.

## Licence

MIT. See [LICENSE](LICENSE).
