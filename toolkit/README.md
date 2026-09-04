# local-llm toolkit

Delegate mechanical work to a local model instead of spending paid API or plan usage on
it — and see exactly what the model was asked and what it said.

Point it at any OpenAI-compatible server (Bionic / LM Studio, llama.cpp, vLLM, Ollama's
compatibility endpoint) with two environment variables. Nothing else is required.

## Install

```bash
pip install -e .                    # core: httpx + pydantic only
pip install -e ".[extraction]"      # + trafilatura, much better article text
pip install -e ".[all]"             # + API, monitoring and MCP server
```

## Configure

Create a `.env` (see `.env.example`):

```env
LOCAL_LLM_URL=http://localhost:1234/v1
LOCAL_LLM_MODEL=qwen/qwen3-14b
```

Every other setting has a working default — see `config.py` for the full list
(concurrency, timeouts, page-size cap, retention, search providers).

## Use

Typed structured output. Define a pydantic model and get an instance back: the JSON
schema is generated from it and sent to the server to constrain generation, and the reply
is validated on the way home.

```python
import asyncio
from pydantic import BaseModel
from local_llm import complete, extract_claims


class Sentiment(BaseModel):
    label: str
    confidence: float


async def main():
    # Structured, validated
    result = await complete(
        [{"role": "user", "content": "Classify: 'shipping took three weeks'"}],
        schema=Sentiment,
        tool="classify",
    )
    print(result.label, result.confidence)   # a Sentiment, not a dict

    # Fetch a page and pull falsifiable claims with supporting quotes
    extraction = await extract_claims(
        "https://en.wikipedia.org/wiki/Payoneer",
        "How do freelancers receive USD payments?",
    )
    for claim in extraction.claims:
        print(f"[{claim.importance}] {claim.claim}")
        print(f"    “{claim.quote}”")


asyncio.run(main())
```

Plain text output, when you don't need a schema:

```python
text = await complete([{"role": "user", "content": "Summarise this in one line: ..."}])
```

## Observability

Every call is recorded with no extra work at the call site:

```python
from local_llm import list_calls, get_payload, stats, read_live

stats()["totals"]        # calls, errors, tokens in/out, total GPU ms
list_calls(limit=20)     # indexed metadata only — cheap
get_payload(call_id)     # the full prompt and response for one call
read_live()              # calls in flight right now, with partial output
```

Design notes, because they matter for a tool that runs for months:

- The prompt is stored **once**, in a side table. Metadata is roughly **200 bytes per
  call**, so history stays cheap to list and query.
- Payloads (page-sized prompts and responses) are pruned independently of metadata, so
  long-term statistics survive while bulky text ages out. See `prune()` and `vacuum()`.
- Live progress is **not** persisted. It is written to one file per in-flight call and
  deleted on completion, so its footprint is bounded by concurrency, not by history.

## Concurrency

Default is 2 simultaneous model calls (`LOCAL_LLM_CONCURRENCY`). On a single GPU, more
than that thrashes VRAM rather than going faster.

## Testing

With the server up and a model loaded:

```bash
python scripts/smoke_test.py
```

Exercises a plain completion, a page fetch, a typed extraction, and the resulting call
records; exits non-zero on failure.
