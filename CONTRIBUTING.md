# Contributing

Contributions are welcome. This file exists mostly to save you from a surprised review:
the codebase has one unusual standard, and it is about comments rather than code.

## Where help would land best

In descending order of usefulness:

1. **A test suite.** There is none. `toolkit/scripts/smoke_test.py` exercises the real
   paths against a live model and is the current check, which means nothing can be verified
   without a GPU and a running model server. The valuable first step is unit coverage for
   the pieces that need neither: `UrlCanonicaliser`, `QuoteVerifier`, `HeadingIndex`,
   `StreamAccumulator`, `JsonResponseParser`, and the SQL dialect differences in
   `database.py`. Every one of those takes plain values and returns plain values, which was
   deliberate.
2. **Verification on Linux and macOS.** Only Windows 11 has been exercised. The Python is
   portable in principle; `Start-Dashboard.cmd` is not, and an equivalent shell script
   would be a genuinely useful contribution. So would a bug report saying what broke.
3. **Semantic deduplication of claims.** Claims are deduplicated by URL only, so the same
   fact from five sources counts as five claims and a report reads as consensus when it is
   one fact echoed. Embedding the claims and collapsing near-duplicates is the fix. This is
   the largest single quality gap in the pipeline.
4. **Search providers.** `search.py` is a base class with one subclass per provider; adding
   one means adding a class and listing it. More coverage directly raises the pipeline's
   ceiling, because search coverage — not extraction quality — is what limits it.
5. **Weighting recency and source quality.** Both are extracted and displayed and neither
   affects ranking, so a 2019 forum post sits level with a 2026 primary source.

## The comment standard, which is the part that trips people up

Read [CLAUDE.md](CLAUDE.md) before writing code. It is the file that governs this
repository, and it asks for something stricter than usual: comments must explain **why**,
and must name **what breaks if the code were not written that way**.

The reasoning is that this code is read cold, months later, by someone who was not in the
conversation where the decision was made. A comment that restates the code is noise; a
comment that records the failure mode is what stops the guard being deleted as pointless.

So this is not useful:

```python
# increment the counter
counter += 1
```

and this is:

```python
# Retry once, because the model server drops the first request after an idle eviction.
# Without this, the first call after a quiet period fails with a bare ConnectionError and
# looks like the server is down.
```

Four more things it asks for, each with a reason:

- **A worked example at every data transformation**, showing the concrete before and after
  shape, so a reader never has to mentally execute the function:
  `# {"choices":[{"delta":{"content":"Hel"}}]} -> "Hel"`
- **Jargon glossed in plain words.** Generator, context manager, back-pressure, SSE,
  schema-constrained decoding — a short ordinary-language note next to the term. Assume a
  reader with no missing intelligence and plenty of missing context.
- **Anything happening "under the hood" flagged.** Where behaviour comes from a library,
  the runtime or the OS rather than the visible code, say so and say what it actually does.
  That is where a reader's mental model silently breaks.
- **The simple construction preferred.** If a clever one-liner needs a comment to be
  readable at all, write the boring version instead.

If you touch an existing module, bringing its comments up to this standard is part of the
change.

## Architecture

Classic object orientation following SOLID, with dependencies injected through
constructors. There are no module-level singletons and no global state.

```text
toolkit/src/local_llm/
  config.py      Settings — data only, no behaviour
  container.py   Toolkit — the composition root, the ONLY place naming concrete classes
  client.py      the model client: streaming, schema-constrained output, recording
  extract.py     page fetching and claim extraction
  fetch_browser.py  the browser fetch path (optional, Playwright)
  membership.py  deterministic "is this term in a list on this page"
  search.py      pluggable search providers
  pipeline.py    the research orchestrator
  routing.py     which model for which role
  loader.py      deliberate model loading and VRAM budgeting
  store.py       call history and live progress
  database.py    per-engine SQL; the only module that knows a dialect
  monitor.py     GPU, host, models, plan usage
  agents.py      which agents are running now
  api.py         the FastAPI surface
  mcp_server.py  the MCP tools
```

Two rules that matter when adding something:

- **Extend by adding a class, not by growing an `if provider ==` chain.** Interchangeable
  implementations — search providers, page sources, extractors, database backends, model
  backends — are a base class with one subclass each.
- **Take collaborators through the constructor.** The test of getting this right is that a
  test can supply a fake without patching a module attribute. `Toolkit` in `container.py`
  is the only place that wires concrete classes together; if you find yourself importing a
  concrete class anywhere else, that is the smell.

Do not build deep inheritance trees or ceremonial abstraction layers in SOLID's name. A
small obvious class beats a clever one — but a class still beats a bag of functions sharing
module state.

## Language split

Python for logic: pipelines, model calls, data processing, persistence, CLI, services.
React and TypeScript on Next.js for anything with an interface.

## Running things

```bash
cd toolkit
pip install -e ".[extraction,api,monitor,search,mcp]"
python scripts/smoke_test.py          # the current end-to-end check; needs a live model
python scripts/agent_probe.py --watch  # agent monitoring, no UI needed
```

```bash
cd dashboard
npx tsc --noEmit    # must stay clean; the project runs TypeScript in strict mode
npm run build
```

## Measurement over assertion

The one cultural thing worth knowing. Nearly every non-obvious decision in this repository
was driven by a measurement, and [toolkit/STATUS.md](toolkit/STATUS.md) records them —
including several that overturned an earlier, confidently held assumption. A 280x latency
penalty from an IPv6 lookup, a 15x throughput regression traced to an inference runtime
rather than a model, a ranking call silently truncating and falling back to unsorted search
results.

If you change something for performance or quality, a number in the pull request is worth
more than an explanation. If you are fixing something that was wrong, adding a line to that
file is welcome.

## Pull requests

- Keep the diff to one concern.
- Say what you measured, if the change is about performance or output quality.
- `npx tsc --noEmit` clean for dashboard changes.
- Do not commit `.env`, or anything under `.local-llm-data/`. Both are gitignored; the
  second contains every prompt and response recorded on that machine.

## Licence

By contributing you agree that your contributions are licensed under the MIT licence, the
same terms as the rest of the project. See [LICENSE](LICENSE).
