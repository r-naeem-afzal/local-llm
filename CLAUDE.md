# Project Instructions — Ideas / Local LLM Toolkit

## Code style: simple and explained

Code here is read cold, months later, by one solo developer. Optimise for that reader,
not for concision or cleverness.

1. **Prefer the simple construction.** If a clever one-liner needs a comment to be
   readable at all, write the boring version instead. Simplicity outranks brevity and
   micro-performance.
2. **Comment the *why*, not the *what*.** `# increment i` is noise. `# retry once
   because the model server drops the first request after an idle eviction` is the point.
3. **At every real decision, record three things:** why this choice, what happens as a
   result, and **what breaks if we don't do it**. The failure mode being defended against
   must be written down — otherwise a future reader deletes the guard as pointless.
4. **Every data transformation gets a worked example in the comment.** Show the concrete
   before and after shape, e.g.
   `# {"choices":[{"delta":{"content":"Hel"}}]}  ->  "Hel"  (we yield just the text)`
   so the reader never has to mentally execute the function.
5. **Gloss jargon in plain words.** Any term of art — generator, context manager,
   back-pressure, WAL, SSE, schema-constrained decoding, editable install — gets a short
   ordinary-language explanation next to it. Assume no missing intelligence, only missing
   context.
6. **Flag anything happening "under the hood."** Where behaviour comes from a library,
   the runtime, or the OS rather than the visible code, say so and say what it actually
   does. That is where a reader's mental model silently breaks.

This standard applies **retroactively**: when you touch an existing module, bring its
comments up to it as part of the change.

## Architecture: classic OO, following SOLID

Write classes with clear responsibilities and injected dependencies — not a flat set of
module-level functions operating on module globals.

- **Single responsibility** — one class, one job. A repository persists; it does not also
  fetch pages or decide retention policy.
- **Open/closed** — extend by adding a class, not by growing an `if provider ==` chain.
  Interchangeable implementations (search providers, page fetchers, model backends) are a
  base class or Protocol with one subclass each.
- **Liskov substitution** — any implementation works wherever the abstraction is declared,
  with no caller checking which one it got.
- **Interface segregation** — narrow, purpose-shaped interfaces over one class exposing
  everything a caller might want.
- **Dependency inversion** — take collaborators through the constructor
  (`def __init__(self, store: CallRepository)`), never reach for a module-level singleton
  inside a method. The marker of getting this right is testability: a test supplies fakes
  without patching globals.

"Classic" means ordinary readable object orientation. Do not build deep inheritance trees
or ceremonial abstraction layers in SOLID's name — a small obvious class beats a clever
one, but a class still beats a bag of functions sharing module state.

## Language split

Python for logic (pipelines, model calls, data processing, persistence, CLI, services).
React + TypeScript on Next.js for any interface.

## Cost discipline

Claude effort `high` at most. No subagent fan-out for cost. Delegate mechanical work to
the local models via the toolkit; Claude spends its usage on judgement only.
