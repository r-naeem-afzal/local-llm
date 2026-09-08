---
name: graphify
description: Build or read the offline code graph of this repository — modules, classes, interfaces and their implementations, constructor dependencies, where each class is constructed, orphan classes and undocumented public definitions. Uses Python's own AST and the TypeScript compiler API, so it costs zero tokens and needs no network. Use this whenever answering a question about the shape of the codebase — what implements an interface, what constructs a class, what depends on what, where a behaviour lives, whether something is dead — instead of grepping or reading many files. Also use it after a refactor to check nothing was orphaned.
---

# graphify — the codebase's structure, for free

## Read the graph before searching for structure

There is a pre-built structural map at **`.graphify/graph.md`**. Read it first when the
question is about shape rather than about the contents of one known function. It answers,
without opening a single source file:

- which classes implement an abstract base
- what each class is handed through its constructor
- where every class is constructed, and whether that is in one place or several
- which classes are never constructed anywhere in the repository
- which public classes and functions have no docstring
- a one-line summary of every module
- what the dashboard's TypeScript exports

Reading one file to answer "what implements `PageSource`" is dramatically cheaper than
grepping for `class .*PageSource` and then opening each hit to check. That is the whole
point of the file existing.

`.graphify/graph.json` holds the same data in machine-readable form — use it when you want
to filter or count rather than read. `.graphify/graph.mmd` is a Mermaid class diagram of
the inheritance hierarchy.

## Rebuild it when it might be stale

```bash
python tools/graphify/graphify.py
```

Takes about a second. It rebuilds automatically after every commit through a `post-commit`
hook, so it is usually current — but it will be stale if you have just edited files in this
session without committing. **If you have changed any Python or TypeScript this session and
then need the graph, rebuild it first.** A stale graph is worse than none, because it looks
authoritative.

To mirror the report into an Obsidian vault as well:

```bash
GRAPHIFY_VAULT=/path/to/vault python tools/graphify/graphify.py
# writes <vault>/03-Build/graphs/<project>/graph.md with Obsidian front matter
```

## Installing the commit hook

Git does not version hooks — deliberately, so cloning a repository cannot make your machine
run code. So it is one command per clone:

```bash
python tools/graphify/install_hook.py                      # rebuild after each commit
python tools/graphify/install_hook.py --vault /path/to/vault # ...and mirror to a vault
python tools/graphify/install_hook.py --uninstall
```

It is a **post**-commit hook, not pre-commit, for two reasons. The output is gitignored, so
a pre-commit run would only add a wait before every commit for a file the commit cannot
contain. And a failing pre-commit hook aborts the commit — this tool is a convenience and
must never stand between someone and committing their work. Git ignores a post-commit
hook's exit status, so the worst case is a stale graph and a warning.

The installer refuses to overwrite a `post-commit` hook it did not write, and will not
accept a `--vault` path that is not a directory (a typo would install a hook that silently
mirrors nothing).

## Why it uses no tokens, and why that is the design rather than a saving

Everything is parsed with **Python's own `ast` module** and, for the dashboard, the
**TypeScript compiler API** already installed in `dashboard/node_modules`. Nothing is
inferred by a model. Three consequences worth knowing:

- **It cannot invent an edge.** Asked which classes implement an interface, a language
  model will occasionally produce a plausible name that does not exist in the codebase.
  A parser only reports what is written. This is the failure the tool exists to prevent —
  a confident wrong answer about structure sends someone editing a class nothing
  constructs.
- **It is deterministic.** The same source always yields the same graph, so a change in the
  graph means the code changed.
- **It is free and offline**, which is what makes "run it on every commit" affordable at
  all.

## How to read the interesting sections

**"Interfaces and their implementations"** is the architecture. This codebase is built as
abstract bases with one subclass per interchangeable behaviour, so this section is the list
of seams where something can be swapped.

**"Where classes are constructed"** finds the composition root. The intended shape here is
that `Toolkit` in `container.py` is the only place naming concrete classes, and everything
else receives its collaborators. So a class with **several construction sites outside
`container.py` is worth looking at** — it may mean a dependency is being reached for rather
than injected.

**"Classes never constructed in this repository"** is a list to *inspect*, not to delete.
Abstract bases, pydantic schemas built by validation, and classes exported for library
users all legitimately appear here. Anything else may be dead.

**"Public definitions with no docstring"** exists because this repository's standard
(`CLAUDE.md`) asks for comments that explain *why* a thing exists. It is the cheapest
possible signal that the standard has slipped.

## Caveats, stated so they are not discovered as bugs

- Constructor calls are detected by the heuristic "a call to a capitalised name", so a
  capitalised *function* is reported as a class. This holds in code following normal Python
  naming and is far cheaper than full type inference.
- Only top-level classes and functions are recorded. A class defined inside a function does
  not appear.
- The TypeScript pass reads syntax only — no type checking — so it reports what each file
  exports and imports, not what those types resolve to.
- A file with a syntax error is listed under "Files that would not parse" and the rest of
  the repository is still mapped, so the graph stays useful mid-edit.
