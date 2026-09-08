"""Build a structural map of this repository from its syntax trees. No model, no tokens.

Everything here is derived by parsing source with Python's own `ast` module — the same
parser the interpreter uses — so the output is deterministic, free, offline, and instant.
That choice is the whole point of the tool rather than an implementation detail:

* **Deterministic.** The same source always produces the same graph, so a diff in the graph
  means the code changed, not that a model answered differently today.
* **Free and offline.** It runs on every commit without spending tokens or needing a
  network, which is what makes "run it always" affordable.
* **Incapable of inventing an edge.** A language model asked "which classes implement this
  interface" will occasionally produce a plausible name that does not exist. `ast` cannot;
  it only reports what is written.

The failure mode being defended against is a confident wrong answer about the shape of the
codebase — the kind that sends someone editing a class that nothing constructs.

## Jargon, in plain words

An **AST** (abstract syntax tree) is the parsed form of source code: instead of characters,
a tree of nodes saying "this is a class, its name is X, it has these methods". Python
exposes its own via the `ast` module, so no third-party parser is needed.

**Composition root** is the one place in a codebase that decides which concrete classes get
used together. Here it is `Toolkit` in `container.py`. Finding it automatically is useful
because everything else is supposed to receive its collaborators rather than construct
them, so a constructor call *outside* the root is worth looking at.

## What comes out

    .graphify/graph.json    the whole graph, machine-readable
    .graphify/graph.md      a summary written for a person (or an agent) to read
    .graphify/graph.mmd     a Mermaid diagram of the class hierarchy

The output directory is gitignored on purpose. It is derived data: committing it would
produce a conflict on almost every merge, and a stale copy in history is worse than no copy
because it looks authoritative.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Directories that never contain source worth mapping. Checked as path parts rather than
# by prefix so a nested `node_modules` deep in a tree is skipped too.
SKIP_DIRS = frozenset({
    ".git", ".graphify", "node_modules", "__pycache__", ".venv", "venv",
    ".next", "dist", "build", ".pytest_cache", ".ruff_cache", ".local-llm-data",
    "site-packages",
})


@dataclass
class FunctionInfo:
    name: str
    line: int
    is_async: bool
    args: list[str]
    returns: str
    documented: bool


@dataclass
class ClassInfo:
    name: str
    line: int
    bases: list[str]
    methods: list[FunctionInfo] = field(default_factory=list)
    documented: bool = False
    # True when any method is decorated `@abstractmethod`. Marks the interfaces, which is
    # what makes the inheritance section readable — this codebase is built as a set of
    # abstract bases with one subclass each, and that is the structure worth seeing.
    is_abstract: bool = False
    # Annotated constructor parameters, i.e. the collaborators this class is *given*.
    # Recorded because the house architecture requires dependencies to arrive through
    # `__init__`; a class with none that still reaches for other classes is the smell.
    injects: list[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: str
    module: str
    docline: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    # Imports split by origin: an edge to another module in this project is architecture,
    # an edge to httpx is a dependency choice. Mixing them makes the graph unreadable.
    internal_imports: list[str] = field(default_factory=list)
    external_imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    # Every `Name(...)` call whose name looks like a class, with the enclosing scope.
    # This is how the composition root is found and how orphan classes are detected.
    instantiates: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str = ""


class PythonModuleParser:
    """Turns one Python file into a `ModuleInfo`.

    A class per concern so each can be exercised on a string of source with no filesystem;
    `parse_source` takes text precisely so a test never needs a temporary directory.
    """

    def __init__(self, project_packages: frozenset[str]) -> None:
        # Used to decide whether an import is internal. Passed in rather than guessed so
        # the tool works on a repository with different package names.
        self._packages = project_packages

    def parse_file(self, path: Path, root: Path) -> ModuleInfo:
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ModuleInfo(path=relative, module=path.stem, docline="",
                              parse_error=f"unreadable: {exc}")
        return self.parse_source(source, relative, path.stem)

    def parse_source(self, source: str, relative: str, module: str) -> ModuleInfo:
        """Parse source text into a module record.

            "class A(B):\\n    def __init__(self, s: Store): ..."
              ->  ModuleInfo(classes=[ClassInfo(name="A", bases=["B"],
                                                injects=["Store"])])

        A syntax error is recorded on the module and the rest of the repository is still
        mapped. Without that, one file mid-edit would empty the whole graph — and the graph
        is most useful exactly when the code is being worked on.
        """
        info = ModuleInfo(path=relative, module=module, docline="")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            info.parse_error = f"syntax error line {exc.lineno}: {exc.msg}"
            return info

        info.docline = self._first_docline(ast.get_docstring(tree))

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._record_import(node, info)
            elif isinstance(node, ast.ClassDef):
                info.classes.append(self._read_class(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info.functions.append(self._read_function(node))
            elif isinstance(node, ast.Assign):
                self._record_exports(node, info)

        info.instantiates = self._find_instantiations(tree)
        return info

    @staticmethod
    def _first_docline(docstring: str | None) -> str:
        """The first sentence of a docstring, for a one-line summary.

            "Fetch web pages and pull claims.\\n\\nLong explanation…"  ->  "Fetch web pages
            and pull claims."

        Only the first line, because the summary tables in `graph.md` need to stay one row
        per module — and a module docstring in this codebase can run to eighty lines.
        """
        if not docstring:
            return ""
        return docstring.strip().split("\n", 1)[0].strip()

    def _record_import(self, node: ast.Import | ast.ImportFrom, info: ModuleInfo) -> None:
        """Split one import statement into internal and external targets.

            "from .store import CallRepository"  ->  internal_imports += ["store"]
            "import httpx"                       ->  external_imports += ["httpx"]

        A relative import (`from .x import y`) is always internal — `node.level > 0` is how
        the AST records the leading dots.
        """
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                target = (node.module or "").split(".")[0]
                info.internal_imports.append(target or ".")
                return
            root = (node.module or "").split(".")[0]
        else:
            root = node.names[0].name.split(".")[0] if node.names else ""

        if not root:
            return
        if root in self._packages:
            info.internal_imports.append(root)
        else:
            info.external_imports.append(root)

    @staticmethod
    def _record_exports(node: ast.Assign, info: ModuleInfo) -> None:
        """Capture `__all__ = [...]`, the declared public surface of a module."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    info.exports = [
                        element.value
                        for element in node.value.elts
                        if isinstance(element, ast.Constant) and isinstance(element.value, str)
                    ]

    def _read_class(self, node: ast.ClassDef) -> ClassInfo:
        info = ClassInfo(
            name=node.name,
            line=node.lineno,
            bases=[self._name_of(base) for base in node.bases],
            documented=bool(ast.get_docstring(node)),
        )
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            method = self._read_function(item)
            info.methods.append(method)
            if any(self._name_of(d) == "abstractmethod" for d in item.decorator_list):
                info.is_abstract = True
            if item.name == "__init__":
                info.injects = self._injected_types(item)
        return info

    def _read_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        return FunctionInfo(
            name=node.name,
            line=node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            args=[a.arg for a in node.args.args if a.arg not in ("self", "cls")],
            returns=self._name_of(node.returns) if node.returns else "",
            documented=bool(ast.get_docstring(node)),
        )

    def _injected_types(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """The annotated types a constructor accepts — the collaborators it is handed.

            "def __init__(self, settings: Settings, repo: CallRepository | None = None)"
              ->  ["Settings", "CallRepository"]

        Only types that start with a capital letter are kept, so `int`, `str` and `float`
        do not clutter the dependency graph with things that are values rather than
        collaborators. Optionals are unwrapped, because `Repo | None` is still a Repo
        dependency — the None only says it has a default.
        """
        found: list[str] = []
        for arg in node.args.args:
            if arg.arg in ("self", "cls") or arg.annotation is None:
                continue
            for name in self._unwrap_annotation(arg.annotation):
                if name and name[0].isupper() and name not in found:
                    found.append(name)
        return found

    def _unwrap_annotation(self, node: ast.expr) -> list[str]:
        """Flatten an annotation into the type names inside it.

            Settings                  ->  ["Settings"]
            CallRepository | None     ->  ["CallRepository", "None"]
            list[SearchProvider]      ->  ["list", "SearchProvider"]
        """
        if isinstance(node, ast.BinOp):
            return self._unwrap_annotation(node.left) + self._unwrap_annotation(node.right)
        if isinstance(node, ast.Subscript):
            return [self._name_of(node.value)] + self._unwrap_annotation(node.slice)
        if isinstance(node, ast.Tuple):
            names: list[str] = []
            for element in node.elts:
                names.extend(self._unwrap_annotation(element))
            return names
        return [self._name_of(node)]

    def _find_instantiations(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Every call to something that looks like a class, with the scope it happens in.

            "def page_fetcher(self): return PageFetcher(settings)"
              ->  [{"class": "PageFetcher", "scope": "page_fetcher", "line": 2}]

        The capital-letter test is a heuristic and is worth naming as one: it will treat a
        capitalised function as a class. It is used anyway because the alternative — full
        type inference — is a different tool entirely, and the heuristic holds in a codebase
        that follows normal Python naming.
        """
        found: list[dict[str, Any]] = []
        scopes: dict[int, str] = {}

        # Record which function each line belongs to, so a call can be attributed to the
        # method that makes it. Walking the tree twice is simpler than threading scope
        # through a recursive visitor, and these files are small.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                for line in range(node.lineno, end + 1):
                    scopes.setdefault(line, node.name)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._name_of(node.func)
            if not name or not name[0].isupper():
                continue
            found.append({
                "class": name,
                "scope": scopes.get(node.lineno, "<module>"),
                "line": node.lineno,
            })
        return found

    @staticmethod
    def _name_of(node: ast.expr | None) -> str:
        """The readable name of an expression node.

            Name(id="Settings")                          ->  "Settings"
            Attribute(value=Name("ast"), attr="parse")    ->  "parse"
            Constant(value="Settings")                    ->  "Settings"   (string annotation)
        """
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ""


class RepositoryScanner:
    """Walks the repository and parses every Python file it should look at."""

    def __init__(self, root: Path, parser: PythonModuleParser) -> None:
        self._root = root
        self._parser = parser

    def scan(self) -> list[ModuleInfo]:
        modules: list[ModuleInfo] = []
        for path in sorted(self._root.rglob("*.py")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            modules.append(self._parser.parse_file(path, self._root))
        return modules


class GraphAnalyser:
    """Derives the interesting relationships from the parsed modules.

    Kept apart from parsing because these are *questions asked of* the syntax, not facts
    read from it — and because each one can be checked against a hand-built list of
    `ModuleInfo` with no files involved.
    """

    def __init__(self, modules: list[ModuleInfo]) -> None:
        self._modules = modules

    def implementations(self) -> dict[str, list[str]]:
        """Base class -> the classes that inherit from it.

        The structure worth seeing in this codebase, which is deliberately built as
        abstract bases with one subclass each per interchangeable behaviour.
        """
        tree: dict[str, list[str]] = {}
        for module in self._modules:
            for klass in module.classes:
                for base in klass.bases:
                    if base and base not in ("object", "BaseModel", "ABC", "Enum"):
                        tree.setdefault(base, []).append(f"{klass.name} ({module.path})")
        return dict(sorted(tree.items()))

    def constructor_sites(self) -> dict[str, list[str]]:
        """Class -> where it is constructed. Reveals the composition root."""
        sites: dict[str, list[str]] = {}
        for module in self._modules:
            for call in module.instantiates:
                sites.setdefault(call["class"], []).append(
                    f"{module.path}:{call['line']} in {call['scope']}()"
                )
        return dict(sorted(sites.items()))

    def orphan_classes(self) -> list[str]:
        """Classes defined here and constructed nowhere here.

        Not automatically dead: a class exported for library users, a pydantic schema built
        by validation, or an abstract base is expected to have no constructor call. So this
        is a list to *look at*, not a list to delete — which is why the report says so.
        """
        defined: dict[str, str] = {}
        for module in self._modules:
            for klass in module.classes:
                if not klass.is_abstract:
                    defined[klass.name] = module.path
        constructed = {call["class"] for m in self._modules for call in m.instantiates}
        return sorted(f"{name} ({path})" for name, path in defined.items()
                      if name not in constructed)

    def import_edges(self) -> list[tuple[str, str]]:
        """Module-to-module edges, deduplicated, for the dependency picture."""
        edges: set[tuple[str, str]] = set()
        for module in self._modules:
            for target in module.internal_imports:
                if target and target != module.module:
                    edges.add((module.module, target))
        return sorted(edges)

    def undocumented(self) -> list[str]:
        """Public classes and functions with no docstring.

        Included because this repository's own standard asks for comments that explain why
        a thing exists. A missing docstring on a public class is the cheapest possible
        signal that the standard has slipped, and it costs nothing to compute.
        """
        gaps: list[str] = []
        for module in self._modules:
            for klass in module.classes:
                if not klass.documented and not klass.name.startswith("_"):
                    gaps.append(f"class {klass.name} ({module.path}:{klass.line})")
            for function in module.functions:
                if not function.documented and not function.name.startswith("_"):
                    gaps.append(f"def {function.name} ({module.path}:{function.line})")
        return gaps

    def external_dependencies(self) -> dict[str, int]:
        """Third-party package -> how many modules import it."""
        counts: dict[str, int] = {}
        for module in self._modules:
            for name in set(module.external_imports):
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


class ReportWriter:
    """Renders the graph as JSON, as markdown, and as a Mermaid diagram.

    Three formats because they have three different readers: a program, a person, and a
    diagram viewer. The markdown one is the file an agent should read, so it leads with the
    things that answer "where does this behaviour live".
    """

    def __init__(self, modules: list[ModuleInfo], analyser: GraphAnalyser) -> None:
        self._modules = modules
        self._analyser = analyser

    def write(self, out_dir: Path, ts_summary: dict[str, Any] | None = None) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [
            self._write_json(out_dir / "graph.json", ts_summary),
            self._write_markdown(out_dir / "graph.md", ts_summary),
            self._write_mermaid(out_dir / "graph.mmd"),
        ]
        return written

    def _write_json(self, path: Path, ts_summary: dict[str, Any] | None) -> Path:
        payload = {
            "generated_by": "tools/graphify/graphify.py (offline AST, no model)",
            "python": {
                "modules": [
                    {
                        "path": m.path,
                        "summary": m.docline,
                        "classes": [
                            {
                                "name": c.name,
                                "line": c.line,
                                "bases": c.bases,
                                "abstract": c.is_abstract,
                                "injects": c.injects,
                                "methods": [f.name for f in c.methods],
                                "documented": c.documented,
                            }
                            for c in m.classes
                        ],
                        "functions": [f.name for f in m.functions],
                        "internal_imports": sorted(set(m.internal_imports)),
                        "external_imports": sorted(set(m.external_imports)),
                        "exports": m.exports,
                        "parse_error": m.parse_error,
                    }
                    for m in self._modules
                ],
                "implementations": self._analyser.implementations(),
                "constructor_sites": self._analyser.constructor_sites(),
                "orphan_classes": self._analyser.orphan_classes(),
                "import_edges": self._analyser.import_edges(),
                "undocumented": self._analyser.undocumented(),
                "external_dependencies": self._analyser.external_dependencies(),
            },
            "typescript": ts_summary or {},
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return path

    def _write_markdown(self, path: Path, ts_summary: dict[str, Any] | None) -> Path:
        modules = [m for m in self._modules if m.classes or m.functions]
        lines: list[str] = [
            "# Repository graph",
            "",
            "Generated from the syntax trees by `tools/graphify/graphify.py`. No model was",
            "involved, so every edge below is written in the source rather than inferred.",
            "Regenerated automatically after each commit; this file is gitignored.",
            "",
            f"{len(self._modules)} Python modules parsed, "
            f"{sum(len(m.classes) for m in self._modules)} classes, "
            f"{sum(len(m.functions) for m in self._modules)} module-level functions.",
            "",
        ]

        errors = [m for m in self._modules if m.parse_error]
        if errors:
            lines += ["## Files that would not parse", ""]
            lines += [f"- `{m.path}` — {m.parse_error}" for m in errors]
            lines.append("")

        lines += ["## Interfaces and their implementations", "",
                  "The abstraction points. Each base class below is the seam where a",
                  "behaviour can be swapped.", ""]
        implementations = self._analyser.implementations()
        if implementations:
            for base, subclasses in implementations.items():
                lines.append(f"- **{base}**")
                lines += [f"  - {s}" for s in subclasses]
        else:
            lines.append("_None found._")
        lines.append("")

        lines += ["## Where classes are constructed", "",
                  "A class constructed in exactly one place, in a composition root, is the",
                  "intended shape here. Several construction sites for the same class is",
                  "worth a look.", ""]
        for name, sites in self._analyser.constructor_sites().items():
            if len(sites) == 1:
                lines.append(f"- `{name}` — {sites[0]}")
            else:
                lines.append(f"- `{name}` — **{len(sites)} sites**")
                lines += [f"  - {s}" for s in sites]
        lines.append("")

        lines += ["## Constructor dependencies", "",
                  "What each class is handed. This is the injected-collaborator graph; a",
                  "class that takes none and still uses others is not following the house",
                  "architecture.", ""]
        for module in modules:
            for klass in module.classes:
                if klass.injects:
                    lines.append(f"- `{klass.name}` ({module.path}) <- "
                                 + ", ".join(f"`{d}`" for d in klass.injects))
        lines.append("")

        orphans = self._analyser.orphan_classes()
        lines += ["## Classes never constructed in this repository", "",
                  "Expected for schemas, exported library types and abstract bases. Worth",
                  "checking for anything else — it may be dead.", ""]
        lines += [f"- {o}" for o in orphans] or ["_None._"]
        lines.append("")

        undocumented = self._analyser.undocumented()
        lines += ["## Public definitions with no docstring", "",
                  f"{len(undocumented)} found. This repository's standard asks for the",
                  "reasoning behind a thing, so these are the cheapest gaps to close.", ""]
        lines += [f"- {u}" for u in undocumented[:40]] or ["_None._"]
        if len(undocumented) > 40:
            lines.append(f"- …and {len(undocumented) - 40} more (see `graph.json`)")
        lines.append("")

        lines += ["## Module summaries", "", "| Module | Classes | Summary |",
                  "| --- | --- | --- |"]
        for module in modules:
            names = ", ".join(c.name for c in module.classes) or "—"
            summary = module.docline[:90].replace("|", "\\|")
            lines.append(f"| `{module.path}` | {names} | {summary} |")
        lines.append("")

        dependencies = self._analyser.external_dependencies()
        lines += ["## Third-party imports", "",
                  ", ".join(f"`{name}` ({count})" for name, count in dependencies.items())
                  or "_None._", ""]

        if ts_summary and ts_summary.get("files"):
            lines += ["## TypeScript", "",
                      f"{len(ts_summary['files'])} files parsed with the TypeScript compiler",
                      "API — also offline, also no model.", "",
                      "| File | Exports |", "| --- | --- |"]
            for entry in ts_summary["files"]:
                exports = ", ".join(entry.get("exports", [])) or "—"
                lines.append(f"| `{entry['path']}` | {exports[:110]} |")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_mermaid(self, path: Path) -> Path:
        """A Mermaid class diagram of the inheritance hierarchy.

        Only inheritance, not imports: the import graph of a package this size is a hairball
        that communicates nothing, whereas the base-to-implementation edges are the actual
        design. Names are sanitised because Mermaid rejects dots and slashes in node ids.
        """
        lines = ["classDiagram"]
        for base, subclasses in self._analyser.implementations().items():
            safe_base = self._safe(base)
            for entry in subclasses:
                child = self._safe(entry.split(" (")[0])
                lines.append(f"    {safe_base} <|-- {child}")
        if len(lines) == 1:
            lines.append("    class NoInheritanceFound")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def _safe(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


class TypeScriptAnalyser:
    """Runs the sibling Node script that parses TypeScript with the compiler API.

    Delegated to Node because the only correct offline TypeScript parser is the TypeScript
    compiler itself, and it is already present in `dashboard/node_modules`. Reimplementing
    it in Python with regular expressions would produce a parser that is wrong in ways
    nobody notices until it silently omits a component.

    Absent Node or an absent dashboard is not an error — the Python graph is the point and
    the TypeScript section is an addition, so a missing toolchain degrades the report rather
    than failing the run.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._script = root / "tools" / "graphify" / "graphify_ts.mjs"

    def analyse(self) -> dict[str, Any]:
        if not self._script.exists():
            return {}
        if not (self._root / "dashboard" / "node_modules" / "typescript").exists():
            return {"skipped": "typescript not installed in dashboard/node_modules"}
        try:
            result = subprocess.run(
                ["node", str(self._script)],
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"skipped": f"node unavailable: {type(exc).__name__}"}
        if result.returncode != 0:
            return {"skipped": f"exit {result.returncode}: {result.stderr[:200]}"}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return {"skipped": f"unparseable output: {exc}"}


class VaultExporter:
    """Copies the readable report into an Obsidian vault, if one is configured.

    Off unless `GRAPHIFY_VAULT` is set, and that default matters: this tool ships in a
    public repository, so it must not assume a vault exists, must not guess at a path on
    someone else's disk, and must not carry the author's own directory layout in its
    source. A contributor who clones this gets the `.graphify/` output and nothing else.

    Where a vault *is* configured, the graph becomes linkable from the notes that discuss
    the design — which is the point of putting it there rather than leaving it in a
    gitignored folder nobody opens. The vault is itself unversioned, so the "output is
    never committed" requirement holds there too, for a different reason.

        GRAPHIFY_VAULT=D:/vault  ->  D:/vault/03-Build/graphs/local-llm/graph.md
    """

    def __init__(self, vault_root: Path | None, project: str) -> None:
        self._vault = vault_root
        self._project = project

    @classmethod
    def from_environment(cls, project: str) -> VaultExporter:
        configured = os.environ.get("GRAPHIFY_VAULT", "").strip()
        root = Path(configured).expanduser() if configured else None
        return cls(root if root and root.is_dir() else None, project)

    def export(self, report: Path) -> Path | None:
        if self._vault is None:
            return None
        target_dir = self._vault / "03-Build" / "graphs" / self._project
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "graph.md"
            # A front-matter header is prepended rather than the file being copied
            # verbatim, because Obsidian uses front matter for its own indexing and a note
            # without it is second-class in search and graph view.
            body = report.read_text(encoding="utf-8")
            front_matter = "\n".join([
                "---",
                f"title: {self._project} code graph",
                "type: reference",
                "generated: true",
                "---",
                "",
                "_Machine-generated by `tools/graphify/graphify.py` from the syntax trees."
                " Overwritten on every commit — do not edit by hand._",
                "",
                "",
            ])
            target.write_text(front_matter + body, encoding="utf-8")
            return target
        except OSError:
            # A vault on a disconnected drive must not fail a commit. The local report is
            # already written by this point, so the useful work is done either way.
            return None


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    packages = frozenset({"local_llm"})

    scanner = RepositoryScanner(root, PythonModuleParser(packages))
    modules = scanner.scan()
    analyser = GraphAnalyser(modules)
    ts_summary = TypeScriptAnalyser(root).analyse()

    written = ReportWriter(modules, analyser).write(root / ".graphify", ts_summary)

    exported = VaultExporter.from_environment(root.name).export(root / ".graphify" / "graph.md")

    print(f"graphify: {len(modules)} python modules, "
          f"{sum(len(m.classes) for m in modules)} classes")
    if ts_summary.get("skipped"):
        print(f"graphify: typescript skipped — {ts_summary['skipped']}")
    elif ts_summary.get("files"):
        print(f"graphify: {len(ts_summary['files'])} typescript files")
    for path in written:
        print(f"graphify: wrote {path.relative_to(root).as_posix()}")
    if exported is not None:
        print(f"graphify: mirrored to vault at {exported}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
