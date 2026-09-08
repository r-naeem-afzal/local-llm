/**
 * Parse the dashboard's TypeScript with the TypeScript compiler API. Offline, no tokens.
 *
 * Run by `graphify.py`, which reads a single JSON object from stdout. It is a separate
 * Node script rather than part of the Python tool for one reason: the only correct
 * offline parser for TypeScript is the TypeScript compiler itself, and it is already
 * sitting in `dashboard/node_modules`. Writing a regex parser in Python instead would
 * produce something that is wrong in ways nobody notices — it would silently omit a
 * component and the graph would quietly under-report the UI.
 *
 * `ts.createSourceFile` is used rather than a full program with type checking. That is a
 * deliberate trade: a full `ts.createProgram` resolves every import and every type across
 * the whole project, which is what `tsc` already does on demand and takes seconds. What
 * this tool needs is structure — what does each file export, what does it import, which
 * components and hooks exist — and all of that is in the syntax alone. Nothing here needs
 * to know that `LiveCall` is an interface with eight fields.
 *
 * Jargon: an *AST* (abstract syntax tree) is source code parsed into a tree of labelled
 * nodes. `forEachChild` walks that tree one level at a time; we only look at the top level
 * of each file, because exports and imports live there.
 */

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const ROOT = process.cwd();
const DASHBOARD = path.join(ROOT, "dashboard");

// Loaded from the dashboard's own install rather than a global one, so the version that
// parses the code is the version that compiles it. A mismatch there would mean the graph
// disagreeing with `tsc` about what is valid syntax.
const ts = require(path.join(DASHBOARD, "node_modules", "typescript"));

// Skipped wholesale. `node_modules` alone is tens of thousands of files, and parsing it
// would take longer than the rest of the repository put together for no benefit.
const SKIP = new Set(["node_modules", ".next", "dist", "build", ".git", ".graphify"]);

/** Every .ts/.tsx file under a directory, recursively, skipping generated trees. */
function collect(dir, found = []) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    // A directory that vanished or cannot be read is not worth failing the run over —
    // the graph is a convenience, and a partial graph beats no graph.
    return found;
  }
  for (const entry of entries) {
    if (SKIP.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      collect(full, found);
    } else if (/\.tsx?$/.test(entry.name) && !entry.name.endsWith(".d.ts")) {
      found.push(full);
    }
  }
  return found;
}

/**
 * Read one file's top-level structure.
 *
 *   "export const LivePanel = memo(LivePanelInner);"
 *     ->  { exports: ["LivePanel"], … }
 *
 *   "export function useDashboardData(): DashboardData { … }"
 *     ->  { exports: ["useDashboardData"], hooks: ["useDashboardData"] }
 *
 * Components and hooks are separated from other exports by naming convention, which is how
 * React itself distinguishes them: a hook must start with `use`, and a component is
 * capitalised. That is a convention rather than a rule the compiler enforces, so it is
 * recorded as a hint and the raw export list is kept alongside it.
 */
function readFile(file) {
  const text = fs.readFileSync(file, "utf8");
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);

  const exports = [];
  const imports = [];
  const interfaces = [];

  const nameOf = (node) => (node && node.name ? node.name.getText(source) : "");

  ts.forEachChild(source, (node) => {
    if (ts.isImportDeclaration(node)) {
      // `node.moduleSpecifier` includes its quotes in the source text, so they are
      // stripped — otherwise every import would be recorded as `"./types"` with quotes.
      const target = node.moduleSpecifier.getText(source).replace(/['"]/g, "");
      imports.push(target);
      return;
    }

    const isExported = (node.modifiers || []).some(
      (m) => m.kind === ts.SyntaxKind.ExportKeyword,
    );

    if (ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node)) {
      interfaces.push(nameOf(node));
      if (isExported) exports.push(nameOf(node));
      return;
    }

    if (ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node)) {
      if (isExported && nameOf(node)) exports.push(nameOf(node));
      return;
    }

    if (ts.isVariableStatement(node)) {
      const exported = (node.modifiers || []).some(
        (m) => m.kind === ts.SyntaxKind.ExportKeyword,
      );
      if (!exported) return;
      for (const declaration of node.declarationList.declarations) {
        const name = declaration.name.getText(source);
        if (name) exports.push(name);
      }
    }
  });

  const relative = path.relative(ROOT, file).split(path.sep).join("/");
  return {
    path: relative,
    exports,
    imports: imports.filter((i) => i.startsWith(".") || i.startsWith("@/")),
    externalImports: imports.filter((i) => !i.startsWith(".") && !i.startsWith("@/")),
    interfaces,
    components: exports.filter((n) => /^[A-Z]/.test(n)),
    hooks: exports.filter((n) => /^use[A-Z]/.test(n)),
  };
}

function main() {
  if (!fs.existsSync(DASHBOARD)) {
    process.stdout.write(JSON.stringify({ skipped: "no dashboard directory" }));
    return;
  }

  const files = collect(DASHBOARD).sort();
  const parsed = [];
  const failures = [];

  for (const file of files) {
    try {
      parsed.push(readFile(file));
    } catch (error) {
      // One unparseable file must not lose the other thirty. Recorded so a silent gap in
      // the report is traceable to a cause.
      failures.push({
        path: path.relative(ROOT, file).split(path.sep).join("/"),
        error: String(error && error.message ? error.message : error),
      });
    }
  }

  process.stdout.write(
    JSON.stringify({
      typescript_version: ts.version,
      files: parsed,
      failures,
      components: parsed.flatMap((f) => f.components),
      hooks: parsed.flatMap((f) => f.hooks),
    }),
  );
}

main();
