/**
 * Small shared building blocks.
 *
 * They exist so a status colour or a bar threshold is defined once. When VRAM being over
 * 90% should read as a warning, that judgement belongs in one place — otherwise each
 * panel invents its own threshold and the colours stop meaning anything consistent.
 */

export function Panel({
  title,
  action,
  children,
}: {
  title: string;
  /** Optional right-aligned content in the heading — a count, a badge, a button. */
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <h2>
        <span>{title}</span>
        {action}
      </h2>
      {children}
    </section>
  );
}

export function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

export function BigMetric({
  label,
  value,
  colour,
}: {
  label: string;
  value: React.ReactNode;
  colour?: string;
}) {
  return (
    <div>
      <div className="big-metric" style={colour ? { color: colour } : undefined}>
        {value}
      </div>
      <div className="big-metric-label">{label}</div>
    </div>
  );
}

/**
 * A usage bar that colours itself by how full it is.
 *
 * The thresholds encode a real constraint rather than being arbitrary. This machine has
 * 16 GB of VRAM and a single 14B model at 32K context occupies about 94% of it, so
 * "above 90%" is not a warning of impending trouble — it is the normal state when a model
 * is loaded, and it means no second model will fit. Amber says "full, as expected"; red at
 * 97% says "not even the KV cache has room".
 */
export function UsageBar({ percent }: { percent: number }) {
  // Clamped because a bar wider than its track would overflow the panel, and a negative
  // width silently fails to render at all.
  const clamped = Math.max(0, Math.min(100, percent));
  const colour =
    clamped >= 97 ? "var(--error)" : clamped >= 90 ? "var(--thinking)" : "var(--ok)";
  return (
    <div className="bar">
      <div className="bar-fill" style={{ width: `${clamped}%`, background: colour }} />
    </div>
  );
}

/** Maps a call status to its colour, so every panel agrees on what "running" looks like. */
export function statusColour(status: string | null): string {
  switch (status) {
    case "ok":
      return "var(--ok)";
    case "error":
      return "var(--error)";
    case "running":
      return "var(--running)";
    case "thinking":
      return "var(--thinking)";
    case "starting":
      return "var(--thinking)";
    default:
      return "var(--idle)";
  }
}

export function StatusBadge({ status }: { status: string | null }) {
  const label = status ?? "unknown";
  // "starting" and "thinking" both mean work in progress, so they share the amber style
  // — the distinction is in the label, which is where it belongs.
  const variant =
    label === "ok" || label === "error" || label === "running" || label === "thinking"
      ? label
      : label === "starting"
        ? "thinking"
        : "idle";
  // Only genuinely active states pulse. A stored "ok" row animating would imply
  // something is still happening when nothing is.
  const isLive = label === "running" || label === "thinking" || label === "starting";
  return (
    <span className={`badge badge-${variant}`}>
      <span
        className={`dot ${isLive ? "dot-live" : ""}`}
        style={{ background: statusColour(label) }}
      />
      {label}
    </span>
  );
}
