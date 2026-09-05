"use client";

import { useEffect, useState } from "react";

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

/**
 * A tiny inline chart of recent values, drawn as an SVG polyline.
 *
 * Hand-drawn rather than pulled from a charting library, for the same reason the rest of
 * this dashboard has no UI framework: it must keep working with no network. A charting
 * library would also be several hundred kilobytes to draw a sixty-pixel line.
 *
 * Its job is to answer a question a single number cannot: *is this moving?* A gauge
 * reading 96% looks identical whether it has been steady for an hour or just jumped from
 * 30%, and that difference is exactly what someone watching a model run wants to see.
 *
 * The data is a plain list of values, oldest first:
 *
 *   [12, 40, 98, 97] with max 100
 *     -> a polyline rising steeply then flattening near the top of the box
 */
export function Sparkline({
  values,
  max,
  colour = "var(--accent)",
  height = 26,
  label,
}: {
  values: number[];
  /** Top of the scale. Fixed rather than derived, so the line is comparable over time. */
  max: number;
  colour?: string;
  height?: number;
  /** Screen-reader description; the drawing itself conveys nothing to a reader. */
  label?: string;
}) {
  // Two points are the minimum for a line. With fewer, render the empty box rather than
  // nothing, so the panel does not visibly resize once the second sample arrives.
  if (values.length < 2) {
    return <div className="spark" style={{ height }} aria-hidden="true" />;
  }

  // A fixed 100-unit viewBox with preserveAspectRatio="none" lets the SVG stretch to
  // whatever width the panel gives it, so no measurement of the container is needed.
  const width = 100;
  const safeMax = max > 0 ? max : 1;
  const step = width / (values.length - 1);

  const points = values
    .map((value, index) => {
      const x = index * step;
      // SVG's y axis grows downward, so a high value must map to a *small* y. Without
      // this inversion the chart would be upside down — a busy GPU drawn as a trough.
      const clamped = Math.max(0, Math.min(safeMax, value));
      const y = height - (clamped / safeMax) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  // Close the path back along the baseline so the area under the line can be filled,
  // which reads as a level far better than a bare stroke at this size.
  const area = `${points} ${width},${height} 0,${height}`;

  return (
    <svg
      className="spark"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      height={height}
      role={label ? "img" : "presentation"}
      aria-label={label}
    >
      <polygon points={area} fill={colour} opacity={0.14} />
      <polyline
        points={points}
        fill="none"
        stroke={colour}
        strokeWidth={1.5}
        // Non-scaling so the stroke stays 1.5px however far the SVG is stretched
        // horizontally; without it the line thins to invisibility on a wide panel.
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * How long ago a reading was taken, refreshed on its own so it counts up in real time.
 *
 * Worth its own component because it is the honest answer to "is this live?". A dashboard
 * that has silently stopped updating looks exactly like one where nothing is happening;
 * a counter that climbs past a few seconds is the difference between the two.
 */
export function Freshness({ ts }: { ts: string | null }) {
  const [, force] = useState(0);

  useEffect(() => {
    // Re-render once a second purely to advance the displayed age. The state value is
    // unused — incrementing it is just the way to ask React to repaint.
    const timer = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  if (!ts) return null;
  const age = Math.max(0, Math.round((Date.now() - Date.parse(ts)) / 1000));

  // Amber past five seconds: the telemetry poll runs every second, so anything older than
  // a few seconds means requests are failing or the tab has been throttled by the browser.
  const stale = age > 5;
  return (
    <span className="faint mono" style={stale ? { color: "var(--thinking)" } : undefined}>
      {age <= 1 ? "just now" : `${age}s ago`}
    </span>
  );
}
