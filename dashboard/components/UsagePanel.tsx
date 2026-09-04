"use client";

import { formatTokens } from "@/lib/format";
import type { AgentUsage, ClaudeUsage } from "@/lib/types";
import { BigMetric, Metric, Panel } from "./ui";

/**
 * Claude plan usage in the current 5-hour window, beside the local model activity.
 *
 * The point of showing it here is accountability for the whole premise: work is supposed
 * to move onto the local GPU so the metered plan is spent only on judgement. That claim is
 * unverifiable unless both numbers sit on one screen.
 *
 * ## Why the four token categories are never summed into one figure
 *
 * They are billed very differently, and the mix is extreme in practice. One measured
 * 5-hour window held 73.5M tokens, of which 72.2M were cache *reads* — billed at a large
 * discount — and only 854 were fresh input. A single "total tokens" number would overstate
 * real spend by roughly fifty times, which makes it worse than no number at all: it looks
 * authoritative and would lead to stopping work that was costing almost nothing.
 *
 * So "fresh input" is the headline, and cache traffic is shown separately as context.
 */
export function UsagePanel({ usage }: { usage: ClaudeUsage | null }) {
  if (!usage) {
    return (
      <Panel title="Claude usage">
        <p className="empty">Loading…</p>
      </Panel>
    );
  }

  if (usage.error) {
    return (
      <Panel title="Claude usage">
        <p className="empty">{usage.error}</p>
      </Panel>
    );
  }

  const sum = (rows: AgentUsage[], field: keyof AgentUsage): number =>
    rows.reduce((total, row) => total + (row[field] as number), 0);

  const all = [...usage.main_loop, ...usage.subagents];
  const freshInput = sum(all, "input_tokens");
  const output = sum(all, "output_tokens");
  const cacheRead = sum(all, "cache_read_tokens");
  const cacheWrite = sum(all, "cache_write_tokens");

  // Subagents are the expensive failure mode: fan-out multiplies plan spend, and the
  // standing rule is not to use it. Any non-zero value here should be noticed
  // immediately, so it is coloured as a warning rather than shown as a neutral count.
  const hasSubagents = usage.subagent_messages > 0;

  return (
    <Panel
      title={`Claude usage · ${usage.window_hours}h window`}
      action={<span className="faint">{usage.sessions} sessions</span>}
    >
      <div className="metric-grid">
        <BigMetric label="fresh input" value={formatTokens(freshInput)} />
        <BigMetric label="output" value={formatTokens(output)} />
        <BigMetric label="messages" value={usage.messages.toLocaleString()} />
        <BigMetric
          label="subagent msgs"
          value={usage.subagent_messages.toLocaleString()}
          colour={hasSubagents ? "var(--thinking)" : "var(--ok)"}
        />
      </div>

      <p className="section-label" style={{ marginTop: 14 }}>
        Cache traffic
      </p>
      {/* Kept visually subordinate on purpose. These are the largest numbers on the panel
          and the least significant per token, so presenting them with equal weight would
          dominate the reader's attention with the cheapest thing on the screen. */}
      <Metric label="Cache reads (discounted)" value={formatTokens(cacheRead)} />
      <Metric label="Cache writes (premium)" value={formatTokens(cacheWrite)} />

      {all.length > 0 && (
        <>
          <p className="section-label" style={{ marginTop: 14 }}>
            By model
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Source</th>
                  <th className="num">Msgs</th>
                  <th className="num">In</th>
                  <th className="num">Out</th>
                </tr>
              </thead>
              <tbody>
                {usage.main_loop.map((row) => (
                  <UsageRow key={`main-${row.model}`} row={row} source="main loop" />
                ))}
                {usage.subagents.map((row) => (
                  <UsageRow key={`sub-${row.model}`} row={row} source="subagent" warn />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  );
}

function UsageRow({
  row,
  source,
  warn,
}: {
  row: AgentUsage;
  source: string;
  warn?: boolean;
}) {
  return (
    <tr>
      <td className="mono">{row.model}</td>
      <td style={warn ? { color: "var(--thinking)" } : undefined}>{source}</td>
      <td className="num">{row.messages.toLocaleString()}</td>
      <td className="num">{formatTokens(row.input_tokens)}</td>
      <td className="num">{formatTokens(row.output_tokens)}</td>
    </tr>
  );
}
