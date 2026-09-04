"use client";

import { memo } from "react";

import { formatBytes, formatDuration, formatTokens } from "@/lib/format";
import type { Stats } from "@/lib/types";
import { BigMetric, Metric, Panel } from "./ui";

/**
 * Lifetime totals for the local models.
 *
 * These survive retention: payloads are pruned but metadata rows are kept forever, so
 * these figures stay correct long after the prompts behind them have gone. That is the
 * whole reason the store splits the two tables.
 */
function StatsPanelInner({ stats }: { stats: Stats | null }) {
  if (!stats) {
    return (
      <Panel title="Local totals">
        <p className="empty">Loading…</p>
      </Panel>
    );
  }

  const { totals } = stats;
  // Error rate is more useful than a raw error count: three failures out of five is a
  // broken setup, three out of three hundred is normal operation, and the count alone
  // cannot tell those apart.
  const errorRate = totals.calls > 0 ? (totals.errors / totals.calls) * 100 : 0;

  return (
    <Panel
      title="Local totals"
      action={<span className="faint">db {formatBytes(stats.db_bytes)}</span>}
    >
      <div className="metric-grid">
        <BigMetric label="calls" value={totals.calls.toLocaleString()} />
        <BigMetric
          label="error rate"
          value={`${errorRate.toFixed(0)}%`}
          colour={errorRate > 20 ? "var(--error)" : errorRate > 5 ? "var(--thinking)" : "var(--ok)"}
        />
        <BigMetric label="tokens in" value={formatTokens(totals.tokens_in)} />
        <BigMetric label="tokens out" value={formatTokens(totals.tokens_out)} />
      </div>

      <Metric label="Total model time" value={formatDuration(totals.ms)} />

      {stats.by_tool.length > 0 && (
        <>
          <p className="section-label" style={{ marginTop: 14 }}>By tool</p>
          {stats.by_tool.map((row) => (
            <Metric key={row.tool ?? "unknown"} label={row.tool ?? "unknown"} value={row.n} />
          ))}
        </>
      )}

      {stats.by_model.length > 0 && (
        <>
          <p className="section-label" style={{ marginTop: 14 }}>By model</p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">Calls</th>
                  <th className="num">Errors</th>
                  <th className="num">Time</th>
                </tr>
              </thead>
              <tbody>
                {stats.by_model.map((row) => (
                  <tr key={row.model ?? "unknown"}>
                    <td className="mono">{row.model ?? "unknown"}</td>
                    <td className="num">{row.calls}</td>
                    <td className="num" style={row.errors > 0 ? { color: "var(--error)" } : undefined}>
                      {row.errors}
                    </td>
                    <td className="num">{formatDuration(row.ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  );
}

/**
 * Memoized so a change in another panel's data cannot re-render this one. Without this,
 * every 900 ms live-progress tick repainted the entire dashboard.
 */
export const StatsPanel = memo(StatsPanelInner);
