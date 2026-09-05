"use client";

import { memo } from "react";

import { formatBytes, formatDuration, formatRate, formatTokens } from "@/lib/format";
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

  // Throughput is computed from model-bearing rows only, not from the lifetime totals.
  //
  // Not every recorded call runs a model: `search:duckduckgo` fetches results over the
  // network and generates no tokens at all. Those rows still contribute their duration to
  // `totals.ms`, so dividing total output tokens by total time charges the model for time
  // it never spent generating. Measured here, that dragged the figure from 52 tok/s down
  // to 47 - a 10% understatement that would look like a hardware regression.
  //
  //   by_model = [{model: "", tokens_out: 0, ms: 4200},        <- search, excluded
  //               {model: "qwen/qwen3-14b", tokens_out: 9k, ms: 173k}]
  //     -> 9000 / 173s = 52 tok/s
  const modelRows = stats.by_model.filter((row) => row.model);
  const modelTokens = modelRows.reduce((sum, row) => sum + row.tokens_out, 0);
  const modelMs = modelRows.reduce((sum, row) => sum + row.ms, 0);

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
        {/* Lifetime average throughput. Worth a headline slot because it is the
            number that makes a degraded setup obvious: this machine sustains roughly
            49 tok/s on a 14B at Q4, so a figure in single digits means the model has
            spilled out of VRAM and is running partly on the CPU. */}
        <BigMetric
          label="throughput"
          value={formatRate(modelTokens, modelMs)}
        />
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
                  <th className="num">Rate</th>
                </tr>
              </thead>
              <tbody>
                {stats.by_model.map((row) => (
                  <tr key={row.model ?? "unknown"}>
                    {/* An empty model is not missing data: search and fetch tools are
                        recorded here too and legitimately run no model. Saying so is
                        better than a blank cell, which reads as a bug. */}
                    <td className="mono">
                      {row.model || (
                        <span className="faint">no model (search / fetch)</span>
                      )}
                    </td>
                    <td className="num">{row.calls}</td>
                    <td className="num" style={row.errors > 0 ? { color: "var(--error)" } : undefined}>
                      {row.errors}
                    </td>
                    <td className="num">{formatDuration(row.ms)}</td>
                    {/* Per model, so a reasoning model's cost is visible next to a
                        coder model's - they differ by more than their weights. */}
                    <td className="num dim">{formatRate(row.tokens_out, row.ms)}</td>
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
