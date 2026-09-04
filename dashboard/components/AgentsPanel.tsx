"use client";

import { memo } from "react";

import { formatDuration, formatTokens } from "@/lib/format";
import type { ActiveAgent, AgentActivity } from "@/lib/types";

import { BigMetric, Panel, StatusBadge } from "./ui";

/**
 * Claude agents running right now — the panel that makes subagent fan-out safe to use.
 *
 * It sits above the local model's Live panel because, under the standing cost rule, a
 * Claude fan-out is the most expensive thing that can be happening on this machine, and
 * "what is happening right now" is the reason the page gets opened.
 *
 * ## Why some token figures carry a `~`
 *
 * Because only some of them are measured. A background agent's completion notification
 * reports its real usage, and those rows are exact. A foreground agent's usage is written
 * to no local file, so its figure is estimated from the size of the prompt in and the
 * report out — which understates by roughly thirteen times, since almost all of an
 * agent's spend is the files it read.
 *
 * Marking the estimates rather than quietly mixing them in is the whole point: the two
 * kinds of number differ by an order of magnitude, so a column that presented them
 * identically would be worse than no column. The `~` marks the row and a footnote
 * explains it once, which keeps the table scannable while leaving the caveat unmissable.
 */
function AgentsPanelInner({
  activity,
  notificationsAction,
}: {
  activity: AgentActivity | null;
  /** The enable/disable notifications control, supplied by the page. */
  notificationsAction?: React.ReactNode;
}) {
  if (activity === null) {
    return (
      <Panel title="Claude agents" action={notificationsAction}>
        <p className="empty">Loading…</p>
      </Panel>
    );
  }

  if (activity.error) {
    // The reader could not find the transcripts at all. Shown as the whole body, matching
    // the usage panel — a broken data source must not be mistaken for "nothing running".
    return (
      <Panel title="Claude agents" action={notificationsAction}>
        <p className="empty">{activity.error}</p>
      </Panel>
    );
  }

  const totalTokens = activity.agents.reduce((total, agent) => total + agent.tokens, 0);

  return (
    <Panel title="Claude agents" action={notificationsAction}>
      <div className="metric-grid">
        <BigMetric
          label="running"
          value={activity.running.toLocaleString()}
          // Coloured only when something is actually running, so a glance at the page
          // distinguishes "spending plan usage" from "idle" without reading the number.
          colour={activity.running > 0 ? "var(--running)" : undefined}
        />
        <BigMetric label="just finished" value={activity.finished.toLocaleString()} />
        <BigMetric
          label="failed"
          value={activity.errored.toLocaleString()}
          colour={activity.errored > 0 ? "var(--error)" : undefined}
        />
        <BigMetric
          label={activity.tokens_estimated ? "tokens (part est.)" : "tokens"}
          value={formatTokens(totalTokens)}
        />
      </div>

      {activity.agents.length === 0 ? (
        <p className="empty">No Claude agents active.</p>
      ) : (
        <div className="table-scroll" style={{ marginTop: 12 }}>
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Agent</th>
                <th>Type</th>
                <th className="num">Elapsed</th>
                <th className="num">Tokens</th>
              </tr>
            </thead>
            <tbody>
              {activity.agents.map((agent) => (
                // Keyed by the tool_use_id, which is stable across polls — so a row keeps
                // its identity as it changes from running to finished instead of being
                // torn down and rebuilt.
                <AgentRow key={agent.key} agent={agent} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activity.tokens_estimated && activity.agents.length > 0 && (
        <p className="section-label" style={{ marginTop: 10 }}>
          Rows marked ~ are estimates from prompt and result size, and understate real
          spend by roughly 13×. Only background agents report measured usage.
        </p>
      )}
    </Panel>
  );
}

function AgentRow({ agent }: { agent: ActiveAgent }) {
  return (
    <tr>
      <td>
        <StatusBadge status={agent.status} />
      </td>
      <td>
        {/* The description leads because it is the label the launcher wrote: three
            "Explore" rows are indistinguishable, three descriptions are not. */}
        <span className="mono">{agent.description || "(no description)"}</span>
        {agent.background && <span className="faint"> · background</span>}
      </td>
      <td className="mono faint">
        {agent.subagent_type}
        {/* An empty model means the agent inherited the parent's, which is left blank
            rather than guessed — a guessed model displayed as fact is worse than none. */}
        {agent.model && ` · ${agent.model}`}
      </td>
      <td className="num">{formatDuration(agent.elapsed_ms)}</td>
      {/* A measured figure is shown plainly; an estimate is prefixed and dimmed, so the
          difference is visible while scanning the column rather than needing the
          footnote. The two differ by an order of magnitude, so they must not look
          alike. */}
      <td className={agent.tokens_measured ? "num" : "num faint"}>
        {agent.tokens_measured ? "" : "~"}
        {formatTokens(agent.tokens)}
      </td>
    </tr>
  );
}

/**
 * Memoized so a change in another panel's data cannot re-render this one. Without this,
 * every 900 ms live-progress tick repainted the entire dashboard.
 */
export const AgentsPanel = memo(AgentsPanelInner);
