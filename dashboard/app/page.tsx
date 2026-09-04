"use client";

import { HistoryPanel } from "@/components/HistoryPanel";
import { LivePanel } from "@/components/LivePanel";
import { StatsPanel } from "@/components/StatsPanel";
import { GpuPanel, HostPanel, ModelsPanel } from "@/components/SystemPanels";
import { UsagePanel } from "@/components/UsagePanel";
import { useDashboardData } from "@/lib/useDashboardData";

/**
 * The dashboard.
 *
 * `"use client"` because everything here is live: it holds state, opens a Server-Sent
 * Events connection and runs timers, none of which a server component can do. There is no
 * server-rendering benefit to give up — the data is only meaningful at the moment it is
 * read, so pre-rendering it on the server would just ship a stale snapshot.
 *
 * ## Panel order
 *
 * Deliberate, and reading top to bottom answers the questions in the order they are
 * actually asked:
 *
 * 1. **Live** — is anything happening right now? The reason to open the page.
 * 2. **GPU / Host / Models** — is the machine able to do the work, and what is resident?
 * 3. **Claude usage / Local totals** — what has the metered plan cost, beside what the
 *    local models did? These sit together because the whole premise is that work moved
 *    from the first to the second, and that is only checkable side by side.
 * 4. **History** — what happened earlier, and what exactly was said?
 */
export default function DashboardPage() {
  const { system, usage, calls, stats, live, loading, error, connected, client, refresh } =
    useDashboardData();

  return (
    <main className="shell">
      <div className="topbar">
        <h1>Local LLM Dashboard</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Whether the change-notification stream is connected. Worth surfacing:
              disconnected means the page has silently stopped updating itself, and
              without this indicator a frozen dashboard is indistinguishable from an
              idle machine — the exact ambiguity this tool exists to remove. */}
          <span className={`badge ${connected ? "badge-ok" : "badge-idle"}`}>
            <span
              className={`dot ${connected ? "dot-live" : ""}`}
              style={{ background: connected ? "var(--ok)" : "var(--idle)" }}
            />
            {connected ? "live" : "not connected"}
          </span>
          {/* A manual refresh, for when the stream is down or you simply do not want to
              wait for the next notification. */}
          <button onClick={refresh}>Refresh</button>
        </div>
      </div>

      {error && <div className="banner">{error}</div>}

      {loading && !error ? (
        <p className="empty">Loading…</p>
      ) : (
        <>
          {/* Live gets the full width: its streaming text is the widest content here, and
              wrapping it into a narrow column would make the tail unreadable. */}
          <div className="grid-wide">
            <LivePanel live={live} />
          </div>

          <div className="grid">
            <GpuPanel snapshot={system} />
            <HostPanel snapshot={system} />
            <ModelsPanel snapshot={system} />
          </div>

          <div className="grid">
            <UsagePanel usage={usage} />
            <StatsPanel stats={stats} />
          </div>

          <div className="grid-wide">
            <HistoryPanel calls={calls} client={client} />
          </div>
        </>
      )}
    </main>
  );
}
