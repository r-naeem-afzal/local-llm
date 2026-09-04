"use client";

import { useMemo } from "react";

import { AgentsPanel } from "@/components/AgentsPanel";
import { HistoryPanel } from "@/components/HistoryPanel";
import { LivePanel } from "@/components/LivePanel";
import { StatsPanel } from "@/components/StatsPanel";
import { GpuPanel, HostPanel, ModelsPanel } from "@/components/SystemPanels";
import { Toasts } from "@/components/Toasts";
import { UsagePanel } from "@/components/UsagePanel";
import { useAgentNotifications } from "@/lib/useAgentNotifications";
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
 * 1. **Claude agents** — is a fan-out running, and what is it costing? First because it
 *    is the most expensive thing that can be happening, and because the standing rule is
 *    that agents may only be used while they are visibly monitored.
 * 2. **Live** — is a local model call in flight right now?
 * 3. **GPU / Host / Models** — is the machine able to do the work, and what is resident?
 * 4. **Claude usage / Local totals** — what has the metered plan cost, beside what the
 *    local models did? These sit together because the whole premise is that work moved
 *    from the first to the second, and that is only checkable side by side.
 * 5. **History** — what happened earlier, and what exactly was said?
 */
export default function DashboardPage() {
  const {
    system,
    usage,
    calls,
    stats,
    live,
    agents,
    callsOffset,
    callsPageSize,
    callsHasMore,
    setCallsOffset,
    loading,
    error,
    connected,
    client,
    refresh,
  } = useDashboardData();

  // Lives here rather than inside AgentsPanel so the toasts keep working while the panel
  // is off screen, and so one place owns the browser-permission state. A panel that owned
  // it would stop notifying the moment it unmounted.
  const notifications = useAgentNotifications(agents);

  // Held stable across renders. Passing `<NotificationsToggle …/>` inline would build a
  // new element object every render, and since AgentsPanel is memoized on its props, a
  // changing element would defeat the memo entirely — the panel would re-render on every
  // live-progress tick, which is exactly what the memo exists to stop.
  const notificationsAction = useMemo(
    () => <NotificationsToggle notifications={notifications} />,
    [notifications],
  );

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
        <div className="layout">
          {/* The main column: everything whose height is stable. */}
          <div>
            <div className="grid-wide">
              <AgentsPanel activity={agents} notificationsAction={notificationsAction} />
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
              <HistoryPanel
                calls={calls}
                client={client}
                offset={callsOffset}
                pageSize={callsPageSize}
                hasMore={callsHasMore}
                onOffsetChange={setCallsOffset}
              />
            </div>
          </div>

          {/* Live gets its own pinned column. It is the only panel whose height changes
              second by second — a call starts, a second joins it, the tail grows, both
              finish — and in the main flow every one of those changes shoved the panels
              below it up and down the page. Nothing shares its vertical axis here, so it
              can resize freely without moving anything. */}
          <div className="live-column">
            <LivePanel live={live} />
          </div>
        </div>
      )}

      {/* Outside the loading branch on purpose: a toast must still be able to appear while
          the first load is in progress, and it is fixed-position so its place in the tree
          does not affect where it is drawn. */}
      <Toasts toasts={notifications.toasts} onDismiss={notifications.dismiss} />
    </main>
  );
}

/**
 * The desktop-notification opt-in, shown in the agents panel's header.
 *
 * Three states rather than two, because "the browser cannot do this" and "you have not
 * turned it on" call for different words — a button that does nothing when clicked is
 * worse than no button.
 *
 * The click matters: browsers only honour a permission request that came from a real user
 * gesture, and they refuse others silently. So the request has to originate here, in an
 * `onClick`, and cannot be moved into an effect on mount however convenient that would be.
 */
function NotificationsToggle({
  notifications,
}: {
  notifications: ReturnType<typeof useAgentNotifications>;
}) {
  if (!notifications.desktopSupported) {
    return <span className="faint">in-page alerts only</span>;
  }

  if (notifications.desktopEnabled) {
    return (
      <button onClick={notifications.disableDesktop} title="Stop desktop notifications">
        desktop alerts on
      </button>
    );
  }

  return (
    <button onClick={notifications.enableDesktop} title="Notify me when an agent starts or ends">
      enable desktop alerts
    </button>
  );
}
