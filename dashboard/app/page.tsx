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
    telemetry,
    telemetryHistory,
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
            {/* Two stacks, not grid rows. Cards in a stack start where the one
                above them ends, so there is no cross-column alignment to leave
                gaps. Grouped by the question each answers: the machine on the
                left, what it has cost on the right. */}
            <div className="columns">
              {/* The rail: gauges, read at a glance. Narrow on purpose - they are a
                  handful of numbers, and the width they used to take belonged to the
                  panels carrying tables. */}
              {/* The rail holds everything read at a glance: what is running, and how
                  the machine is doing. The main column holds everything read as a
                  table. Grouping them this way also makes the two sides come out
                  roughly the same height, which is what stops one of them ending
                  early and leaving a hole. */}
              <div className="col col-rail">
                <GpuPanel
                  snapshot={system}
                  telemetry={telemetry}
                  history={telemetryHistory}
                />
                <HostPanel
                  snapshot={system}
                  telemetry={telemetry}
                  history={telemetryHistory}
                />

                {/* Agents and Live come last in the rail because they are the only two
                    panels whose height changes: an agent starting adds rows, and a model
                    call adds a status block and a text tail.

                    Placed above the gauges, as they were, every one of those changes
                    pushed the GPU and Host cards down and back up - measured at 236px of
                    page movement as a call went from idle to thinking to answering. From
                    the bottom of the rail they grow into empty space and nothing moves.

                    The general rule this follows: variable-height content goes below
                    fixed-height content, never above it. */}
                <AgentsPanel activity={agents} notificationsAction={notificationsAction} />
              </div>

              {/* Flowed rather than placed. Three panels of unequal height cannot fill
                  two explicit columns evenly - one always ends short - so the browser
                  balances them instead. */}
              <div className="dense">
                {/* The two narrow tables share a row; the wide one keeps the full
                    width it actually uses. */}
                <div className="dense-pair">
                  <ModelsPanel snapshot={system} />
                  <UsagePanel usage={usage} />
                </div>
                <StatsPanel stats={stats} />

                {/* Live calls sit here, in the main column, rather than in the rail.
                    Two reasons. It is the panel worth the most space when something is
                    happening - a status line, a subject, and streaming text are wider
                    than a 300px rail can show without wrapping every line - and this is
                    where the page had a large empty region below the totals.

                    Being last in its column also means it can grow without moving
                    anything: an extraction going from starting to thinking to answering
                    used to shift the cards below it by a measured 236px. */}
                <LivePanel live={live} />
              </div>
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
