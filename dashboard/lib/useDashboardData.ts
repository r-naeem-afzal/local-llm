"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiClient } from "./ApiClient";
import type { CallRecord, ClaudeUsage, LiveCall, Stats, SystemSnapshot } from "./types";

/**
 * Everything the dashboard shows, refreshed when the server says something changed.
 *
 * ## Why this is one hook rather than one per panel
 *
 * The panels are read together and shown together. Given separate hooks, each would open
 * its own SSE connection and run its own timer, so a five-panel dashboard would hold five
 * streams and the panels could disagree — VRAM sampled before a model was evicted next to
 * a model list sampled after. One hook means one stream, one refresh, one consistent view.
 *
 * ## Two update paths, for two different kinds of data
 *
 * - **Server-Sent Events** drive history, stats and system state. The server tells us
 *   when a call starts or finishes, so nothing is polled speculatively.
 * - **A short timer** drives the live-progress panel while something is generating. This
 *   is the one thing SSE cannot cover: a running call produces new text continuously
 *   without any discrete "changed" moment to notify, and notifying per token would be
 *   thousands of events. The timer only runs while a call is in flight, so an idle
 *   dashboard makes no requests at all.
 */

/** How often to re-read live progress while something is generating. */
const LIVE_POLL_MS = 900;

export interface DashboardData {
  system: SystemSnapshot | null;
  usage: ClaudeUsage | null;
  calls: CallRecord[];
  stats: Stats | null;
  live: LiveCall[];
  /** True until the first load completes, so panels can show a skeleton not an error. */
  loading: boolean;
  /** Set when the API is unreachable — almost always "the server is not running". */
  error: string | null;
  /** Whether the change-notification stream is currently connected. */
  connected: boolean;
  client: ApiClient;
  refresh: () => void;
}

export function useDashboardData(): DashboardData {
  // Built once and reused. Constructing it inline would make a new client on every
  // render, and since it is a dependency of the effects below, that would tear down and
  // reopen the SSE connection on every render — a reconnect loop.
  const client = useMemo(() => new ApiClient(), []);

  const [system, setSystem] = useState<SystemSnapshot | null>(null);
  const [usage, setUsage] = useState<ClaudeUsage | null>(null);
  const [calls, setCalls] = useState<CallRecord[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [live, setLive] = useState<LiveCall[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  // Tracks whether the component is still mounted. Without it, a fetch that resolves
  // after the user navigates away would call setState on an unmounted component — a
  // React warning at best, and a memory leak in a long-lived tab.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  /**
   * Reload the four "settled" datasets together.
   *
   * `Promise.allSettled`, not `Promise.all`: `all` rejects as soon as one request fails,
   * which would discard three good responses because one panel's endpoint was briefly
   * unhappy. Settled lets each panel succeed or fail on its own — the point of the
   * backend's best-effort probes, carried through to the UI.
   */
  const refresh = useCallback(async () => {
    const [systemResult, usageResult, callsResult, statsResult] = await Promise.allSettled([
      client.system(),
      client.usage(),
      client.calls(100),
      client.stats(),
    ]);

    if (!mounted.current) return;

    if (systemResult.status === "fulfilled") setSystem(systemResult.value);
    if (usageResult.status === "fulfilled") setUsage(usageResult.value);
    if (callsResult.status === "fulfilled") setCalls(callsResult.value.calls);
    if (statsResult.status === "fulfilled") setStats(statsResult.value);

    // Only report an error when *everything* failed, which means the API itself is down.
    // One failing endpoint is a degraded panel, not a broken dashboard, and showing a
    // page-level error for it would hide the panels that are working.
    const allFailed = [systemResult, usageResult, callsResult, statsResult].every(
      (result) => result.status === "rejected",
    );
    setError(
      allFailed
        ? "Cannot reach the API. Start it with: uvicorn local_llm.api:app --port 7878"
        : null,
    );
    setLoading(false);
  }, [client]);

  const refreshLive = useCallback(async () => {
    try {
      const response = await client.live();
      if (mounted.current) setLive(response.live);
    } catch {
      // Live progress is the most disposable data here — it is replaced within a second.
      // Surfacing a transient failure would flicker an error over a working dashboard.
    }
  }, [client]);

  // First load.
  useEffect(() => {
    void refresh();
    void refreshLive();
  }, [refresh, refreshLive]);

  // The change-notification stream.
  useEffect(() => {
    // EventSource is the browser's built-in SSE client: it opens a long-lived HTTP
    // request, fires `onmessage` for each `data:` line the server writes, and reconnects
    // by itself if the connection drops.
    const source = new EventSource(client.eventsUrl);

    source.onopen = () => {
      if (mounted.current) setConnected(true);
    };

    source.onmessage = () => {
      // The event only says "something changed" — it carries no data. So we refetch
      // through the normal endpoints, which keeps one definition of each payload shape
      // instead of a second copy embedded in the event stream. It also means a tab that
      // was backgrounded and missed events simply reads current state rather than
      // replaying a backlog.
      void refresh();
      void refreshLive();
    };

    source.onerror = () => {
      // Fired on a dropped connection *and* while EventSource is retrying, so this is
      // not necessarily fatal — hence marking disconnected without closing. Closing here
      // would defeat the automatic reconnect that is the main reason to use SSE.
      if (mounted.current) setConnected(false);
    };

    return () => source.close();
  }, [client, refresh, refreshLive]);

  // Poll live progress only while something is actually generating.
  useEffect(() => {
    if (live.length === 0) {
      // Nothing in flight: no timer at all, so an idle dashboard is genuinely idle and
      // does not sit making a request every second for no reason.
      return;
    }
    const timer = setInterval(() => void refreshLive(), LIVE_POLL_MS);
    return () => clearInterval(timer);
    // Depends on `live.length`, not `live`: the array is replaced on every poll, so
    // depending on the array itself would clear and recreate the timer each time.
  }, [live.length, refreshLive]);

  return {
    system,
    usage,
    calls,
    stats,
    live,
    loading,
    error,
    connected,
    client,
    refresh: () => {
      void refresh();
      void refreshLive();
    },
  };
}
