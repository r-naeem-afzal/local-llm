"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiClient } from "./ApiClient";
import type {
  AgentActivity,
  CallRecord,
  ClaudeUsage,
  LiveCall,
  Stats,
  SystemSnapshot,
  Telemetry,
  TelemetrySample,
} from "./types";

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

/**
 * How often to sample GPU, CPU and RAM.
 *
 * This timer runs unconditionally, because these are *sampled* values rather than
 * event-driven ones: GPU load and VRAM drift continuously whether or not a model call is
 * happening. They were previously refreshed only when the change-stream fired, and that
 * stream fires on call activity — so an idle machine, or one long generation with no call
 * boundary, left the gauges frozen at whatever they read when the last call started. They
 * looked live only by coincidence.
 *
 * One second is affordable because `/telemetry` is deliberately cheap: about 25 ms, against
 * roughly 550 ms for a full `/system`. Getting there needed two backend fixes — a cached
 * reachability probe and a cached process-table scan — without which this poll would have
 * kept a CPU core busy doing nothing but monitoring.
 */
const TELEMETRY_POLL_MS = 1000;

/**
 * How often to re-read the slow-moving inventory: resident models, installed models, the
 * RAM held by inference processes.
 *
 * Still refreshed on every change-stream event as well. The timer is a floor, not the
 * primary path — it exists so that a model loaded or evicted by something other than this
 * dashboard (the Bionic GUI, a TTL expiry) shows up within a few seconds rather than
 * waiting for the next model call to happen.
 */
const SYSTEM_POLL_MS = 8000;

/**
 * How many telemetry samples the sparklines keep.
 *
 * 90 at one second apart is a minute and a half of history — long enough to see a
 * generation start and finish, short enough that the array stays trivial to re-render.
 */
const TELEMETRY_HISTORY = 90;

/**
 * How long to keep polling after the live list empties.
 *
 * Without this, the timer stopped the instant the list came back empty — and a single
 * empty reading is not proof that nothing is running. A progress file can be missed for
 * one poll, or a call can sit between two generations. The panel then went idle and
 * stayed idle until the next Server-Sent Event happened to revive it, turning one dropped
 * frame into a visible gap. Six seconds of coasting costs a handful of requests and makes
 * the panel stop blinking.
 */
const LIVE_IDLE_GRACE_MS = 6000;

/**
 * How often to re-read Claude agent activity.
 *
 * This timer runs **unconditionally**, unlike the live-progress one below which only runs
 * while something is in flight. That difference is the whole point of the panel: the event
 * most worth knowing about is an agent *starting*, and that by definition happens while
 * the list is empty. A timer that only ran when agents were already known could never
 * discover the first one, and the panel would stay empty until some unrelated change
 * happened to trigger an SSE refresh.
 *
 * Slower than the live poll because agents last tens of seconds to minutes, not
 * milliseconds, and each request makes the server stat a directory of transcripts.
 */
const AGENTS_POLL_MS = 2000;

/**
 * How many history rows to show at once.
 *
 * The panel used to render every one of the last 100 calls. That is a lot of DOM to
 * rebuild whenever a call finishes, and — more to the point — a wall of rows nobody reads
 * past the top of. A page of 25 fits a screen, and the endpoint has supported `limit` and
 * `offset` since it was written, so paging costs no server work.
 */
const CALLS_PAGE_SIZE = 25;

/**
 * Update state only when the value actually changed.
 *
 * This is the fix for the dashboard visibly flickering whenever anything updated. The
 * polling loops refetch every dataset on a timer, and `setState` with a *newly parsed*
 * object is always a change as far as React is concerned — a fresh `JSON.parse` produces
 * a new reference even when every byte is identical. So a live-progress tick every 900 ms
 * replaced `system`, `usage`, `calls` and `stats` as well, and re-rendered the GPU panel,
 * the usage tables and a hundred rows of history along with it. The screen was rebuilding
 * itself roughly once a second, all of it, for data that had not moved.
 *
 * Comparing the serialised form and keeping the previous reference means an unchanged
 * dataset produces no state update at all, so React never re-renders the panels that read
 * it. Together with `React.memo` on the panels themselves, only the panel whose data
 * genuinely moved repaints.
 *
 * The cost is a `JSON.stringify` per dataset per poll. That is far cheaper than the
 * render it prevents, and it runs on data that was just parsed from exactly this format.
 */
function useChangeGuard<T>(setter: (value: T) => void): (value: T) => void {
  const previous = useRef<string | undefined>(undefined);
  return useCallback(
    (value: T) => {
      const serialised = JSON.stringify(value);
      if (serialised === previous.current) return;
      previous.current = serialised;
      setter(value);
    },
    [setter],
  );
}

export interface DashboardData {
  system: SystemSnapshot | null;
  /** GPU, CPU and RAM, resampled every second. Null until the first reading lands. */
  telemetry: Telemetry | null;
  /** A bounded ring of recent samples, for the sparklines. Oldest first. */
  telemetryHistory: TelemetrySample[];
  usage: ClaudeUsage | null;
  calls: CallRecord[];
  stats: Stats | null;
  live: LiveCall[];
  /** Claude agents running now, plus any that finished within the server's window. */
  agents: AgentActivity | null;
  /** Index of the first history row currently shown. */
  callsOffset: number;
  /** How many rows a page holds. */
  callsPageSize: number;
  /** Whether the server has older rows beyond this page. */
  callsHasMore: boolean;
  /** Jump to a page. Clamped at zero; triggers an immediate refresh. */
  setCallsOffset: (offset: number) => void;
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
  const [agents, setAgents] = useState<AgentActivity | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetrySample[]>([]);

  // Every dataset goes through a change guard, so a poll that returns identical data
  // causes no state update and therefore no re-render anywhere downstream.
  const putSystem = useChangeGuard(setSystem);
  const putUsage = useChangeGuard(setUsage);
  const putCalls = useChangeGuard(setCalls);
  const putStats = useChangeGuard(setStats);
  const putLive = useChangeGuard(setLive);
  const putAgents = useChangeGuard(setAgents);
  // Telemetry deliberately does NOT go through the change guard. Its whole purpose is to
  // show movement, and two consecutive identical readings are meaningful information —
  // "the GPU is genuinely steady" — not a redundant update to suppress. Guarding it would
  // also freeze the "updated Xs ago" indicator whenever the numbers happened to repeat.

  const [callsOffset, setCallsOffsetState] = useState(0);
  const [callsHasMore, setCallsHasMore] = useState(false);

  /**
   * The offset the next fetch should use.
   *
   * Mirrored into a ref because `refresh` must not depend on it. If the paging offset
   * were a dependency, `refresh` would be a new function on every page change, and the
   * Server-Sent Events effect that depends on `refresh` would tear down and reopen its
   * connection each time someone clicked "older" — reconnecting the live stream to turn a
   * page.
   */
  const callsOffsetRef = useRef(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  // Tracks whether the component is still mounted. Without it, a fetch that resolves
  // after the user navigates away would call setState on an unmounted component — a
  // React warning at best, and a memory leak in a long-lived tab.
  // Guards against overlapping agent polls — see refreshAgents. A ref rather than state
  // because changing it must not re-render.
  const agentsInFlight = useRef(false);

  // When something was last actually in flight, so the live poll can coast for a few
  // seconds afterwards instead of stopping dead on one empty reading.
  const lastLiveSeen = useRef(0);

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
      client.calls(CALLS_PAGE_SIZE, callsOffsetRef.current),
      client.stats(),
    ]);

    if (!mounted.current) return;

    if (systemResult.status === "fulfilled") putSystem(systemResult.value);
    if (usageResult.status === "fulfilled") putUsage(usageResult.value);
    if (callsResult.status === "fulfilled") {
      putCalls(callsResult.value.calls);
      setCallsHasMore(callsResult.value.has_more);
    }
    if (statsResult.status === "fulfilled") putStats(statsResult.value);

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
  }, [client, putSystem, putUsage, putCalls, putStats]);

  const refreshLive = useCallback(async () => {
    try {
      const response = await client.live();
      if (response.live.length > 0) lastLiveSeen.current = Date.now();
      if (mounted.current) putLive(response.live);
    } catch {
      // Live progress is the most disposable data here — it is replaced within a second.
      // Surfacing a transient failure would flicker an error over a working dashboard.
    }
  }, [client, putLive]);

  const refreshTelemetry = useCallback(async () => {
    try {
      const reading = await client.telemetry();
      if (!mounted.current) return;
      setTelemetry(reading);
      setTelemetryHistory((previous) => {
        const sample: TelemetrySample = {
          t: Date.parse(reading.ts) || Date.now(),
          gpuPct: reading.gpu.utilisation_pct,
          vramPct: reading.gpu.used_pct,
          cpuPct: reading.cpu_pct,
          tempC: reading.gpu.temperature_c,
        };
        // Append and drop the oldest, so the array is bounded no matter how long the tab
        // stays open. Without the slice, a dashboard left open overnight would accumulate
        // tens of thousands of samples and the sparkline would redraw all of them.
        const next = [...previous, sample];
        return next.length > TELEMETRY_HISTORY ? next.slice(-TELEMETRY_HISTORY) : next;
      });
    } catch {
      // Swallowed like the other pollers: this is replaced within a second, so a
      // transient failure must not flash an error over a working page. The staleness
      // indicator in the header is what surfaces a sustained outage.
    }
  }, [client]);

  const refreshAgents = useCallback(async () => {
    // Skip if the previous request has not come back yet. Each `/agents` call makes the
    // server stat a directory of transcripts, so if one ever takes longer than the poll
    // interval, requests would pile up and their responses could be applied out of order
    // — briefly rendering an older agent list over a newer one.
    if (agentsInFlight.current) return;
    agentsInFlight.current = true;
    try {
      const response = await client.agents();
      if (mounted.current) putAgents(response);
    } catch {
      // Swallowed for the same reason as live progress: this is replaced within two
      // seconds, so a transient failure must not flash an error over a working page.
      // Deliberately *not* cleared to null either — keeping the last known agent list is
      // better than blanking a panel someone is watching because one poll failed.
    } finally {
      // `finally`, so a thrown request cannot leave the flag stuck true and stop the
      // panel updating for the rest of the page's life.
      agentsInFlight.current = false;
    }
  }, [client, putAgents]);

  const setCallsOffset = useCallback(
    (offset: number) => {
      const next = Math.max(0, offset);
      callsOffsetRef.current = next;
      setCallsOffsetState(next);
      // Fetch immediately rather than waiting for the next event, so turning a page feels
      // like a click rather than like a delay.
      void refresh();
    },
    [refresh],
  );

  // First load.
  useEffect(() => {
    void refresh();
    void refreshLive();
    void refreshAgents();
    void refreshTelemetry();
  }, [refresh, refreshLive, refreshAgents, refreshTelemetry]);

  // Sample the machine continuously. See TELEMETRY_POLL_MS for why this is unconditional.
  useEffect(() => {
    const timer = setInterval(() => void refreshTelemetry(), TELEMETRY_POLL_MS);
    return () => clearInterval(timer);
  }, [refreshTelemetry]);

  // A slow floor under the inventory, so a model loaded or evicted outside this dashboard
  // appears without waiting for the next model call to fire the change stream.
  useEffect(() => {
    const timer = setInterval(() => void refresh(), SYSTEM_POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

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

  // Poll live progress while something is generating, and for a short while after.
  useEffect(() => {
    if (live.length === 0 && Date.now() - lastLiveSeen.current > LIVE_IDLE_GRACE_MS) {
      // Genuinely idle: no timer at all, so an idle dashboard makes no requests.
      return;
    }
    const timer = setInterval(() => void refreshLive(), LIVE_POLL_MS);
    return () => clearInterval(timer);
    // Depends on `live.length`, not `live`: the array is replaced on every poll, so
    // depending on the array itself would clear and recreate the timer each time.
  }, [live.length, refreshLive]);

  // Poll Claude agent activity always — see AGENTS_POLL_MS for why this one is not
  // conditional. The cost of being wrong in the other direction is the panel never
  // noticing an agent start, which is the single thing it exists to do.
  useEffect(() => {
    const timer = setInterval(() => void refreshAgents(), AGENTS_POLL_MS);
    return () => clearInterval(timer);
  }, [refreshAgents]);

  return {
    system,
    telemetry,
    telemetryHistory,
    usage,
    calls,
    stats,
    live,
    agents,
    callsOffset,
    callsPageSize: CALLS_PAGE_SIZE,
    callsHasMore,
    setCallsOffset,
    loading,
    error,
    connected,
    client,
    refresh: () => {
      void refresh();
      void refreshLive();
      void refreshAgents();
      void refreshTelemetry();
    },
  };
}
