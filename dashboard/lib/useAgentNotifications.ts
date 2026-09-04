"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { ActiveAgent, AgentActivity } from "./types";

/**
 * Turns successive agent snapshots into notifications about what changed.
 *
 * ## Why a diff is needed at all
 *
 * Every other panel on this dashboard is a pure function of the current snapshot — it
 * renders what is true now and nothing else. That is not enough here. "An agent started"
 * and "an agent finished" are *transitions*, and a transition is invisible to code that
 * only ever sees one snapshot at a time. So this hook keeps the previous set of agent keys
 * and compares.
 *
 * The comparison is on `key` (the `tool_use_id`), which is why the server sends one: it is
 * stable across polls and unique across sessions, so a key appearing means a genuinely new
 * agent rather than a re-render or a reordered list.
 *
 * ## Two delivery layers, because one of them can be refused
 *
 * 1. In-page toasts, always on. This layer always works, needs no permission, and is what
 *    makes the feature testable at all.
 * 2. Browser notifications, opt in. These survive a tab that is not in focus, which is the
 *    point when the whole idea is to launch a fan-out and go and do something else.
 *
 * If the browser layer were the only one, a refused permission prompt would leave the
 * feature silently doing nothing, and the user would have no way to tell that from "no
 * agents ran".
 *
 * ## The limitation, stated rather than hidden
 *
 * State is replaced wholesale on each poll, so an agent that starts *and* finishes inside
 * one poll interval is never seen by this hook. The server keeps finished agents in its
 * response for a grace window specifically to make that unlikely — a real agent lives for
 * tens of seconds — but it is not impossible, and this hook cannot detect what it was
 * never shown.
 */

/** How long a toast stays on screen. Long enough to read a line, short enough to ignore. */
const TOAST_TTL_MS = 6000;

/** Remembered across reloads so the permission question is asked once, not every visit. */
const STORAGE_KEY = "local-llm.agent-notifications";

export interface AgentToast {
  id: string;
  kind: "started" | "finished" | "errored";
  title: string;
  detail: string;
}

export interface AgentNotifications {
  toasts: AgentToast[];
  dismiss: (id: string) => void;
  /** Whether desktop notifications are both permitted and switched on. */
  desktopEnabled: boolean;
  /** Null when the browser has no Notification API at all, so the UI can hide the button. */
  desktopSupported: boolean;
  /** Asks the browser for permission. Must be called from a real click — see below. */
  enableDesktop: () => void;
  disableDesktop: () => void;
}

/**
 * Read the stored preference.
 *
 * Wrapped in try/catch and guarded on `typeof window` because this file runs during
 * Next.js server rendering too, where `window` and `localStorage` do not exist. An
 * unguarded read there is not a caught error — it breaks the render, and the page fails
 * with a hydration mismatch rather than anything that points at storage. The catch covers
 * the separate case of a browser configured to block site data, which throws on access
 * rather than returning null.
 */
function readStoredPreference(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "on";
  } catch {
    return false;
  }
}

function writeStoredPreference(enabled: boolean): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, enabled ? "on" : "off");
  } catch {
    // Storage blocked. The preference then lasts for this page view only, which is a
    // sane degradation — losing a remembered toggle is not worth breaking the page over.
  }
}

function describe(agent: ActiveAgent): string {
  // The type alone is useless when three "Explore" agents are running; the description is
  // the label the launcher actually wrote, so it leads.
  const label = agent.description || agent.subagent_type;
  return agent.model ? `${label} (${agent.subagent_type}, ${agent.model})` : label;
}

function formatElapsed(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

/**
 * Compare this snapshot's agents against the *statuses* seen last time.
 *
 *   seen = {a: running},            agents = [a running, b running]  ->  b started
 *   seen = {a: running, b: running} agents = [a ok, b running]       ->  a finished
 *   seen = {a: running},            agents = [a error]               ->  a errored
 *   seen = {a: ok},                 agents = [a ok]                  ->  nothing
 *
 * The last line is why this takes statuses and not merely a set of keys. An earlier
 * version remembered presence alone, and a completion was "a key we knew, no longer
 * running" — which meant opening the dashboard within the server's grace window after a
 * fan-out had already finished announced every one of those agents as if it had just
 * completed. Remembering the status makes a completion an actual `running` -> terminal
 * transition, which is the thing worth announcing.
 *
 * A plain function outside the hook because it is the rule being implemented and needs
 * neither React nor a clock — so it can be reasoned about by handing it two values.
 */
function buildEvents(agents: ActiveAgent[], seen: Map<string, string>): AgentToast[] {
  const events: AgentToast[] = [];

  for (const agent of agents) {
    const previousStatus = seen.get(agent.key);
    const running = agent.status === "running";

    if (previousStatus === undefined && running) {
      events.push({
        id: `${agent.key}:started`,
        kind: "started",
        title: "Agent started",
        detail: describe(agent),
      });
      continue;
    }

    // Detected from the status change rather than from the row disappearing, because the
    // server keeps finished agents for a grace window — that grace is precisely what
    // makes the completion observable instead of a row that silently vanishes.
    if (previousStatus === "running" && !running) {
      const errored = agent.status === "error";
      events.push({
        id: `${agent.key}:${agent.status}`,
        kind: errored ? "errored" : "finished",
        title: errored ? "Agent failed" : "Agent finished",
        detail: `${describe(agent)} — ${formatElapsed(agent.elapsed_ms)}`,
      });
    }
  }

  return events;
}

export function useAgentNotifications(activity: AgentActivity | null): AgentNotifications {
  const [toasts, setToasts] = useState<AgentToast[]>([]);
  const [desktopOn, setDesktopOn] = useState(false);
  const [desktopSupported, setDesktopSupported] = useState(false);

  // Read on mount rather than in the initial state, because the first render also happens
  // on the server where neither `window` nor `Notification` exists. Initialising state
  // from them directly would make the server and client render differently, which React
  // reports as a hydration error.
  useEffect(() => {
    setDesktopSupported(typeof window !== "undefined" && "Notification" in window);
    // `typeof Notification` rather than `Notification?.permission`. Optional chaining
    // guards a declared identifier holding null or undefined; it does *not* guard an
    // identifier that is absent from the global scope entirely, which is the case in a
    // Firefox private window and various Android WebViews. There, evaluating
    // `Notification` throws ReferenceError from this mount effect, and React turns that
    // into an uncaught error that blanks the whole page.
    setDesktopOn(
      readStoredPreference() &&
        typeof Notification !== "undefined" &&
        Notification.permission === "granted",
    );
  }, []);

  /**
   * Each agent's status as of the previous poll, keyed by its `tool_use_id`.
   *
   * A ref, not state: writing it must not trigger a re-render, or every poll would render
   * twice. `undefined` until the first snapshot arrives, which is what distinguishes "the
   * page just loaded and three agents are already running" from "three agents just
   * started" — without that distinction, opening the dashboard mid-fan-out would fire a
   * burst of start notifications for agents that started minutes ago.
   *
   * Statuses rather than just keys, so a completion is a real `running` -> finished
   * transition. See `buildEvents` for the bug that distinction fixes.
   */
  const previousStatuses = useRef<Map<string, string> | undefined>(undefined);

  /**
   * Event ids already announced, so nothing is announced twice.
   *
   * Never pruned. It holds a couple of short strings per agent ever launched in this page
   * view, so the memory is negligible, and pruning would reintroduce the exact bug it
   * exists to prevent — an id dropped from here while its row is still on the server would
   * be announced all over again.
   */
  const announced = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (activity === null) return;

    const currentStatuses = new Map(
      activity.agents.map((agent) => [agent.key, agent.status] as const),
    );
    const seen = previousStatuses.current;
    previousStatuses.current = currentStatuses;

    // First snapshot: record what is already there and announce none of it.
    if (seen === undefined) return;

    const events = buildEvents(activity.agents, seen);

    // Announce each event once and only once.
    //
    // The ledger is a ref rather than being derived from the toasts on screen, for two
    // separate reasons. First, the completion branch fires on *every* poll for as long as
    // the finished row lingers on the server, so without a memory a finished agent would
    // toast every two seconds for its whole grace window. Second, deriving it from
    // current state would mean doing the check inside a state updater — and React invokes
    // updaters twice in development StrictMode, which would fire every desktop
    // notification twice. A ref is read and written exactly once per effect run.
    const fresh = events.filter((event) => !announced.current.has(event.id));
    if (fresh.length === 0) return;
    for (const event of fresh) {
      announced.current.add(event.id);
    }

    setToasts((current) => [...current, ...fresh]);

    for (const event of fresh) {
      // Self-expiring rather than swept by a second timer. Ids are unique per event, so a
      // late expiry can never remove a newer toast.
      window.setTimeout(() => {
        setToasts((current) => current.filter((toast) => toast.id !== event.id));
      }, TOAST_TTL_MS);

      if (desktopOn && Notification.permission === "granted") {
        // Tagged by id so the browser replaces rather than stacks a repeat of the same
        // event, using its own de-duplication.
        new Notification(event.title, { body: event.detail, tag: event.id });
      }
    }
  }, [activity, desktopOn]);

  const enableDesktop = useCallback(() => {
    if (!("Notification" in window)) return;
    // Must run inside a real click handler. Browsers reject a permission request that was
    // not started by a user gesture, and they reject it *silently* — so calling this on
    // mount would look like the user had denied it.
    void Notification.requestPermission().then((permission) => {
      const granted = permission === "granted";
      setDesktopOn(granted);
      writeStoredPreference(granted);
    });
  }, []);

  const disableDesktop = useCallback(() => {
    setDesktopOn(false);
    writeStoredPreference(false);
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  // Memoized so the returned object keeps its identity between renders. The page passes
  // this straight into a memoized panel, and a fresh object every render would make that
  // panel's props look changed on every tick — defeating the memo it depends on.
  return useMemo(
    () => ({
      toasts,
      dismiss,
      desktopEnabled: desktopOn,
      desktopSupported,
      enableDesktop,
      disableDesktop,
    }),
    [toasts, dismiss, desktopOn, desktopSupported, enableDesktop, disableDesktop],
  );
}
