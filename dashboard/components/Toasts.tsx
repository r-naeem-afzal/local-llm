"use client";

import type { AgentToast } from "@/lib/useAgentNotifications";

/**
 * The in-page notification stack.
 *
 * This is the layer that always works. Desktop notifications are better when the tab is
 * not in focus — which is the point of launching a fan-out and walking away — but they can
 * be refused, and a feature whose only output can be silently switched off gives no way to
 * tell "notifications are blocked" from "no agents ran". So the toasts are unconditional
 * and the desktop layer is an addition to them, not a replacement.
 *
 * Fixed to the corner rather than pushed into the page flow, so a toast appearing cannot
 * shift the panel someone is reading.
 */
export function Toasts({
  toasts,
  onDismiss,
}: {
  toasts: AgentToast[];
  onDismiss: (id: string) => void;
}) {
  if (toasts.length === 0) {
    // Nothing rendered at all, not an empty container: an empty fixed-position element
    // still sits over the page and can swallow clicks meant for what is underneath.
    return null;
  }

  return (
    // aria-live so a screen reader announces a toast when it appears. "polite" rather than
    // "assertive" because an agent starting is informational and should not interrupt
    // whatever is being read.
    <div className="toast-stack" aria-live="polite">
      {toasts.map((toast) => (
        <button
          key={toast.id}
          type="button"
          className={`toast toast-${toast.kind}`}
          onClick={() => onDismiss(toast.id)}
          // A button rather than a div with a click handler, so it is reachable and
          // dismissible by keyboard for free instead of needing a role and a key handler.
          title="Dismiss"
        >
          <span className="toast-title">{toast.title}</span>
          <span className="toast-detail">{toast.detail}</span>
        </button>
      ))}
    </div>
  );
}
