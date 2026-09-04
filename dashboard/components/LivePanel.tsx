"use client";

import { memo } from "react";

import { formatDuration } from "@/lib/format";
import type { LiveCall } from "@/lib/types";
import { Panel, StatusBadge } from "./ui";

/**
 * What the local models are doing *right now*.
 *
 * This is the panel the whole toolkit exists for. Without it a 12-second extraction and a
 * hung request look identical from outside, because a history row does not appear until
 * the call finishes — so a busy machine and a stuck one are indistinguishable.
 *
 * Three phases are shown distinctly, because each means something different and each has
 * a different expected duration:
 *
 * - **starting** — accepted, no tokens yet. Almost always a model load, ~18s here. There
 *   is no output to show, so the panel says what is happening instead of showing nothing.
 * - **thinking** — producing reasoning, not yet an answer. A reasoning model can spend a
 *   long time here: Qwen3 14B used 203 output tokens simply to say "OK". Without this
 *   phase being labelled, the answer character count sits at zero and the call looks
 *   stalled when it is working normally.
 * - **running** — producing the answer.
 */
function LivePanelInner({ live }: { live: LiveCall[] }) {
  return (
    <Panel
      title="Live"
      action={
        live.length > 0 ? (
          <span className="badge badge-running">
            <span className="dot dot-live" style={{ background: "var(--running)" }} />
            {live.length} in flight
          </span>
        ) : (
          <span className="badge badge-idle">
            <span className="dot" style={{ background: "var(--idle)" }} />
            idle
          </span>
        )
      }
    >
      {/* A fixed-minimum body, so starting or finishing a call does not resize the panel
          and shove the rest of the page up and down. The content here changes every
          second; its footprint should not. */}
      <div className="live-body">
        {live.length === 0 ? (
          <p className="empty">No model calls in flight.</p>
        ) : (
          live.map((call) => <LiveCallRow key={call.id} call={call} />)
        )}
      </div>
    </Panel>
  );
}

function LiveCallRow({ call }: { call: LiveCall }) {
  const thinking = call.status === "thinking";
  const starting = call.status === "starting";

  return (
    <div className="live-call">
      <div className="metric-row">
        <span>
          <StatusBadge status={call.status} />{" "}
          <span className="mono">{call.tool}</span>{" "}
          <span className="faint mono">{call.model}</span>
        </span>
        <span className="metric-value">{formatDuration(call.elapsed_ms)}</span>
      </div>

      {/* What this call is actually working on. The research pipeline runs two
          extractions at once, and without this both rows read "extract_claims /
          qwen3-14b" and look like one call rendered twice. */}
      {call.subject && <div className="live-subject">{shorten(call.subject)}</div>}

      <div className="metric-row">
        <span className="metric-label">
          {starting
            ? // No token counts exist yet, so state the cause instead of showing zeros
              // that would read as "producing nothing".
              "loading model into VRAM…"
            : thinking
              ? // During thinking the answer is genuinely empty, so the reasoning length
                // is the only honest measure of progress.
                `thinking · ${(call.reasoning_chars ?? 0).toLocaleString()} chars`
              : `${call.chars.toLocaleString()} chars · ${call.chunks} chunks`}
        </span>
        {!starting && !thinking && (call.reasoning_chars ?? 0) > 0 && (
          // Once an answer is coming through, reasoning becomes context rather than the
          // headline — but it is still worth seeing, because it is what the token budget
          // was spent on.
          <span className="faint">
            +{(call.reasoning_chars ?? 0).toLocaleString()} reasoning
          </span>
        )}
      </div>

      {call.tail && <div className="tail">{call.tail}</div>}
    </div>
  );
}

/**
 * Shorten a subject for a narrow column.
 *
 *   "https://www.payoneer.com/resources/business/guide-to-payment-gateways-in-pakistan/"
 *     ->  "payoneer.com/…/guide-to-payment-gateways-in-pakistan"
 *
 * The host and the last path segment are what identify a page to a reader; the middle is
 * where the length lives. A question is passed through unchanged apart from a length cap,
 * since it has no structure to exploit.
 */
function shorten(subject: string): string {
  const match = /^https?:\/\/(?:www\.)?([^/]+)(\/.*)?$/.exec(subject);
  if (!match) return subject.length > 70 ? `${subject.slice(0, 69)}…` : subject;

  const [, host, path = ""] = match;
  const segments = path.split("/").filter(Boolean);
  if (segments.length === 0) return host;
  const last = segments[segments.length - 1];
  return segments.length > 1 ? `${host}/…/${last}` : `${host}/${last}`;
}

/**
 * Memoized so a change in another panel's data cannot re-render this one. Without this,
 * every 900 ms live-progress tick repainted the entire dashboard.
 */
export const LivePanel = memo(LivePanelInner);
