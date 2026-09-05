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

/**
 * Turn streaming structured output into something a person can read.
 *
 * Extraction calls are schema-constrained, so what streams back is raw JSON:
 *
 *   {"claim": "Dedicated GPUs use on-board RAM…", "quote": "…", "importance": "central"},
 *
 * Shown verbatim, the live panel filled with braces, escaped quotes and field names -
 * three cards of it side by side - which is noise wearing the costume of detail. What is
 * actually worth seeing is how many claims have landed and what the newest one says.
 *
 * Falls back to the raw tail for plain-text calls, which stream prose and are already
 * readable.
 *
 *   '…"claim": "A", …"claim": "B"'  ->  { count: 2, latest: "B" }
 *   'Lightweight, embedded…'        ->  { count: 0, latest: null }
 */
function readableTail(tail: string): { count: number; latest: string | null } {
  // The tail is a *window* onto the stream, so the first match is usually cut off
  // mid-string. Only fully-closed values are taken, which is why the count can lag the
  // true total by one - an honest undercount beats showing a truncated fragment.
  // Matches a complete "claim": "…" pair. The inner alternation accepts any character
  // that is not a quote or a backslash, or any backslash-escaped character — which is
  // what allows a claim containing an escaped quote to still match as one whole value
  // rather than being cut short at the quote inside it.
  const matches = [...tail.matchAll(/"claim"\s*:\s*"((?:[^"\\]|\\.)*)"/g)];
  if (matches.length === 0) return { count: 0, latest: null };

  // Undo JSON escaping so the text reads as prose instead of showing \" and \n.
  // Backslash last: doing it first would turn \\" into \" and then into a bare quote,
  // corrupting text that legitimately contained a backslash.
  const latest = matches[matches.length - 1][1]
    .replace(/\\"/g, '"')
    .replace(/\\n/g, " ")
    .replace(/\\\\/g, "\\");
  return { count: matches.length, latest };
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

      {/* The streaming text, and ONLY while an answer is actually streaming.
          During the thinking phase the tail is the model's private monologue, and
          showing it was a mistake on two counts. It is noise - nobody needs to read a
          14B model talking itself through a schema - and because thinking can run for
          tens of seconds, the box appeared, grew, and vanished again, shoving every
          card below it down by about 250px and back. The panel sits at the top of the
          rail, so that was the whole page moving.
          The line above already reports "thinking · N chars", which is the part worth
          knowing: it is working, and how far along it is. */}
      {!thinking && !starting && call.tail && <AnswerTail tail={call.tail} />}
    </div>
  );
}

/**
 * The streaming answer, rendered as prose or as claim progress depending on its shape.
 */
function AnswerTail({ tail }: { tail: string }) {
  const { count, latest } = readableTail(tail);

  if (count === 0) {
    // Plain-text generation: the stream is already readable, so show it as it arrives.
    return <div className="tail">{tail}</div>;
  }

  return (
    <div className="tail-claims">
      <div className="tail-claims-count">
        {count} claim{count === 1 ? "" : "s"} so far
      </div>
      {latest && <div className="tail-claims-latest">{latest}</div>}
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
