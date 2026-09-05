"use client";

import { useEffect, useState, memo } from "react";

import type { ApiClient } from "@/lib/ApiClient";
import { formatDuration, formatRate, formatTime, formatTokens } from "@/lib/format";
import type { CallPayload, CallRecord } from "@/lib/types";
import { Panel, StatusBadge } from "./ui";

/**
 * Call history, with the full prompt available on demand.
 *
 * The two-step — a light table, then a payload fetched only when a row is opened — mirrors
 * the split in the store, and it is the reason the table stays fast. A history row is
 * about 200 bytes of metadata; the prompt behind a claim-extraction row is roughly 55 KB.
 * Sending payloads with the list would mean about 5.5 MB for one screen of history, nearly
 * all of it never read.
 */
function HistoryPanelInner({
  calls,
  client,
  offset,
  pageSize,
  hasMore,
  onOffsetChange,
}: {
  calls: CallRecord[];
  client: ApiClient;
  offset: number;
  pageSize: number;
  hasMore: boolean;
  onOffsetChange: (offset: number) => void;
}) {
  const [selected, setSelected] = useState<CallRecord | null>(null);

  // 1-based and inclusive, because "showing 26–50" is what a reader expects to see, not
  // the zero-based offset the API works in.
  const first = calls.length === 0 ? 0 : offset + 1;
  const last = offset + calls.length;

  return (
    <Panel
      title="Call history"
      action={
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="faint">
            {first}–{last}
          </span>
          <button
            onClick={() => onOffsetChange(offset - pageSize)}
            disabled={offset === 0}
            title="Newer calls"
          >
            ← newer
          </button>
          <button
            onClick={() => onOffsetChange(offset + pageSize)}
            disabled={!hasMore}
            title="Older calls"
          >
            older →
          </button>
        </span>
      }
    >
      {calls.length === 0 ? (
        <p className="empty">No calls recorded yet.</p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Tool</th>
                <th>Model</th>
                <th>Status</th>
                <th className="num">Duration</th>
                <th className="num">In</th>
                <th className="num">Out</th>
                <th className="num">Rate</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {calls.map((call) => (
                <tr
                  key={call.id}
                  className="row-clickable"
                  onClick={() => setSelected(call)}
                  // Keyboard access: a clickable <tr> is invisible to keyboard users
                  // without an explicit tab stop and an Enter handler, which would make
                  // the payload view unreachable without a mouse.
                  tabIndex={0}
                  role="button"
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelected(call);
                    }
                  }}
                >
                  <td className="mono dim">{formatTime(call.ts)}</td>
                  <td className="mono">{call.tool}</td>
                  <td className="mono faint">{call.model}</td>
                  <td>
                    <StatusBadge status={call.status} />
                  </td>
                  <td className="num">{formatDuration(call.duration_ms)}</td>
                  <td className="num">{formatTokens(call.tokens_in)}</td>
                  <td className="num">{formatTokens(call.tokens_out)}</td>
                  {/* Per call, because a slow one is usually slow for a knowable
                      reason - a cold model load, or a long page - and seeing the rate
                      beside the duration separates "lots of work" from "running slowly". */}
                  <td className="num dim">
                    {formatRate(call.tokens_out, call.duration_ms)}
                  </td>
                  <td className="dim">
                    {/* An error is the most important thing on the row, so it takes
                        precedence over the metadata summary. */}
                    {call.error ? (
                      <span style={{ color: "var(--error)" }}>{truncate(call.error, 70)}</span>
                    ) : (
                      <span className="faint">{summariseMeta(call.meta)}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <PayloadDrawer call={selected} client={client} onClose={() => setSelected(null)} />
      )}
    </Panel>
  );
}

/**
 * Turn a call's metadata into one readable line.
 *
 * Each tool records different context, so rather than a fixed set of columns the
 * interesting field is picked per tool:
 *
 *   { url: "https://en.wikipedia.org/wiki/Payoneer", page_chars: 18738 }
 *     ->  "en.wikipedia.org · 18738 chars"
 *   { file: "config.py", from_line: 71 }  ->  "config.py:71"
 */
function summariseMeta(meta: Record<string, unknown> | null): string {
  if (!meta) return "";

  if (typeof meta.url === "string") {
    let host = meta.url;
    try {
      host = new URL(meta.url).hostname;
    } catch {
      // A malformed URL is not worth failing a table row over — fall back to the raw
      // string, truncated below.
    }
    const chars = typeof meta.page_chars === "number" ? ` · ${meta.page_chars} chars` : "";
    return `${truncate(host, 40)}${chars}`;
  }

  if (typeof meta.file === "string") {
    const line = typeof meta.from_line === "number" ? `:${meta.from_line}` : "";
    return `${meta.file}${line}`;
  }

  if (typeof meta.question === "string") return truncate(meta.question, 60);

  // Unknown shape: show the keys rather than nothing, so a new tool's metadata is at
  // least visible without a code change here.
  return truncate(Object.keys(meta).join(", "), 60);
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

/**
 * The prompt, answer and reasoning for one call.
 *
 * Reasoning is collapsed by default. It is routinely several times longer than the answer
 * — Qwen3 produced 1,008 characters of thinking to reply "OK" — so showing it expanded
 * would bury the thing you opened the row to read.
 */
function PayloadDrawer({
  call,
  client,
  onClose,
}: {
  call: CallRecord;
  client: ApiClient;
  onClose: () => void;
}) {
  const [payload, setPayload] = useState<CallPayload | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "pruned" | "error">("loading");
  const [showReasoning, setShowReasoning] = useState(false);

  useEffect(() => {
    // Guards against a race: if the drawer is closed, or another row opened, before this
    // request resolves, the late response must not overwrite the newer state.
    let active = true;

    client
      .payload(call.id)
      .then((result) => {
        if (!active) return;
        // null means the API answered 404, which is an expected state rather than a
        // failure: retention prunes payloads after a week but keeps the metadata row
        // forever. Saying "pruned" is the difference between an explanation and a bug.
        setPayload(result);
        setState(result === null ? "pruned" : "ready");
      })
      .catch(() => {
        if (active) setState("error");
      });

    return () => {
      active = false;
    };
  }, [call.id, client]);

  // Escape to close. Expected of any overlay, and without it a keyboard user has to tab
  // to the close button.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      {/* Stop propagation so clicking inside the drawer does not reach the backdrop and
          immediately close it. */}
      <div className="drawer" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <div className="mono" style={{ fontSize: 15 }}>
              {call.tool}
            </div>
            <div className="faint mono" style={{ fontSize: 12 }}>
              {call.id} · {call.model} · {formatTime(call.ts)} ·{" "}
              {formatDuration(call.duration_ms)}
            </div>
          </div>
          <button onClick={onClose}>Close</button>
        </div>

        {/* One scroll region for the whole payload, rather than a scrollbar inside every
            block. The blocks used to be capped at 340px each and scrolled internally, so a
            long prompt sat in a short box with half the drawer below it left empty — the
            reader had to scroll a small window while a screenful of space went unused.
            Now each block is as tall as its content and this container scrolls once the
            content is taller than the screen: fill the space first, scroll second. */}
        <div className="drawer-body">
        {call.error && (
          <>
            <p className="section-label">Error</p>
            <pre className="payload-block" style={{ color: "var(--error)" }}>
              {call.error}
            </pre>
          </>
        )}

        {state === "loading" && <p className="empty">Loading payload…</p>}
        {state === "error" && <p className="empty">Could not load the payload.</p>}
        {state === "pruned" && (
          <p className="empty">
            The payload for this call has been pruned by retention. Its metadata is kept
            indefinitely, but the prompt and response text are removed after the retention
            window.
          </p>
        )}

        {state === "ready" && payload && (
          <>
            {payload.prompt?.map((message, index) => (
              <div key={index}>
                <p className="section-label">{message.role}</p>
                <pre className="payload-block">{message.content}</pre>
              </div>
            ))}

            {payload.response && (
              <>
                <p className="section-label">Response</p>
                <pre className="payload-block">{payload.response}</pre>
              </>
            )}

            {payload.reasoning && (
              <>
                <p className="section-label">
                  <button
                    onClick={() => setShowReasoning((shown) => !shown)}
                    style={{ marginRight: 8 }}
                  >
                    {showReasoning ? "Hide" : "Show"}
                  </button>
                  Reasoning ({payload.reasoning.length.toLocaleString()} chars)
                </p>
                {showReasoning && (
                  <pre className="payload-block dim">{payload.reasoning}</pre>
                )}
              </>
            )}
          </>
        )}
        </div>
      </div>
    </div>
  );
}

/**
 * Memoized so a change in another panel's data cannot re-render this one. Without this,
 * every 900 ms live-progress tick repainted the entire dashboard.
 */
export const HistoryPanel = memo(HistoryPanelInner);
