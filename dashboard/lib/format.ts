/**
 * Display formatting, kept out of the components so every panel renders a number the
 * same way. A dashboard that shows "8571 MiB" in one panel and "8.4 GB" in another makes
 * the reader do conversion arithmetic to compare them.
 */

/** Bytes to a human size. 32768 -> "32.0 KB"; 1310720 -> "1.2 MB" */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

/** MiB to GiB once the number gets unwieldy. 8571 -> "8.4 GiB"; 80 -> "80 MiB" */
export function formatMib(mib: number): string {
  return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${mib} MiB`;
}

/**
 * Milliseconds to a duration that stays readable across four orders of magnitude —
 * model calls here range from 3 s to several minutes.
 *
 *   850 -> "850ms"   13560 -> "13.6s"   182076 -> "3m 2s"
 */
export function formatDuration(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/**
 * Token counts, abbreviated. Cache reads run to tens of millions, and a raw
 * "73751370" is unreadable at a glance.
 *
 *   854 -> "854"   516369 -> "516.4K"   73751370 -> "73.8M"
 */
export function formatTokens(tokens: number | null): string {
  if (tokens === null) return "—";
  if (tokens < 1000) return `${tokens}`;
  if (tokens < 1_000_000) return `${(tokens / 1000).toFixed(1)}K`;
  return `${(tokens / 1_000_000).toFixed(1)}M`;
}

/** Seconds to a compact clock, used for model eviction TTLs. 3375 -> "56m" */
export function formatSeconds(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

/**
 * An ISO timestamp as a local wall-clock time.
 *
 * Local, not UTC, because the reader is sitting at this machine comparing "when did that
 * run" against their own memory of the last few minutes. The API returns UTC, and the
 * offset here is +5, so showing UTC would make every entry look five hours stale.
 */
export function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** How long ago, for "last used" columns. 45 -> "45s ago" */
export function formatAgo(iso: string | null): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86_400)}d ago`;
}

/**
 * Generation throughput in tokens per second.
 *
 * The single most useful health number for a local model, and the one that makes a
 * regression obvious: this machine sustains roughly 49 tok/s on a 14B at Q4, so a reading
 * of 8 means the model has partially spilled to CPU and is running from system RAM.
 * Neither the token count nor the duration alone would show that.
 *
 * Measured over the whole call, so it includes the time the model spent loading and
 * thinking, not just emitting. That understates raw decode speed and is the honest number
 * for "how long will this take" — which is the question being asked.
 *
 *   tokens=589, ms=11389  ->  "52 tok/s"
 *   tokens=0               ->  "—"    (nothing was generated, so there is no rate)
 */
export function formatRate(tokens: number | null, ms: number | null): string {
  // Guard both the missing case and the divide-by-zero. A call recorded with no duration
  // would otherwise render "Infinity tok/s", which looks like a bug in the dashboard
  // rather than a gap in the data.
  if (!tokens || !ms || ms <= 0) return "—";
  const rate = tokens / (ms / 1000);
  // Below ten, one decimal matters - the difference between 2 and 2.5 tok/s is the
  // difference between usable and not.
  return rate < 10 ? `${rate.toFixed(1)} tok/s` : `${Math.round(rate)} tok/s`;
}
