/**
 * Response shapes from the Python API, declared once.
 *
 * These are hand-written rather than generated, because the API is small and stable and a
 * code-generation step would be more machinery than the problem deserves. The trade is
 * that they can drift from the server: if a panel starts rendering `undefined`, check
 * these first.
 *
 * Fields that the API can genuinely return as null or absent are typed that way rather
 * than being optimistically declared non-null. `strict` mode then forces every consumer
 * to handle the missing case, which is the whole reason for using TypeScript here — a
 * GPU probe that failed, or a payload that retention has pruned, are both normal states
 * the UI must display rather than crash on.
 */

/** A call that has finished, as listed in history. Metadata only — never the prompt. */
export interface CallRecord {
  id: string;
  ts: string;
  tool: string | null;
  model: string | null;
  /** "running" | "ok" | "error" in practice, but the server is free to add more. */
  status: string | null;
  duration_ms: number | null;
  tokens_in: number | null;
  tokens_out: number | null;
  error: string | null;
  stage: string | null;
  /** Per-tool context, e.g. { url, question, page_chars } for a claim extraction. */
  meta: Record<string, unknown> | null;
}

export interface CallsPage {
  calls: CallRecord[];
  has_more: boolean;
  limit: number;
  offset: number;
}

/** The bulky text for one call, fetched only when a row is opened. */
export interface CallPayload {
  prompt: { role: string; content: string }[] | null;
  response: string | null;
  /** A reasoning model's private thinking. Null for models that do not produce it. */
  reasoning: string | null;
}

/**
 * A call that is in flight right now.
 *
 * `status` distinguishes three phases, and the difference matters to the reader:
 * - "starting"  — accepted, but no tokens yet. Usually an ~18 s model load.
 * - "thinking"  — producing reasoning, not yet an answer. Busy, not stuck.
 * - "running"   — producing the answer.
 */
export interface LiveCall {
  id: string;
  ts: string;
  tool: string | null;
  model: string | null;
  status: "starting" | "thinking" | "running" | string;
  chunks: number;
  chars: number;
  reasoning_chars?: number;
  /** Null while starting, because nothing has been generated to time yet. */
  elapsed_ms: number | null;
  tail: string;
}

export interface LiveResponse {
  live: LiveCall[];
  count: number;
}

export interface GpuInfo {
  available: boolean;
  name: string;
  total_mib: number;
  used_mib: number;
  free_mib: number;
  used_pct: number;
  utilisation_pct: number;
  temperature_c: number | null;
  error: string;
}

export interface HostInfo {
  available: boolean;
  cpu_pct: number;
  ram_total_mib: number;
  ram_used_mib: number;
  /** System RAM held by the inference processes — the model file is memory-mapped. */
  inference_rss_mib: number;
  inference_processes: number;
  error: string;
}

export interface LoadedModel {
  key: string;
  display_name: string;
  size_mib: number;
  context_length: number;
  max_context_length: number;
  status: string;
  quantisation: string;
  /** Seconds until an idle model is evicted to free VRAM. Null if not reported. */
  ttl_remaining_s: number | null;
  queued: number;
  parallel: number;
}

export interface InstalledModel {
  key: string;
  type: string;
  display_name: string;
  size_mib: number;
  params: string;
  architecture: string;
  max_context_length: number | null;
}

export interface SystemSnapshot {
  ts: string;
  server_up: boolean;
  gpu: GpuInfo;
  host: HostInfo;
  loaded_models: LoadedModel[];
  /**
   * VRAM the loaded models account for. The gap against `gpu.used_mib` is the desktop,
   * the browser and the KV cache — which is why a 9 GB model leaves far less than 7 GB
   * free on a 16 GB card.
   */
  model_vram_mib: number;
  installed_models: InstalledModel[];
}

/**
 * Claude usage for one model, in one bucket.
 *
 * The four token categories stay separate because they are billed very differently and
 * the mix is extreme: one measured 5-hour window held 73.5M tokens of which 72.2M were
 * cache *reads* and only 854 were fresh input. A single total would overstate real spend
 * by roughly fifty times.
 */
export interface AgentUsage {
  model: string;
  messages: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  total_tokens: number;
}

export interface ClaudeUsage {
  window_hours: number;
  window_start: string;
  sessions: number;
  messages: number;
  /**
   * Always 0 on this machine, and *not* because subagents are free.
   *
   * Claude Code 2.1.260 records no sidechain messages and writes no transcript for a
   * subagent, so their tokens are billed to the 5-hour window but appear in no local
   * file. These figures are therefore a floor on real spend whenever agents have run.
   * `ActiveAgent` below is the live view built on the tool-call records that *are*
   * written, and its token numbers are explicitly estimates.
   */
  subagent_messages: number;
  total_tokens: number;
  main_loop: AgentUsage[];
  subagents: AgentUsage[];
  error: string;
}

/**
 * One Claude Code agent that is running now, or finished moments ago.
 *
 * Not to be confused with `AgentUsage` above, which it sits next to: that is
 * retrospective token accounting per model over a 5-hour window, this is one live
 * invocation of the `Agent` tool. The distinction matters because only one of them
 * carries measured token counts, and it is not this one.
 */
export interface ActiveAgent {
  /** The `tool_use_id`. Stable and unique, so it is what identifies a row across polls. */
  key: string;
  project: string;
  session: string;
  /** e.g. "Explore", "general-purpose" — the agent type that was launched. */
  subagent_type: string;
  /** The short human label the launcher wrote. Empty is possible. */
  description: string;
  /** Empty means the agent inherited the parent's model rather than naming one. */
  model: string;
  background: boolean;
  /** "running" | "ok" | "error" — the same vocabulary `StatusBadge` already colours. */
  status: string;
  started_ts: string;
  /** Null while still running. */
  finished_ts: string | null;
  elapsed_ms: number;
  /**
   * Tokens the agent spent — real when `tokens_measured`, otherwise a floor.
   *
   * A background agent's completion notification reports its true usage, so those rows
   * are measured. A foreground agent's usage is recorded nowhere, so its figure is
   * estimated from the size of the prompt in and the report out — which understates by
   * roughly thirteen times, because an agent's spend is mostly the files it read.
   */
  tokens: number;
  /** Whether `tokens` is a measurement. False means it is the estimate described above. */
  tokens_measured: boolean;
}

export interface AgentActivity {
  window_s: number;
  agents: ActiveAgent[];
  running: number;
  finished: number;
  errored: number;
  /** True when at least one listed agent's tokens are an estimate rather than measured. */
  tokens_estimated: boolean;
  error: string;
}

export interface ToolCount {
  tool: string | null;
  n: number;
}

export interface ModelStats {
  model: string | null;
  calls: number;
  errors: number;
  tokens_in: number;
  tokens_out: number;
  ms: number;
  last_ts: string | null;
}

export interface Stats {
  totals: {
    calls: number;
    errors: number;
    tokens_in: number;
    tokens_out: number;
    ms: number;
  };
  by_tool: ToolCount[];
  by_model: ModelStats[];
  db_bytes: number;
}

export interface Health {
  ok: boolean;
  version: string;
  model: string;
  endpoint: string;
  db_bytes: number;
}
