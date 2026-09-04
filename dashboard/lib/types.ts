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
  subagent_messages: number;
  total_tokens: number;
  main_loop: AgentUsage[];
  /** Non-empty means subagents ran — which the standing cost rules say should be rare. */
  subagents: AgentUsage[];
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
