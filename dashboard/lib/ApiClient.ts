import type {
  AgentActivity,
  CallPayload,
  CallsPage,
  ClaudeUsage,
  Health,
  LiveResponse,
  Stats,
  SystemSnapshot,
} from "./types";

/**
 * The single place that knows how to talk to the Python API.
 *
 * A class taking its base URL through the constructor, rather than a set of exported
 * `fetch` functions reading a module-level constant. The practical benefit is that the
 * base URL is a parameter: a test, a Storybook story, or a second instance pointed at
 * another machine constructs its own client instead of patching a global. Nothing in the
 * components mentions a URL or a fetch.
 */
export class ApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl?: string) {
    // Resolution order matters. The env var lets a build point at another machine, and
    // the default keeps the common case — API and dashboard on the same box — working
    // with no configuration at all.
    //
    // NEXT_PUBLIC_ is required for the value to exist in the browser: Next.js only
    // inlines env vars with that prefix into client-side code. Without the prefix the
    // variable would read as undefined in the browser and silently fall back.
    const configured = baseUrl ?? process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:7878";
    // Strip a trailing slash so paths can be appended with a single one. A configured
    // "http://host:7878/" would otherwise produce "//health", which some servers route
    // as a different, non-existent path and answer with 404.
    this.baseUrl = configured.replace(/\/+$/, "");
  }

  /** The SSE endpoint URL. Exposed because `EventSource` is constructed by the caller. */
  get eventsUrl(): string {
    return `${this.baseUrl}/events`;
  }

  health(): Promise<Health> {
    return this.get<Health>("/health");
  }

  system(): Promise<SystemSnapshot> {
    return this.get<SystemSnapshot>("/system");
  }

  usage(hours = 5): Promise<ClaudeUsage> {
    // 5 hours by default because that is the period the plan allowance resets on, so it
    // is the only window that answers "how much have I got left".
    return this.get<ClaudeUsage>(`/usage?hours=${hours}`);
  }

  calls(limit = 100, offset = 0): Promise<CallsPage> {
    return this.get<CallsPage>(`/calls?limit=${limit}&offset=${offset}`);
  }

  live(): Promise<LiveResponse> {
    return this.get<LiveResponse>("/live");
  }

  /**
   * Claude agents running now, plus those that finished within the display window.
   *
   * No argument: the server's configured window decides how long a finished agent
   * lingers. Passing it from here would put the same policy in two places and let them
   * disagree, and the browser has no better information about it than the server does.
   */
  agents(): Promise<AgentActivity> {
    return this.get<AgentActivity>("/agents");
  }

  stats(): Promise<Stats> {
    return this.get<Stats>("/stats");
  }

  /**
   * Fetch one call's prompt, response and reasoning.
   *
   * Returns null for a 404 rather than throwing, because a missing payload is an
   * expected state, not a failure: retention prunes payloads after a week while keeping
   * the metadata row forever. The UI needs to say "pruned", and it can only do that if
   * absence is a value rather than an exception.
   */
  async payload(callId: string): Promise<CallPayload | null> {
    const response = await fetch(`${this.baseUrl}/calls/${encodeURIComponent(callId)}/payload`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(`payload ${callId}: ${response.status} ${response.statusText}`);
    }
    return (await response.json()) as CallPayload;
  }

  /**
   * One GET, with errors turned into thrown `Error`s carrying the status.
   *
   * `fetch` does not reject on a 500 — it resolves with `ok: false` — so without this
   * check a failed request would flow onward as `undefined` and surface much later as an
   * unrelated rendering error. Checking here means a failure is reported where it
   * happened.
   *
   * `cache: "no-store"` is essential: this is live monitoring data, and Next.js caches
   * fetches aggressively by default, so without it panels would freeze at their first
   * reading and look like a broken dashboard rather than a cached one.
   */
  private async get<T>(path: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`${path}: ${response.status} ${response.statusText}`);
    }
    return (await response.json()) as T;
  }
}
