# Local LLM Dashboard

Live view of what the local models are doing and which Claude agents are running, beside
what the Claude plan has cost in the current 5-hour window — with the caveat that the plan
figure is a floor, since a foreground subagent's usage is billed to the window but written
to no local transcript. React + TypeScript on Next.js; it is a pure client of the Python
API in `../toolkit`.

## Running

From the repository root, `Start-Local-LLM-Dashboard.cmd` starts the model server, the API
and this dashboard together. To run it by hand:

```bash
# 1. the API (from ../toolkit)
uvicorn local_llm.api:app --port 7878

# 2. this dashboard
npm install          # first time only
npm run dev          # development, hot reload
# or
npm run build && npm start
```

Then open http://localhost:3000.

Point it at an API on another machine with `NEXT_PUBLIC_API_URL`. The `NEXT_PUBLIC_`
prefix is required — Next.js only inlines env vars with that prefix into browser code, so
without it the value reads as undefined in the browser and silently falls back to
localhost. Note the API's CORS allowlist only permits localhost origins, so serving the
dashboard from a different host means widening that list in `api.py` too.

## How it updates

Two paths, for two kinds of data:

- **Server-Sent Events** (`/events`) drive history, stats and system state. The server
  notifies on change, so nothing is polled speculatively. The "live / not connected" badge
  in the header shows whether that stream is attached — a disconnected dashboard has
  silently stopped updating, and without the badge that is indistinguishable from an idle
  machine.
- **A short timer** drives the live-progress panel, but *only while a call is in flight*.
  This is the one thing SSE cannot cover: a running generation produces new text
  continuously with no discrete "changed" moment, and notifying per token would be
  thousands of events. An idle dashboard makes no requests at all.

## Why the panels are ordered as they are

Top to bottom answers the questions in the order they get asked: is a Claude fan-out
running and what is it costing (Claude agents) → is a local call in flight (Live) → can
the machine do the work (GPU/Host/Models) → what did it cost, locally versus on the plan
(Usage/Totals) → what happened earlier (History).

Claude agents comes first because, under the standing cost rule, it is the most expensive
thing that can be happening — and because fan-out is only permitted while it is being
watched, which makes this panel the thing that licenses the practice rather than merely
reporting on it.

Usage and Local totals sit together deliberately. The whole premise is that mechanical
work moved off the metered plan onto the GPU, and that is only checkable with both numbers
on one screen. The comparison is directional rather than exact: the local figure is
complete and the plan figure is a floor, so the real gap is at least as large as it looks
and never smaller.

## Things the display is deliberate about

- **The three live phases are labelled differently.** `starting` means accepted but no
  tokens yet — almost always an ~18s model load. `thinking` means a reasoning model is
  producing its private monologue, during which the answer is genuinely empty; without
  this label the character count sits at zero and a working call looks stalled. `running`
  means the answer is streaming.
- **Claude tokens are never summed into one figure.** One measured window held 73.5M
  tokens of which 72.2M were discounted cache *reads* and only 854 were fresh input. A
  single total would overstate real spend by roughly fifty times, so fresh input is the
  headline and cache traffic is shown separately.
- **The usage panel's subagent message count is known to be broken, and is labelled so.**
  It counts messages flagged `isSidechain`, and that flag is never set on this version of
  Claude Code, so the figure is structurally zero whether ten agents ran or none. A zero
  there means "not visible", not "not incurred". The Claude agents panel is what actually
  shows agent activity.
- **Only some agent token figures are measurements.** A background agent reports its real
  usage when it completes, so those rows are exact. A foreground agent's usage exists in no
  local record, so its figure is estimated from prompt and result size — which understates
  by roughly thirteen times. Estimated rows carry a `~` and are dimmed, because two numbers
  an order of magnitude apart must not look alike in the same column.
- **The Claude agents panel polls unconditionally, unlike the live panel.** The live-progress
  timer only runs while a call is in flight, which is right for it. The agents timer cannot
  work that way: the event worth being notified about is an agent *starting*, which by
  definition happens while the list is empty, so a conditional timer could never discover
  the first one.
- **A finished agent lingers for about a minute before disappearing.** Not tidiness — a row
  that vanished the instant it completed could never be *seen* to complete, so the finish
  notification would have nothing to fire on.
- **Notifications come in two layers.** In-page toasts are always on; desktop notifications
  are opt-in behind a button, because browsers only honour a permission request that came
  from a real click. The toasts exist so that a refused permission leaves the feature
  visibly working rather than silently doing nothing.
- **VRAM above 90% is amber, not red.** On a 16 GB card a single 14B model at 32K context
  occupies about 94%. That is the normal loaded state, and it means no second model will
  fit — not that something is wrong. Red starts at 97%.
- **A pruned payload says so.** Retention removes prompt text after a week but keeps the
  metadata row forever, so a 404 is an explanation rather than an error.
- **Reasoning is collapsed by default.** It runs several times longer than the answer —
  1,008 characters of thinking to reply "OK" — so expanding it by default would bury what
  you opened the row to read.

## No UI framework

A CDN-loaded framework was rejected: this is a local monitoring tool and it has to keep
working when the connection is down, which is one of the times you most want to look at
it. A build-time framework was skipped too — the whole dashboard is a handful of panels,
so it would be more machinery than the styling it replaces.
