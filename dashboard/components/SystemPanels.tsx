"use client";

import { memo } from "react";

import { formatMib, formatSeconds } from "@/lib/format";
import type { SystemSnapshot, Telemetry, TelemetrySample } from "@/lib/types";
import {
  BigMetric,
  Freshness,
  Metric,
  Panel,
  Sparkline,
  StatusBadge,
  UsageBar,
} from "./ui";

/**
 * The machine panels: GPU, host, and which models are resident.
 *
 * Split into three exported components rather than one, so the page decides the layout.
 *
 * ## Two sources, on purpose
 *
 * The GPU and Host panels read their live numbers from `telemetry`, resampled every
 * second, and their slow-moving context — resident model sizes, inference process
 * memory — from `snapshot`, refreshed on the change stream and a slow timer.
 *
 * That split is the fix for the gauges not being live. They previously read everything
 * from the snapshot, which only refreshed when the change stream fired, and that stream
 * fires on *model-call* activity. So an idle machine, or one long generation with no
 * call boundary, left GPU load and VRAM frozen at whatever they read when the last call
 * started. They looked live only by coincidence.
 *
 * Falling back to the snapshot's copy while telemetry has not arrived keeps the first
 * paint populated rather than empty.
 */

function GpuPanelInner({
  snapshot,
  telemetry,
  history,
}: {
  snapshot: SystemSnapshot | null;
  telemetry: Telemetry | null;
  history: TelemetrySample[];
}) {
  // Telemetry first, snapshot as the fallback for the first paint before the first
  // one-second sample lands.
  const gpu = telemetry?.gpu ?? snapshot?.gpu;

  if (!gpu?.available) {
    return (
      <Panel title="GPU">
        <p className="empty">{gpu?.error || "No GPU reading available."}</p>
      </Panel>
    );
  }

  // The gap between VRAM in use and the models' on-disk size. Worth showing because it
  // explains an otherwise confusing observation: a 9 GB model on a 16 GB card leaves far
  // less than 7 GB free.
  //
  // Measured with Qwen3 14B resident at 32K context: 2,240 MiB is the desktop and
  // browser, and roughly 5,100 MiB is the KV cache. The cache dominates, and it is
  // linear in context — Qwen3 14B stores 160 KiB per token at fp16, so 32,768 tokens
  // costs 5,120 MiB. Halving the context would free half of that, which is the lever
  // to pull if a second model needs to fit alongside.
  const modelVram = snapshot?.model_vram_mib ?? 0;
  const overhead = Math.max(0, gpu.used_mib - modelVram);

  return (
    <Panel
      title="GPU"
      action={
        <span style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
          <span className="faint mono">{gpu.name.replace("NVIDIA ", "")}</span>
          {/* The age of the reading, counting up in real time. This is the honest
              answer to "is this live?" — a frozen dashboard and an idle machine look
              identical without it, which is exactly the failure being fixed here. */}
          <Freshness ts={telemetry?.ts ?? null} />
        </span>
      }
    >
      <div className="metric-grid">
        <BigMetric
          label="VRAM used"
          value={`${gpu.used_pct}%`}
          colour={
            gpu.used_pct >= 97
              ? "var(--error)"
              : gpu.used_pct >= 90
                ? "var(--thinking)"
                : "var(--ok)"
          }
        />
        <BigMetric label="GPU load" value={`${gpu.utilisation_pct}%`} />
        <BigMetric
          label="Temp"
          // Null when the driver refuses the reading, which some cards do. An em dash
          // says "not reported" rather than implying a temperature of zero.
          value={gpu.temperature_c === null ? "—" : `${gpu.temperature_c}°C`}
        />
      </div>

      <UsageBar percent={gpu.used_pct} />

      {/* Two charts, because the numbers behave completely differently and each is
          misleading alone. VRAM is a step function - it jumps when a model loads and then
          sits flat - so a steady 96% says nothing about whether work is happening. GPU
          load is spiky and is the one that actually shows a generation running. Seeing
          them together distinguishes "a model is resident but idle" from "resident and
          busy", which is the question this panel exists to answer. */}
      <div className="spark-row">
        <div>
          <div className="spark-label">
            <span>GPU load</span>
            <span className="mono">{gpu.utilisation_pct}%</span>
          </div>
          <Sparkline
            values={history.map((sample) => sample.gpuPct)}
            max={100}
            colour="var(--running)"
            label="GPU load over recent samples"
          />
        </div>
        <div>
          <div className="spark-label">
            <span>VRAM</span>
            <span className="mono">{gpu.used_pct}%</span>
          </div>
          <Sparkline
            values={history.map((sample) => sample.vramPct)}
            max={100}
            colour="var(--thinking)"
            label="VRAM use over recent samples"
          />
        </div>
      </div>

      <Metric
        label="Used / total"
        value={`${formatMib(gpu.used_mib)} / ${formatMib(gpu.total_mib)}`}
      />
      <Metric label="Free" value={formatMib(gpu.free_mib)} />
      <Metric label="Model weights" value={formatMib(modelVram)} />
      {/* Named "KV cache + overhead" rather than the vaguer "other", because the
          measurement showed the cache is the bulk of it and the desktop is a small
          share. A label that implies otherwise sends the reader to close browser
          tabs when the actual lever is the context length. */}
      <Metric
        label="KV cache + overhead"
        value={<span className="dim">{formatMib(overhead)}</span>}
      />
    </Panel>
  );
}

function HostPanelInner({
  snapshot,
  telemetry,
  history,
}: {
  snapshot: SystemSnapshot | null;
  telemetry: Telemetry | null;
  history: TelemetrySample[];
}) {
  const host = snapshot?.host;

  if (!host?.available) {
    return (
      <Panel title="Host">
        <p className="empty">{host?.error || "No host reading available."}</p>
      </Panel>
    );
  }

  // CPU and RAM come from telemetry so they move every second. The inference-process
  // figures below stay on the snapshot, because that scan walks every process on the
  // machine and costs about 260 ms - far too slow to repeat once a second.
  const cpuPct = telemetry?.cpu_pct ?? host.cpu_pct;
  const ramUsed = telemetry?.ram_used_mib ?? host.ram_used_mib;
  const ramTotal = telemetry?.ram_total_mib ?? host.ram_total_mib;
  const ramPct = ramTotal ? Math.round((ramUsed / ramTotal) * 100) : 0;
  // Also from telemetry, so stopping or starting the model server shows within a
  // second or two rather than at the next model call. The probe behind it is cached
  // server-side, so asking every second costs nothing.
  const serverUp = telemetry?.server_up ?? snapshot?.server_up ?? false;

  return (
    <Panel
      title="Host"
      action={
        <span className={`badge ${serverUp ? "badge-ok" : "badge-error"}`}>
          <span
            className="dot"
            style={{ background: serverUp ? "var(--ok)" : "var(--error)" }}
          />
          {serverUp ? "model server up" : "model server down"}
        </span>
      }
    >
      <div className="metric-grid">
        <BigMetric label="CPU" value={`${Math.round(cpuPct)}%`} />
        <BigMetric label="RAM" value={`${ramPct}%`} />
      </div>

      <UsageBar percent={ramPct} />

      <div className="spark-label" style={{ marginTop: 8 }}>
        <span>CPU</span>
        <span className="mono">{Math.round(cpuPct)}%</span>
      </div>
      <Sparkline
        values={history.map((sample) => sample.cpuPct)}
        max={100}
        colour="var(--ok)"
        label="CPU load over recent samples"
      />

      <Metric
        label="RAM used / total"
        value={`${formatMib(ramUsed)} / ${formatMib(ramTotal)}`}
      />
      {/* System RAM held by the inference processes. It matters even though the model
          runs on the GPU: the runtime memory-maps the model file, so a 9 GB model also
          appears in host RAM — and a machine that starts swapping presents as the model
          having mysteriously slowed down. */}
      <Metric
        label={`Inference RSS (${host.inference_processes} proc)`}
        value={formatMib(host.inference_rss_mib)}
      />
    </Panel>
  );
}

function ModelsPanelInner({ snapshot }: { snapshot: SystemSnapshot | null }) {
  const loaded = snapshot?.loaded_models ?? [];
  const installed = snapshot?.installed_models ?? [];

  return (
    <Panel
      title="Models"
      action={
        <span className="faint">
          {loaded.length} loaded / {installed.length} on disk
        </span>
      }
    >
      {loaded.length === 0 ? (
        // Not an error state. Only one 14B model fits in 16 GB here, and an idle model is
        // evicted when its TTL expires, so "nothing loaded" is the normal resting state
        // between runs — the message says so rather than looking like a fault.
        <p className="empty">
          Nothing resident. The next call will load a model, which takes about 18s.
        </p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>State</th>
                <th className="num">Size</th>
                <th className="num">Context</th>
                {/* TTL explains an otherwise baffling observation: a model loaded moments
                    ago is suddenly gone and the next call pays the load again. */}
                <th className="num">Evicts in</th>
              </tr>
            </thead>
            <tbody>
              {loaded.map((model) => (
                <tr key={model.key}>
                  <td className="mono">{model.key}</td>
                  <td>
                    <StatusBadge status={model.status === "idle" ? "idle" : "running"} />
                  </td>
                  <td className="num">{formatMib(model.size_mib)}</td>
                  <td className="num">
                    {model.context_length.toLocaleString()}
                    <span className="faint">
                      {" "}
                      / {model.max_context_length.toLocaleString()}
                    </span>
                  </td>
                  <td className="num">{formatSeconds(model.ttl_remaining_s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {installed.length > 0 && (
        <>
          <p className="section-label" style={{ marginTop: 14 }}>
            On disk
          </p>
          <div className="table-scroll">
            <table>
              <tbody>
                {installed.map((model) => (
                  <tr key={model.key}>
                    <td className="mono">{model.key}</td>
                    <td className="dim">
                      {model.params || model.type}
                      {model.architecture ? ` · ${model.architecture}` : ""}
                    </td>
                    <td className="num">{formatMib(model.size_mib)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  );
}

/**
 * Memoized so a change in another panel's data cannot re-render this one. Without this,
 * every 900 ms live-progress tick repainted the entire dashboard.
 */
export const GpuPanel = memo(GpuPanelInner);

/**
 * Memoized so a change in another panel's data cannot re-render this one. Without this,
 * every 900 ms live-progress tick repainted the entire dashboard.
 */
export const HostPanel = memo(HostPanelInner);

/**
 * Memoized so a change in another panel's data cannot re-render this one. Without this,
 * every 900 ms live-progress tick repainted the entire dashboard.
 */
export const ModelsPanel = memo(ModelsPanelInner);
