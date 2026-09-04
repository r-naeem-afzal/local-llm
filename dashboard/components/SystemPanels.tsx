"use client";

import { formatMib, formatSeconds } from "@/lib/format";
import type { SystemSnapshot } from "@/lib/types";
import { BigMetric, Metric, Panel, StatusBadge, UsageBar } from "./ui";

/**
 * The machine panels: GPU, host, and which models are resident.
 *
 * Split into three exported components rather than one, so the page decides the layout.
 * They share a props type because they are always rendered from the same snapshot — the
 * pieces are read together, and reading them from one object is what stops the panels
 * disagreeing with each other.
 */

export function GpuPanel({ snapshot }: { snapshot: SystemSnapshot | null }) {
  const gpu = snapshot?.gpu;

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
      action={<span className="faint mono">{gpu.name.replace("NVIDIA ", "")}</span>}
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

export function HostPanel({ snapshot }: { snapshot: SystemSnapshot | null }) {
  const host = snapshot?.host;

  if (!host?.available) {
    return (
      <Panel title="Host">
        <p className="empty">{host?.error || "No host reading available."}</p>
      </Panel>
    );
  }

  const ramPct = host.ram_total_mib
    ? Math.round((host.ram_used_mib / host.ram_total_mib) * 100)
    : 0;

  return (
    <Panel
      title="Host"
      action={
        <span className={`badge ${snapshot?.server_up ? "badge-ok" : "badge-error"}`}>
          <span
            className="dot"
            style={{ background: snapshot?.server_up ? "var(--ok)" : "var(--error)" }}
          />
          {snapshot?.server_up ? "model server up" : "model server down"}
        </span>
      }
    >
      <div className="metric-grid">
        <BigMetric label="CPU" value={`${Math.round(host.cpu_pct)}%`} />
        <BigMetric label="RAM" value={`${ramPct}%`} />
      </div>

      <UsageBar percent={ramPct} />

      <Metric
        label="RAM used / total"
        value={`${formatMib(host.ram_used_mib)} / ${formatMib(host.ram_total_mib)}`}
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

export function ModelsPanel({ snapshot }: { snapshot: SystemSnapshot | null }) {
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
