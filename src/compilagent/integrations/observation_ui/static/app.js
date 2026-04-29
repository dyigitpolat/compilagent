// compilagent · observe — single-file vanilla SPA.
//
// Top-level structure:
//   1. State + DOM refs.
//   2. Fetch helpers.
//   3. Bootstrap (load registry snapshot, populate selectors).
//   4. Run lifecycle (start, WebSocket, event dispatch).
//   5. Per-EventKind handlers (timeline cards, leaderboard, artifacts).
//   6. Sidebar / right-pane renderers.
//   7. GPU telemetry poll.

"use strict";

// ----------------------------------------------------------------- state

const state = {
  runtime: null,
  workloads: [],
  backends: [],
  harnesses: [],
  diagnostics: [],
  recentRuns: [],

  activeRunId: null,
  ws: null,
  wsLastError: null,

  // candidateId -> {description, status, timing, speedup, plan, artifacts:[]}
  candidates: new Map(),
  // The right-pane leaderboard rows (latest leaderboard.updated payload).
  leaderboard: [],
  // run-level artifact paths (deduped).
  artifacts: [],
  selectedArtifactPath: null,

  // Per-stream UI: which streaming card to append a delta to.
  // key = `${kind}:${part_index}` -> HTMLElement of the card body.
  streamingCards: new Map(),
  // candidateId -> compile placeholder card element
  compileCards: new Map(),

  gpu: { gpus: [], lastFetchedAt: null, hasGpu: false },

  selectedWorkloadId: null,
};

// ----------------------------------------------------------------- DOM refs

const $ = (id) => document.getElementById(id);

const ui = {
  selWorkload: $("sel-workload"),
  selHarness: $("sel-harness"),
  inpModel: $("inp-model"),
  inpCandidates: $("inp-candidates"),
  btnStart: $("btn-start"),
  runForm: $("run-form"),
  runIdChip: $("run-id-chip"),
  wsState: $("ws-state"),

  listBackends: $("list-backends"),
  listHarnesses: $("list-harnesses"),
  listWorkloads: $("list-workloads"),
  listDiagnostics: $("list-diagnostics"),
  listRuns: $("list-runs"),
  diagnosticsSection: $("diagnostics-section"),
  countBackends: $("count-backends"),
  countHarnesses: $("count-harnesses"),
  countWorkloads: $("count-workloads"),
  countRuns: $("count-runs"),
  btnShowSource: $("btn-show-source"),

  timeline: $("timeline"),
  btnClear: $("btn-clear"),

  tblLeaderboard: $("tbl-leaderboard").querySelector("tbody"),
  listArtifacts: $("list-artifacts"),
  artifactPreview: $("artifact-preview"),
  gpuSection: $("gpu-section"),
  listGpus: $("list-gpus"),

  sourceModal: $("source-modal"),
  sourceTitle: $("source-title"),
  sourceCode: $("source-code"),
  toastRail: $("toast-rail"),
};

// ----------------------------------------------------------------- fetch

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && body.detail) msg = body.detail;
    } catch (_) { /* swallow */ }
    throw new Error(`${path}: ${msg}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

function toast(message, level) {
  const el = document.createElement("div");
  el.className = `toast ${level || ""}`;
  el.textContent = message;
  ui.toastRail.appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

function fmtMs(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(2);
}

function fmtSpeedup(value) {
  if (value === null || value === undefined) return "—";
  return `×${Number(value).toFixed(2)}`;
}

function fmtTimestamp(value) {
  if (!value) return "";
  try {
    const d = new Date(value * 1000);
    return d.toTimeString().split(" ")[0];
  } catch (_) { return ""; }
}

// ----------------------------------------------------------------- bootstrap

async function bootstrap() {
  try {
    const cfg = await api("/api/runtime/config");
    state.runtime = cfg.settings || {};
    state.backends = cfg.backends || [];
    state.harnesses = cfg.harnesses || [];
    state.workloads = cfg.workloads || [];

    renderRegistries();
    populateSelectors();

    try {
      const diag = await api("/api/workloads/diagnostics");
      state.diagnostics = diag.startup_errors || [];
      renderDiagnostics();
    } catch (_) { /* tolerable */ }

    refreshRuns().catch(() => { /* tolerable */ });
  } catch (exc) {
    toast(`bootstrap failed: ${exc.message}`, "err");
  }

  pollGpuTelemetry();
}

function renderRegistries() {
  ui.countBackends.textContent = state.backends.length;
  ui.countHarnesses.textContent = state.harnesses.length;
  ui.countWorkloads.textContent = state.workloads.length;

  ui.listBackends.innerHTML = "";
  for (const b of state.backends) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${b.id}</span><span class="meta">${(b.artifact_stages || []).join(", ")}</span>`;
    ui.listBackends.appendChild(li);
  }
  if (state.backends.length === 0) {
    ui.listBackends.innerHTML = `<li class="meta">none registered</li>`;
  }

  ui.listHarnesses.innerHTML = "";
  for (const h of state.harnesses) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${h.id}</span><span class="meta">${(h.supported_providers || []).join(", ")}</span>`;
    ui.listHarnesses.appendChild(li);
  }
  if (state.harnesses.length === 0) {
    ui.listHarnesses.innerHTML = `<li class="meta">none registered</li>`;
  }

  ui.listWorkloads.innerHTML = "";
  for (const w of state.workloads) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${w.id}</span><span class="meta">${w.backend_id}</span>`;
    li.title = w.description || w.title || "";
    li.addEventListener("click", () => selectWorkload(w.id));
    ui.listWorkloads.appendChild(li);
  }
  if (state.workloads.length === 0) {
    ui.listWorkloads.innerHTML = `<li class="meta">none registered</li>`;
  }
}

function renderDiagnostics() {
  if (!state.diagnostics.length) {
    ui.diagnosticsSection.hidden = true;
    return;
  }
  ui.diagnosticsSection.hidden = false;
  ui.listDiagnostics.innerHTML = "";
  for (const d of state.diagnostics) {
    const li = document.createElement("li");
    li.className = "diag";
    li.innerHTML = `<span>${d.module}</span><span class="meta">${d.error_type}: ${d.message}</span>`;
    li.title = d.traceback || "";
    ui.listDiagnostics.appendChild(li);
  }
}

function populateSelectors() {
  ui.selWorkload.innerHTML = "";
  for (const w of state.workloads) {
    const opt = document.createElement("option");
    opt.value = w.id;
    opt.textContent = `${w.id} (${w.backend_id})`;
    ui.selWorkload.appendChild(opt);
  }
  if (!state.workloads.length) {
    const opt = document.createElement("option");
    opt.disabled = true;
    opt.textContent = "(no workloads registered)";
    ui.selWorkload.appendChild(opt);
  }

  ui.selHarness.innerHTML = "";
  for (const h of state.harnesses) {
    const opt = document.createElement("option");
    opt.value = h.id;
    opt.textContent = h.id;
    ui.selHarness.appendChild(opt);
  }

  if (state.runtime) {
    ui.inpModel.value = state.runtime.model_name || "";
    ui.inpCandidates.value = state.runtime.max_candidates || 4;
    if (state.runtime.harness && state.harnesses.find(h => h.id === state.runtime.harness)) {
      ui.selHarness.value = state.runtime.harness;
    }
  }
}

function selectWorkload(workloadId) {
  state.selectedWorkloadId = workloadId;
  for (const li of ui.listWorkloads.children) {
    li.classList.toggle("selected", li.firstElementChild?.textContent === workloadId);
  }
  ui.selWorkload.value = workloadId;
  ui.btnShowSource.hidden = false;
}

ui.btnShowSource.addEventListener("click", async () => {
  const wid = state.selectedWorkloadId;
  if (!wid) return;
  try {
    const data = await api(`/api/workloads/${encodeURIComponent(wid)}/source`);
    ui.sourceTitle.textContent = `${wid} · ${data.source_path || "(in-memory)"}`;
    ui.sourceCode.textContent = data.source || "(no source available)";
    ui.sourceModal.hidden = false;
  } catch (exc) {
    toast(exc.message, "err");
  }
});

document.addEventListener("click", (ev) => {
  if (ev.target.matches("[data-close-modal]")) {
    ui.sourceModal.hidden = true;
  }
});

// ----------------------------------------------------------------- runs

async function refreshRuns() {
  try {
    const data = await api("/api/runs");
    state.recentRuns = data.runs || [];
    renderRuns();
  } catch (_) { /* tolerable */ }
}

function renderRuns() {
  ui.countRuns.textContent = state.recentRuns.length;
  ui.listRuns.innerHTML = "";
  for (const r of state.recentRuns.slice(0, 25)) {
    const li = document.createElement("li");
    const status = r.status === "finished" ? "done" : r.status === "failed" ? "fail" : "live";
    li.innerHTML = `<span>${r.run_id}</span><span class="meta">${r.workload_id || "?"} · ${status}</span>`;
    if (r.run_id === state.activeRunId) li.classList.add("selected");
    li.addEventListener("click", () => attachToRun(r.run_id));
    ui.listRuns.appendChild(li);
  }
  if (!state.recentRuns.length) {
    ui.listRuns.innerHTML = `<li class="meta">no runs yet</li>`;
  }
}

ui.runForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const workloadId = ui.selWorkload.value;
  const harness = ui.selHarness.value;
  if (!workloadId) { toast("pick a workload first", "warn"); return; }
  if (!harness) { toast("no harness available", "warn"); return; }

  ui.btnStart.disabled = true;
  try {
    const body = {
      workload_id: workloadId,
      harness,
      model_id: ui.inpModel.value.trim() || undefined,
      max_candidates: parseInt(ui.inpCandidates.value, 10) || 4,
    };
    const result = await api("/api/runs/workload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    toast(`started ${result.run_id}`);
    attachToRun(result.run_id, { live: true });
    refreshRuns().catch(() => {});
  } catch (exc) {
    toast(exc.message, "err");
  } finally {
    ui.btnStart.disabled = false;
  }
});

ui.btnClear.addEventListener("click", () => {
  state.candidates.clear();
  state.leaderboard = [];
  state.artifacts = [];
  state.streamingCards.clear();
  state.compileCards.clear();
  ui.timeline.innerHTML = "";
  ui.tblLeaderboard.innerHTML = "";
  ui.listArtifacts.innerHTML = "";
  ui.artifactPreview.hidden = true;
});

// ----------------------------------------------------------------- websocket

function attachToRun(runId, opts) {
  const live = !!(opts && opts.live);
  state.activeRunId = runId;
  ui.runIdChip.hidden = false;
  ui.runIdChip.textContent = runId;
  ui.btnClear.click();
  if (state.ws) {
    try { state.ws.close(); } catch (_) { /* swallow */ }
    state.ws = null;
  }
  // Hydrate from history when not in live-only mode.
  if (!live) {
    api(`/api/runs/${encodeURIComponent(runId)}/events`)
      .then(({ events }) => (events || []).forEach(handleEvent))
      .catch((exc) => toast(`history: ${exc.message}`, "warn"));
  }
  api(`/api/runs/${encodeURIComponent(runId)}/leaderboard`)
    .then(({ rows }) => { if (rows && rows.length) { state.leaderboard = rows; renderLeaderboard(); } })
    .catch(() => {});
  openWebSocket(runId, live);
  renderRuns();
}

function openWebSocket(runId, live) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({ run_id: runId });
  if (live) params.set("live", "1");
  const url = `${proto}//${window.location.host}/ws?${params}`;
  setWsState("connecting", "dim");
  try {
    state.ws = new WebSocket(url);
  } catch (exc) {
    setWsState("ws: error", "err");
    toast(exc.message, "err");
    return;
  }
  state.ws.addEventListener("open", () => setWsState("ws: live", "live"));
  state.ws.addEventListener("close", () => setWsState("ws: closed", "dim"));
  state.ws.addEventListener("error", () => setWsState("ws: error", "err"));
  state.ws.addEventListener("message", (ev) => {
    let parsed;
    try {
      parsed = JSON.parse(ev.data);
    } catch (_) { return; }
    handleEvent(parsed);
  });
}

function setWsState(text, kind) {
  ui.wsState.textContent = text;
  ui.wsState.className = `chip ${kind || ""}`;
}

// ----------------------------------------------------------------- timeline

function appendCard(kindClass, kindLabel, body, ts) {
  const card = document.createElement("div");
  card.className = `card ${kindClass}`;
  const head = document.createElement("div");
  head.className = "card-head";
  head.innerHTML = `<span class="kind">${kindLabel}</span><span class="timestamp">${fmtTimestamp(ts)}</span>`;
  const bodyEl = document.createElement("div");
  bodyEl.className = "card-body";
  if (typeof body === "string") bodyEl.textContent = body;
  else if (body) bodyEl.appendChild(body);
  card.append(head, bodyEl);
  ui.timeline.appendChild(card);
  ui.timeline.scrollTop = ui.timeline.scrollHeight;
  return { card, body: bodyEl };
}

// ----------------------------------------------------------------- handlers

const HANDLERS = {
  "session.started": (ev) => {
    appendCard("session", "session.started",
      `workload=${ev.payload.workload_id} backend=${ev.payload.backend_id} ` +
      `device=${ev.payload?.device?.arch || "?"} max_candidates=${ev.payload.max_candidates}`,
      ev.timestamp);
  },
  "session.finished": (ev) => {
    appendCard("session", "session.finished",
      `successful=${ev.payload?.successful_count} failed=${ev.payload?.failed_attempts} ` +
      `episode=${ev.payload?.episode_path || "—"}`,
      ev.timestamp);
  },
  "session.failed": (ev) => {
    appendCard("fail", "session.failed",
      `${ev.payload?.error_type || "Error"}: ${ev.payload?.error_message || ev.payload?.message || "(no detail)"}`,
      ev.timestamp);
  },

  "compile.started": (ev) => {
    const { card, body } = appendCard("compile", "compile.started",
      `cand=${ev.candidate_id || ev.payload?.candidate_id || "?"} compiling…`,
      ev.timestamp);
    state.compileCards.set(ev.candidate_id || ev.payload?.candidate_id, body);
  },
  "compile.completed": (ev) => {
    const cid = ev.candidate_id || ev.payload?.candidate_id;
    const ok = ev.payload?.ok ? "ok" : "FAILED";
    const elapsed = ev.payload?.elapsed_ms;
    const placeholder = state.compileCards.get(cid);
    const txt = `cand=${cid} ${ok} elapsed=${fmtMs(elapsed)}ms` +
                (ev.payload?.diagnostics ? `\n${ev.payload.diagnostics}` : "");
    if (placeholder) {
      placeholder.textContent = txt;
      state.compileCards.delete(cid);
    } else {
      appendCard("compile", "compile.completed", txt, ev.timestamp);
    }
  },
  "compiler.pass": (ev) => {
    const p = ev.payload || {};
    appendCard("compiler-pass", "compiler.pass",
      `${p.stage}/${p.name} ${p.action} ${fmtMs(p.duration_ms)}ms` +
      (p.error ? ` error=${p.error}` : ""), ev.timestamp);
  },

  "artifact.created": (ev) => {
    const path = ev.payload?.path;
    if (path && !state.artifacts.includes(path)) {
      state.artifacts.push(path);
      renderArtifacts();
    }
    appendCard("artifact", "artifact.created",
      `${ev.payload?.stage || "?"}: ${path || "?"}`, ev.timestamp);
  },

  "benchmark.started": (ev) => {
    appendCard("bench", "benchmark.started",
      `cand=${ev.candidate_id || "?"} timing…`, ev.timestamp);
  },
  "benchmark.completed": (ev) => {
    const p = ev.payload || {};
    appendCard("bench", "benchmark.completed",
      `cand=${p.candidate_id || ev.candidate_id || "?"}  median=${fmtMs(p.median_ms)}ms  ` +
      `speedup=${fmtSpeedup(p.speedup_vs_baseline)}  correctness=${p.correctness_ok}`,
      ev.timestamp);
  },

  "search_space.derived": (ev) => {
    appendCard("session", "search_space.derived",
      `lever_count=${ev.payload?.lever_count}  backend=${ev.payload?.backend_id}`,
      ev.timestamp);
  },

  "candidate.proposed": (ev) => {
    const p = ev.payload || {};
    state.candidates.set(p.candidate_id, p);
    const ivs = (p.interventions || []).map((iv) =>
      `${iv.target?.kind}(${iv.target?.selector})=${JSON.stringify(iv.payload)}`).join("\n  ");
    appendCard("cand", "candidate.proposed",
      `${p.candidate_id}: ${p.description || "(no description)"}\n  ${ivs}`, ev.timestamp);
  },
  "candidate.validated": (ev) => {
    appendCard("cand", "candidate.validated", `cand=${ev.candidate_id || "?"} ok`, ev.timestamp);
  },
  "candidate.rejected": (ev) => {
    appendCard("cand-rej", "candidate.rejected",
      `cand=${ev.candidate_id || "?"} reason=${ev.payload?.reason}`, ev.timestamp);
  },

  "run.progress": (ev) => {
    const p = ev.payload || {};
    appendCard("progress", "run.progress",
      `${p.successful_count}/${p.max_candidates} successful · ${p.failed_attempts} failed · ` +
      `${p.slots_remaining} slots remaining`, ev.timestamp);
  },
  "leaderboard.updated": (ev) => {
    state.leaderboard = (ev.payload || {}).rows || [];
    renderLeaderboard();
  },

  "agent.thinking.started": (ev) => startStream("thinking", ev),
  "agent.thinking.delta":   (ev) => extendStream("thinking", ev, ev.payload?.text || ""),
  "agent.text.started":     (ev) => startStream("text", ev),
  "agent.text.delta":       (ev) => extendStream("text", ev, ev.payload?.text || ""),

  "tool.call.started": (ev) => {
    const p = ev.payload || {};
    appendCard("tool", `tool.call ${p.tool_name}`,
      `args=${JSON.stringify(p.args)}`, ev.timestamp);
  },
  "tool.call.completed": (ev) => {
    const p = ev.payload || {};
    const result = (p.result || "").toString();
    const truncated = result.length > 800 ? result.slice(0, 800) + " …(truncated)" : result;
    appendCard("tool", `tool.result ${p.tool_name}`, truncated || "(empty)", ev.timestamp);
  },
  "tool.call.failed": (ev) => {
    const p = ev.payload || {};
    appendCard("tool-err", `tool.error ${p.tool_name}`,
      `${p.error_type}: ${p.error_message}`, ev.timestamp);
  },

  "log.line": (ev) => {
    const level = (ev.payload?.level || "info").toLowerCase();
    if (level === "warn" || level === "error") {
      toast(ev.payload?.message || "", level === "error" ? "err" : "warn");
    }
  },
};

function startStream(kind, ev) {
  const key = `${kind}:${ev.payload?.part_index ?? "?"}`;
  const className = kind === "thinking" ? "thinking" : "text";
  const { card, body } = appendCard(className, `agent.${kind}`, "", ev.timestamp);
  card.classList.add("streaming");
  state.streamingCards.set(key, { card, body });
}

function extendStream(kind, ev, text) {
  const key = `${kind}:${ev.payload?.part_index ?? "?"}`;
  let entry = state.streamingCards.get(key);
  if (!entry) {
    startStream(kind, ev);
    entry = state.streamingCards.get(key);
  }
  if (entry && text) {
    entry.body.textContent += text;
    ui.timeline.scrollTop = ui.timeline.scrollHeight;
  }
}

function handleEvent(ev) {
  // Filter: only events for the active run.
  if (state.activeRunId && ev.run_id && ev.run_id !== state.activeRunId) return;
  const handler = HANDLERS[ev.kind];
  if (handler) {
    try {
      handler(ev);
    } catch (exc) {
      console.warn("handler failed", ev.kind, exc);
    }
  } else {
    // Unknown event kind — surface as a faint card so we know something arrived.
    appendCard("progress", ev.kind || "unknown", JSON.stringify(ev.payload || {}, null, 2), ev.timestamp);
  }
}

// ----------------------------------------------------------------- right pane

function renderLeaderboard() {
  ui.tblLeaderboard.innerHTML = "";
  const rows = state.leaderboard.slice().sort((a, b) => {
    const am = a.median_ms === null || a.median_ms === undefined ? Infinity : a.median_ms;
    const bm = b.median_ms === null || b.median_ms === undefined ? Infinity : b.median_ms;
    return am - bm;
  });
  let bestSeen = false;
  for (const row of rows) {
    const tr = document.createElement("tr");
    const isBaseline = row.candidate_id === "baseline";
    const isBad = row.correctness_ok === false;
    if (isBaseline) tr.classList.add("baseline");
    else if (!isBad && !bestSeen) {
      tr.classList.add("best");
      bestSeen = true;
    } else if (isBad) tr.classList.add("bad");
    tr.innerHTML =
      `<td>${row.candidate_id}</td>` +
      `<td>${fmtMs(row.median_ms)}</td>` +
      `<td>${row.candidate_id === "baseline" ? "1.00×" : fmtSpeedup(row.speedup_vs_baseline)}</td>` +
      `<td>${row.correctness_ok === null || row.correctness_ok === undefined ? "?" : row.correctness_ok ? "✓" : "✗"}</td>`;
    ui.tblLeaderboard.appendChild(tr);
  }
}

function renderArtifacts() {
  ui.listArtifacts.innerHTML = "";
  for (const path of state.artifacts) {
    const li = document.createElement("li");
    const display = path.length > 50 ? "…" + path.slice(-50) : path;
    li.innerHTML = `<span title="${path}">${display}</span>`;
    if (path === state.selectedArtifactPath) li.classList.add("selected");
    li.addEventListener("click", () => previewArtifact(path));
    ui.listArtifacts.appendChild(li);
  }
}

async function previewArtifact(absPath) {
  state.selectedArtifactPath = absPath;
  renderArtifacts();
  // Resolve to a path relative to the workspace root. The trace store
  // payloads emit absolute paths; for the preview endpoint we want the
  // suffix after the last `<workspace_dir_name>/` segment.
  let relPath = absPath;
  const wsName = (state.runtime && state.runtime.workspace_dir_name) || ".compilagent";
  const idx = absPath.lastIndexOf(`/${wsName}/`);
  if (idx >= 0) relPath = absPath.slice(idx + wsName.length + 2);
  try {
    const data = await api(`/api/artifacts/preview/${encodeURI(relPath)}`);
    const lang = data.language || "text";
    ui.artifactPreview.hidden = false;
    ui.artifactPreview.textContent = data.text || "(empty)";
    ui.artifactPreview.dataset.language = lang;
  } catch (exc) {
    ui.artifactPreview.hidden = false;
    ui.artifactPreview.textContent = `<error: ${exc.message}>`;
  }
}

// ----------------------------------------------------------------- gpu

async function pollGpuTelemetry() {
  try {
    const data = await api("/api/telemetry/gpu");
    const gpus = data.gpus || [];
    state.gpu.gpus = gpus;
    state.gpu.lastFetchedAt = data.fetched_at || Date.now() / 1000;
    state.gpu.hasGpu = gpus.length > 0;
    renderGpus();
  } catch (_) {
    // Hide the panel on persistent failure — the endpoint already returns
    // {gpus: []} on missing nvidia-smi, so this branch is rare.
    ui.gpuSection.hidden = true;
  }
  setTimeout(pollGpuTelemetry, 4000);
}

function renderGpus() {
  if (!state.gpu.hasGpu) {
    ui.gpuSection.hidden = true;
    return;
  }
  ui.gpuSection.hidden = false;
  ui.listGpus.innerHTML = "";
  for (const gpu of state.gpu.gpus) {
    const li = document.createElement("li");
    const utilPct = gpu.utilization_gpu_pct;
    const memPct = gpu.memory_total_mib
      ? Math.round((gpu.memory_used_mib / gpu.memory_total_mib) * 100)
      : 0;
    const memUsed = gpu.memory_used_mib != null ? `${(gpu.memory_used_mib / 1024).toFixed(1)} GB` : "?";
    const memTotal = gpu.memory_total_mib != null ? `${(gpu.memory_total_mib / 1024).toFixed(0)} GB` : "?";
    const tempClass = gpu.temperature_c == null ? "" : gpu.temperature_c > 80 ? "hot" : gpu.temperature_c > 65 ? "warn" : "";
    li.innerHTML = `
      <span class="meta">#${gpu.index}</span>
      <div class="gpu-bars">
        <div>${gpu.name}  ${gpu.power_w != null ? gpu.power_w.toFixed(0) + "W" : ""}  ${gpu.temperature_c != null ? gpu.temperature_c.toFixed(0) + "°C" : ""}</div>
        <div>util ${utilPct ?? "—"}%</div>
        <div class="bar"><span style="width:${Math.min(utilPct ?? 0, 100)}%"></span></div>
        <div>mem ${memUsed} / ${memTotal}</div>
        <div class="bar ${tempClass}"><span style="width:${Math.min(memPct, 100)}%"></span></div>
      </div>`;
    ui.listGpus.appendChild(li);
  }
}

// ----------------------------------------------------------------- go

bootstrap();
