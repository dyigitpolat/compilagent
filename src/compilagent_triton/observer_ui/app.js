// Compilagent Triton observer — single-stream conversation over WebSocket.

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = {
  examples: [],
  selectedExampleId: null,
  runtimeConfig: null,
  runs: {},
  activeRunId: null,
  filter: 'all',
  items: [],
  itemByKey: new Map(),
  thinkingActiveKey: null,
  textActiveKey: null,
  candidates: new Map(),
  passes: new Map(),
  telemetry: [],
  artifacts: [],
  artifactManifest: [],
  finalSummary: null,
  bestSpeedupSoFar: null,
  source: null,
  irRuns: [],
  ws: null,
  wsBackoff: 250,
  collapsed: new Set(),
  followMode: true,
  smoothScrollRaf: 0,
  workloads: [],
  selectedWorkloadId: null,
};

// ----- helpers -----

function escapeHtml(v) {
  if (v == null) return '';
  return String(v)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}
function escapeAttr(v) { return escapeHtml(v); }
function fmtMs(v) {
  if (typeof v !== 'number') return '—';
  if (v < 1) return `${(v * 1000).toFixed(1)}μs`;
  if (v < 1000) return `${v.toFixed(3)}ms`;
  return `${(v / 1000).toFixed(2)}s`;
}
function fmtSpeedup(v) { return typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(3)}×` : '—'; }
function fmtBandwidth(v) { return typeof v === 'number' ? `${v.toFixed(0)} GB/s` : '—'; }
function fmtTime(ts) { if (!ts) return ''; return new Date(ts).toLocaleTimeString(); }
function renderMarkdown(t) {
  if (!t) return '';
  if (typeof window.marked !== 'undefined') return window.marked.parse(String(t), { breaks: true, gfm: true });
  return `<p>${escapeHtml(t).replace(/\n/g, '<br />')}</p>`;
}
function renderCode(code, language = 'python') {
  if (!code) return '';
  if (typeof window.hljs !== 'undefined') {
    try { return window.hljs.highlight(code, { language, ignoreIllegals: true }).value; } catch { return escapeHtml(code); }
  }
  return escapeHtml(code);
}
function renderJson(value) {
  if (value == null) return '';
  let text;
  if (typeof value === 'string') {
    try { text = JSON.stringify(JSON.parse(value), null, 2); }
    catch { return `<pre>${escapeHtml(value)}</pre>`; }
  } else {
    try { text = JSON.stringify(value, null, 2); } catch { return `<pre>${escapeHtml(String(value))}</pre>`; }
  }
  return `<pre><code class="language-json">${renderCode(text, 'json')}</code></pre>`;
}
async function fetchJson(url, init) {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
function parseList(v) {
  return String(v || '').split(',').map((s) => s.trim()).filter(Boolean)
    .map(Number).filter(Number.isFinite);
}
function parseStringList(v) {
  return String(v || '').split(',').map((s) => s.trim()).filter(Boolean)
    .map((s) => (s === 'none' ? '' : s));
}

// ----- init -----

async function init() {
  bindUi();
  bindScrollFollow();
  await Promise.all([
    loadRuntimeConfig(),
    loadWorkloads(),
    loadTelemetry(),
  ]);
  connectWebSocket();
  setInterval(loadTelemetry, 4000);
  renderRuntimeChrome();
}

function bindUi() {
  $('#harness-select').addEventListener('change', onConfigChange);
  $('#model-select').addEventListener('change', onConfigChange);
  $('#workload-select').addEventListener('change', () => selectWorkload($('#workload-select').value));
  $('#max-candidates-input').addEventListener('change', onConfigChange);
  $('#run-button').addEventListener('click', startRun);
  $('#clear-stream').addEventListener('click', clearStream);
  $$('#filter-chips .chip').forEach((c) => c.addEventListener('click', () => {
    state.filter = c.dataset.filter;
    $$('#filter-chips .chip').forEach((cc) => cc.classList.toggle('active', cc === c));
    applyFilter();
  }));
  $$('#artifact-preview [data-modal-close]').forEach((el) =>
    el.addEventListener('click', () => $('#artifact-preview').classList.add('hidden')),
  );
}

function clearStream() {
  state.items = [];
  state.itemByKey.clear();
  state.thinkingActiveKey = null;
  state.textActiveKey = null;
  state.candidates.clear();
  state.passes.clear();
  state.collapsed.clear();
  state.irRuns = [];
  state.artifactManifest = [];
  state.finalSummary = null;
  state.bestSpeedupSoFar = null;
  $('#stream').innerHTML = '';
  // Always seed with the source kernel as the first card.
  insertSourceCard();
  updateMetrics();
}

async function onConfigChange() {
  const harness = $('#harness-select').value;
  const model = $('#model-select').value;
  const max_candidates = Math.max(1, Math.min(32, Number($('#max-candidates-input').value) || 4));
  try {
    state.runtimeConfig = await fetchJson('/api/runtime/config', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ harness, model, mode: 'optimize', max_candidates }),
    });
    renderRuntimeChrome();
  } catch (err) { console.error(err); }
}

async function loadRuntimeConfig() {
  try { state.runtimeConfig = await fetchJson('/api/runtime/config'); } catch (e) { console.error(e); }
}

async function loadWorkloads() {
  const sel = $('#workload-select');
  try {
    const data = await fetchJson('/api/workloads');
    state.workloads = data.workloads || [];
    sel.innerHTML = state.workloads.map((w) => {
      const backend = w.backend?.id || w.backend_id;
      const label = `${w.title} · ${backend}`;
      return `<option value="${escapeAttr(w.id)}">${escapeHtml(label)}</option>`;
    }).join('');
    if (state.workloads.length) {
      const def = state.workloads.find((w) => w.id === 'vit_block') || state.workloads[0];
      await selectWorkload(def.id);
    } else {
      // Empty registry — fetch diagnostics so we can tell the user WHY.
      sel.innerHTML = '<option value="">(no workloads registered)</option>';
      try {
        const diag = await fetchJson('/api/workloads/diagnostics');
        const errs = diag.startup_errors || [];
        if (errs.length) {
          $('#run-status').textContent = `Workload registry empty — ${errs.length} startup error(s); see console`;
          console.warn('compilagent-observe startup errors:\n' + errs.join('\n'));
        } else {
          $('#run-status').textContent = 'Workload registry empty (no startup errors reported); restart the server.';
        }
      } catch { /* ignore */ }
    }
  } catch (err) {
    console.error('loadWorkloads', err);
    sel.innerHTML = '<option value="">(failed to load /api/workloads)</option>';
    $('#run-status').textContent = `Failed to load workloads: ${err.message}`;
  }
}

async function selectWorkload(id) {
  state.selectedWorkloadId = id;
  $('#workload-select').value = id;
  const w = state.workloads.find((x) => x.id === id);
  if (!w) return;
  $('#backend-pill').textContent = `backend: ${w.backend?.id || w.backend_id}`;
  $('#run-status').textContent = `Workload selected: ${w.title}.`;
  // For triton-kernel workloads we still have an /api/examples/{id}/kernel
  // preview available; for full-model workloads there's no single-symbol view
  // (the agent will see the FX graph + output_code via the IR browser).
  state.source = null;
  if (w.kind === 'kernel') {
    // Triton kernel workloads still come with a Triton-aware AST extractor
    // that returns just the @triton.jit function body.
    try {
      state.source = await fetchJson(`/api/examples/${encodeURIComponent(id)}/kernel`);
    } catch { state.source = null; }
  }
  // For everything else (full_model / fused_subgraph), fall back to the
  // workload module's full Python source — that's the "JIT source" the user
  // means: the nn.Module + builder code that gets handed to torch.compile.
  if (!state.source || !state.source.source) {
    try {
      state.source = await fetchJson(`/api/workloads/${encodeURIComponent(id)}/source`);
    } catch { state.source = state.source || null; }
  }
  insertSourceCard();
}

// `loadExamples` / `selectExample` / `setRunControls` are gone — the
// example dropdown was retired in favour of the workload registry.

async function loadTelemetry() {
  try {
    const data = await fetchJson('/api/telemetry/gpu');
    state.telemetry = data.gpus || [];
    renderTelemetry();
  } catch { /* ignore */ }
}

// ----- WebSocket -----

function connectWebSocket() {
  if (state.ws) { try { state.ws.close(); } catch {} }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  // live=1 → server skips historical events; we only see what happens after connect.
  const url = `${proto}://${location.host}/ws?live=1`;
  let ws;
  try { ws = new WebSocket(url); }
  catch (e) { console.error('ws ctor failed', e); scheduleReconnect(); return; }
  state.ws = ws;
  $('#conn-pill').textContent = 'connecting…';
  ws.addEventListener('open', () => {
    $('#conn-pill').textContent = 'connected';
    $('#conn-pill').classList.remove('pill-muted');
    state.wsBackoff = 250;
  });
  ws.addEventListener('message', (e) => {
    let event;
    try { event = JSON.parse(e.data); } catch { return; }
    ingestEvent(event);
  });
  ws.addEventListener('close', () => {
    $('#conn-pill').textContent = 'reconnecting…';
    $('#conn-pill').classList.add('pill-muted');
    scheduleReconnect();
  });
  ws.addEventListener('error', () => {
    $('#conn-pill').textContent = 'reconnecting…';
    $('#conn-pill').classList.add('pill-muted');
  });
}
function scheduleReconnect() {
  const wait = Math.min(8000, state.wsBackoff);
  state.wsBackoff = Math.min(8000, state.wsBackoff * 2);
  setTimeout(connectWebSocket, wait);
}

// ----- ingestion -----

function ingestEvent(ev) {
  const kind = ev.kind;
  const payload = ev.payload || {};
  const ts = ev.timestamp;

  if (kind === 'agent.run_started' || kind === 'run.started') {
    if (state.activeRunId !== payload.run_id) clearStream();
    state.activeRunId = payload.run_id;
    state.runs[payload.run_id] = {
      ...state.runs[payload.run_id], status: 'running',
      kind: kind === 'agent.run_started' ? 'agent' : 'benchmark',
      max_candidates: payload.max_candidates ?? state.runs[payload.run_id]?.max_candidates,
      successful_count: 0,
      failed_attempts: 0,
    };
    // Reset the best-so-far tracker; "new best found" cards stream during
    // the run, while the full Insights / IR / Artifacts cards are only
    // appended at terminal events.
    state.bestSpeedupSoFar = null;
    pushItem({
      key: `run-${payload.run_id}`, kind: 'system', filter: 'all', ts,
      title: kind === 'agent.run_started' ? 'Agent run started' : 'Benchmark run started',
      pillLabel: 'run', pillClass: 'pill-loop',
      contentHtml: `<small class="muted">${escapeHtml(payload.run_id)}${payload.model ? ` · model ${escapeHtml(payload.model)}` : ''}${payload.reasoning_effort ? ` · effort ${escapeHtml(payload.reasoning_effort)}` : ''}</small>`,
    });
  } else if (kind === 'agent.run_completed') {
    if (state.runs[payload.run_id]) state.runs[payload.run_id].status = 'completed';
    pushItem({
      key: `run-end-${payload.run_id}`, kind: 'system', filter: 'all', ts,
      title: 'Agent run completed', pillLabel: 'run', pillClass: 'pill-loop',
      contentHtml: `<small class="muted">elapsed ${fmtMs(payload.elapsed_ms)}</small>`,
    });
    appendIrCard();
    appendArtifactsCard();
    appendInsightsCard();
    if (payload.final_text) appendFinalSummaryCard(payload.final_text);
  } else if (kind === 'run.completed') {
    if (state.runs[payload.run_id]) state.runs[payload.run_id].status = 'completed';
    appendIrCard();
    appendArtifactsCard();
    appendInsightsCard();
    if (payload.final_text) appendFinalSummaryCard(payload.final_text);
  } else if (kind === 'run.progress') {
    const r = state.runs[payload.run_id] || (state.runs[payload.run_id] = {});
    r.successful_count = payload.successful_count ?? r.successful_count ?? 0;
    r.failed_attempts = payload.failed_attempts ?? r.failed_attempts ?? 0;
    if (typeof payload.max_candidates === 'number') r.max_candidates = payload.max_candidates;
  } else if (kind === 'agent.run_failed' || kind === 'run.failed') {
    if (state.runs[payload.run_id]) state.runs[payload.run_id].status = 'failed';
    // Surface partial insights / IR / artifacts even on failure so the user
    // sees what was collected before the run blew up.
    appendIrCard();
    appendArtifactsCard();
    appendInsightsCard();
    pushItem({
      key: `run-fail-${payload.run_id}-${ev.event_id}`, kind: 'error', filter: 'all', ts,
      title: 'Run failed', pillLabel: 'fail', pillClass: 'pill-status failed',
      contentHtml: `<div class="msg-content">${escapeHtml(payload.error_type || 'Error')}: ${escapeHtml(payload.error || '')}</div>`,
    });

  } else if (kind === 'agent.thinking_started') {
    // Use the server-assigned monotonic part_id so each new model response
    // opens a fresh card. Defer card creation until first delta — avoids
    // empty caret cards.
    const sub = (payload.part_id != null) ? `p${payload.part_id}` : `i${payload.index ?? 0}`;
    state.thinkingActiveKey = `thk-${payload.run_id}-${sub}`;
  } else if (kind === 'agent.thinking_delta') {
    const sub = (payload.part_id != null) ? `p${payload.part_id}` : `i${payload.index ?? 0}`;
    const key = `thk-${payload.run_id}-${sub}`;
    state.thinkingActiveKey = key;
    if (!(payload.delta || '').length) return;
    appendToItem(key, payload.delta || '', {
      kind: 'thinking', filter: 'thinking', ts,
      title: 'thinking', pillLabel: 'thinking', pillClass: 'pill-thinking',
    });
  } else if (kind === 'agent.text_started') {
    const sub = (payload.part_id != null) ? `p${payload.part_id}` : `i${payload.index ?? 0}`;
    state.textActiveKey = `txt-${payload.run_id}-${sub}`;
  } else if (kind === 'agent.text_delta') {
    const sub = (payload.part_id != null) ? `p${payload.part_id}` : `i${payload.index ?? 0}`;
    const key = `txt-${payload.run_id}-${sub}`;
    state.textActiveKey = key;
    if (!(payload.delta || '').length) return;
    appendToItem(key, payload.delta || '', {
      kind: 'text', filter: 'text', ts,
      title: 'assistant', pillLabel: 'assistant', pillClass: 'pill-text',
    });

  } else if (kind === 'agent.tool_call') {
    finalizeLive(state.thinkingActiveKey); state.thinkingActiveKey = null;
    finalizeLive(state.textActiveKey); state.textActiveKey = null;
    pushItem({
      key: `tool-${payload.tool_call_id}`, kind: 'tool', filter: 'tool', ts,
      title: payload.tool, pillLabel: 'tool', pillClass: 'pill-tool',
      args: payload.args, status: 'running', collapsedDefault: true,
    });
  } else if (kind === 'agent.tool_result') {
    updateItem(`tool-${payload.tool_call_id}`, (it) => {
      it.status = 'returned'; it.preview = payload.preview;
    });
  } else if (kind === 'tool.failed') {
    pushItem({
      key: `toolfail-${ev.event_id}`, kind: 'error', filter: 'tool', ts,
      title: `${payload.tool} failed`, pillLabel: 'fail', pillClass: 'pill-status failed',
      contentHtml: `<div class="msg-content"><strong>${escapeHtml(payload.error_type || 'Error')}</strong>: ${escapeHtml(payload.error || '')}</div>`,
    });

  } else if (kind === 'hypothesis.recorded') {
    pushItem({
      key: `hyp-${payload.hypothesis_id || ev.event_id}`, kind: 'hypothesis',
      filter: 'reasoning', ts,
      title: 'hypothesis', pillLabel: 'hypothesis', pillClass: 'pill-hypothesis',
      contentHtml:
        `<div class="msg-content"><strong>${escapeHtml(payload.statement || '')}</strong></div>` +
        (payload.expected_effect
          ? `<div class="msg-content"><em>expected</em> ${escapeHtml(payload.expected_effect)}</div>`
          : ''),
    });
  } else if (kind === 'agent.reasoning_summary') {
    pushItem({
      key: `sum-${payload.summary_id || ev.event_id}`, kind: 'summary',
      filter: 'reasoning', ts,
      title: 'reasoning summary', pillLabel: 'summary', pillClass: 'pill-summary',
      contentHtml:
        `<div class="msg-content">${renderMarkdown(payload.summary || '')}</div>` +
        (payload.next_step ? `<div class="msg-content"><em>next</em> ${escapeHtml(payload.next_step)}</div>` : ''),
    });

  } else if (kind === 'candidate.proposed') {
    (payload.candidates || []).forEach((c) => {
      const key = `cand-${c.id}`;
      state.candidates.set(c.id, key);
      pushItem({
        key, kind: 'candidate', filter: 'candidate', ts,
        title: c.description || c.id, pillLabel: c.kind || 'candidate', pillClass: 'pill-status proposed',
        candidate: { ...c, status: 'proposed' },
      });
    });
  } else if (kind === 'candidate.validated') {
    const key = state.candidates.get(payload.candidate_id);
    if (key) updateItem(key, (it) => {
      if (it.candidate) {
        it.candidate.validated = payload.ok;
        it.candidate.status = payload.ok ? 'validated' : 'rejected';
        it.candidate.validation_diagnostics = payload.diagnostics;
      }
    });
  } else if (kind === 'candidate.judged') {
    const key = state.candidates.get(payload.candidate_id);
    if (key) updateItem(key, (it) => {
      if (it.candidate) {
        it.candidate.status = payload.verdict || 'judged';
        it.candidate.rationale = payload.rationale;
      }
    });
  } else if (kind === 'candidate.rationale') {
    const key = state.candidates.get(payload.candidate_id);
    if (key) updateItem(key, (it) => { if (it.candidate) it.candidate.rationale = payload.rationale; });

  } else if (kind === 'benchmark.started') {
    pushItem({
      key: `benchstart-${ev.event_id}`, kind: 'benchmark', filter: 'benchmark', ts,
      title: 'benchmark started', pillLabel: 'bench', pillClass: 'pill-bench',
      contentHtml: `<small class="muted">${escapeHtml(payload.example_id || payload.kernel_id || '')}</small>`,
      collapsedDefault: true,
    });
  } else if (kind === 'benchmark.completed') {
    pushItem({
      key: `bench-${ev.event_id}`, kind: 'benchmark', filter: 'benchmark', ts,
      title: 'benchmark completed', pillLabel: 'bench', pillClass: 'pill-bench',
      benchmark: payload,
    });
    const cid = payload.candidate_id || (payload.best && payload.best.candidate_id);
    if (cid) {
      const key = state.candidates.get(cid);
      if (key) updateItem(key, (it) => {
        if (it.candidate) {
          const best = payload.best || {};
          it.candidate.median_ms = best.median_ms;
          it.candidate.speedup_vs_baseline = best.speedup_vs_baseline;
          it.candidate.bandwidth_gbps = best.bandwidth_gbps;
          it.candidate.profile_metrics = best.profile_metrics;
        }
      });
    }
    // Stream a lightweight "new best found" card whenever the latest
    // benchmark beats the prior best. The full performance-insights summary
    // is only appended at terminal events; here we just flag the milestone.
    const sp = (payload.best && payload.best.speedup_vs_baseline)
            ?? payload.speedup_vs_baseline;
    if (typeof sp === 'number' && sp > 1.0
        && (state.bestSpeedupSoFar === null || sp > state.bestSpeedupSoFar)) {
      state.bestSpeedupSoFar = sp;
      const pct = ((sp - 1.0) * 100).toFixed(1);
      const sign = sp >= 1.0 ? '+' : '';
      pushItem({
        key: `best-${ev.event_id}`, kind: 'best', filter: 'benchmark', ts,
        title: `New best · ${cid || ''} (${sign}${pct}%)`,
        pillLabel: 'best', pillClass: 'pill-insight',
        contentHtml: `<div class="msg-content"><strong>${fmtSpeedup(sp)}</strong> · median ${fmtMs(payload.best && payload.best.median_ms)}</div>`,
      });
    }
  } else if (kind === 'comparison.created') {
    pushItem({
      key: `cmp-${ev.event_id}`, kind: 'comparison', filter: 'benchmark', ts,
      title: payload.conclusion || 'comparison', pillLabel: 'compare', pillClass: 'pill-bench',
      comparison: payload,
    });
  } else if (kind === 'decision_trace.created') {
    pushItem({
      key: `dec-${ev.event_id}`, kind: 'decision', filter: 'compiler', ts,
      title: `${payload.kind || 'decision'} · ${payload.op_name || ''}`,
      pillLabel: 'decision', pillClass: 'pill-trace',
      decision: payload,
    });
  } else if (kind === 'compiler.pass') {
    const candKey = payload.candidate_id || 'baseline';
    const key = `passes-${candKey}`;
    let item = state.itemByKey.get(key);
    if (!item) {
      item = pushItem({
        key, kind: 'compiler', filter: 'compiler', ts,
        title: `compiler passes · ${candKey}`,
        pillLabel: 'compiler', pillClass: 'pill-code',
        passes: [], collapsedDefault: true,
      });
      state.passes.set(candKey, key);
    }
    item.passes.push(payload);
    rerenderItem(item);
  } else if (kind === 'artifact.created') {
    if (payload.path) state.artifacts = Array.from(new Set([payload.path, ...state.artifacts])).slice(0, 64);
    if (payload.path && !state.artifactManifest.find((a) => a.path === payload.path)) {
      state.artifactManifest.push({
        run_id: payload.run_id || '',
        kernel_id: payload.kernel_id || '',
        stage: payload.stage || '',
        path: payload.path,
      });
    }
    // Pick up both Triton MLIR artifacts and TorchInductor python/log artifacts.
    const isInductor = /(output_code|fx_graph|schedule|fusion)\.(py|log)$/i.test(payload.path || '');
    const isTriton = /\.(ttir|ttgir|llir|ptx|mlir)$/i.test(payload.path || '');
    if (payload.run_id && payload.path && (isInductor || isTriton)) {
      if (!state.irRuns.find((r) => r.run_id === payload.run_id)) {
        state.irRuns.push({ run_id: payload.run_id, kernel_id: payload.kernel_id || '' });
      }
    }
  }
  updateMetrics();
}

// ----- stream items -----

function pushItem(item) {
  if (state.itemByKey.has(item.key)) {
    const existing = state.itemByKey.get(item.key);
    Object.assign(existing, item);
    rerenderItem(existing);
    return existing;
  }
  if (item.collapsedDefault) state.collapsed.add(item.key);
  state.items.push(item);
  state.itemByKey.set(item.key, item);
  const el = renderItem(item);
  item.el = el;
  $('#stream').appendChild(el);
  applyFilterToEl(item, el);
  maybeAutoscroll();
  return item;
}

function appendToItem(key, delta, fallback) {
  let it = state.itemByKey.get(key);
  if (!it) it = pushItem({ ...fallback, key, contentRaw: '', live: true });
  it.contentRaw = (it.contentRaw || '') + delta;
  // For streaming text/thinking, mutate the existing .msg-content's textContent
  // in place rather than replacing the whole card — this avoids the blink and
  // dramatically reduces DOM churn at high delta rates. Markdown rendering is
  // skipped during streaming (applied on finalizeLive instead).
  if ((it.kind === 'thinking' || it.kind === 'text') && it.el) {
    const content = it.el.querySelector('.msg-content');
    if (content) {
      content.textContent = it.contentRaw;
      maybeAutoscroll();
      return;
    }
  }
  rerenderItem(it);
  maybeAutoscroll();
}

function finalizeLive(key) {
  if (!key) return;
  const it = state.itemByKey.get(key);
  if (it) { it.live = false; rerenderItem(it); }  // re-render once with markdown
}

function updateItem(key, mutator) {
  const it = state.itemByKey.get(key);
  if (!it) return;
  mutator(it);
  rerenderItem(it);
}

function rerenderItem(it) {
  const old = it.el;
  if (!old) return;
  const fresh = renderItem(it);  // sets it.el = fresh internally
  old.replaceWith(fresh);
}

function renderItem(it) {
  const el = document.createElement('div');
  const collapsed = state.collapsed.has(it.key);
  el.className = `msg msg-${it.kind} ${it.candidate ? (it.candidate.status || '') : ''} ${collapsed ? 'collapsed' : ''}`.trim();
  el.dataset.key = it.key;
  el.innerHTML = `
    <div class="msg-time">${escapeHtml(fmtTime(it.ts))}</div>
    <div class="msg-body">
      <div class="msg-head" data-toggle="${escapeAttr(it.key)}">
        <span class="msg-toggle">▾</span>
        <span class="pill ${it.pillClass || 'pill-muted'}">${escapeHtml(it.pillLabel || it.kind)}</span>
        <strong>${escapeHtml(it.title || '')}</strong>
        ${renderItemHeadExtra(it)}
      </div>
      ${renderItemContent(it)}
    </div>`;
  el.querySelector('.msg-head').addEventListener('click', () => toggleCollapsed(it.key));
  it.el = el;
  // Wire up any nested controls (IR loader)
  if (it.kind === 'ir') {
    el.querySelector('[data-action="load-ir"]')?.addEventListener('click', () => loadIrInline(it));
  }
  if (it.kind === 'source' && state.source && state.source.source) {
    // re-highlight after replace
    el.querySelectorAll('pre code').forEach((c) => {
      try { window.hljs?.highlightElement(c); } catch {}
    });
  }
  return el;
}

function toggleCollapsed(key) {
  if (state.collapsed.has(key)) state.collapsed.delete(key);
  else state.collapsed.add(key);
  const it = state.itemByKey.get(key);
  if (it) rerenderItem(it);
}

function renderItemHeadExtra(it) {
  if (it.kind === 'tool' && it.status) {
    return `<span class="pill pill-status ${it.status === 'returned' ? 'accepted' : 'running'}">${escapeHtml(it.status)}</span>`;
  }
  if (it.kind === 'candidate' && it.candidate) {
    const s = it.candidate.status || 'proposed';
    return `<span class="pill pill-status ${s}">${escapeHtml(s)}</span>`;
  }
  return '';
}

function renderItemContent(it) {
  if (it.kind === 'thinking' || it.kind === 'text') {
    // While streaming, render plain text into a single node so we can mutate
    // textContent in-place without churning markdown / DOM. Once finalized we
    // upgrade to rendered markdown.
    if (it.live) {
      return `<div class="msg-content live-caret" style="white-space:pre-wrap;">${escapeHtml(it.contentRaw || '')}</div>`;
    }
    return `<div class="msg-content">${renderMarkdown(it.contentRaw || '')}</div>`;
  }
  if (it.kind === 'tool') {
    const argHtml = it.args ? renderJson(it.args) : '<pre><code class="muted">(no args)</code></pre>';
    let preview = '<pre><code class="muted">(awaiting result…)</code></pre>';
    if (it.preview) {
      const text = String(it.preview).trim();
      if (text.startsWith('{') || text.startsWith('[')) preview = renderJson(text);
      else preview = `<pre>${escapeHtml(text.slice(0, 1500))}</pre>`;
    }
    return `<div class="tool-grid">
      <div class="tool-col"><em>args</em>${argHtml}</div>
      <div class="tool-col"><em>result</em>${preview}</div>
    </div>`;
  }
  if (it.kind === 'candidate' && it.candidate) {
    const c = it.candidate;
    const changes = c.changes ? renderJson(c.changes) : '';
    const stats = [
      typeof c.speedup_vs_baseline === 'number' ? `<div class="stat">speedup <strong>${fmtSpeedup(c.speedup_vs_baseline)}</strong></div>` : '',
      typeof c.median_ms === 'number' ? `<div class="stat">median <strong>${fmtMs(c.median_ms)}</strong></div>` : '',
      typeof c.bandwidth_gbps === 'number' ? `<div class="stat">bw <strong>${fmtBandwidth(c.bandwidth_gbps)}</strong></div>` : '',
    ].filter(Boolean).join('');
    const profileRow = c.profile_metrics && Object.keys(c.profile_metrics).length
      ? `<div class="candidate-stats">${Object.entries(c.profile_metrics).map(([k, v]) =>
          `<div class="stat">${escapeHtml(k)} <strong>${typeof v === 'number' ? v.toFixed(2) : escapeHtml(String(v))}</strong></div>`).join('')}</div>`
      : '';
    return [
      c.expected_effect ? `<div class="msg-content"><em>expected</em> ${escapeHtml(c.expected_effect)}</div>` : '',
      changes,
      stats ? `<div class="candidate-stats">${stats}</div>` : '',
      profileRow,
      c.rationale ? `<div class="msg-content"><em>rationale</em> ${escapeHtml(c.rationale)}</div>` : '',
    ].join('');
  }
  if (it.kind === 'benchmark' && it.benchmark) {
    const best = it.benchmark.best || {};
    const series = (it.benchmark.results || []).slice()
      .sort((a, b) => (a.median_ms || 0) - (b.median_ms || 0)).slice(0, 16);
    const max = Math.max(...series.map((r) => r.median_ms || 0), 1);
    const bestId = best.candidate_id;
    const rows = series.map((r) => {
      const cid = r.candidate_id || 'baseline';
      const isBest = bestId && cid === bestId;
      const sp = typeof r.speedup_vs_baseline === 'number' ? fmtSpeedup(r.speedup_vs_baseline) : '—';
      const bw = typeof r.bandwidth_gbps === 'number' ? r.bandwidth_gbps.toFixed(0) : '—';
      return `
        <tr class="${isBest ? 'row-best' : ''}">
          <td class="cand" title="${escapeAttr(cid)}">${escapeHtml(cid.length > 40 ? '…' + cid.slice(-38) : cid)}</td>
          <td class="bar"><div class="bar-track"><div class="bar-fill" style="width:${((r.median_ms || 0) / max) * 100}%"></div></div></td>
          <td class="num">${fmtMs(r.median_ms)}</td>
          <td class="num">${escapeHtml(sp)}</td>
          <td class="num">${escapeHtml(bw)}</td>
        </tr>`;
    }).join('');
    return [
      `<div class="candidate-stats">
        <div class="stat">median <strong>${fmtMs(best.median_ms)}</strong></div>
        ${typeof best.speedup_vs_baseline === 'number' ? `<div class="stat">speedup <strong>${fmtSpeedup(best.speedup_vs_baseline)}</strong></div>` : ''}
        ${typeof best.bandwidth_gbps === 'number' ? `<div class="stat">bw <strong>${fmtBandwidth(best.bandwidth_gbps)}</strong></div>` : ''}
       </div>`,
      `<table class="bench-table">
        <thead>
          <tr>
            <th>candidate</th>
            <th class="bar"></th>
            <th class="num">median</th>
            <th class="num">speedup</th>
            <th class="num">GB/s</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
       </table>`,
    ].join('');
  }
  if (it.kind === 'comparison' && it.comparison) {
    const c = it.comparison;
    return `<div class="candidate-stats">
      <div class="stat">speedup <strong>${fmtSpeedup(c.speedup_vs_baseline)}</strong></div>
      ${typeof c.delta_percent === 'number' ? `<div class="stat">Δ <strong>${c.delta_percent.toFixed(2)}%</strong></div>` : ''}
      ${c.candidate_id ? `<div class="stat">cand <strong>${escapeHtml(c.candidate_id)}</strong></div>` : ''}
    </div>`;
  }
  if (it.kind === 'decision' && it.decision) {
    const d = it.decision;
    const stats = [
      d.tensor_shape ? `<div class="stat">shape <strong>${escapeHtml(JSON.stringify(d.tensor_shape))}</strong></div>` : '',
      typeof d.num_warps === 'number' ? `<div class="stat">warps <strong>${d.num_warps}</strong></div>` : '',
      typeof d.threads_per_warp === 'number' ? `<div class="stat">threads/warp <strong>${d.threads_per_warp}</strong></div>` : '',
      d.chosen_order ? `<div class="stat">order <strong>${escapeHtml(JSON.stringify(d.chosen_order))}</strong></div>` : '',
      d.size_per_thread ? `<div class="stat">per-thread <strong>${escapeHtml(JSON.stringify(d.size_per_thread))}</strong></div>` : '',
      d.mma_version ? `<div class="stat">mma <strong>${escapeHtml(String(d.mma_version))}</strong></div>` : '',
      d.op_location ? `<div class="stat">@<strong>${escapeHtml(String(d.op_location))}</strong></div>` : '',
    ].filter(Boolean).join('');
    const meta = d.metadata && Object.keys(d.metadata).length ? renderJson(d.metadata) : '';
    return `
      ${stats ? `<div class="candidate-stats">${stats}</div>` : ''}
      ${d.evidence ? `<pre>${escapeHtml(String(d.evidence).slice(0, 600))}</pre>` : ''}
      ${meta}`;
  }
  if (it.kind === 'compiler' && it.passes) {
    const max = Math.max(...it.passes.map((p) => p.duration_ms || 0), 0.01);
    return it.passes.map((p) => {
      const pct = ((p.duration_ms || 0) / max) * 100;
      const cls = p.action === 'skip' ? 'skipped' : '';
      return `<div class="pass-row ${cls}">
        <span title="${escapeAttr(p.error || '')}">${escapeHtml(p.stage || '')} · ${escapeHtml(p.name || p.pass || '')}${p.error ? ' ⚠' : ''}</span>
        <div class="pass-bar"><div class="pass-bar-fill" style="width:${pct}%"></div></div>
        <span>${(p.duration_ms || 0).toFixed(2)}ms</span>
      </div>`;
    }).join('');
  }
  if (it.kind === 'source') {
    const src = state.source;
    if (!src) return `<div class="msg-content muted">Select an example to see its kernel.</div>`;
    if (src.source_kind === 'missing') {
      return `<div class="msg-content"><span class="pill pill-status rejected">missing kernel_symbol</span><br>${escapeHtml(src.warning || '')}</div>`;
    }
    return `
      <div class="msg-content muted">${escapeHtml(src.source_path || '')}${src.line_start ? ` lines ${src.line_start}-${src.line_end}` : ''}</div>
      <pre><code class="language-${escapeAttr(src.language || 'python')}">${renderCode(src.source || '', src.language || 'python')}</code></pre>`;
  }
  if (it.kind === 'ir') {
    const opts = state.irRuns.map((r) => `<option value="${escapeAttr(r.run_id)}">${escapeHtml(r.kernel_id || 'kernel')} · ${escapeHtml(r.run_id)}</option>`).join('');
    return `
      <div class="msg-content muted" id="ir-meta-${escapeAttr(it.key)}">${escapeHtml(it.irMeta || 'select a compile run + stage')}</div>
      <div class="ir-controls">
        <select data-role="ir-run">${opts || '<option value="">no compile runs yet</option>'}</select>
        <select data-role="ir-stage">
          <optgroup label="Triton">
            <option value="ttir">ttir</option>
            <option value="ttgir" selected>ttgir</option>
            <option value="llir">llir</option>
            <option value="ptx">ptx</option>
          </optgroup>
          <optgroup label="Inductor">
            <option value="output_code">output_code (.py)</option>
            <option value="fx_graph">fx_graph (.py)</option>
            <option value="schedule_log">schedule (.log)</option>
          </optgroup>
        </select>
        <button class="btn btn-ghost" type="button" data-action="load-ir">load</button>
      </div>
      <pre><code class="language-mlir">${escapeHtml(it.irText || '// IR will load on demand.')}</code></pre>`;
  }
  if (it.kind === 'insight') {
    const i = it.insight || {};
    const tile = (label, value, sub = '') =>
      `<div class="insight-tile"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong>${sub ? `<div class="insight-sub">${escapeHtml(sub)}</div>` : ''}</div>`;
    const tiles = [
      tile('best speedup', fmtSpeedup(i.bestSpeedup), i.bestCandidateId || ''),
      tile('best median', fmtMs(i.bestMedian)),
      tile('candidates', String(i.totalCandidates ?? 0), `${i.acceptedCandidates ?? 0} accepted`),
      tile('passes run', String(i.totalPasses ?? 0), `${i.passMs ? i.passMs.toFixed(1) + ' ms compile' : ''}`),
    ].join('');
    const speedupPlot = renderSpeedupPlot(i.candidatesByTime || []);
    const passPlot = renderPassPlot(i.passDurations || []);
    return `
      <div class="msg-content">${escapeHtml(i.summary || '')}</div>
      <div class="insight-grid">${tiles}</div>
      <div class="insight-grid" style="margin-top:8px;">
        <div class="insight-plot"><h4>Speedup over candidates</h4>${speedupPlot}</div>
        <div class="insight-plot"><h4>Per-pass duration (top 18)</h4>${passPlot}</div>
      </div>`;
  }
  if (it.kind === 'artifacts') {
    const groups = (it.artifactManifest && it.artifactManifest.groups) || {};
    const keys = Object.keys(groups);
    if (!keys.length) {
      return `<div class="msg-content muted">No compiled artifacts emitted yet.</div>`;
    }
    // Sort so "baseline" comes first, then candidates by id.
    keys.sort((a, b) => (a === 'baseline' ? -1 : b === 'baseline' ? 1 : a.localeCompare(b)));
    const sections = keys.map((runId) => {
      const items = groups[runId] || [];
      const rows = items.map((a) => {
        const name = (a.path || '').split('/').pop() || a.path;
        const previewHref = artifactPreviewUrl(a.path);
        const downloadHref = artifactDownloadUrl(a.path);
        return `<div class="artifact-row">
          <span class="artifact-stage">${escapeHtml(a.stage || '')}</span>
          <a class="artifact-name" href="${previewHref}" target="_blank" rel="noopener" title="${escapeAttr(a.path)}">${escapeHtml(name)}</a>
          <a class="artifact-dl" href="${downloadHref}" download title="download">⬇</a>
        </div>`;
      }).join('');
      return `<div class="artifact-group">
        <h4>${escapeHtml(runId)} <small class="muted">(${items.length})</small></h4>
        ${rows}
      </div>`;
    }).join('');
    return `<div class="artifact-list">${sections}</div>`;
  }
  if (it.kind === 'final') {
    const txt = it.finalText || '';
    if (!txt.trim()) {
      return `<div class="msg-content muted">No final report emitted.</div>`;
    }
    return `<div class="msg-content">${renderMarkdown(txt)}</div>`;
  }
  if (it.contentHtml) return it.contentHtml;
  return '';
}

function maybeAutoscroll() {
  if (state.followMode) startSmoothFollow();
  else showMoreUpdatesPill();
}

function isAtBottom() {
  const stream = $('#stream');
  return stream.scrollHeight - stream.scrollTop - stream.clientHeight < 24;
}

function showMoreUpdatesPill() {
  const pill = $('#more-updates');
  if (pill) pill.classList.remove('hidden');
}
function hideMoreUpdatesPill() {
  const pill = $('#more-updates');
  if (pill) pill.classList.add('hidden');
}

// Smoothly slide the stream's scroll position toward the bottom. Each frame
// moves a fraction of the remaining distance (exponential ease-out), so when
// new deltas arrive the target gets pushed further and the animation glides
// without re-starting.
function startSmoothFollow() {
  const stream = $('#stream');
  if (state.smoothScrollRaf) return;
  const tick = () => {
    if (!state.followMode) {
      state.smoothScrollRaf = 0;
      return;
    }
    const target = stream.scrollHeight - stream.clientHeight;
    const current = stream.scrollTop;
    const distance = target - current;
    if (Math.abs(distance) < 0.5) {
      stream.scrollTop = target;
      state.smoothScrollRaf = 0;
      return;
    }
    const step = Math.sign(distance) * Math.max(1, Math.abs(distance) * 0.22);
    stream.scrollTop = current + step;
    state.smoothScrollRaf = requestAnimationFrame(tick);
  };
  state.smoothScrollRaf = requestAnimationFrame(tick);
}

function disengageFollow() {
  state.followMode = false;
  if (state.smoothScrollRaf) {
    cancelAnimationFrame(state.smoothScrollRaf);
    state.smoothScrollRaf = 0;
  }
  showMoreUpdatesPill();
}

function bindScrollFollow() {
  const stream = $('#stream');
  // User input intent: any of these means the user wants to take over scroll.
  // We disengage follow-mode immediately rather than trying to disambiguate
  // programmatic vs. user scrolls inside the scroll handler — that race is
  // unwinnable while the smooth-follow animation is writing scrollTop every frame.
  stream.addEventListener('wheel', disengageFollow, { passive: true });
  stream.addEventListener('touchstart', disengageFollow, { passive: true });
  stream.addEventListener('keydown', (e) => {
    const keys = ['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' '];
    if (keys.includes(e.key)) disengageFollow();
  });
  // Scroll handler only re-engages follow mode when the user lands back at the bottom.
  stream.addEventListener('scroll', () => {
    if (isAtBottom() && !state.followMode) {
      state.followMode = true;
      hideMoreUpdatesPill();
      startSmoothFollow();
    }
  });
  $('#more-updates')?.addEventListener('click', () => {
    state.followMode = true;
    hideMoreUpdatesPill();
    startSmoothFollow();
  });
}

function applyFilter() { for (const it of state.items) applyFilterToEl(it, it.el); }
function applyFilterToEl(it, el) {
  if (!el) return;
  if (state.filter === 'all') { el.style.display = ''; return; }
  el.style.display = (it.filter === state.filter) ? '' : 'none';
}

// ----- source / IR / insights cards -----

function insertSourceCard() {
  // Always force the source card to be the first item.
  const key = 'source-card';
  if (state.itemByKey.has(key)) {
    const existing = state.itemByKey.get(key);
    rerenderItem(existing);
    return;
  }
  const it = {
    key, kind: 'source', filter: 'all',
    ts: new Date().toISOString(),
    title: state.source?.symbol || 'kernel source',
    pillLabel: 'kernel', pillClass: 'pill-code',
  };
  state.collapsed.add(key);
  state.items.unshift(it);
  state.itemByKey.set(key, it);
  const el = renderItem(it);
  it.el = el;
  $('#stream').prepend(el);
  applyFilterToEl(it, el);
}

function appendIrCard() {
  const key = 'ir-card';
  if (state.itemByKey.has(key)) {
    rerenderItem(state.itemByKey.get(key));
    return;
  }
  pushItem({
    key, kind: 'ir', filter: 'compiler',
    ts: new Date().toISOString(),
    title: 'IR browser', pillLabel: 'IR', pillClass: 'pill-ir',
  });
}

function appendArtifactsCard() {
  // Group every artifact emitted during the run by its owning run_id (e.g.
  // "baseline" or "cand-..."). The card lists each compiled artifact with a
  // download link so the user can grab the produced output_code, ttgir, etc.
  const groups = {};
  for (const a of state.artifactManifest) {
    const key = a.run_id || 'unknown';
    (groups[key] ||= []).push(a);
  }
  const payload = { groups };
  const cardKey = 'artifacts-card';
  if (state.itemByKey.has(cardKey)) {
    updateItem(cardKey, (it) => { it.artifactManifest = payload; });
    return;
  }
  pushItem({
    key: cardKey, kind: 'artifacts', filter: 'all',
    ts: new Date().toISOString(),
    title: 'Compiled artifacts',
    pillLabel: 'artifacts', pillClass: 'pill-ir',
    artifactManifest: payload,
  });
}

function appendFinalSummaryCard(text) {
  state.finalSummary = text;
  const key = 'final-summary-card';
  if (state.itemByKey.has(key)) {
    updateItem(key, (it) => { it.finalText = text; });
    return;
  }
  pushItem({
    key, kind: 'final', filter: 'all',
    ts: new Date().toISOString(),
    title: 'Final report', pillLabel: 'report', pillClass: 'pill-summary',
    finalText: text,
  });
}

function appendInsightsCard() {
  const insight = computeInsights();
  const key = 'insights-card';
  if (state.itemByKey.has(key)) {
    updateItem(key, (it) => { it.insight = insight; });
    return;
  }
  pushItem({
    key, kind: 'insight', filter: 'all',
    ts: new Date().toISOString(),
    title: 'Performance insights',
    pillLabel: 'insights', pillClass: 'pill-insight',
    insight,
  });
}

function computeInsights() {
  const cands = Array.from(state.candidates.values())
    .map((k) => state.itemByKey.get(k))
    .filter((it) => it && it.candidate);
  let bestSpeedup = null, bestMedian = null, bestCandidateId = null;
  let accepted = 0;
  const candidatesByTime = [];
  for (const it of cands) {
    const c = it.candidate;
    if (c.status === 'accepted') accepted++;
    if (typeof c.speedup_vs_baseline === 'number') {
      candidatesByTime.push({ id: c.id, speedup: c.speedup_vs_baseline });
      if (bestSpeedup === null || c.speedup_vs_baseline > bestSpeedup) {
        bestSpeedup = c.speedup_vs_baseline;
        bestMedian = c.median_ms;
        bestCandidateId = c.id;
      }
    }
  }
  // Aggregate pass durations across all compiler timelines.
  const passDurations = [];
  let passMs = 0, totalPasses = 0;
  for (const k of state.passes.values()) {
    const it = state.itemByKey.get(k);
    if (!it || !it.passes) continue;
    for (const p of it.passes) {
      totalPasses++;
      passMs += p.duration_ms || 0;
      passDurations.push({ name: p.name || p.pass, ms: p.duration_ms || 0, action: p.action });
    }
  }
  passDurations.sort((a, b) => b.ms - a.ms);
  return {
    bestSpeedup, bestMedian, bestCandidateId,
    totalCandidates: cands.length, acceptedCandidates: accepted,
    totalPasses, passMs,
    candidatesByTime, passDurations: passDurations.slice(0, 18),
    summary: bestSpeedup
      ? `Best candidate ${bestCandidateId || ''} at ${fmtSpeedup(bestSpeedup)} (median ${fmtMs(bestMedian)}). ${totalPasses} compiler passes ran in ${passMs.toFixed(1)} ms.`
      : `Run finished with ${cands.length} candidates and ${totalPasses} compiler passes.`,
  };
}

function renderSpeedupPlot(points) {
  const w = 360, h = 140, padL = 36, padR = 14, padT = 14, padB = 22;
  if (!points.length) return `<svg viewBox="0 0 ${w} ${h}" width="100%"><text x="${w/2}" y="${h/2}" text-anchor="middle" fill="#5b6478" font-size="10">no measured candidates</text></svg>`;
  // Min-max scale that ALWAYS includes 1.0 (the baseline reference) so the
  // viewer can see at a glance whether each candidate is above or below it.
  const speedups = points.map((p) => p.speedup);
  const dataMax = Math.max(...speedups, 1.0);
  const dataMin = Math.min(...speedups, 1.0);
  const span = Math.max(dataMax - dataMin, 1e-3);
  const headroom = span * 0.12;
  const maxY = dataMax + headroom;
  const minY = dataMin - headroom;
  const xs = (i) => padL + (points.length === 1 ? (w - padL - padR) / 2 : (i * (w - padL - padR)) / (points.length - 1));
  const ys = (v) => h - padB - ((v - minY) / (maxY - minY)) * (h - padT - padB);
  // Color each dot by win/loss vs. baseline. Bright cyan = win, soft red = regression.
  const dotColor = (sp) => sp >= 1.0 ? '#22d3ee' : '#f87171';
  // Connecting line uses a neutral, less-bright color so the data points pop.
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xs(i).toFixed(1)} ${ys(p.speedup).toFixed(1)}`).join(' ');
  const refY = ys(1.0).toFixed(1);
  // Y-axis ticks at min, 1.0, max.
  const fmt = (v) => v.toFixed(2) + '×';
  const yMinTxt = fmt(minY + headroom);
  const yMaxTxt = fmt(maxY - headroom);
  const dots = points.map((p, i) => {
    const cx = xs(i).toFixed(1);
    const cy = ys(p.speedup).toFixed(1);
    const pct = (p.speedup - 1.0) * 100;
    const sign = pct >= 0 ? '+' : '';
    const label = `${sign}${pct.toFixed(1)}%`;
    const color = dotColor(p.speedup);
    // Place the label above the point if it's below the line; below if it's above.
    const below = p.speedup < 1.0;
    const labelY = below ? (parseFloat(cy) + 12) : (parseFloat(cy) - 7);
    return `
      <circle cx="${cx}" cy="${cy}" r="3.6" fill="${color}" stroke="#0b0d12" stroke-width="0.6"/>
      <text x="${cx}" y="${labelY.toFixed(1)}" text-anchor="middle"
            fill="${color}" font-size="9" font-weight="600"
            font-family="ui-monospace, monospace">${label}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">
    <line x1="${padL}" x2="${w - padR}" y1="${refY}" y2="${refY}" stroke="#5b6478" stroke-dasharray="2,3" stroke-width="0.8"/>
    <text x="${padL - 4}" y="${(parseFloat(refY) + 3).toFixed(1)}" text-anchor="end" fill="#8c97ac" font-size="9">1.00×</text>
    <text x="${padL - 4}" y="${(padT + 4).toFixed(1)}" text-anchor="end" fill="#5b6478" font-size="8">${yMaxTxt}</text>
    <text x="${padL - 4}" y="${(h - padB + 3).toFixed(1)}" text-anchor="end" fill="#5b6478" font-size="8">${yMinTxt}</text>
    <path d="${path}" fill="none" stroke="#3b82f6" stroke-width="1.2" opacity="0.55"/>
    ${dots}
  </svg>`;
}

function renderPassPlot(passes) {
  const w = 320, h = Math.max(60, 12 * passes.length);
  if (!passes.length) return `<svg viewBox="0 0 ${w} ${h}" width="100%"><text x="${w/2}" y="${h/2}" text-anchor="middle" fill="#5b6478" font-size="10">no passes</text></svg>`;
  const max = Math.max(...passes.map((p) => p.ms), 0.01);
  const rowH = 11;
  const rows = passes.map((p, i) => {
    const y = i * rowH + 2;
    const barW = (p.ms / max) * (w - 130);
    const fill = p.action === 'skip' ? '#5b6478' : '#a78bfa';
    const label = p.name.length > 30 ? p.name.slice(0, 28) + '…' : p.name;
    return `
      <g>
        <text x="0" y="${y + 8}" fill="#c5cee0" font-size="9" font-family="monospace">${escapeHtml(label)}</text>
        <rect x="125" y="${y + 2}" width="${barW.toFixed(1)}" height="6" fill="${fill}" rx="2"/>
        <text x="${(125 + barW + 4).toFixed(1)}" y="${y + 8}" fill="#8c97ac" font-size="9" font-family="monospace">${p.ms.toFixed(2)}ms</text>
      </g>`;
  }).join('');
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">${rows}</svg>`;
}

async function loadIrInline(it) {
  const card = it.el;
  const runId = card.querySelector('[data-role="ir-run"]')?.value;
  const stage = card.querySelector('[data-role="ir-stage"]')?.value;
  if (!runId) { it.irMeta = 'select a compile run first'; rerenderItem(it); return; }
  // Map UI stage names to filename predicates for both Triton and Inductor.
  const matchers = {
    ttir: (p) => p.toLowerCase().endsWith('.ttir'),
    ttgir: (p) => p.toLowerCase().endsWith('.ttgir'),
    llir: (p) => p.toLowerCase().endsWith('.llir'),
    ptx: (p) => p.toLowerCase().endsWith('.ptx'),
    output_code: (p) => /output_code\.py$/i.test(p),
    fx_graph: (p) => /fx_graph\.py$/i.test(p),
    schedule_log: (p) => /schedule\.log$/i.test(p),
  };
  const matcher = matchers[stage] || ((p) => p.toLowerCase().endsWith('.' + stage.toLowerCase()));
  const artifact = state.artifacts.find((p) => p.includes(runId) && matcher(p))
    || state.artifacts.find((p) => matcher(p));
  if (!artifact) { it.irMeta = `no ${stage} artifact for ${runId}`; it.irText = ''; rerenderItem(it); return; }
  try {
    const data = await fetchJson(`${artifactPreviewUrl(artifact)}?max_chars=80000`);
    it.irMeta = `${stage} · ${data.size_bytes} bytes · ${artifact}`;
    it.irText = data.text || '(empty)';
    rerenderItem(it);
  } catch (e) { it.irMeta = e.message; rerenderItem(it); }
}

function artifactPreviewUrl(path) {
  const segs = String(path || '').split('/').filter(Boolean).map(encodeURIComponent);
  return `/api/artifacts/preview/${segs.join('/')}`;
}

function artifactDownloadUrl(path) {
  const segs = String(path || '').split('/').filter(Boolean).map(encodeURIComponent);
  return `/api/artifacts/${segs.join('/')}`;
}

// ----- chrome -----

function updateMetrics() {
  let bestSpeedup = null, bestMedian = null;
  for (const k of state.candidates.values()) {
    const it = state.itemByKey.get(k);
    const c = it && it.candidate;
    if (c && typeof c.speedup_vs_baseline === 'number' &&
        (bestSpeedup === null || c.speedup_vs_baseline > bestSpeedup)) {
      bestSpeedup = c.speedup_vs_baseline;
      bestMedian = c.median_ms;
    }
  }
  $('#metric-speedup').textContent = fmtSpeedup(bestSpeedup);
  $('#metric-median').textContent = fmtMs(bestMedian);
  $('#metric-candidates').textContent = state.candidates.size;
  let passCount = 0;
  for (const k of state.passes.values()) {
    const it = state.itemByKey.get(k);
    if (it && it.passes) passCount += it.passes.length;
  }
  $('#metric-passes').textContent = passCount;
  $('#event-count').textContent = `${state.items.length} cards`;
  const runId = state.activeRunId;
  const run = runId ? state.runs[runId] : null;
  $('#run-id').textContent = runId || '';
  // Progress = successful trials / max_candidates. Terminal states clamp to 100%
  // so the bar visibly completes regardless of whether all slots were spent.
  let pct = 0;
  if (run) {
    const max = Math.max(1, run.max_candidates || 0);
    const succ = Math.min(max, run.successful_count || 0);
    pct = Math.round((succ / max) * 100);
    if (run.status === 'completed' || run.status === 'failed') pct = 100;
    const succLabel = `${succ}/${max} successful`;
    const failPart = (run.failed_attempts || 0) > 0 ? ` · ${run.failed_attempts} failed` : '';
    $('#run-status').textContent =
      run.status === 'failed' ? `Failed (${succLabel}${failPart}).`
      : run.status === 'completed' ? `Completed (${succLabel}${failPart}).`
      : `Running… ${succLabel}${failPart}`;
  }
  $('#run-progress-fill').style.width = `${pct}%`;
  $('#run-progress-pct').textContent = `${pct}%`;
}

function renderRuntimeChrome() {
  const c = state.runtimeConfig || {};
  $('#harness-select').value = c.harness || 'pydantic_ai';
  // Reflect the active model from the runtime config; fall back to whatever the
  // dropdown is showing if the config doesn't echo it back yet.
  if (c.model) {
    const sel = $('#model-select');
    if ([...sel.options].some((o) => o.value === c.model)) {
      sel.value = c.model;
    }
  }
  if (typeof c.max_candidates === 'number' && c.max_candidates >= 1) {
    $('#max-candidates-input').value = c.max_candidates;
  }
  $('#model-pill').textContent = c.model || 'model n/a';
  $('#effort-pill').textContent = c.reasoning_effort ? `effort: ${c.reasoning_effort}` : 'effort n/a';
}

function renderTelemetry() {
  const root = $('#topbar-gpus');
  if (!state.telemetry.length) { root.innerHTML = ''; return; }
  root.innerHTML = state.telemetry.map((g) => {
    const util = g.utilization_gpu_pct ?? 0;
    const used = g.memory_used_mib ?? 0;
    const total = g.memory_total_mib || 1;
    const memPct = (used / total) * 100;
    const temp = g.temperature_c ?? '—';
    const power = g.power_w ?? '—';
    const usedGib = (used / 1024).toFixed(1);
    const totalGib = (total / 1024).toFixed(1);
    return `
      <div class="gpu-card">
        <div class="gpu-name">${escapeHtml(g.name || 'GPU')}</div>
        <div class="gpu-meta">#${escapeHtml(String(g.index ?? 0))} · ${escapeHtml(String(temp))}°C · ${escapeHtml(String(power))} W</div>
        <div class="gpu-bars">
          <div class="gpu-bar"><span class="bar-label">util</span>
            <div class="bar-track"><div class="bar-fill" style="width:${util}%"></div></div>
            <span class="bar-val">${util}%</span></div>
          <div class="gpu-bar mem"><span class="bar-label">mem</span>
            <div class="bar-track"><div class="bar-fill" style="width:${memPct.toFixed(1)}%"></div></div>
            <span class="bar-val">${usedGib}/${totalGib} GiB</span></div>
        </div>
      </div>`;
  }).join('');
}

// ----- run start -----

async function startRun() {
  const workloadId = state.selectedWorkloadId;
  // Wipe the conversation client-side immediately so old cards disappear before the
  // first event of the new run lands.
  clearStream();
  const button = $('#run-button');
  button.disabled = true;
  $('#run-status').textContent = 'Submitting run…';
  try {
    if (workloadId) {
      const max_candidates = Math.max(1, Math.min(32, Number($('#max-candidates-input').value) || 4));
      const harness = $('#harness-select').value || (state.runtimeConfig && state.runtimeConfig.harness) || 'pydantic_ai';
      const r = await fetchJson('/api/runs/workload', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ workload_id: workloadId, mode: 'optimize', max_candidates, harness }),
      });
      state.activeRunId = r.run_id;
      $('#run-id').textContent = r.run_id;
      $('#run-status').textContent = `Queued ${r.run_id}`;
    } else {
      $('#run-status').textContent = 'No workload selected.';
    }
  } catch (e) {
    $('#run-status').textContent = e.message;
  } finally {
    button.disabled = false;
  }
}

init();
