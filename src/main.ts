import { webgpuEngine } from './webgpu.js';
import { initNaso } from './naso.js';
import { quantizationPipeline } from './pipeline.js';
import { loadModelWeights, type Tensor } from './model.js';
import { loadTokenizer, type SimpleTokenizer } from './tokenizer.js';
import { logger } from './logger.js';
import { MODELS, type ModelConfig } from './types.js';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

function setDot(id: string, state: 'idle' | 'ok' | 'err' | 'busy', text?: string) {
  const dot = $(id);
  dot.className = `dot ${state === 'ok' ? 'ok' : state === 'err' ? 'err' : state === 'busy' ? 'busy' : ''}`;
  if (text) {
    const label = dot.nextElementSibling as HTMLElement | null;
    if (label) label.textContent = text;
  }
}

const logEl = $('log');
function appendLog(entry: { level: string; source: string; message: string }) {
  const div = document.createElement('div');
  div.className = `lv-${entry.level}`;
  const t = new Date().toISOString().slice(11, 23);
  div.textContent = `${t} [${entry.source}] ${entry.message}`;
  logEl.appendChild(div);
  while (logEl.childElementCount > 300) logEl.removeChild(logEl.firstChild!);
  logEl.scrollTop = logEl.scrollHeight;
}
for (const source of ['main', 'webgpu', 'naso'] as const) {
  logger.subscribe(source, e => appendLog(e));
}

function renderKernels() {
  const container = $('kernels');
  container.textContent = '';
  if (quantizationPipeline.kernels.size === 0) {
    container.innerHTML = '<p class="muted">Compiling…</p>';
    return;
  }
  for (const [name, k] of quantizationPipeline.kernels) {
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = `${name} → ${k.abi.entryPoint}  ·  ${k.abi.bindings.length} storage · ${k.abi.scalars.length} scalar · ${k.abi.dispatchGroups} wg × ${k.abi.workgroupSize}`;
    details.appendChild(summary);

    const table = document.createElement('table');
    table.innerHTML = '<thead><tr><th>binding</th><th>name</th><th>kind</th><th>type</th><th class="num">bytes</th></tr></thead>';
    const tbody = document.createElement('tbody');
    for (const b of k.abi.bindings) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${b.index}</td><td>${b.name}</td><td>storage</td><td>${b.elem} (${b.access})</td><td class="num">${b.bytes}</td>`;
      tbody.appendChild(tr);
    }
    for (const s of k.abi.scalars) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${s.index}</td><td>${s.name}</td><td>uniform</td><td>${s.type}</td><td class="num">16</td>`;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    details.appendChild(table);

    const pre = document.createElement('pre');
    pre.textContent = k.wgsl;
    details.appendChild(pre);
    container.appendChild(details);
  }
}

async function boot() {
  setDot('dot-webgpu', 'busy', 'WebGPU: initializing…');
  const gpuOk = await webgpuEngine.init();
  setDot('dot-webgpu', gpuOk ? 'ok' : 'err', gpuOk ? 'WebGPU: ready' : 'WebGPU: unavailable');

  setDot('dot-naso', 'busy', 'Naso: loading…');
  try {
    await initNaso();
    setDot('dot-naso', 'ok', 'Naso: ready');
  } catch (e) {
    setDot('dot-naso', 'err', 'Naso: failed');
    logger.error('main', `Naso init failed: ${e}`);
    $('hint').textContent = 'The Naso WASM compiler failed to load. Check that public/pkg/nasoc_wasm_bg.wasm is served.';
    return;
  }

  if (!gpuOk) {
    $('hint').textContent = 'WebGPU is required to run the kernels. The shaders below are still generated in-browser by the Naso WASM compiler, so the demo works up to the dispatch.';
    try {
      await quantizationPipeline.compile();
      renderKernels();
    } catch (e) {
      logger.error('main', `Kernel compilation failed: ${e}`);
    }
    return;
  }

  try {
    await quantizationPipeline.prepare();
    renderKernels();
    $<HTMLButtonElement>('run').disabled = false;
    $('hint').textContent = `Ready. Compiled ${quantizationPipeline.kernels.size} kernels from Naso source in-browser.`;
  } catch (e) {
    logger.error('main', `Kernel preparation failed: ${e}`);
    $('hint').textContent = `Kernel compilation failed: ${e}`;
  }
}

const model: ModelConfig = MODELS[0];
let tokenizer: SimpleTokenizer | null = null;
let tensors: Map<string, Tensor> | null = null;

async function ensureModel() {
  if (tensors) return;
  setDot('dot-model', 'busy', 'Model: downloading…');
  const progress = $('progress').firstElementChild as HTMLElement;
  tensors = await loadModelWeights(model, (loaded, total) => {
    if (total > 0) progress.style.width = `${Math.min(100, (loaded / total) * 100).toFixed(0)}%`;
    $('model-state').textContent = `Model: ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
  });
  progress.style.width = '100%';
  tokenizer = await loadTokenizer(model.repo);
  setDot('dot-model', 'ok', `Model: ${model.name}`);
  logger.success('main', `Model ready: ${tensors.size} tensors`);
}

async function run() {
  const input = $<HTMLInputElement>('prompt');
  $<HTMLButtonElement>('run').disabled = true;
  try {
    await ensureModel();
    const ids = tokenizer!.encodeSimple(input.value);
    logger.info('main', `tokenized ${input.value.length} chars -> ${ids.length} ids`);

    const result = await quantizationPipeline.run(model, tensors!, ids);

    const reportsEl = $('reports');
    reportsEl.textContent = '';
    const table = document.createElement('table');
    table.innerHTML = '<thead><tr><th>tensor</th><th class="num">elements</th><th class="num">scale</th><th class="num">max |error|</th><th class="num">bound (s/2)</th><th>holds</th></tr></thead>';
    const tbody = document.createElement('tbody');
    let allHold = true;
    for (const r of result.reports) {
      if (!r.boundHolds) allHold = false;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${r.tensorName}</td><td class="num">${r.elements}</td>` +
        `<td class="num">${r.scale.toExponential(3)}</td><td class="num">${r.maxAbsError.toExponential(3)}</td>` +
        `<td class="num">${r.promisedBound.toExponential(3)}</td>` +
        `<td class="${r.boundHolds ? 'ok' : 'bad'}">${r.boundHolds ? 'yes' : 'NO'}</td>`;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    reportsEl.appendChild(table);
    const summary = document.createElement('p');
    summary.className = allHold ? 'ok' : 'bad';
    summary.style.marginTop = '.6rem';
    summary.textContent = allHold
      ? `All ${result.reports.length} tensors satisfy the half-step bound.`
      : `Bound violated on ${result.reports.filter(r => !r.boundHolds).length} tensors.`;
    reportsEl.appendChild(summary);
    $('results-card').style.display = '';

    const vocab = model.vocabSize;
    const seq = Math.floor(result.logits.length / vocab);
    const last = result.logits.subarray((seq - 1) * vocab);
    const idx = Array.from({ length: vocab }, (_, i) => i);
    idx.sort((a, b) => last[b] - last[a]);
    const top = idx.slice(0, 10);
    const maxL = last[top[0]];
    const exps = top.map(i => Math.exp(last[i] - maxL));
    const sum = exps.reduce((a, b) => a + b, 0);

    const out = $('output');
    out.textContent = '';
    const best = document.createElement('div');
    best.className = 'out';
    best.textContent = JSON.stringify(tokenizer!.decode([top[0]]));
    out.appendChild(best);

    const dl = document.createElement('table');
    dl.innerHTML = '<thead><tr><th>rank</th><th>id</th><th>token</th><th class="num">logit</th><th class="num">p</th></tr></thead>';
    const dlb = document.createElement('tbody');
    top.forEach((id, rank) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${rank + 1}</td><td>${id}</td><td>${JSON.stringify(tokenizer!.decode([id]))}</td>` +
        `<td class="num">${last[id].toFixed(4)}</td><td class="num">${(exps[rank] / sum).toFixed(4)}</td>`;
      dlb.appendChild(tr);
    });
    dl.appendChild(dlb);
    out.appendChild(dl);
    $('output-card').style.display = '';

    logger.success('main', `Forward pass complete. Top token id ${top[0]} = ${JSON.stringify(tokenizer!.decode([top[0]]))}`);

    // Independently check the logits against the golden vector produced by
    // tools/reference_llama.py (a separate NumPy implementation). This is what
    // turns "it produced numbers" into "it produced the RIGHT numbers", in the
    // same page, with no extra tooling.
    try {
      const goldenRes = await fetch('golden.json');
      if (!goldenRes.ok) throw new Error(`HTTP ${goldenRes.status}`);
      const golden = await goldenRes.json() as { inputIds: number[]; top: { id: number; logit: number }[] };

      const matchIds = golden.inputIds.length === ids.length && golden.inputIds.every((v, i) => v === ids[i]);
      if (!matchIds) {
        logger.warn('main', `golden check skipped: golden was computed for inputIds [${golden.inputIds.join(', ')}], prompt tokenized to [${ids.join(', ')}]. Type "The capital of France is" or run with no edits to reproduce.`);
      } else {
        let worst = 0;
        for (const g of golden.top) worst = Math.max(worst, Math.abs(last[g.id] - g.logit));
        const topMatches = golden.top[0].id === top[0];
        logger.info('main', `Golden check: max |logit - reference| = ${worst.toExponential(3)} over top-${golden.top.length}; top-1 ${topMatches ? 'matches' : 'MISMATCH'}`);
        const card = $('output');
        const p = document.createElement('p');
        p.className = worst < 1e-3 && topMatches ? 'ok' : 'bad';
        p.style.marginTop = '.6rem';
        p.textContent = worst < 1e-3 && topMatches
          ? `Verified against the independent NumPy reference: max |Δlogit| = ${worst.toExponential(2)}, top-1 token matches.`
          : `Golden mismatch: max |Δlogit| = ${worst.toExponential(2)}${topMatches ? '' : ', top-1 differs'}.`;
        card.appendChild(p);
      }
    } catch (e) {
      logger.debug('main', `golden check unavailable: ${e}`);
    }
  } catch (e) {
    logger.error('main', `Run failed: ${e}`);
    $('run-hint').textContent = `Failed: ${e}`;
  } finally {
    $<HTMLButtonElement>('run').disabled = false;
  }
}

$('run').addEventListener('click', () => { void run(); });

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').then(
      () => logger.info('main', 'Service worker registered (offline cache ready)'),
      e => logger.warn('main', `Service worker registration failed: ${e}`),
    );
  });
}

void boot();
