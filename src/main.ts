/**
 * Browser entry point: chat UI over the KV-cache generation loop, plus the
 * quantisation check and the in-browser Naso kernel compilation.
 *
 * Everything here runs client-side. After the first load the service worker
 * serves the model, tokenizer, kernels and WASM from cache, so the chat works
 * with the network off.
 */
import { MODELS, DEFAULT_MODEL, CHAT_MAX_SEQ, HF_BASE, type ModelConfig } from './types.js';
import { logger } from './logger.js';
import { initNaso, compileKernel } from './naso.js';
import { KERNEL_SPECS, loadKernelSource } from './kernels.js';
import { webgpuEngine } from './webgpu.js';
import { parseSafetensors, parseNPQ, fetchWithProgress, fetchWithRetry, type Tensor } from './model.js';
import { BPETokenizer, type ChatMessage } from './tokenizer.js';
import { createKVCache, prefill, decodeFrom, type KVCache } from './generate.js';
import { quantizeRows, quantError, packedBytes, type QuantizedMatrix } from './quantize.js';

const SYSTEM_PROMPT = 'You are a helpful AI assistant.';

/**
 * The Cache Storage bucket the service worker reads for the checkpoint and
 * tokenizer (cache-first). The page writes into the same bucket so a first
 * visit is cached even if the worker was not yet controlling the page.
 */
const MODEL_CACHE = 'naso-llm-models-v1';

const $ = <T extends HTMLElement = HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

function log(source: string, message: string, level: string) {
  const el = $('log');
  const row = document.createElement('div');
  row.className = `lv-${level}`;
  row.textContent = `${new Date().toISOString().slice(11, 23)} [${source}] ${message}`;
  el.appendChild(row);
  while (el.childElementCount > 300) el.removeChild(el.firstChild!);
  el.scrollTop = el.scrollHeight;
}

logger.subscribe('main', (e) => log('main', e.message, e.level));
logger.subscribe('webgpu', (e) => log('webgpu', e.message, e.level));
logger.subscribe('naso', (e) => log('naso', e.message, e.level));

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
let config: ModelConfig = DEFAULT_MODEL;
let tensors: Record<string, Tensor> = {};
let tokenizer: BPETokenizer | null = null;
let cache: KVCache | null = null;
let history: ChatMessage[] = [];
let generating = false;
let stopRequested = false;
let loaded = false;
const quantized: Record<string, QuantizedMatrix> = {};

function setDot(id: string, state: 'idle' | 'ok' | 'err' | 'busy') {
  $(id).className = `dot${state === 'idle' ? '' : ' ' + state}`;
}

// ---------------------------------------------------------------------------
// model loading
// ---------------------------------------------------------------------------
// `fetchWithProgress`, the safetensors reader and the NPQ1 reader live in
// model.ts, where the decode paths are unit-tested; main.ts only wires them in.

/**
 * Look for the local pre-quantised checkpoint (NPQ1) served from the app's own
 * origin (see the naso-local-model plugin in vite.config.ts).
 *
 * A HEAD request decides, so an absent artifact costs one cheap round-trip and
 * simply falls through to the published safetensors path. The URL is
 * base-relative so it resolves under both '/' and the GitHub Pages subpath, and
 * no-cors/opaque failures are treated as "absent" rather than as load errors.
 */
async function probeLocalArtifact(path: string): Promise<{ url: string; bytes: number } | null> {
  const url = new URL(path.replace(/^\//, ''), document.baseURI).href;
  try {
    const res = await fetch(url, { method: 'HEAD' });
    if (!res.ok) return null;
    const bytes = Number(res.headers.get('Content-Length') ?? 0);
    // A wrongly-served index.html would also be 200; an HTML payload is not a
    // checkpoint. Anything implausibly small is not one either.
    const type = res.headers.get('Content-Type') ?? '';
    if (type.includes('text/html') || bytes < 1e6) return null;
    return { url, bytes };
  } catch {
    return null;
  }
}

async function loadModel() {
  const btn = $<HTMLButtonElement>('load');
  const bar = $('progress').firstElementChild as HTMLElement;
  btn.disabled = true;
  setDot('dot-model', 'busy');
  const t0 = performance.now();
  const phase = (text: string) => { $('model-state').textContent = text; };

  try {
    // config.json drives the architecture and is fetched rather than hardcoded,
    // so the two cannot drift apart.
    const cfgRes = await fetchWithRetry(`${HF_BASE}/${config.repo}/resolve/main/config.json`);
    if (!cfgRes.ok) throw new Error(`config.json -> HTTP ${cfgRes.status}`);
    const c = await cfgRes.json() as Record<string, number>;
    config = {
      ...config,
      hiddenSize: c.hidden_size,
      intermediateSize: c.intermediate_size,
      numLayers: c.num_hidden_layers,
      numHeads: c.num_attention_heads,
      numKvHeads: c.num_key_value_heads,
      vocabSize: c.vocab_size,
      rmsNormEps: c.rms_norm_eps,
      ropeTheta: c.rope_theta ?? 10000.0,
    };
    logger.info('main', `${config.name}: hidden=${config.hiddenSize} layers=${config.numLayers} heads=${config.numHeads}/${config.numKvHeads} vocab=${config.vocabSize}`);

    // Prefer the local pre-quantised checkpoint when it is available. It is 1.65x
    // smaller than the bf16 safetensors and its per-row scales are already
    // applied. It is served from a local mount only (see vite.config.ts), so a
    // HEAD probe decides: 200 -> use it, anything else -> the published f32 path.
    // The quantisation check below still runs either way, so the panel keeps
    // measuring rather than reporting a stored number.
    let buf: ArrayBuffer | null = null;
    const npq = config.quantizedLocal ? await probeLocalArtifact(config.quantizedLocal) : null;
    if (npq) {
      logger.info('main', `local pre-quantised checkpoint available: ${npq.url} (${(npq.bytes / 1e6).toFixed(1)} MB)`);
      phase('downloading int8 checkpoint…');
      buf = await fetchWithProgress(npq.url, (loaded_, total) => {
        bar.style.width = `${total ? ((loaded_ / total) * 100).toFixed(1) : 0}%`;
        phase(`downloading int8 ${(loaded_ / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB`);
      }, MODEL_CACHE);
      logger.info('main', `downloaded int8 checkpoint in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

      phase('expanding int8…');
      tensors = Object.fromEntries(parseNPQ(buf, (done, total) => {
        bar.style.width = `${((done / total) * 100).toFixed(1)}%`;
        if (done % 24 === 0 || done === total) phase(`expanding ${done}/${total} tensors`);
      }));
      logger.success('main', `int8 checkpoint ready: ${Object.keys(tensors).length} tensors`);
    } else {
      const weightsUrl = `${HF_BASE}/${config.repo}/resolve/main/model.safetensors`;
      logger.info('main', `fetching ${weightsUrl}`);
      phase('downloading…');
      buf = await fetchWithProgress(weightsUrl, (loaded_, total) => {
        bar.style.width = `${total ? ((loaded_ / total) * 100).toFixed(1) : 0}%`;
        phase(`downloading ${(loaded_ / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB`);
      }, MODEL_CACHE);
      logger.info('main', `downloaded ${((performance.now() - t0) / 1000).toFixed(1)}s`);

      phase('decoding…');
      const tDecode = performance.now();
      tensors = Object.fromEntries(parseSafetensors(buf, (done, total) => {
        bar.style.width = `${((done / total) * 100).toFixed(1)}%`;
        // Only touch the DOM a few times; a text write per tensor is noise.
        if (done % 24 === 0 || done === total) phase(`decoding ${done}/${total} tensors`);
      }));
      logger.info('main', `decoded ${Object.keys(tensors).length} tensors in ${((performance.now() - tDecode) / 1000).toFixed(1)}s`);
    }
    // The raw checkpoint is dead weight once the tensors are decoded (each byte
    // of bf16 became four, so the decoded form is what matters). Dropping the
    // reference now lets the GC reclaim the download before the quantisation pass.
    buf = null;
    logger.success('main', `parsed ${Object.keys(tensors).length} tensors`);

    const tokRes = await fetchWithRetry(`${HF_BASE}/${config.repo}/resolve/main/tokenizer.json`);
    if (!tokRes.ok) throw new Error(`tokenizer.json -> HTTP ${tokRes.status}`);
    tokenizer = BPETokenizer.fromJSON(await tokRes.json());
    logger.success('main', `tokenizer ready: vocab ${tokenizer.size}, eos ${tokenizer.eosId}`);

    // Tokenizer regression check against ids from the real HF `tokenizers`
    // library, run in the page so a port regression is visible here too.
    const probes: Array<[string, number[]]> = [
      ['hello', [28120]],
      ['Hello, world!', [19556, 28, 905, 17]],
      ['2 + 2 = 4', [34, 1232, 216, 34, 446, 216, 36]],
    ];
    let ok = 0;
    for (const [text, want] of probes) {
      const got = tokenizer.encode(text);
      if (JSON.stringify(got) === JSON.stringify(want)) ok++;
      else logger.warn('main', `tokenizer mismatch on ${JSON.stringify(text)}: got ${got}, want ${want}`);
    }
    logger.info('main', `tokenizer self-check ${ok}/${probes.length} against HF reference ids`);

    cache = createKVCache(config, CHAT_MAX_SEQ);
    history = [];
    renderChat();
    runQuantizationCheck();

    loaded = true;
    setDot('dot-model', 'ok');
    // The forward pass is ready the moment the model is, on whichever backend
    // the label states (here "cpu f32"). Leaving this grey until the first
    // generation reads as "something is broken" when nothing is, which is
    // exactly how this was first reported.
    setDot('dot-gpu', 'ok');
    phase(`${config.name} · ready`);
    $<HTMLTextAreaElement>('input').disabled = false;
    $<HTMLButtonElement>('send').disabled = false;
    $('load-note').textContent = 'Model loaded. Everything runs in this tab — no server, no API key. Reloads are served from the offline cache.';
    logger.success('main', `ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  } catch (e) {
    setDot('dot-model', 'err');
    phase('load failed — press Load model to retry');
    logger.error('main', `model load failed: ${e}`);
    // A failed load must be recoverable from the page: the common cause is a
    // transient network/rate-limit failure, not a broken model.
    $<HTMLButtonElement>('load').textContent = 'Retry load';
  } finally {
    btn.disabled = false;
    bar.style.width = '0';
  }
}

// ---------------------------------------------------------------------------
// quantization check
// ---------------------------------------------------------------------------
const PROJ = [
  'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
  'self_attn.o_proj.weight',
  'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
];

function runQuantizationCheck() {
  const el = $('reports');
  const names = Object.keys(tensors).filter((n) => PROJ.some((s) => n.endsWith(s)));
  if (names.length === 0) {
    el.innerHTML = '<span class="muted">No projection matrices found.</span>';
    return;
  }

  let totalWeights = 0;
  let f32Bytes = 0;
  let int8Bytes = 0;
  let violations = 0;
  let worstRatio = 0;
  let worstName = '';
  const t0 = performance.now();

  for (const name of names) {
    const t = tensors[name];
    const [n, k] = t.shape;
    if (k % 4 !== 0) continue;
    const q = quantizeRows(t.data, n, k);
    const rep = quantError(t.data, q);
    quantized[name] = q;
    totalWeights += n * k;
    f32Bytes += n * k * 4;
    int8Bytes += packedBytes(q);
    violations += rep.violated;
    const ratio = rep.maxBound > 0 ? rep.maxError / rep.maxBound : 0;
    if (ratio > worstRatio) { worstRatio = ratio; worstName = name; }
  }
  const ms = performance.now() - t0;
  const boundOk = worstRatio <= 1 && violations === 0;

  el.innerHTML = `
    <table>
      <tr><th>matrices quantized</th><td class="num">${Object.keys(quantized).length}</td></tr>
      <tr><th>weights</th><td class="num">${(totalWeights / 1e6).toFixed(2)}M</td></tr>
      <tr><th>worst max|W − W′| ÷ (scale/2)</th><td class="num ${boundOk ? 'ok' : 'bad'}">${worstRatio.toFixed(6)} ${boundOk ? '≤ 1 ✓' : '> 1 ✗'}</td></tr>
      <tr><th>rows violating the bound</th><td class="num ${violations === 0 ? 'ok' : 'bad'}">${violations}</td></tr>
      <tr><th>worst matrix</th><td class="num">${worstName || '—'}</td></tr>
      <tr><th>size</th><td class="num">f32 ${(f32Bytes / 1e6).toFixed(1)} MB → int8 ${(int8Bytes / 1e6).toFixed(1)} MB (${(f32Bytes / int8Bytes).toFixed(2)}×)</td></tr>
      <tr><th>time</th><td class="num">${ms.toFixed(0)} ms</td></tr>
    </table>`;
  logger.success('main', `quantization: ${Object.keys(quantized).length} matrices, worst ratio ${worstRatio.toFixed(6)}, ${violations} violations, ${(f32Bytes / 1e6).toFixed(1)} MB -> ${(int8Bytes / 1e6).toFixed(1)} MB`);
}

// ---------------------------------------------------------------------------
// Naso kernels
// ---------------------------------------------------------------------------
async function compileKernels(): Promise<number> {
  const el = $('kernels');
  el.innerHTML = '';
  let okCount = 0;

  for (const spec of KERNEL_SPECS) {
    try {
      const source = await loadKernelSource(spec);
      const { kernel, diagnostics } = await compileKernel(source, spec.name);
      if (!kernel) {
        throw new Error(diagnostics.map((d) => `${d.severity}: ${d.message}`).join('; ') || 'compiler refused');
      }
      okCount++;

      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = `${spec.name} → ${kernel.abi.entryPoint}  ·  ${kernel.abi.workgroupSize}/workgroup  ·  ${kernel.abi.dispatchGroups} workgroup(s)  ✓`;
      details.appendChild(summary);

      const abi = document.createElement('pre');
      abi.textContent = `// ABI emitted by the compiler\n${JSON.stringify(kernel.abi, null, 1)}`;
      details.appendChild(abi);

      const wgsl = document.createElement('pre');
      wgsl.textContent = kernel.wgsl;
      details.appendChild(wgsl);
      el.appendChild(details);
    } catch (e) {
      const div = document.createElement('div');
      div.className = 'bad';
      div.textContent = `${spec.name}: ${e}`;
      el.appendChild(div);
    }
  }
  setDot('dot-naso', okCount === KERNEL_SPECS.length ? 'ok' : 'err');
  return okCount;
}

// ---------------------------------------------------------------------------
// chat
// ---------------------------------------------------------------------------
function renderChat() {
  const el = $('chat');
  el.innerHTML = '';
  if (history.length === 0) {
    el.innerHTML = '<div class="empty">Ask something. Everything runs in this tab — no server, no API key.</div>';
    return;
  }
  for (const m of history) addMessage(m.role, m.content);
}

function addMessage(role: string, content: string): HTMLElement {
  const el = $('chat');
  el.querySelector('.empty')?.remove();
  const wrap = document.createElement('div');
  wrap.className = `msg ${role} fade-in`;
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = role;
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = content;
  wrap.append(who, body);
  el.appendChild(wrap);
  el.scrollTop = el.scrollHeight;
  return body;
}

async function send() {
  if (!loaded || generating || !tokenizer || !cache) return;
  const inputEl = $<HTMLTextAreaElement>('input');
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';

  history.push({ role: 'user', content: text });
  addMessage('user', text);

  generating = true;
  stopRequested = false;
  $<HTMLButtonElement>('send').disabled = true;
  $<HTMLButtonElement>('stop').disabled = false;
  setDot('dot-gpu', 'busy');

  const body = addMessage('assistant', '');
  // Loading placeholder: animated "thinking" dots until the first token lands.
  // onToken replaces this via `body.textContent = soFar` on the first tick, so
  // the dots vanish precisely when streaming begins -- the perceived latency of
  // the model "starting to think" is made visible rather than blank.
  const thinking = document.createElement('span');
  thinking.className = 'thinking';
  thinking.innerHTML = '<span></span><span></span><span></span>';
  body.appendChild(thinking);

  const caret = document.createElement('span');
  caret.className = 'caret';
  caret.textContent = ' ';
  body.appendChild(caret);

  try {
    // Incremental context, in the model's real ChatML format. The turn markers
    // are load-bearing: feeding SmolLM2-Instruct bare `user\n...\nassistant\n`
    // text without them makes it loop and hallucinate fake turns instead of
    // answering. Each turn appends one user block and the assistant opener, so
    // the cache's contents are identical to rendering the whole conversation
    // and prefilling it from scratch. The system block is emitted only when the
    // cache is empty, otherwise it would repeat every turn.
    const IM_START = '\u003c\u007cim_start\u007c\u003e';
    const IM_END = '\u003c\u007cim_end\u007c\u003e';
    if (cache.pos === 0) {
      prefill(tensors, config, tokenizer.encode(`${IM_START}system\n${SYSTEM_PROMPT}${IM_END}\n`, true), cache);
    }
    let logits = prefill(tensors, config, tokenizer.encode(`${IM_START}user\n${text}${IM_END}\n`, true), cache);
    logits = prefill(tensors, config, tokenizer.encode(`${IM_START}assistant\n`, true), cache);

    const maxTokens = Number($<HTMLInputElement>('max-tokens').value) || 96;
    const temperature = Number($<HTMLInputElement>('temperature').value) || 0;
    const topK = Number($<HTMLInputElement>('top-k').value) || 1;

    const ids = decodeFrom(tensors, config, tokenizer, logits, cache, {
      maxTokens,
      temperature,
      topK,
      seed: 1234,
      shouldStop: () => stopRequested,
      onToken: (_id, soFar) => {
        body.textContent = soFar;
        body.appendChild(caret);
        $('chat').scrollTop = $('chat').scrollHeight;
      },
    });

    caret.remove();
    if (ids.length > 0) {
      const reply = body.textContent ?? '';
      history.push({ role: 'assistant', content: reply });
      body.textContent = reply;
      // Close the assistant turn with the real marker so the next turn continues
      // from a well-formed boundary.
      prefill(tensors, config, tokenizer.encode(`${IM_END}\n`, true), cache);
    } else {
      body.textContent = '(no tokens — raise max tokens or rephrase)';
    }
    logger.info('main', `${ids.length} tokens; ${cache.pos}/${cache.maxSeq} positions used`);
    setDot('dot-gpu', 'ok');
  } catch (e) {
    logger.error('main', `generation failed: ${e}`);
    body.textContent = `error: ${e}`;
    setDot('dot-gpu', 'err');
  } finally {
    generating = false;
    $<HTMLButtonElement>('send').disabled = false;
    $<HTMLButtonElement>('stop').disabled = true;
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
function populateModels() {
  const sel = $<HTMLSelectElement>('model-select');
  for (const m of MODELS) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.name} (${(m.weightsBytes / 1e6).toFixed(0)} MB)`;
    sel.appendChild(opt);
  }
  sel.value = DEFAULT_MODEL.id;
  sel.addEventListener('change', () => {
    config = MODELS.find((x) => x.id === sel.value) ?? DEFAULT_MODEL;
    loaded = false;
    $<HTMLButtonElement>('send').disabled = true;
  });
}

/**
 * Wait until the service worker controls this page.
 *
 * This exists because of a real, reported bug: on the FIRST visit the worker is
 * still installing when the 269 MB checkpoint fetch fires, so nothing
 * intercepts it and nothing caches it. The second visit then downloads the
 * whole checkpoint AGAIN. `register()` alone does not order this -- the page
 * must wait for control (the worker calls skipWaiting + clients.claim).
 *
 * Bounded, and never fatal: if the worker cannot be installed the app still
 * runs -- the weights fetch writes into the same cache bucket itself, so the
 * next visit is served from disk either way.
 */
async function waitForServiceWorkerControl(timeoutMs = 15000): Promise<boolean> {
  if (!('serviceWorker' in navigator)) return false;
  if (navigator.serviceWorker.controller) return true;

  // Register relative to the page, not '/sw.js': the deployed app lives under a
  // repo subpath on GitHub Pages, where a root-absolute URL 404s and offline
  // support silently never activates.
  const swUrl = new URL('./sw.js', document.baseURI).href;
  try {
    await navigator.serviceWorker.register(swUrl);
  } catch (e) {
    logger.warn('main', `service worker registration failed: ${e}`);
    return false;
  }

  if (!navigator.serviceWorker.controller) {
    await Promise.race([
      new Promise<void>((resolve) => {
        navigator.serviceWorker.addEventListener('controllerchange', () => resolve(), { once: true });
      }),
      navigator.serviceWorker.ready.then(() => undefined),
      new Promise<void>((r) => setTimeout(r, timeoutMs)),
    ]);
  }
  const controlled = !!navigator.serviceWorker.controller;
  logger.info('main', controlled
    ? 'service worker is controlling this page'
    : 'service worker did not take control in time; caching the checkpoint directly');
  return controlled;
}

async function boot() {
  // Disarm the index.html boot watchdog on the first line: the bundle has
  // demonstrably executed, so any later slowness is progress the UI reports
  // itself, not a silent failure.
  window.clearTimeout((window as any).__nasoWatchdog);
  populateModels();
  logger.info('main', 'booting');

  // Naso compiler (WASM). A failure here is reported but not fatal: the chat
  // runs on the host f32 path and the quantisation check is host-side too.
  setDot('dot-naso', 'busy');
  try {
    await initNaso();
    logger.success('naso', 'WASM compiler loaded');
    const n = await compileKernels();
    logger.info('naso', `${n}/${KERNEL_SPECS.length} kernels compiled in-browser`);
  } catch (e) {
    setDot('dot-naso', 'err');
    logger.error('naso', `compiler unavailable: ${e}`);
  }

  // WebGPU is reported honestly: the forward pass in this build is host-side
  // f32, so a working device does not mean the kernels were dispatched.
  try {
    const ok = await webgpuEngine.init();
    setDot('dot-gpu', ok ? 'ok' : 'idle');
    $('gpu-state').textContent = ok ? 'wgpu' : 'cpu f32';
    $('hint').textContent = ok
      ? 'WebGPU device ready. The forward pass still runs on the CPU in f32 in this build — the Naso kernels are compiled and validated here, but their dispatch is not yet wired into generation (see README).'
      : 'Running on the CPU (f32) — this container has no WebGPU adapter. The Naso kernels are still compiled and validated by the WASM compiler in this tab.';
  } catch (e) {
    setDot('dot-gpu', 'err');
    logger.error('webgpu', `init failed: ${e}`);
  }

  $<HTMLButtonElement>('load').addEventListener('click', () => { void loadModel(); });
  $<HTMLButtonElement>('send').addEventListener('click', () => { void send(); });
  $<HTMLButtonElement>('stop').addEventListener('click', () => { stopRequested = true; });
  $<HTMLTextAreaElement>('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  });

  // Watchdog: boot() sets window.__nasoBooted on its first line. If the page is
  // still grey after this long the bundle never executed at all (a 404 under the
  // deployed base path, a syntax error, ...) -- the exact silent failure this UI
  // was reported for. index.html owns the timer because code in this file cannot
  // report its own absence; here we only disarm it.
  window.clearTimeout((window as any).__nasoWatchdog);

  // Order matters: take control BEFORE the checkpoint fetch, or the first
  // visit's 269 MB never reaches Cache Storage and every revisit re-downloads
  // it. The await is bounded, so a blocked worker cannot stall the app.
  await waitForServiceWorkerControl();

  // Auto-start the default model so the demo is one click, not four.
  void loadModel();
}

void boot();
