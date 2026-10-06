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
import { parseSafetensors, type Tensor } from './model.js';
import { BPETokenizer, type ChatMessage } from './tokenizer.js';
import { createKVCache, prefill, decodeFrom, type KVCache } from './generate.js';
import { quantizeRows, quantError, packedBytes, type QuantizedMatrix } from './quantize.js';

const SYSTEM_PROMPT = 'You are a helpful AI assistant.';

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
async function fetchWithProgress(
  url: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? '0');
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

async function loadModel() {
  const btn = $<HTMLButtonElement>('load');
  const bar = $('progress').firstElementChild as HTMLElement;
  btn.disabled = true;
  setDot('dot-model', 'busy');
  $('model-state').textContent = 'loading…';

  try {
    // config.json drives the architecture and is fetched rather than hardcoded,
    // so the two cannot drift apart.
    const cfgRes = await fetch(`${HF_BASE}/${config.repo}/resolve/main/config.json`);
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

    const weightsUrl = `${HF_BASE}/${config.repo}/resolve/main/model.safetensors`;
    logger.info('main', `fetching ${weightsUrl}`);
    const buf = await fetchWithProgress(weightsUrl, (loaded_, total) => {
      bar.style.width = `${total ? ((loaded_ / total) * 100).toFixed(1) : 0}%`;
      $('model-state').textContent = `${(loaded_ / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB`;
    });
    tensors = Object.fromEntries(parseSafetensors(buf));
    logger.success('main', `parsed ${Object.keys(tensors).length} tensors`);

    const tokRes = await fetch(`${HF_BASE}/${config.repo}/resolve/main/tokenizer.json`);
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
    $('model-state').textContent = config.name;
    $<HTMLTextAreaElement>('input').disabled = false;
    $<HTMLButtonElement>('send').disabled = false;
  } catch (e) {
    setDot('dot-model', 'err');
    $('model-state').textContent = 'load failed';
    logger.error('main', `model load failed: ${e}`);
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
  wrap.className = `msg ${role}`;
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
  const caret = document.createElement('span');
  caret.className = 'caret';
  caret.textContent = ' ';
  body.appendChild(caret);

  try {
    // Incremental context. Every turn appends exactly one user turn followed by
    // the opening assistant turn, so the cache's contents are identical to
    // rendering the whole conversation and prefilling it from scratch. The
    // system block is emitted only when the cache is empty, otherwise it would
    // be repeated on every turn.
    if (cache.pos === 0) {
      prefill(tensors, config, tokenizer.encode(`<|im_start|>system\n${SYSTEM_PROMPT}<|im_end|>\n`, true), cache);
    }
    let logits = prefill(tensors, config, tokenizer.encode(`<|im_start|>user\n${text}<|im_end|>\n`, true), cache);
    logits = prefill(tensors, config, tokenizer.encode('<|im_start|>assistant\n', true), cache);

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
      // Close the assistant turn so the next one continues from here.
      prefill(tensors, config, tokenizer.encode(`<|im_end|>\n`, true), cache);
    } else {
      body.textContent = '(no tokens — raise max tokens or rephrase)';
    }
    logger.info('main', `${ids.length} tokens; ${cache.pos}/${cache.maxSeq} positions used`);
    setDot('dot-gpu', loaded ? 'ok' : 'idle');
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

async function boot() {
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
    $('gpu-state').textContent = ok ? 'gpu ready' : 'cpu';
    $('hint').textContent = ok
      ? 'WebGPU device ready. The forward pass still runs on the CPU in f32 in this build — the Naso kernels are compiled and validated here, but their dispatch is not yet wired into generation (see README).'
      : 'No WebGPU adapter: the chat runs on the CPU. The Naso kernels are still compiled by the WASM compiler in this tab.';
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

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').then(
      () => logger.info('main', 'service worker registered (offline cache ready)'),
      (e) => logger.warn('main', `service worker registration failed: ${e}`),
    );
  }

  // Auto-start the default model so the demo is one click, not four.
  void loadModel();
}

void boot();
