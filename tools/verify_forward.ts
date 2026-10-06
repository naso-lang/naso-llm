/**
 * Node harness: run the real forward passes over a real checkpoint and diff
 * against an independent NumPy reference (tools/reference_smol.py).
 *
 * Three things must hold for the chat demo to be trustworthy:
 *
 *   1. `forward()` (llama.ts, full-sequence reference) reproduces NumPy.
 *   2. `forwardToken()` (generate.ts, KV cache) reproduces NumPy too -- a
 *      cache bug that reused a stale position or mis-strided K/V would show up
 *      here and nowhere else.
 *   3. The KV-cache path reproduces the full-sequence path, token for token,
 *      over a multi-token prompt. That is the invariant the chat loop relies on.
 *
 * Bundled with esbuild and run by `npm run verify`.
 */
import { readFileSync } from 'node:fs';
import { forward, mse } from '../src/llama.js';
import { createKVCache, forwardToken, prefill, decodeFrom } from '../src/generate.js';
import { BPETokenizer } from '../src/tokenizer.js';
import { parseSafetensors, type Tensor } from '../src/model.js';
import { quantizeRows, dequantizeRows, quantError, packedBytes } from '../src/quantize.js';
import { logger } from '../src/logger.js';
import type { ModelConfig } from '../src/types.js';

logger.subscribe('main', () => {});
logger.subscribe('webgpu', () => {});
logger.subscribe('naso', () => {});

const DIR = process.argv[2] ?? '/var/tmp/smol';
const REF = process.argv[3] ?? `${DIR}/ref_logits.npy`;

function loadNpy(path: string): Float32Array {
  const buf = readFileSync(path);
  if (buf[0] !== 0x93 || buf[1] !== 0x4e) throw new Error(`not an .npy file: ${path}`);
  const headerLen = buf.readUInt16LE(8);
  const dataStart = 10 + headerLen;
  const header = buf.subarray(10, dataStart).toString('latin1');
  if (!header.includes("'<f4'") && !header.includes('float32')) {
    throw new Error(`expected float32 npy, header: ${header}`);
  }
  const count = (buf.length - dataStart) / 4;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = buf.readFloatLE(dataStart + i * 4);
  return out;
}

function configFromJson(json: Record<string, unknown>): ModelConfig {
  return {
    id: 'smollm2-135m-instruct',
    name: 'SmolLM2-135M-Instruct',
    repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    hiddenSize: json.hidden_size as number,
    intermediateSize: json.intermediate_size as number,
    numLayers: json.num_hidden_layers as number,
    numHeads: json.num_attention_heads as number,
    numKvHeads: json.num_key_value_heads as number,
    vocabSize: json.vocab_size as number,
    rmsNormEps: json.rms_norm_eps as number,
    ropeTheta: (json.rope_theta as number) ?? 10000.0,
    weightsBytes: 269060552,
  };
}

function argmax(x: Float32Array): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function main() {
  const config = configFromJson(JSON.parse(readFileSync(`${DIR}/config.json`, 'utf8')));
  const raw = readFileSync(`${DIR}/model.safetensors`);
  const ab = new Uint8Array(raw).buffer;
  const tensors = parseSafetensors(ab);
  const t: Record<string, Tensor> = Object.fromEntries(tensors);
  console.log(`loaded ${tensors.size} tensors (hidden=${config.hiddenSize}, layers=${config.numLayers}, kv=${config.numKvHeads})`);

  const idsArg = process.argv.slice(4).map(Number);
  let inputIds = idsArg;
  if (inputIds.length === 0) {
    const promptFile = `${DIR}/prompt.json`;
    inputIds = JSON.parse(readFileSync(promptFile, 'utf8')).ids;
    console.log(`prompt: ${inputIds.length} tokens from ${promptFile}`);
  }

  const ref = loadNpy(REF);
  let failures = 0;

  // ---- 1. full-sequence reference vs NumPy --------------------------------
  const t0 = performance.now();
  const full = forward(t, config, inputIds);
  const fullMs = performance.now() - t0;
  const seq = Math.floor(full.logits.length / config.vocabSize);
  const fullLast = full.logits.subarray((seq - 1) * config.vocabSize);
  const dFull = maxAbsDiff(fullLast, ref);
  const okFull = dFull < 1e-4 && argmax(fullLast) === argmax(ref);
  console.log(`\n[1] full-sequence forward() vs numpy`);
  console.log(`    ${fullMs.toFixed(0)} ms; max|Δ| = ${dFull.toExponential(3)}; MSE = ${mse(fullLast, ref).toExponential(3)}`);
  console.log(`    argmax ts=${argmax(fullLast)} numpy=${argmax(ref)} ${argmax(fullLast) === argmax(ref) ? 'MATCH' : 'MISMATCH'}`);
  if (!okFull) failures++;

  // ---- 2. KV-cache path vs NumPy -----------------------------------------
  const cache = createKVCache(config, 256);
  let last = new Float32Array(config.vocabSize);
  const t1 = performance.now();
  for (const id of inputIds) last = forwardToken(t, config, id, cache);
  const kvMs = performance.now() - t1;
  const dKv = maxAbsDiff(last, ref);
  const okKv = dKv < 1e-4 && argmax(last) === argmax(ref);
  console.log(`\n[2] KV-cache forwardToken() vs numpy`);
  console.log(`    ${kvMs.toFixed(0)} ms; max|Δ| = ${dKv.toExponential(3)}; cache.pos = ${cache.pos}`);
  console.log(`    argmax ts=${argmax(last)} numpy=${argmax(ref)} ${argmax(last) === argmax(ref) ? 'MATCH' : 'MISMATCH'}`);
  if (!okKv) failures++;

  // ---- 3. KV cache vs full sequence, per position ------------------------
  // The chat loop only ever reads the LAST token's logits, but an off-by-one in
  // the cache would still be latent. Compare every position's logits.
  console.log(`\n[3] KV-cache path vs full-sequence path, all ${seq} positions`);
  const cache2 = createKVCache(config, 256);
  let worst = 0;
  let worstPos = -1;
  let argmaxMismatch = 0;
  for (let s = 0; s < seq; s++) {
    const got = forwardToken(t, config, inputIds[s], cache2);
    const want = full.logits.subarray(s * config.vocabSize, (s + 1) * config.vocabSize);
    const d = maxAbsDiff(got, want);
    if (d > worst) { worst = d; worstPos = s; }
    if (argmax(got) !== argmax(want)) argmaxMismatch++;
  }
  console.log(`    max|Δ| = ${worst.toExponential(3)} at position ${worstPos}; argmax mismatches = ${argmaxMismatch}`);
  if (worst > 1e-3 || argmaxMismatch > 0) failures++;

  // ---- 4. int8 quantisation: the error bound holds -----------------------
  // Quantise every projection matrix in the model and assert |W - W'| <= scale/2.
  const projSuffixes = [
    'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
  ];
  console.log(`\n[4] int8 quantisation error bound (|W - W'| <= scale/2)`);
  let matrices = 0;
  let worstRatio = 0;
  let totalElements = 0;
  let f32Bytes = 0;
  let int8Bytes = 0;
  let violations = 0;
  const skipped: string[] = [];
  for (const [name, tensor] of Object.entries(t)) {
    if (!projSuffixes.some((s) => name.endsWith(s))) continue;
    const [n, k] = tensor.shape;
    if (k % 4 !== 0) { skipped.push(`${name} (k=${k})`); continue; }
    const q = quantizeRows(tensor.data, n, k);
    const rep = quantError(tensor.data, q);
    matrices++;
    totalElements += n * k;
    f32Bytes += n * k * 4;
    int8Bytes += packedBytes(q);
    violations += rep.violated;
    // Report how close the worst element came to its bound (should be <= 1).
    const ratio = rep.maxBound > 0 ? rep.maxError / rep.maxBound : 0;
    if (ratio > worstRatio) worstRatio = ratio;
  }
  console.log(`    ${matrices} matrices, ${(totalElements / 1e6).toFixed(1)}M weights`);
  console.log(`    worst max|W - W'| / (scale/2) = ${worstRatio.toFixed(6)}  (must be <= 1)`);
  console.log(`    rows violating the bound: ${violations}`);
  console.log(`    f32 ${(f32Bytes / 1e6).toFixed(1)} MB -> int8 ${(int8Bytes / 1e6).toFixed(1)} MB (${(f32Bytes / int8Bytes).toFixed(2)}x)`);
  if (skipped.length) console.log(`    skipped (k not a multiple of 4): ${skipped.join(', ')}`);
  if (worstRatio > 1 || violations > 0) failures++;

  // ---- 5. accuracy cost: run the model with dequantised int8 weights -----
  // This is what the GPU path computes: W' instead of W. If the reply's token
  // distribution survives this, the quantisation is safe in practice, not just
  // in bound.
  console.log(`\n[5] forward pass with int8-dequantised weights vs f32`);
  const qTensors: Record<string, Tensor> = { ...t };
  let quantized = 0;
  for (const [name, tensor] of Object.entries(t)) {
    if (!projSuffixes.some((s) => name.endsWith(s))) continue;
    const [n, k] = tensor.shape;
    if (k % 4 !== 0) continue;
    const q = quantizeRows(tensor.data, n, k);
    qTensors[name] = { dtype: 'F32', shape: tensor.shape, data: dequantizeRows(q) };
    quantized++;
  }
  const qOut = forward(qTensors, config, inputIds);
  const qLast = qOut.logits.subarray((seq - 1) * config.vocabSize);
  let dotProd = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < ref.length; i++) {
    dotProd += fullLast[i] * qLast[i];
    na += fullLast[i] * fullLast[i];
    nb += qLast[i] * qLast[i];
  }
  const cosine = dotProd / (Math.sqrt(na) * Math.sqrt(nb));
  const top1Agree = argmax(qLast) === argmax(fullLast);
  // Top-5 agreement is the honest metric: greedy decoding only needs the argmax,
  // but a near-tie in the top-5 is what tells you the distribution is intact.
  const top5 = (x: Float32Array) => Array.from(x.keys()).sort((a, b) => x[b] - x[a]).slice(0, 5).join(',');
  const t5Agree = top5(qLast) === top5(fullLast);
  console.log(`    ${quantized} matrices dequantised; max|Δ| vs f32 = ${maxAbsDiff(qLast, fullLast).toExponential(3)}`);
  console.log(`    cosine similarity = ${cosine.toFixed(8)}`);
  console.log(`    argmax f32=${argmax(fullLast)} int8=${argmax(qLast)} ${top1Agree ? 'MATCH' : 'MISMATCH'}`);
  console.log(`    top-5 ids identical: ${t5Agree ? 'yes' : 'no'}`);
  if (!top1Agree || cosine < 0.999) failures++;

  // ---- 6. incremental chat context == a full re-render -------------------
  // The UI keeps the cache across turns and only feeds the new text. That is an
  // optimisation, so it must be indistinguishable from rendering the whole
  // conversation and prefilling it from scratch. Any drift here (a missing
  // control token, a mis-spliced boundary) would silently change replies over a
  // long conversation, which is the hardest kind of bug to notice.
  console.log(`\n[6] incremental multi-turn prefill == full re-render`);
  const tok = BPETokenizer.fromJSON(JSON.parse(readFileSync(`${DIR}/tokenizer.json`, 'utf8')));
  const CT = '<|im_' + 'start|>';
  const CTE = '<|im_' + 'end|>';
  const SYS = 'You are a helpful AI assistant.';
  // A two-turn conversation with a REAL generated reply in between, driven
  // through exactly the sequence the UI uses: the system block once, then per
  // turn a user block, an assistant opener, the decoded reply, and the closing
  // token. This is what catches a missing turn-boundary token, which a
  // single-turn test cannot see.
  const inc = createKVCache(config, 512);
  const q1 = 'What is the capital of France?';
  const q2 = 'And what is its population?';

  // turn 1
  prefill(t, config, tok.encode(`<|im_start|>system\n${SYS}<|im_end|>\n`, true), inc);
  let l1 = prefill(t, config, tok.encode(`<|im_start|>user\n${q1}<|im_end|>\n`, true), inc);
  l1 = prefill(t, config, tok.encode('<|im_start|>assistant\n', true), inc);
  const replyIds = decodeFrom(t, config, tok, l1, inc, { maxTokens: 24, temperature: 0, topK: 1, seed: 1234 });
  const reply = tok.decode(replyIds);
  prefill(t, config, tok.encode(`<|im_end|>\n`, true), inc);

  // turn 2
  let incLogits = prefill(t, config, tok.encode(`<|im_start|>user\n${q2}<|im_end|>\n`, true), inc);
  incLogits = prefill(t, config, tok.encode('<|im_start|>assistant\n', true), inc);

  // The same conversation rendered whole and prefilled from scratch.
  const chatml = `<|im_start|>system\n${SYS}<|im_end|>\n` +
    `<|im_start|>user\n${q1}<|im_end|>\n` +
    `<|im_start|>assistant\n${reply}<|im_end|>\n` +
    `<|im_start|>user\n${q2}<|im_end|>\n` +
    `<|im_start|>assistant\n`;
  const fresh = createKVCache(config, 512);
  const wholeIds = tok.encode(chatml, true);
  let freshLogits = new Float32Array(0);
  for (let i = 0; i < wholeIds.length; i++) {
    freshLogits = forwardToken(t, config, wholeIds[i], fresh, i < wholeIds.length - 1);
  }

  console.log(`    reply to turn 1: ${JSON.stringify(reply)}`);
  if (inc.pos !== fresh.pos) {
    console.log(`    FAIL: cache positions differ (incremental ${inc.pos} vs fresh ${fresh.pos})`);
    failures++;
  } else if (incLogits.length !== freshLogits.length) {
    console.log(`    FAIL: logits length differs (${incLogits.length} vs ${freshLogits.length})`);
    failures++;
  } else {
    const d = maxAbsDiff(incLogits, freshLogits);
    const same = argmax(incLogits) === argmax(freshLogits);
    console.log(`    ${inc.pos} cached positions; max|Δ| = ${d.toExponential(3)}`);
    console.log(`    argmax incremental=${argmax(incLogits)} fresh=${argmax(freshLogits)} ${same ? 'MATCH' : 'MISMATCH'}`);
    if (d !== 0 || !same) failures++;
  }

  console.log(failures === 0 ? '\nRESULT: PASS' : `\nRESULT: FAIL (${failures} check(s))`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
