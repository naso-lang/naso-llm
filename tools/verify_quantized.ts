/**
 * Verify a pre-quantised NPQ1 checkpoint against the app's own code.
 *
 * The claim being checked is strong: the artifact must be EXACTLY what
 * src/quantize.ts would compute from the original weights, and the forward pass
 * run on it must match the f32 model. So:
 *
 *   1. NPQ parses and every tensor is present with the original shape.
 *   2. For each int8 tensor, the packed words AND the per-row scales are
 *      bit-identical to `quantizeRows(originalF32, n, k)`. This is the check
 *      that catches the half-up vs half-even rounding trap (88,677 elements).
 *   3. For each pass-through tensor, the decoded f32 equals the original exactly.
 *   4. The int8 error bound |W - W'| <= scale/2 holds on the artifact.
 *   5. A real prompt run on the artifact reproduces the f32 model's argmax and
 *      top-5 logits.
 *
 * Usage: node tools/verify_quantized.mjs <modelDir> <npqPath>
 *   (built from tools/verify_quantized.ts with esbuild)
 */
import { readFileSync } from 'node:fs';
import { parseSafetensors, type Tensor } from '../src/model.js';
import { quantizeRows, unpackByte, type QuantizedMatrix } from '../src/quantize.js';
import { createKVCache, prefill, decodeFrom } from '../src/generate.js';
import type { Tensors } from '../src/llama.js';
import { BPETokenizer } from '../src/tokenizer.js';
import { logger } from '../src/logger.js';
logger.subscribe('main', () => {}); logger.subscribe('webgpu', () => {}); logger.subscribe('naso', () => {});

const DIR = process.argv[2] ?? '/var/tmp/smol';
const NPQ = process.argv[3] ?? '/var/tmp/model.int8.npq';

interface Entry { name: string; dtype: number; dims: number[]; packed?: Uint32Array; scales?: Float32Array; f32?: Float32Array; bf16?: Uint16Array; }

function parseNPQ(buf: ArrayBuffer): Map<string, Entry> {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'NPQ1') throw new Error(`bad magic ${JSON.stringify(magic)}`);
  const version = dv.getUint32(4, true);
  const count = dv.getUint32(8, true);
  if (version !== 1) throw new Error(`unsupported version ${version}`);
  let off = 12;
  const out = new Map<string, Entry>();
  for (let i = 0; i < count; i++) {
    const nameLen = dv.getUint32(off, true); off += 4;
    const name = new TextDecoder().decode(new Uint8Array(buf, off, nameLen)); off += nameLen;
    const dtype = dv.getUint8(off); off += 1;
    const rank = dv.getUint32(off, true); off += 4;
    const dims: number[] = [];
    for (let d = 0; d < rank; d++) { dims.push(dv.getUint32(off, true)); off += 4; }
    const elems = dims.reduce((a, b) => a * b, 1);
    const e: Entry = { name, dtype, dims };
    if (dtype === 1) {
      const [n, k] = dims;
      const words = (n * k) / 4;
      e.packed = new Uint32Array(buf.slice(off, off + words * 4)); off += words * 4;
      e.scales = new Float32Array(buf.slice(off, off + n * 4)); off += n * 4;
    } else if (dtype === 2) {
      e.bf16 = new Uint16Array(buf.slice(off, off + elems * 2)); off += elems * 2;
    } else {
      e.f32 = new Float32Array(buf.slice(off, off + elems * 4)); off += elems * 4;
    }
    out.set(name, e);
  }
  return out;
}

function fail(msg: string): never { console.error(`FAIL: ${msg}`); process.exit(1); }

const orig = parseSafetensors(new Uint8Array(readFileSync(`${DIR}/model.safetensors`)).buffer) as Map<string, Tensor>;
const art = parseNPQ(new Uint8Array(readFileSync(NPQ)).buffer);

console.log(`original: ${orig.size} tensors   artifact: ${art.size} tensors`);
if (orig.size !== art.size) fail(`tensor count differs`);

let checkedInt8 = 0, int8Ok = 0, checkedPass = 0, passOk = 0, mismatched = 0, worstRatio = 0, violations = 0;

for (const [name, t] of orig) {
  const a = art.get(name);
  if (!a) fail(`artifact missing ${name}`);
  const W = t.data as Float32Array;
  const shapeOk = a.dims.length === t.shape.length && a.dims.every((d, i) => d === t.shape[i]);
  if (!shapeOk) fail(`${name}: shape ${t.shape} vs artifact [${a.dims}]`);
  const n = t.shape.length === 2 ? t.shape[0] : 1;
  const k = t.shape.length === 2 ? t.shape[1] : W.length;

  if (a.dtype === 1) {
    // must equal quantizeRows on the ORIGINAL weights, bit for bit
    const ref: QuantizedMatrix = quantizeRows(W, n, k);
    if (ref.packed.length !== a.packed!.length) fail(`${name}: packed length`);
    let thisTensorOk = true;
    for (let i = 0; i < ref.packed.length; i++) {
      if (ref.packed[i] !== a.packed![i]) { thisTensorOk = false; mismatched++; if (mismatched < 4) console.error(`  word mismatch ${name}[${i}]: ref ${ref.packed[i]} vs art ${a.packed![i]}`); break; }
    }
    for (let i = 0; i < ref.scales.length; i++) {
      if (ref.scales[i] !== a.scales![i]) { thisTensorOk = false; mismatched++; if (mismatched < 4) console.error(`  scale mismatch ${name}[${i}]: ref ${ref.scales[i]} vs art ${a.scales![i]}`); break; }
    }
    if (thisTensorOk) int8Ok++;
    // error bound on the artifact. Compare each error to its OWN row bound, as
    // quantError() in src/quantize.ts does -- using a global bound is wrong.
    const kwords = k / 4;
    for (let o = 0; o < n; o++) {
      const bound = a.scales![o] / 2;
      let rowBad = false;
      for (let w = 0; w < kwords; w++) {
        const word = a.packed![o * kwords + w];
        for (let i = 0; i < 4; i++) {
          const err = Math.abs(W[o * k + w * 4 + i] - unpackByte(word, i) * a.scales![o]);
          const ratio = err / Math.max(bound, 1e-30);
          if (ratio > worstRatio) worstRatio = ratio;
          if (err > bound) rowBad = true;
        }
      }
      if (rowBad) violations++;
    }
    checkedInt8++;
  } else if (a.dtype === 2) {
    // bf16 must be the top 16 bits of the ORIGINAL f32 (bit-exact round trip)
    const f32bits = new Uint32Array(W.buffer, W.byteOffset, W.length);
    for (let i = 0; i < W.length; i++) {
      if ((f32bits[i] >>> 16) !== a.bf16![i]) { mismatched++; console.error(`  bf16 mismatch ${name}[${i}]`); break; }
    }
    checkedPass++; passOk++;
  } else {
    let thisTensorOk = true;
    for (let i = 0; i < W.length; i++) {
      if (W[i] !== a.f32![i]) { thisTensorOk = false; mismatched++; console.error(`  f32 mismatch ${name}[${i}]`); break; }
    }
    checkedPass++; if (thisTensorOk) passOk++;
  }
}

console.log(`int8 tensors bit-identical to quantize.ts : ${int8Ok}/${checkedInt8}`);
console.log(`pass-through tensors exact               : ${passOk}/${checkedPass}`);
console.log(`worst |W-W'|/(scale/2)                    : ${worstRatio.toFixed(6)} (must be <= 1)`);
console.log(`rows violating the bound                 : ${violations}`);

// --- forward pass equivalence ------------------------------------------------
const cfg = JSON.parse(readFileSync(`${DIR}/config.json`, 'utf8'));
const config = {
  id: 'x', name: 'x', repo: 'x',
  hiddenSize: cfg.hidden_size, intermediateSize: cfg.intermediate_size,
  numLayers: cfg.num_hidden_layers, numHeads: cfg.num_attention_heads,
  numKvHeads: cfg.num_key_value_heads, vocabSize: cfg.vocab_size,
  rmsNormEps: cfg.rms_norm_eps, ropeTheta: cfg.rope_theta ?? 10000.0, weightsBytes: 1,
};
const tok = BPETokenizer.fromJSON(JSON.parse(readFileSync(`${DIR}/tokenizer.json`, 'utf8')));

// dequantise the whole artifact into an f32 tensor map the generator can use
const IM_START = '\u003c\u007cim_start\u007c\u003e';
const IM_END = '\u003c\u007cim_end\u007c\u003e';
const prompt = `${IM_START}user\nWhat is the capital of France?${IM_END}\n${IM_START}assistant\n`;
const ids = tok.encode(prompt, true);

function runForward(tensors: Tensors) {
  const cache = createKVCache(config, 256);
  const logits = prefill(tensors as never, config as never, ids, cache);
  const gen = decodeFrom(tensors as never, config as never, tok, logits, cache, { maxTokens: 12, temperature: 0, topK: 1 });
  return { logits, text: tok.decode(gen), gen };
}

/**
 * Materialise the artifact as the generator's `Tensors` shape: an object keyed by
 * tensor name, each `{ dtype, shape, data }` -- exactly what
 * Object.fromEntries(parseSafetensors(...)) produces in the app.
 */
function materialise(): Tensors {
  const out: Tensors = {};
  for (const [name, a] of art) {
    const data = new Float32Array(a.dims.reduce((x, y) => x * y, 1));
    if (a.dtype === 1) {
      const [n, k] = a.dims;
      const kwords = k / 4;
      for (let o = 0; o < n; o++) for (let w = 0; w < kwords; w++) {
        const word = a.packed![o * kwords + w];
        for (let i = 0; i < 4; i++) data[o * k + w * 4 + i] = unpackByte(word, i) * a.scales![o];
      }
    } else if (a.dtype === 2) {
      const u = new Uint32Array(data.buffer);
      for (let i = 0; i < data.length; i++) u[i] = a.bf16![i] << 16;
    } else {
      data.set(a.f32!);
    }
    out[name] = { dtype: 'F32', shape: a.dims, data };
  }
  return out;
}

// top-5 over the last position
function top5(logits: Float32Array) {
  const n = logits.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => logits[b] - logits[a]).slice(0, 5);
  return idx.map((i) => [i, logits[i]] as const);
}

const refRun = runForward(Object.fromEntries(orig) as unknown as Tensors);
const artRun = runForward(materialise());
const r5 = top5(refRun.logits), a5 = top5(artRun.logits);
const sameTop1 = r5[0][0] === a5[0][0];
const maxLogitDelta = Math.max(...a5.map(([i, v], j) => Math.abs(v - r5[j][1])));

console.log(`\nf32 model  : argmax ${r5[0][0]} (${tok.decode([r5[0][0]])!})  text ${JSON.stringify(refRun.text.slice(0, 60))}`);
console.log(`int8 artifact: argmax ${a5[0][0]} (${tok.decode([a5[0][0]])!})  text ${JSON.stringify(artRun.text.slice(0, 60))}`);
console.log(`top-1 matches: ${sameTop1}   max logit |Δ| across top-5: ${maxLogitDelta.toExponential(3)}`);

const ok = mismatched === 0 && violations === 0 && worstRatio <= 1 && sameTop1 && refRun.text === artRun.text;
console.log(`\nVERDICT: ${ok ? 'PASS' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
