/**
 * Node harness: run src/llama.ts's forward() on the real checkpoint and diff it
 * against the independent NumPy reference (tools/reference_llama.py).
 *
 * Bundled with esbuild and executed with `npm run verify`. It is NOT part of the
 * browser app; it exists so the forward pass the demo shows is checkable against
 * a second implementation rather than merely "it ran without throwing".
 */
import { readFileSync } from 'node:fs';
import { forward, mse } from '../src/llama.js';
import { parseSafetensors } from '../src/model.js';
import { logger } from '../src/logger.js';
import type { ModelConfig } from '../src/types.js';
import type { Tensor } from '../src/model.js';

// Keep output to the numbers.
logger.subscribe('main', () => {});
logger.subscribe('webgpu', () => {});
logger.subscribe('naso', () => {});

const DIR = process.argv[2] ?? '/var/tmp/tiny';

function loadNpy(path: string): Float32Array {
  const buf = readFileSync(path);
  if (buf[0] !== 0x93 || buf[1] !== 0x4e) throw new Error(`not an .npy file: ${path}`);
  const headerLen = buf.readUInt16LE(8);
  const headerStart = 10;
  const dataStart = headerStart + headerLen;
  // Header is a Python dict repr; we only need it to be present.
  const header = buf.subarray(headerStart, dataStart).toString('latin1');
  if (!header.includes("'<f4'") && !header.includes('float32')) throw new Error(`expected float32 npy, header: ${header}`);
  const count = (buf.length - dataStart) / 4;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = buf.readFloatLE(dataStart + i * 4);
  return out;
}

function configFromJson(json: any): ModelConfig {
  return {
    id: 'tiny-random-llama',
    name: 'tiny-random-LlamaForCausalLM',
    repo: 'hf-internal-testing/tiny-random-LlamaForCausalLM',
    hiddenSize: json.hidden_size,
    intermediateSize: json.intermediate_size,
    numLayers: json.num_hidden_layers,
    numHeads: json.num_attention_heads,
    numKvHeads: json.num_key_value_heads,
    vocabSize: json.vocab_size,
    rmsNormEps: json.rms_norm_eps,
    ropeTheta: json.rope_theta ?? 10000.0,
    weightsBytes: 4131280,
  };
}

function main() {
  const config = configFromJson(JSON.parse(readFileSync(`${DIR}/config.json`, 'utf8')));

  // Read the safetensors bytes into a standalone ArrayBuffer (a Node Buffer may
  // be a view into a larger pool, so copy rather than trusting .buffer).
  const raw = readFileSync(`${DIR}/model.safetensors`);
  const ab = new Uint8Array(raw).buffer;
  const tensors: Map<string, Tensor> = parseSafetensors(ab);
  console.log(`loaded ${tensors.size} tensors (hidden=${config.hiddenSize}, layers=${config.numLayers})`);

  const ids = process.argv.slice(3).map(Number);
  const inputIds = ids.length > 0 ? ids : [1, 1229, 3000];

  const t0 = performance.now();
  const { logits } = forward(Object.fromEntries(tensors), config, inputIds);
  const ms = performance.now() - t0;
  console.log(`forward() ran in ${ms.toFixed(0)} ms over ${inputIds.length} token(s)`);

  const ref = loadNpy(`${DIR}/ref_logits.npy`);
  const seq = Math.floor(logits.length / config.vocabSize);
  const last = logits.subarray((seq - 1) * config.vocabSize);
  if (last.length !== ref.length) throw new Error(`length mismatch: ts ${last.length} vs ref ${ref.length}`);

  let maxAbs = 0;
  let argmaxTs = 0;
  let argmaxRef = 0;
  for (let i = 0; i < ref.length; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(last[i] - ref[i]));
    if (last[i] > last[argmaxTs]) argmaxTs = i;
    if (ref[i] > ref[argmaxRef]) argmaxRef = i;
  }
  console.log(`max |ts - numpy| = ${maxAbs.toExponential(3)}`);
  console.log(`MSE = ${mse(last, ref).toExponential(3)}`);
  console.log(`argmax: ts=${argmaxTs} numpy=${argmaxRef} ${argmaxTs === argmaxRef ? 'MATCH' : 'MISMATCH'}`);
  console.log(`logit[argmax]: ts=${last[argmaxTs].toFixed(6)} numpy=${ref[argmaxRef].toFixed(6)}`);

  // f32 accumulation over a 16-wide layer should be exact to ~1e-6; a real
  // disagreement (layout/orientation bug) shows up orders of magnitude larger.
  const ok = maxAbs < 1e-4 && argmaxTs === argmaxRef;
  console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
  process.exit(ok ? 0 : 1);
}

main();
