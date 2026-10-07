// Loader integration test: prove the APP'S OWN loader reproduces the f32 model.
//
// This is the missing coverage for "Local loader please": main.ts loads the
// pre-quantised checkpoint via `parseNPQ` from src/model.ts. The app's parser
// dequantises int8 straight into dense f32 Tensors (it does not keep packed
// words around -- there is nothing to inspect at runtime); so the proof here
// is behavioural, not byte-level:
//
//   * src/model.parseNPQ(artifact) parses every tensor into the SAME
//     {dtype: 'F32'|'BF16', shape, data} structure parseSafetensors produces.
//   * A real forward pass (src/generate.prefill/decodeFrom/createKVCache) over
//     those tensors yields the SAME top-1 token and the SAME first generated
//     text as the f32 reference safetensors.
//
// (The byte-level "packed words are identical to quantize.ts" check lives in
// tools/verify_quantized.ts, which needs a parser that exposes packed words --
// the runtime loader intentionally does not.)
//
// Built via esbuild (see scripts/verify-forward.mjs for the recipe) so the
// relative `../src/*.ts` imports resolve into one node-runnable bundle.
import { readFileSync } from 'node:fs';

import { parseNPQ, parseSafetensors } from '../src/model.ts';
import { logger } from '../src/logger.ts';
import { createKVCache, prefill, decodeFrom, type KVCache } from '../src/generate.ts';
import { BPETokenizer, type ChatMessage } from '../src/tokenizer.ts';
import type { ModelConfig, Tensors } from '../src/types.ts';

// Silence the app logger in esbuild's bundle of logger.ts so its stdout stays
// parseable by the harness (which looks for `VERDICT:`).
logger.subscribe('main', () => {});

const args = process.argv.slice(2);
const DIR = args[0] ?? process.env.NASO_MODEL_DIR ?? '/var/tmp/smol';
const NPQ = args[1] ?? '/var/tmp/model.int8.npq';
const PROMPT = args[2] ?? 'What is the capital of France?';

const cfg = JSON.parse(readFileSync(`${DIR}/config.json`, 'utf8'));
const CONFIG: ModelConfig = {
  id: 'x',
  name: 'x',
  repo: 'x',
  hiddenSize: cfg.hidden_size,
  intermediateSize: cfg.intermediate_size,
  numLayers: cfg.num_hidden_layers,
  numHeads: cfg.num_attention_heads,
  numKvHeads: cfg.num_key_value_heads,
  vocabSize: cfg.vocab_size,
  rmsNormEps: cfg.rms_norm_eps,
  ropeTheta: cfg.rope_theta ?? 1e4,
  weightsBytes: 1,
};

function fail(msg: string): never {
  console.error(`FAIL: ${msg}`);
  process.exit(1);
}

// ---- Load reference (f32) and artifact (via the APP's loader) ------------

const ref: Tensors = Object.fromEntries(
  parseSafetensors(new Uint8Array(readFileSync(`${DIR}/model.safetensors`)).buffer),
);
const art: Tensors = Object.fromEntries(
  parseNPQ(new Uint8Array(readFileSync(NPQ)).buffer),
);

for (const name of Object.keys(ref)) {
  const a = art[name];
  if (!a) fail(`loader dropped tensor ${name}`);
  const r = ref[name];
  if (a.shape.length !== r.shape.length || !a.shape.every((d, i) => d === r.shape[i])) {
    fail(`${name}: shape ${JSON.stringify(r.shape)} vs loader ${JSON.stringify(a.shape)}`);
  }
  // Every tensor the loader yields is f32 (int8 dequantised, bf16 promoted to f32
  // for compute). That is the dense view the generator consumes.
  if (a.dtype !== 'F32') {
    fail(`${name}: loader returned dtype ${a.dtype}, expected F32`);
  }
}
console.log(`loader parsed ${Object.keys(art).length}/${Object.keys(ref).length} tensors as F32`);

// ---- Forward pass: does the loader materialise the same computation? -----

const tok = BPETokenizer.fromJSON(JSON.parse(readFileSync(`${DIR}/tokenizer.json`, 'utf8')));
// Build the chat template via the tokenizer's own method, so the prompt matches
// exactly what the browser sends (and so we don't hand-roll ChatML markers).
const messages: ChatMessage[] = [{ role: 'user', content: PROMPT }];
const prompt = BPETokenizer.chatTemplate(messages, true);
const ids = tok.encode(prompt, true);

function runForward(tensors: Tensors) {
  const cache: KVCache = createKVCache(CONFIG, 256);
  const logits = prefill(tensors, CONFIG, ids, cache);
  const gen = decodeFrom(tensors, CONFIG, tok, logits, cache, {
    maxTokens: 12, temperature: 0, topK: 1,
  });
  return { logits, text: tok.decode(gen), gen };
}

function argmax(x: Float32Array) {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

const refRun = runForward(ref);
const artRun = runForward(art);
const refTop1 = argmax(refRun.logits);
const artTop1 = argmax(artRun.logits);
const delta = Math.abs(refRun.logits[refTop1] - artRun.logits[artTop1]);

console.log(`
f32 model    : argmax ${refTop1} (${tok.decode([refTop1])})  text ${JSON.stringify(refRun.text.slice(0, 60))}`);
console.log(`int8 loader  : argmax ${artTop1} (${tok.decode([artTop1])})  text ${JSON.stringify(artRun.text.slice(0, 60))}`);
console.log(`top-1 matches : ${refTop1 === artTop1}   |logit delta|: ${delta.toExponential(3)}`);

const ok = refTop1 === artTop1 && refRun.text === artRun.text;
console.log(`
VERDICT: ${ok ? 'PASS' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
