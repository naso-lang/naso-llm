/**
 * Multi-turn chat harness. Exercises the same incremental-prefill path the UI
 * uses (only the new text is fed to the cache between turns) so the reply for a
 * follow-up question can be checked without a browser.
 *
 * Usage: npm run chat2
 */
import { readFileSync } from 'node:fs';
import { BPETokenizer } from '../src/tokenizer.js';
import { createKVCache, prefill, decodeFrom } from '../src/generate.js';
import { parseSafetensors } from '../src/model.js';
import { logger } from '../src/logger.js';
import type { ModelConfig } from '../src/types.js';

for (const ch of ['main', 'webgpu', 'naso'] as const) logger.subscribe(ch, () => {});

const DIR = process.env.MODEL_DIR ?? '/var/tmp/smol';
const CT = '<|im_' + 'start|>';
const CTE = '<|im_' + 'end|>';
const SYS = 'You are a helpful AI assistant.';

const c = JSON.parse(readFileSync(`${DIR}/config.json`, 'utf8'));
const config: ModelConfig = {
  id: 'smollm2-135m-instruct', name: 'SmolLM2-135M-Instruct',
  repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
  hiddenSize: c.hidden_size, intermediateSize: c.intermediate_size,
  numLayers: c.num_hidden_layers, numHeads: c.num_attention_heads,
  numKvHeads: c.num_key_value_heads, vocabSize: c.vocab_size,
  rmsNormEps: c.rms_norm_eps, ropeTheta: c.rope_theta,
  weightsBytes: 269060552,
};
const raw = readFileSync(`${DIR}/model.safetensors`);
const t = Object.fromEntries(parseSafetensors(new Uint8Array(raw).buffer));
const tok = BPETokenizer.fromJSON(JSON.parse(readFileSync(`${DIR}/tokenizer.json`, 'utf8')));

const cache = createKVCache(config, 512);
let logits = new Float32Array(0);
const questions = [
  'What is the capital of France?',
  'And what is its population?',
];

for (const q of questions) {
  const t0 = performance.now();
  if (cache.pos === 0) {
    const rendered = BPETokenizer.chatTemplate([], false, SYS);
    logits = prefill(t, config, tok.encode(`${rendered}${CT}user\n`, true), cache);
    logits = prefill(t, config, tok.encode(q, false), cache);
  } else {
    logits = prefill(t, config, tok.encode(`${q}${CTE}\n`, true), cache);
  }
  logits = prefill(t, config, tok.encode(`${CT}assistant\n`, true), cache);
  const prefillMs = performance.now() - t0;

  let reply = '';
  const ids = decodeFrom(t, config, tok, logits, cache, {
    maxTokens: 40, temperature: 0, topK: 1, seed: 1234,
    onToken: (_id, text) => { reply = text; },
  });
  const ms = performance.now() - t0;
  console.log(`\n> ${q}`);
  console.log(`  ${reply}`);
  console.log(`  [${ids.length} tokens; prefill ${prefillMs.toFixed(0)} ms; total ${(ms / 1000).toFixed(1)} s; cache ${cache.pos}/${cache.maxSeq}]`);

  // Close the assistant turn so the next question continues from here.
  prefill(t, config, tok.encode(`${CTE}\n`, true), cache);
}

// A leak check: the cache must only grow by what was actually appended.
console.log(`\nfinal cache position: ${cache.pos}/${cache.maxSeq}`);
