/**
 * End-to-end chat harness (Node): tokenize a real chat prompt, generate a reply
 * with the KV-cache path, and print it. This is the same code the browser runs,
 * minus the GPU -- it exists so the reply can be eyeballed and its speed
 * measured without a browser.
 *
 * Usage: npm run chat -- "What is the capital of France?"
 *        npm run chat -- --tokens 48 --temp 0 "Tell me a joke"
 */
import { readFileSync } from 'node:fs';
import { BPETokenizer, type ChatMessage } from '../src/tokenizer.js';
import { createKVCache, generate } from '../src/generate.js';
import { parseSafetensors, type Tensor } from '../src/model.js';
import { logger } from '../src/logger.js';
import type { ModelConfig } from '../src/types.js';

for (const ch of ['main', 'webgpu', 'naso'] as const) logger.subscribe(ch, () => {});

const DIR = process.env.MODEL_DIR ?? '/var/tmp/smol';

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

const argv = process.argv.slice(2);
let maxTokens = 40;
let temperature = 0;
let topK = 1;
let seed = 1234;
const words: string[] = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--tokens') maxTokens = Number(argv[++i]);
  else if (argv[i] === '--temp') temperature = Number(argv[++i]);
  else if (argv[i] === '--topk') topK = Number(argv[++i]);
  else if (argv[i] === '--seed') seed = Number(argv[++i]);
  else words.push(argv[i]);
}
const question = words.join(' ') || 'What is the capital of France?';

const config = configFromJson(JSON.parse(readFileSync(`${DIR}/config.json`, 'utf8')));
const raw = readFileSync(`${DIR}/model.safetensors`);
const tensors: Map<string, Tensor> = parseSafetensors(new Uint8Array(raw).buffer);
const t = Object.fromEntries(tensors);
const tok = BPETokenizer.fromJSON(JSON.parse(readFileSync(`${DIR}/tokenizer.json`, 'utf8')));

const messages: ChatMessage[] = [
  { role: 'system', content: 'You are a helpful AI assistant.' },
  { role: 'user', content: question },
];
const prompt = BPETokenizer.chatTemplate(messages, true);
const promptIds = tok.encode(prompt, true);
console.log(`prompt (${promptIds.length} tokens): ${JSON.stringify(prompt)}`);

const cache = createKVCache(config, 512);
const t0 = performance.now();
let render = '';
const out = generate(t, config, tok, promptIds, cache, {
  maxTokens,
  temperature,
  topK,
  seed,
  onToken: (_id, text) => { render = text; },
});
const ms = performance.now() - t0;
const newTokens = out.length;
console.log(`\nreply: ${JSON.stringify(render)}`);
console.log(`\n${newTokens} tokens in ${(ms / 1000).toFixed(2)} s  (${(ms / newTokens).toFixed(0)} ms/token, ${(newTokens / (ms / 1000)).toFixed(1)} tok/s)`);
console.log(`cache.pos = ${cache.pos} / ${cache.maxSeq}`);
