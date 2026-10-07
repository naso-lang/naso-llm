import { type ModelConfig } from './types.js';
import { linear, rmsNorm, silu, applyRope, getTensor, type Tensors } from './llama.js';

/**
 * Incremental (KV-cache) Llama forward pass and a sampling generation loop.
 *
 * `llama.ts` is the honest reference: it runs one full pass over the whole
 * prompt. That is correct but re-computes every earlier position on every new
 * token, which is O(T^2) per token and far too slow for a chat demo. This module
 * keeps the keys and values of every past position, so each new token costs one
 * position's work and attention reads the cache.
 *
 * The two paths must agree exactly on the logits for the same prefix.
 * `tools/verify_forward.ts` asserts that: it runs the full-sequence reference and
 * this incremental path over the same prompt and diffs the logits, so a cache
 * bug (wrong stride, stale position, missing RoPE rotation) cannot hide behind a
 * plausible-looking reply.
 */

export interface KVCache {
  readonly maxSeq: number;
  /** Number of tokens already written. Also the position of the NEXT token. */
  pos: number;
  /** Per layer, [maxSeq * kvDim] keys. */
  readonly k: Float32Array[];
  /** Per layer, [maxSeq * kvDim] values. */
  readonly v: Float32Array[];
}

/** Allocate an empty cache for `maxSeq` tokens. */
export function createKVCache(config: ModelConfig, maxSeq: number): KVCache {
  const headDim = config.hiddenSize / config.numHeads;
  const kvDim = config.numKvHeads * headDim;
  const k: Float32Array[] = [];
  const v: Float32Array[] = [];
  for (let l = 0; l < config.numLayers; l++) {
    k.push(new Float32Array(maxSeq * kvDim));
    v.push(new Float32Array(maxSeq * kvDim));
  }
  return { maxSeq, pos: 0, k, v };
}

/** Rewind the cache without reallocating (used between chat turns). */
export function resetKVCache(cache: KVCache): void {
  cache.pos = 0;
}

/**
 * Run one token at `cache.pos` through the model, append its K/V to the cache,
 * and return logits for the next token. Deterministic f32.
 */
export function forwardToken(
  t: Tensors,
  config: ModelConfig,
  tokenId: number,
  cache: KVCache,
  skipLogits = false,
): Float32Array<ArrayBuffer> {
  const {
    hiddenSize: H, intermediateSize: I, numLayers: L, numHeads: NH,
    numKvHeads: NKV, rmsNormEps: eps, ropeTheta,
  } = config;
  const headDim = H / NH;
  const kvDim = NKV * headDim;
  const kvRepeat = NH / NKV;
  const pos = cache.pos;
  if (pos >= cache.maxSeq) throw new Error(`KV cache is full (${cache.maxSeq} tokens)`);

  const embed = getTensor(t, 'model.embed_tokens.weight');
  let h = new Float32Array(H);
  h.set(embed.subarray(tokenId * H, tokenId * H + H));

  const normed = new Float32Array(H);
  const q = new Float32Array(H);
  const k = new Float32Array(kvDim);
  const v = new Float32Array(kvDim);
  const attn = new Float32Array(H);
  const proj = new Float32Array(H);
  const gate = new Float32Array(I);
  const up = new Float32Array(I);
  const mlp = new Float32Array(I);
  const down = new Float32Array(H);

  for (let layer = 0; layer < L; layer++) {
    const p = `model.layers.${layer}`;
    const Wq = getTensor(t, `${p}.self_attn.q_proj.weight`);
    const Wk = getTensor(t, `${p}.self_attn.k_proj.weight`);
    const Wv = getTensor(t, `${p}.self_attn.v_proj.weight`);
    const Wo = getTensor(t, `${p}.self_attn.o_proj.weight`);
    const Wgate = getTensor(t, `${p}.mlp.gate_proj.weight`);
    const Wup = getTensor(t, `${p}.mlp.up_proj.weight`);
    const Wdown = getTensor(t, `${p}.mlp.down_proj.weight`);
    const ln1 = getTensor(t, `${p}.input_layernorm.weight`);
    const ln2 = getTensor(t, `${p}.post_attention_layernorm.weight`);

    const cacheK = cache.k[layer];
    const cacheV = cache.v[layer];

    // --- attention ---
    const residual = h;
    rmsNorm(residual, ln1, eps, normed);
    linear(Wq, normed, H, H, q);
    linear(Wk, normed, kvDim, H, k);
    linear(Wv, normed, kvDim, H, v);

    // RoPE at the ABSOLUTE position -- the whole point of passing `pos` rather
    // than 0, and the bug that a cache-less reference cannot catch.
    applyRope(q, 1, NH, headDim, pos, ropeTheta);
    applyRope(k, 1, NKV, headDim, pos, ropeTheta);

    cacheK.set(k, pos * kvDim);
    cacheV.set(v, pos * kvDim);

    const scale = 1 / Math.sqrt(headDim);
    for (let head = 0; head < NH; head++) {
      const kvHead = Math.floor(head / kvRepeat);
      const scores = new Float32Array(pos + 1);
      let maxScore = -Infinity;
      for (let j = 0; j <= pos; j++) {
        let dot = 0;
        const kBase = j * kvDim + kvHead * headDim;
        const qBase = head * headDim;
        for (let d = 0; d < headDim; d++) dot += q[qBase + d] * cacheK[kBase + d];
        scores[j] = dot * scale;
        if (scores[j] > maxScore) maxScore = scores[j];
      }
      // Stable softmax (max-subtracted). Every position 0..pos is visible, so the
      // causal mask is implicit in the loop bound.
      let sum = 0;
      for (let j = 0; j <= pos; j++) {
        scores[j] = Math.exp(scores[j] - maxScore);
        sum += scores[j];
      }
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let j = 0; j <= pos; j++) acc += (scores[j] / sum) * cacheV[j * kvDim + kvHead * headDim + d];
        attn[head * headDim + d] = acc;
      }
    }

    linear(Wo, attn, H, H, proj);
    h = new Float32Array(H);
    for (let i = 0; i < H; i++) h[i] = residual[i] + proj[i];

    // --- MLP (SwiGLU) ---
    const residual2 = h;
    rmsNorm(residual2, ln2, eps, normed);
    linear(Wgate, normed, I, H, gate);
    linear(Wup, normed, I, H, up);
    for (let i = 0; i < I; i++) mlp[i] = silu(gate[i]) * up[i];
    linear(Wdown, mlp, H, I, down);
    h = new Float32Array(H);
    for (let i = 0; i < H; i++) h[i] = residual2[i] + down[i];
  }

  // --- final norm + tied LM head ---
  // `skipLogits` is for prefill: the caller only needs the K/V entries and the
  // last token's logits, and the LM head is a [vocab x hidden] matvec -- the
  // single most expensive operation in the layer stack. Skipping it for the
  // first T-1 prompt tokens is what makes a long prompt affordable.
  cache.pos = pos + 1;
  if (skipLogits) return new Float32Array(0);

  const finalNorm = getTensor(t, 'model.norm.weight');
  const hidden = new Float32Array(H);
  rmsNorm(h, finalNorm, eps, hidden);

  const lmHead = t['lm_head.weight'] ? getTensor(t, 'lm_head.weight') : embed;
  const logits = new Float32Array(config.vocabSize);
  linear(lmHead, hidden, config.vocabSize, H, logits);

  return logits;
}

/** Minimal tokenizer surface this module needs (avoids a circular import). */
export interface Decoder {
  decode(ids: number[]): string;
  readonly eosId: number;
}

export interface GenerateOptions {
  maxTokens?: number;
  /** 0 (or omitted alongside topK=1) means greedy. */
  temperature?: number;
  topK?: number;
  topP?: number;
  seed?: number;
  /** Called after each token with the id and the decoded text so far. */
  onToken?: (id: number, text: string) => void;
  /** Return true to stop early (a UI stop button). */
  shouldStop?: () => boolean;
}

/** Deterministic PRNG, so a seed reproduces a run exactly. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Greedy argmax. Ties resolve to the lowest id, matching the reference. */
export function argmax(x: Float32Array): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

/**
 * Draw a token from `logits`. `temperature <= 0` with `topK <= 1` is greedy, so
 * the default path is fully deterministic and reproducible.
 */
export function sampleToken(
  logits: Float32Array,
  opts: { temperature?: number; topK?: number; topP?: number; rand?: () => number },
): number {
  const temperature = opts.temperature ?? 0;
  const topK = opts.topK ?? 1;
  if (temperature <= 0 || topK <= 1) return argmax(logits);

  // top-k over (index, score), highest first.
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  let candidates = order.slice(0, Math.max(1, topK));

  if (opts.topP !== undefined && opts.topP > 0 && opts.topP < 1) {
    const maxScore = logits[candidates[0]];
    let cdf = 0;
    const kept: number[] = [];
    for (const id of candidates) {
      cdf += Math.exp((logits[id] - maxScore) / temperature);
      kept.push(id);
      if (cdf >= opts.topP) break;
    }
    candidates = kept;
  }

  const maxScore = logits[candidates[0]];
  const weights = candidates.map((id) => Math.exp((logits[id] - maxScore) / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  const rand = opts.rand ?? mulberry32(0x9e3779b9);
  let r = rand() * total;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i];
    if (r <= 0) return candidates[i];
  }
  return candidates[candidates.length - 1];
}

/**
 * Generate a reply for `promptIds` (already chat-templated). Stops at
 * `decoder.eosId`, `maxTokens`, or a `shouldStop()` signal.
 */
export function generate(
  t: Tensors,
  config: ModelConfig,
  decoder: Decoder,
  promptIds: number[],
  cache: KVCache,
  opts: GenerateOptions = {},
): number[] {
  const maxTokens = opts.maxTokens ?? 128;
  const rand = mulberry32(opts.seed ?? 1234);
  const out: number[] = [];

  // Prefill. Only the LAST prompt token needs the LM head; every earlier token
  // only has to deposit its K/V into the cache.
  let logits: Float32Array<ArrayBuffer> = new Float32Array(config.vocabSize);
  for (let i = 0; i < promptIds.length; i++) {
    logits = forwardToken(t, config, promptIds[i], cache, i < promptIds.length - 1);
  }

  return decodeFrom(t, config, decoder, logits, cache, opts);
}

/**
 * Prefill `ids` into `cache` without generating. Returns the last token's
 * logits (empty if `ids` is empty). Used to extend a conversation: the cache
 * already holds everything processed so far, so only the new text is fed.
 */
export function prefill(
  t: Tensors,
  config: ModelConfig,
  ids: number[],
  cache: KVCache,
): Float32Array {
  let logits = new Float32Array(0);
  for (let i = 0; i < ids.length; i++) {
    logits = forwardToken(t, config, ids[i], cache, i < ids.length - 1);
  }
  return logits;
}

/**
 * Async variant of `prefill` that yields to the event loop after each prompt
 * token, so a long prompt doesn't hard-freeze the page. The browser chat uses
 * this so the "thinking" indicator stays painted and animated while the prompt
 * is encoded into the KV cache, and the page stays responsive (the Stop button
 * works) during the prompt. A microtask (`await Promise.resolve()`) would NOT
 * allow a repaint or event-loop turn; `setTimeout(0)` is a macrotask, so it
 * lets a refresh frame through between prompt tokens. The deterministic CPU
 * reference tests keep the synchronous `prefill`.
 */
export async function prefillAsync(
  t: Tensors,
  config: ModelConfig,
  ids: number[],
  cache: KVCache,
): Promise<Float32Array> {
  let logits = new Float32Array(0);
  for (let i = 0; i < ids.length; i++) {
    logits = forwardToken(t, config, ids[i], cache, i < ids.length - 1);
    await new Promise((r) => setTimeout(r, 0));
  }
  return logits;
}

/**
 * Sample tokens starting from an existing logits vector, appending every
 * generated token to `cache`. This is the decode half of `generate`, exposed so
 * a conversation can continue from an already-populated cache. Synchronous --
 * the deterministic CPU reference tests use this; the browser chat uses
 * `decodeFromAsync` instead so it can yield between tokens.
 */
export function decodeFrom(
  t: Tensors,
  config: ModelConfig,
  decoder: Decoder,
  firstLogits: Float32Array,
  cache: KVCache,
  opts: GenerateOptions = {},
): number[] {
  const maxTokens = opts.maxTokens ?? 128;
  const rand = mulberry32(opts.seed ?? 1234);
  const out: number[] = [];
  let logits = firstLogits;

  for (let step = 0; step < maxTokens; step++) {
    if (opts.shouldStop?.()) break;
    const next = sampleToken(logits, {
      temperature: opts.temperature,
      topK: opts.topK,
      topP: opts.topP,
      rand,
    });
    if (next === decoder.eosId) break;
    out.push(next);
    opts.onToken?.(next, decoder.decode(out));
    if (cache.pos >= cache.maxSeq) break;
    logits = forwardToken(t, config, next, cache);
  }
  return out;
}

/**
 * Async variant of `decodeFrom` that yields to the event loop after each token,
 * so the in-browser chat can repaint the streaming caret / "thinking" indicator
 * between tokens. Behaviour is otherwise IDENTICAL to `decodeFrom` (same sampler,
 * same cache writes, same `onToken` decoded text). `generate.ts` tests keep the
 * synchronous `decodeFrom`; `main.ts` uses this one so the renderer is not starved
 * by the decode loop and the user actually sees tokens stream.
 */
export async function decodeFromAsync(
  t: Tensors,
  config: ModelConfig,
  decoder: Decoder,
  firstLogits: Float32Array,
  cache: KVCache,
  opts: GenerateOptions = {},
): Promise<number[]> {
  const maxTokens = opts.maxTokens ?? 128;
  const rand = mulberry32(opts.seed ?? 1234);
  const out: number[] = [];
  let logits = firstLogits;

  for (let step = 0; step < maxTokens; step++) {
    if (opts.shouldStop?.()) break;
    const next = sampleToken(logits, {
      temperature: opts.temperature,
      topK: opts.topK,
      topP: opts.topP,
      rand,
    });
    if (next === decoder.eosId) break;
    out.push(next);
    opts.onToken?.(next, decoder.decode(out));
    if (cache.pos >= cache.maxSeq) break;
    logits = forwardToken(t, config, next, cache);
    // Yield to a RENDER FRAME (not a microtask): `await Promise.resolve()` only
    // drains the microtask queue, so the renderer never paints between tokens
    // and the caret / streamed text never appear until the whole loop returns.
    // rAF hands control back at the next paint frame so each token is visible.
    await new Promise((r) => requestAnimationFrame(r));
  }
  return out;
}
