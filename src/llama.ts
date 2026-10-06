import { type ModelConfig } from './types.js';
import { type Tensor } from './model.js';

/**
 * A complete, honest Llama forward pass.
 *
 * This is the reference implementation: plain f32, no GPU, no shortcuts. Its
 * only job is to be CORRECT, so that the quantized GPU path (pipeline.ts) can be
 * checked against it. A demo that only ran the GPU path could report a wrong
 * tensor and look fine; running both and comparing is what makes the numerical
 * claim checkable.
 *
 * Supported architecture (LlamaForCausalLM): RMSNorm, fused QKV projections,
 * rotary position embedding, grouped-query attention, SwiGLU MLP, and a tied or
 * untied LM head. Multi-token input with a full causal mask is supported -- no
 * KV cache, so this is exactly one forward pass over the prompt.
 */

export interface Tensors {
  [name: string]: Tensor;
}

function get(t: Tensors, name: string): Float32Array {
  const tensor = t[name];
  if (!tensor) throw new Error(`missing tensor "${name}"`);
  return tensor.data;
}

/** out[o] = sum_j W[o, inDim] * x[j], with W row-major [outDim, inDim]. */
export function linear(W: Float32Array, x: Float32Array, outDim: number, inDim: number, out: Float32Array): void {
  for (let o = 0; o < outDim; o++) {
    let acc = 0;
    const row = o * inDim;
    for (let j = 0; j < inDim; j++) acc += W[row + j] * x[j];
    out[o] = acc;
  }
}

/** RMSNorm: x / sqrt(mean(x^2) + eps) * weight. */
export function rmsNorm(x: Float32Array, weight: Float32Array, eps: number, out: Float32Array): void {
  const n = x.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += x[i] * x[i];
  const inv = 1 / Math.sqrt(sum / n + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * weight[i];
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

/**
 * Rotary position embedding, applied in place to a [seq, heads, headDim] tensor.
 * `start` is the absolute position of the first row, so a KV cache could be
 * appended to later without recomputing earlier positions.
 */
export function applyRope(
  x: Float32Array, seq: number, heads: number, headDim: number, start: number, theta: number,
): void {
  const half = headDim / 2;
  for (let s = 0; s < seq; s++) {
    const pos = start + s;
    for (let h = 0; h < heads; h++) {
      const base = (s * heads + h) * headDim;
      for (let i = 0; i < half; i++) {
        const freq = 1 / Math.pow(theta, (2 * i) / headDim);
        const angle = pos * freq;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const a = x[base + i];
        const b = x[base + i + half];
        x[base + i] = a * cos - b * sin;
        x[base + i + half] = a * sin + b * cos;
      }
    }
  }
}

export interface ForwardResult {
  /** [seq, vocabSize] next-token logits, flattened. */
  logits: Float32Array;
  /** Final hidden state after the output norm, [seq, hidden]. */
  hidden: Float32Array;
}

/**
 * One forward pass over `inputIds` (length T). Returns logits for every row; the
 * last row's argmax is the next token.
 */
export function forward(t: Tensors, config: ModelConfig, inputIds: number[]): ForwardResult {
  const { hiddenSize: H, intermediateSize: I, numLayers: L, numHeads: NH, numKvHeads: NKV, rmsNormEps: eps, ropeTheta } = config;
  const T = inputIds.length;
  const headDim = H / NH;
  const kvDim = NKV * headDim;
  const kvRepeat = NH / NKV;

  const embed = get(t, 'model.embed_tokens.weight');

  // h: [T, H]
  let h = new Float32Array(T * H);
  for (let s = 0; s < T; s++) {
    const id = inputIds[s];
    embed.subarray(id * H, id * H + H).forEach((v, i) => { h[s * H + i] = v; });
  }

  let normed = new Float32Array(T * H);
  const q = new Float32Array(T * H);
  const k = new Float32Array(T * kvDim);
  const v = new Float32Array(T * kvDim);
  const attn = new Float32Array(T * H);
  const proj = new Float32Array(T * H);
  const gate = new Float32Array(T * I);
  const up = new Float32Array(T * I);
  const mlp = new Float32Array(T * I);
  const down = new Float32Array(T * H);

  for (let layer = 0; layer < L; layer++) {
    const p = `model.layers.${layer}`;
    const Wq = get(t, `${p}.self_attn.q_proj.weight`);
    const Wk = get(t, `${p}.self_attn.k_proj.weight`);
    const Wv = get(t, `${p}.self_attn.v_proj.weight`);
    const Wo = get(t, `${p}.self_attn.o_proj.weight`);
    const Wgate = get(t, `${p}.mlp.gate_proj.weight`);
    const Wup = get(t, `${p}.mlp.up_proj.weight`);
    const Wdown = get(t, `${p}.mlp.down_proj.weight`);
    const ln1 = get(t, `${p}.input_layernorm.weight`);
    const ln2 = get(t, `${p}.post_attention_layernorm.weight`);

    // --- attention block ---
    const residual = h; // keep for the skip connection
    for (let s = 0; s < T; s++) {
      rmsNorm(residual.subarray(s * H, s * H + H), ln1, eps, normed.subarray(s * H, s * H + H));
      linear(Wq, normed.subarray(s * H, s * H + H), H, H, q.subarray(s * H, s * H + H));
      linear(Wk, normed.subarray(s * H, s * H + H), kvDim, H, k.subarray(s * kvDim, s * kvDim + kvDim));
      linear(Wv, normed.subarray(s * H, s * H + H), kvDim, H, v.subarray(s * kvDim, s * kvDim + kvDim));
    }

    applyRope(q, T, NH, headDim, 0, ropeTheta);
    applyRope(k, T, NKV, headDim, 0, ropeTheta);

    // Causal attention; no KV cache, so scores are computed for the full prompt.
    const scale = 1 / Math.sqrt(headDim);
    for (let s = 0; s < T; s++) {
      for (let head = 0; head < NH; head++) {
        const kvHead = Math.floor(head / kvRepeat);
        // scores over positions 0..s
        const scores = new Float32Array(s + 1);
        let maxScore = -Infinity;
        for (let j = 0; j <= s; j++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += q[(s * NH + head) * headDim + d] * k[(j * NKV + kvHead) * headDim + d];
          }
          scores[j] = dot * scale;
          if (scores[j] > maxScore) maxScore = scores[j];
        }
        let sum = 0;
        for (let j = 0; j <= s; j++) { scores[j] = Math.exp(scores[j] - maxScore); sum += scores[j]; }
        for (let d = 0; d < headDim; d++) {
          let acc = 0;
          for (let j = 0; j <= s; j++) acc += (scores[j] / sum) * v[(j * NKV + kvHead) * headDim + d];
          attn[(s * NH + head) * headDim + d] = acc;
        }
      }
    }

    for (let s = 0; s < T; s++) {
      linear(Wo, attn.subarray(s * H, s * H + H), H, H, proj.subarray(s * H, s * H + H));
    }
    // h = residual + proj
    h = new Float32Array(T * H);
    for (let i = 0; i < T * H; i++) h[i] = residual[i] + proj[i];

    // --- MLP block ---
    const residual2 = h;
    for (let s = 0; s < T; s++) {
      rmsNorm(residual2.subarray(s * H, s * H + H), ln2, eps, normed.subarray(s * H, s * H + H));
      linear(Wgate, normed.subarray(s * H, s * H + H), I, H, gate.subarray(s * I, s * I + I));
      linear(Wup, normed.subarray(s * H, s * H + H), I, H, up.subarray(s * I, s * I + I));
    }
    for (let i = 0; i < T * I; i++) mlp[i] = silu(gate[i]) * up[i];
    for (let s = 0; s < T; s++) {
      linear(Wdown, mlp.subarray(s * I, s * I + I), H, I, down.subarray(s * H, s * H + H));
    }
    h = new Float32Array(T * H);
    for (let i = 0; i < T * H; i++) h[i] = residual2[i] + down[i];
  }

  // --- final norm + LM head ---
  const finalNorm = get(t, 'model.norm.weight');
  const hidden = new Float32Array(T * H);
  for (let s = 0; s < T; s++) {
    rmsNorm(h.subarray(s * H, s * H + H), finalNorm, eps, hidden.subarray(s * H, s * H + H));
  }

  // lm_head may be tied to the embedding (no separate weight).
  const lmHead = t['lm_head.weight'] ? get(t, 'lm_head.weight') : embed;
  const logits = new Float32Array(T * config.vocabSize);
  for (let s = 0; s < T; s++) {
    linear(lmHead, hidden.subarray(s * H, s * H + H), config.vocabSize, H, logits.subarray(s * config.vocabSize, (s + 1) * config.vocabSize));
  }

  return { logits, hidden };
}

/** Largest finite value, for a sanity check (NaN detection). */
export function maxAbs(x: Float32Array): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i]);
    if (a > m) m = a;
  }
  return m;
}

/** Mean squared error between two equal-length arrays. */
export function mse(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`mse: length mismatch ${a.length} vs ${b.length}`);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s / a.length;
}

export { get as getTensor };
