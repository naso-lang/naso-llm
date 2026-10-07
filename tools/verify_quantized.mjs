// tools/verify_quantized.ts
import { readFileSync } from "node:fs";

// src/logger.ts
var Logger = class {
  logs = /* @__PURE__ */ new Map();
  maxLogs = 500;
  listeners = /* @__PURE__ */ new Map();
  constructor() {
    this.logs.set("main", []);
    this.logs.set("webgpu", []);
    this.logs.set("naso", []);
  }
  log(source, level, message) {
    const entry = {
      timestamp: Date.now(),
      level,
      source,
      message
    };
    const arr = this.logs.get(source) || [];
    arr.push(entry);
    if (arr.length > this.maxLogs) arr.shift();
    this.logs.set(source, arr);
    const listener = this.listeners.get(source);
    if (listener) listener(entry);
    const prefix = `[${source.toUpperCase()}]`;
    switch (level) {
      case "error":
        console.error(prefix, message);
        break;
      case "warn":
        console.warn(prefix, message);
        break;
      case "debug":
        console.debug(prefix, message);
        break;
      default:
        console.log(prefix, message);
    }
  }
  info(source, message) {
    this.log(source, "info", message);
  }
  success(source, message) {
    this.log(source, "success", message);
  }
  warn(source, message) {
    this.log(source, "warn", message);
  }
  error(source, message) {
    this.log(source, "error", message);
  }
  debug(source, message) {
    this.log(source, "debug", message);
  }
  getLogs(source) {
    return this.logs.get(source) || [];
  }
  subscribe(source, callback) {
    this.listeners.set(source, callback);
  }
  unsubscribe(source) {
    this.listeners.delete(source);
  }
  clear(source) {
    this.logs.set(source, []);
  }
};
var logger = new Logger();

// src/types.ts
var MODELS = [
  /**
   * The chat model. 135M params, real instruct tuning (SmolLM2's SFT+DPO mix),
   * ChatML template, tied embeddings. 269 MB in bf16, which is small enough to
   * cache for offline use and small enough to run client-side. GQA (9 query
   * heads, 3 KV heads) and rope_theta 100000 are both non-default, so this
   * exercises the same code paths a larger Llama would.
   */
  {
    id: "smollm2-135m-instruct",
    name: "SmolLM2-135M-Instruct",
    repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
    hiddenSize: 576,
    intermediateSize: 1536,
    numLayers: 30,
    numHeads: 9,
    numKvHeads: 3,
    vocabSize: 49152,
    rmsNormEps: 1e-5,
    ropeTheta: 1e5,
    weightsBytes: 269060552
  }
];
var DEFAULT_MODEL = MODELS[0];

// src/model.ts
function f16BitsToF32Bits(h) {
  const sign = (h & 32768) << 16;
  const exp = h >> 10 & 31;
  const man = h & 1023;
  if (exp === 0) {
    if (man === 0) return sign;
    let e = 0;
    let m = man;
    while ((m & 1024) === 0) {
      e++;
      m <<= 1;
    }
    return sign | 113 - e << 23 | (m & 1023) << 13;
  }
  if (exp === 31) return sign | 2139095040 | man << 13;
  return sign | exp + 112 << 23 | man << 13;
}
function decode(dtype, bytes, elements) {
  const out = new Float32Array(elements);
  switch (dtype) {
    case "F32":
      new Uint8Array(out.buffer).set(bytes.subarray(0, elements * 4));
      return out;
    case "F16": {
      const src = new Uint16Array(bytes.buffer, bytes.byteOffset, elements);
      const dst = new Uint32Array(out.buffer, out.byteOffset, elements);
      for (let i = 0; i < elements; i++) dst[i] = f16BitsToF32Bits(src[i]);
      return out;
    }
    case "BF16": {
      const src = new Uint16Array(bytes.buffer, bytes.byteOffset, elements);
      const dst = new Uint32Array(out.buffer, out.byteOffset, elements);
      for (let i = 0; i < elements; i++) dst[i] = src[i] << 16;
      return out;
    }
    default:
      throw new Error(`unsupported dtype "${dtype}"`);
  }
}
function parseSafetensors(buffer, onProgress) {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
  const header = JSON.parse(headerJson);
  const base = 8 + headerLen;
  const tensors = /* @__PURE__ */ new Map();
  const entries = Object.entries(header).filter(([name]) => name !== "__metadata__");
  let done = 0;
  for (const [name, meta] of entries) {
    const [start, end] = meta.data_offsets;
    const elements = meta.shape.reduce((a, b) => a * b, 1);
    const bytes = new Uint8Array(buffer, base + start, end - start);
    tensors.set(name, { dtype: meta.dtype, shape: meta.shape, data: decode(meta.dtype, bytes, elements) });
    onProgress?.(++done, entries.length);
  }
  logger.info("main", `parsed ${tensors.size} tensors from safetensors`);
  return tensors;
}

// src/quantize.ts
function computeScales(W, n, k) {
  const scales = new Float32Array(n);
  for (let o = 0; o < n; o++) {
    let max = 0;
    const row = o * k;
    for (let j = 0; j < k; j++) {
      const a = Math.abs(W[row + j]);
      if (a > max) max = a;
    }
    scales[o] = max === 0 ? 1 : max / 127;
  }
  return scales;
}
function quantizeRows(W, n, k) {
  if (k % 4 !== 0) throw new Error(`quantizeRows: k=${k} must be a multiple of 4 for u32 packing`);
  const scales = computeScales(W, n, k);
  const kwords = k / 4;
  const packed = new Uint32Array(n * kwords);
  for (let o = 0; o < n; o++) {
    const row = o * k;
    const scale = scales[o];
    const base = o * kwords;
    for (let w = 0; w < kwords; w++) {
      let word = 0;
      for (let i = 0; i < 4; i++) {
        const v = W[row + w * 4 + i] / scale;
        let q = Math.round(v);
        if (q > 127) q = 127;
        else if (q < -127) q = -127;
        word |= (q & 255) << i * 8;
      }
      packed[base + w] = word >>> 0;
    }
  }
  return { n, k, packed, scales };
}
function unpackByte(word, i) {
  const b = word >>> i * 8 & 255;
  return b < 128 ? b : b - 256;
}

// src/llama.ts
function get(t, name) {
  const tensor = t[name];
  if (!tensor) throw new Error(`missing tensor "${name}"`);
  return tensor.data;
}
function linear(W, x, outDim, inDim, out) {
  for (let o = 0; o < outDim; o++) {
    let acc = 0;
    const row = o * inDim;
    for (let j = 0; j < inDim; j++) acc += W[row + j] * x[j];
    out[o] = acc;
  }
}
function rmsNorm(x, weight, eps, out) {
  const n = x.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += x[i] * x[i];
  const inv = 1 / Math.sqrt(sum / n + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * weight[i];
}
function silu(x) {
  return x / (1 + Math.exp(-x));
}
function applyRope(x, seq, heads, headDim, start, theta) {
  const half = headDim / 2;
  for (let s = 0; s < seq; s++) {
    const pos = start + s;
    for (let h = 0; h < heads; h++) {
      const base = (s * heads + h) * headDim;
      for (let i = 0; i < half; i++) {
        const freq = 1 / Math.pow(theta, 2 * i / headDim);
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

// src/generate.ts
function createKVCache(config2, maxSeq) {
  const headDim = config2.hiddenSize / config2.numHeads;
  const kvDim = config2.numKvHeads * headDim;
  const k = [];
  const v = [];
  for (let l = 0; l < config2.numLayers; l++) {
    k.push(new Float32Array(maxSeq * kvDim));
    v.push(new Float32Array(maxSeq * kvDim));
  }
  return { maxSeq, pos: 0, k, v };
}
function forwardToken(t, config2, tokenId, cache, skipLogits = false) {
  const {
    hiddenSize: H,
    intermediateSize: I,
    numLayers: L,
    numHeads: NH,
    numKvHeads: NKV,
    rmsNormEps: eps,
    ropeTheta
  } = config2;
  const headDim = H / NH;
  const kvDim = NKV * headDim;
  const kvRepeat = NH / NKV;
  const pos = cache.pos;
  if (pos >= cache.maxSeq) throw new Error(`KV cache is full (${cache.maxSeq} tokens)`);
  const embed = get(t, "model.embed_tokens.weight");
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
    const Wq = get(t, `${p}.self_attn.q_proj.weight`);
    const Wk = get(t, `${p}.self_attn.k_proj.weight`);
    const Wv = get(t, `${p}.self_attn.v_proj.weight`);
    const Wo = get(t, `${p}.self_attn.o_proj.weight`);
    const Wgate = get(t, `${p}.mlp.gate_proj.weight`);
    const Wup = get(t, `${p}.mlp.up_proj.weight`);
    const Wdown = get(t, `${p}.mlp.down_proj.weight`);
    const ln1 = get(t, `${p}.input_layernorm.weight`);
    const ln2 = get(t, `${p}.post_attention_layernorm.weight`);
    const cacheK = cache.k[layer];
    const cacheV = cache.v[layer];
    const residual = h;
    rmsNorm(residual, ln1, eps, normed);
    linear(Wq, normed, H, H, q);
    linear(Wk, normed, kvDim, H, k);
    linear(Wv, normed, kvDim, H, v);
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
      let sum = 0;
      for (let j = 0; j <= pos; j++) {
        scores[j] = Math.exp(scores[j] - maxScore);
        sum += scores[j];
      }
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let j = 0; j <= pos; j++) acc += scores[j] / sum * cacheV[j * kvDim + kvHead * headDim + d];
        attn[head * headDim + d] = acc;
      }
    }
    linear(Wo, attn, H, H, proj);
    h = new Float32Array(H);
    for (let i = 0; i < H; i++) h[i] = residual[i] + proj[i];
    const residual2 = h;
    rmsNorm(residual2, ln2, eps, normed);
    linear(Wgate, normed, I, H, gate);
    linear(Wup, normed, I, H, up);
    for (let i = 0; i < I; i++) mlp[i] = silu(gate[i]) * up[i];
    linear(Wdown, mlp, H, I, down);
    h = new Float32Array(H);
    for (let i = 0; i < H; i++) h[i] = residual2[i] + down[i];
  }
  cache.pos = pos + 1;
  if (skipLogits) return new Float32Array(0);
  const finalNorm = get(t, "model.norm.weight");
  const hidden = new Float32Array(H);
  rmsNorm(h, finalNorm, eps, hidden);
  const lmHead = t["lm_head.weight"] ? get(t, "lm_head.weight") : embed;
  const logits = new Float32Array(config2.vocabSize);
  linear(lmHead, hidden, config2.vocabSize, H, logits);
  return logits;
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = a + 1831565813 >>> 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function argmax(x) {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}
function sampleToken(logits, opts) {
  const temperature = opts.temperature ?? 0;
  const topK = opts.topK ?? 1;
  if (temperature <= 0 || topK <= 1) return argmax(logits);
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  let candidates = order.slice(0, Math.max(1, topK));
  if (opts.topP !== void 0 && opts.topP > 0 && opts.topP < 1) {
    const maxScore2 = logits[candidates[0]];
    let cdf = 0;
    const kept = [];
    for (const id of candidates) {
      cdf += Math.exp((logits[id] - maxScore2) / temperature);
      kept.push(id);
      if (cdf >= opts.topP) break;
    }
    candidates = kept;
  }
  const maxScore = logits[candidates[0]];
  const weights = candidates.map((id) => Math.exp((logits[id] - maxScore) / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  const rand = opts.rand ?? mulberry32(2654435769);
  let r = rand() * total;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i];
    if (r <= 0) return candidates[i];
  }
  return candidates[candidates.length - 1];
}
function prefill(t, config2, ids2, cache) {
  let logits = new Float32Array(0);
  for (let i = 0; i < ids2.length; i++) {
    logits = forwardToken(t, config2, ids2[i], cache, i < ids2.length - 1);
  }
  return logits;
}
function decodeFrom(t, config2, decoder, firstLogits, cache, opts = {}) {
  const maxTokens = opts.maxTokens ?? 128;
  const rand = mulberry32(opts.seed ?? 1234);
  const out = [];
  let logits = firstLogits;
  for (let step = 0; step < maxTokens; step++) {
    if (opts.shouldStop?.()) break;
    const next = sampleToken(logits, {
      temperature: opts.temperature,
      topK: opts.topK,
      topP: opts.topP,
      rand
    });
    if (next === decoder.eosId) break;
    out.push(next);
    opts.onToken?.(next, decoder.decode(out));
    if (cache.pos >= cache.maxSeq) break;
    logits = forwardToken(t, config2, next, cache);
  }
  return out;
}

// src/tokenizer.ts
var GPT2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
function buildByteMaps() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar = new Array(256);
  const charToByte = /* @__PURE__ */ new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}
var IM_START = "<|im_start|>";
var IM_END = "<|im_end|>";
var BPETokenizer = class _BPETokenizer {
  vocab;
  ranks;
  idToToken;
  byteToChar;
  charToByte;
  /** Special strings (e.g. `<|im_start|>`) matched literally, longest first. */
  specials;
  /** Token closing an assistant turn; generation stops on it. */
  eosId;
  imStartId;
  constructor(json) {
    const { byteToChar, charToByte } = buildByteMaps();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;
    this.vocab = new Map(Object.entries(json.model.vocab));
    const maxId = Math.max(...Object.values(json.model.vocab));
    this.idToToken = new Array(maxId + 1);
    for (const [tok2, id] of this.vocab) this.idToToken[id] = tok2;
    this.ranks = /* @__PURE__ */ new Map();
    json.model.merges.forEach((m, i) => {
      const sp = m.indexOf(" ");
      this.ranks.set(m.slice(0, sp) + " " + m.slice(sp + 1), i);
    });
    const added = json.added_tokens ?? [];
    for (const a of added) {
      this.vocab.set(a.content, a.id);
      this.idToToken[a.id] = a.content;
    }
    this.specials = added.filter((a) => a.special).map((a) => a.content).sort((a, b) => b.length - a.length);
    this.eosId = this.vocab.get(IM_END) ?? this.vocab.get("<|endoftext|>") ?? 2;
    this.imStartId = this.vocab.get(IM_START) ?? 0;
  }
  static fromJSON(json) {
    return new _BPETokenizer(json);
  }
  /** Split into pre-tokens: Digits first, then GPT-2's byte-level rule. */
  preTokenize(text) {
    const digitPieces = [];
    let run = "";
    for (const ch of text) {
      if (ch >= "0" && ch <= "9") {
        if (run) {
          digitPieces.push(run);
          run = "";
        }
        digitPieces.push(ch);
      } else {
        run += ch;
      }
    }
    if (run) digitPieces.push(run);
    const out = [];
    for (const piece of digitPieces) {
      const words = piece.match(GPT2_SPLIT);
      if (words) out.push(...words);
      else if (piece) out.push(piece);
    }
    return out;
  }
  /** Greedy lowest-rank pair merging -- the reference BPE algorithm. */
  bpe(token) {
    let word = [...token];
    if (word.length < 2) return word;
    while (word.length > 1) {
      let bestRank = Infinity;
      let bestIndex = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(word[i] + " " + word[i + 1]);
        if (rank !== void 0 && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex === -1) break;
      const pair = word[bestIndex] + " " + word[bestIndex + 1];
      const merged = [];
      for (let i = 0; i < word.length; ) {
        if (i < word.length - 1 && word[i] + " " + word[i + 1] === pair) {
          merged.push(word[i] + word[i + 1]);
          i += 2;
        } else {
          merged.push(word[i]);
          i += 1;
        }
      }
      word = merged;
    }
    return word;
  }
  /** UTF-8 bytes of `text`, each mapped to its printable byte-level char. */
  toByteLevel(text) {
    const bytes = new TextEncoder().encode(text);
    let out = "";
    for (const b of bytes) out += this.byteToChar[b];
    return out;
  }
  /**
   * Encode `text`. Special tokens (the chat control strings) are matched
   * literally when `allowSpecial` is set, so `<|im_start|>` becomes one id
   * rather than nine byte-level merges.
   */
  encode(text, allowSpecial = false) {
    const ids2 = [];
    const chunks = [];
    if (allowSpecial && this.specials.length) {
      let rest = text;
      while (rest) {
        let hit = -1;
        let hitTok = "";
        for (const s of this.specials) {
          const at = rest.indexOf(s);
          if (at !== -1 && (hit === -1 || at < hit)) {
            hit = at;
            hitTok = s;
          }
        }
        if (hit === -1) {
          chunks.push({ text: rest, special: false });
          break;
        }
        if (hit > 0) chunks.push({ text: rest.slice(0, hit), special: false });
        chunks.push({ text: hitTok, special: true });
        rest = rest.slice(hit + hitTok.length);
      }
    } else {
      chunks.push({ text, special: false });
    }
    for (const chunk of chunks) {
      if (chunk.special) {
        const id = this.vocab.get(chunk.text);
        if (id !== void 0) ids2.push(id);
        continue;
      }
      for (const word of this.preTokenize(chunk.text)) {
        for (const sym of this.bpe(this.toByteLevel(word))) {
          const id = this.vocab.get(sym);
          if (id !== void 0) ids2.push(id);
        }
      }
    }
    return ids2;
  }
  /** Decode ids to text (byte-level chars -> bytes -> UTF-8). */
  decode(ids2) {
    let byteStr = "";
    for (const id of ids2) {
      const tok2 = this.idToToken[id];
      if (tok2 === void 0) continue;
      if (this.specials.includes(tok2)) {
        byteStr += tok2;
        continue;
      }
      for (const ch of tok2) {
        const b = this.charToByte.get(ch);
        if (b !== void 0) byteStr += String.fromCharCode(b);
      }
    }
    const bytes = new Uint8Array(byteStr.length);
    for (let i = 0; i < byteStr.length; i++) bytes[i] = byteStr.charCodeAt(i) & 255;
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  }
  get size() {
    return this.idToToken.length;
  }
  idOf(token) {
    return this.vocab.get(token);
  }
  /**
   * ChatML template (SmolLM2 / Qwen family), matching the model's
   * `tokenizer_config.json`. The `` / `` markers are required, not
   * cosmetic: without them the model does not recognise turn boundaries at all
   * and generates fake `user`/`assistant` lines inside its own reply.
   *
   * `addGenerationPrompt` opens the assistant turn so the model continues from it.
   */
  static chatTemplate(messages, addGenerationPrompt = true, systemPrompt) {
    let out = "";
    const sys = systemPrompt ?? messages.find((m) => m.role === "system")?.content;
    if (sys) out += `${IM_START}system
${sys}${IM_END}
`;
    for (const m of messages) {
      if (m.role === "system") continue;
      out += `${IM_START}${m.role}
${m.content}${IM_END}
`;
    }
    if (addGenerationPrompt) out += `${IM_START}assistant
`;
    return out;
  }
};

// tools/verify_quantized.ts
logger.subscribe("main", () => {
});
logger.subscribe("webgpu", () => {
});
logger.subscribe("naso", () => {
});
var DIR = process.argv[2] ?? "/var/tmp/smol";
var NPQ = process.argv[3] ?? "/var/tmp/model.int8.npq";
function parseNPQ(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "NPQ1") throw new Error(`bad magic ${JSON.stringify(magic)}`);
  const version = dv.getUint32(4, true);
  const count = dv.getUint32(8, true);
  if (version !== 1) throw new Error(`unsupported version ${version}`);
  let off = 12;
  const out = /* @__PURE__ */ new Map();
  for (let i = 0; i < count; i++) {
    const nameLen = dv.getUint32(off, true);
    off += 4;
    const name = new TextDecoder().decode(new Uint8Array(buf, off, nameLen));
    off += nameLen;
    const dtype = dv.getUint8(off);
    off += 1;
    const rank = dv.getUint32(off, true);
    off += 4;
    const dims = [];
    for (let d = 0; d < rank; d++) {
      dims.push(dv.getUint32(off, true));
      off += 4;
    }
    const elems = dims.reduce((a, b) => a * b, 1);
    const e = { name, dtype, dims };
    if (dtype === 1) {
      const [n, k] = dims;
      const words = n * k / 4;
      e.packed = new Uint32Array(buf.slice(off, off + words * 4));
      off += words * 4;
      e.scales = new Float32Array(buf.slice(off, off + n * 4));
      off += n * 4;
    } else if (dtype === 2) {
      e.bf16 = new Uint16Array(buf.slice(off, off + elems * 2));
      off += elems * 2;
    } else {
      e.f32 = new Float32Array(buf.slice(off, off + elems * 4));
      off += elems * 4;
    }
    out.set(name, e);
  }
  return out;
}
function fail(msg) {
  console.error(`FAIL: ${msg}`);
  process.exit(1);
}
var orig = parseSafetensors(new Uint8Array(readFileSync(`${DIR}/model.safetensors`)).buffer);
var art = parseNPQ(new Uint8Array(readFileSync(NPQ)).buffer);
console.log(`original: ${orig.size} tensors   artifact: ${art.size} tensors`);
if (orig.size !== art.size) fail(`tensor count differs`);
var checkedInt8 = 0;
var int8Ok = 0;
var checkedPass = 0;
var passOk = 0;
var mismatched = 0;
var worstRatio = 0;
var violations = 0;
for (const [name, t] of orig) {
  const a = art.get(name);
  if (!a) fail(`artifact missing ${name}`);
  const W = t.data;
  const shapeOk = a.dims.length === t.shape.length && a.dims.every((d, i) => d === t.shape[i]);
  if (!shapeOk) fail(`${name}: shape ${t.shape} vs artifact [${a.dims}]`);
  const n = t.shape.length === 2 ? t.shape[0] : 1;
  const k = t.shape.length === 2 ? t.shape[1] : W.length;
  if (a.dtype === 1) {
    const ref = quantizeRows(W, n, k);
    if (ref.packed.length !== a.packed.length) fail(`${name}: packed length`);
    let thisTensorOk = true;
    for (let i = 0; i < ref.packed.length; i++) {
      if (ref.packed[i] !== a.packed[i]) {
        thisTensorOk = false;
        mismatched++;
        if (mismatched < 4) console.error(`  word mismatch ${name}[${i}]: ref ${ref.packed[i]} vs art ${a.packed[i]}`);
        break;
      }
    }
    for (let i = 0; i < ref.scales.length; i++) {
      if (ref.scales[i] !== a.scales[i]) {
        thisTensorOk = false;
        mismatched++;
        if (mismatched < 4) console.error(`  scale mismatch ${name}[${i}]: ref ${ref.scales[i]} vs art ${a.scales[i]}`);
        break;
      }
    }
    if (thisTensorOk) int8Ok++;
    const kwords = k / 4;
    for (let o = 0; o < n; o++) {
      const bound = a.scales[o] / 2;
      let rowBad = false;
      for (let w = 0; w < kwords; w++) {
        const word = a.packed[o * kwords + w];
        for (let i = 0; i < 4; i++) {
          const err = Math.abs(W[o * k + w * 4 + i] - unpackByte(word, i) * a.scales[o]);
          const ratio = err / Math.max(bound, 1e-30);
          if (ratio > worstRatio) worstRatio = ratio;
          if (err > bound) rowBad = true;
        }
      }
      if (rowBad) violations++;
    }
    checkedInt8++;
  } else if (a.dtype === 2) {
    const f32bits = new Uint32Array(W.buffer, W.byteOffset, W.length);
    for (let i = 0; i < W.length; i++) {
      if (f32bits[i] >>> 16 !== a.bf16[i]) {
        mismatched++;
        console.error(`  bf16 mismatch ${name}[${i}]`);
        break;
      }
    }
    checkedPass++;
    passOk++;
  } else {
    let thisTensorOk = true;
    for (let i = 0; i < W.length; i++) {
      if (W[i] !== a.f32[i]) {
        thisTensorOk = false;
        mismatched++;
        console.error(`  f32 mismatch ${name}[${i}]`);
        break;
      }
    }
    checkedPass++;
    if (thisTensorOk) passOk++;
  }
}
console.log(`int8 tensors bit-identical to quantize.ts : ${int8Ok}/${checkedInt8}`);
console.log(`pass-through tensors exact               : ${passOk}/${checkedPass}`);
console.log(`worst |W-W'|/(scale/2)                    : ${worstRatio.toFixed(6)} (must be <= 1)`);
console.log(`rows violating the bound                 : ${violations}`);
var cfg = JSON.parse(readFileSync(`${DIR}/config.json`, "utf8"));
var config = {
  id: "x",
  name: "x",
  repo: "x",
  hiddenSize: cfg.hidden_size,
  intermediateSize: cfg.intermediate_size,
  numLayers: cfg.num_hidden_layers,
  numHeads: cfg.num_attention_heads,
  numKvHeads: cfg.num_key_value_heads,
  vocabSize: cfg.vocab_size,
  rmsNormEps: cfg.rms_norm_eps,
  ropeTheta: cfg.rope_theta ?? 1e4,
  weightsBytes: 1
};
var tok = BPETokenizer.fromJSON(JSON.parse(readFileSync(`${DIR}/tokenizer.json`, "utf8")));
var IM_START2 = "<|im_start|>";
var IM_END2 = "<|im_end|>";
var prompt = `${IM_START2}user
What is the capital of France?${IM_END2}
${IM_START2}assistant
`;
var ids = tok.encode(prompt, true);
function runForward(tensors) {
  const cache = createKVCache(config, 256);
  const logits = prefill(tensors, config, ids, cache);
  const gen = decodeFrom(tensors, config, tok, logits, cache, { maxTokens: 12, temperature: 0, topK: 1 });
  return { logits, text: tok.decode(gen), gen };
}
function materialise() {
  const out = {};
  for (const [name, a] of art) {
    const data = new Float32Array(a.dims.reduce((x, y) => x * y, 1));
    if (a.dtype === 1) {
      const [n, k] = a.dims;
      const kwords = k / 4;
      for (let o = 0; o < n; o++) for (let w = 0; w < kwords; w++) {
        const word = a.packed[o * kwords + w];
        for (let i = 0; i < 4; i++) data[o * k + w * 4 + i] = unpackByte(word, i) * a.scales[o];
      }
    } else if (a.dtype === 2) {
      const u = new Uint32Array(data.buffer);
      for (let i = 0; i < data.length; i++) u[i] = a.bf16[i] << 16;
    } else {
      data.set(a.f32);
    }
    out[name] = { dtype: "F32", shape: a.dims, data };
  }
  return out;
}
function top5(logits) {
  const n = logits.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => logits[b] - logits[a]).slice(0, 5);
  return idx.map((i) => [i, logits[i]]);
}
var refRun = runForward(Object.fromEntries(orig));
var artRun = runForward(materialise());
var r5 = top5(refRun.logits);
var a5 = top5(artRun.logits);
var sameTop1 = r5[0][0] === a5[0][0];
var maxLogitDelta = Math.max(...a5.map(([i, v], j) => Math.abs(v - r5[j][1])));
console.log(`
f32 model  : argmax ${r5[0][0]} (${tok.decode([r5[0][0]])})  text ${JSON.stringify(refRun.text.slice(0, 60))}`);
console.log(`int8 artifact: argmax ${a5[0][0]} (${tok.decode([a5[0][0]])})  text ${JSON.stringify(artRun.text.slice(0, 60))}`);
console.log(`top-1 matches: ${sameTop1}   max logit |\u0394| across top-5: ${maxLogitDelta.toExponential(3)}`);
var ok = mismatched === 0 && violations === 0 && worstRatio <= 1 && sameTop1 && refRun.text === artRun.text;
console.log(`
VERDICT: ${ok ? "PASS" : "FAIL"}`);
process.exit(ok ? 0 : 1);
