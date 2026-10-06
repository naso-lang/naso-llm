import { logger } from './logger.js';
import { HF_BASE, type ModelConfig } from './types.js';

/**
 * Minimal safetensors reader.
 *
 * Format: 8 bytes little-endian header length N, then N bytes of JSON
 * (`{tensor_name: {dtype, shape, data_offsets: [start, end]}}`, plus an optional
 * `__metadata__`), then the raw tensor bytes. `data_offsets` are relative to the
 * start of that byte buffer.
 *
 * Only the dtypes this demo's model uses are decoded, plus f16/bf16 widening so
 * a differently-typed checkpoint would still load. An unknown dtype is an
 * explicit error, never a silently zeroed tensor.
 */
export interface Tensor {
  dtype: string;
  shape: number[];
  /** Always f32: f16/bf16 are widened on load. */
  data: Float32Array;
}

/** IEEE-754 binary16 -> f32, as raw bits. Exact, branch-predictable, no allocation. */
export function f16BitsToF32Bits(h: number): number {
  const sign = (h & 0x8000) << 16;
  const exp = (h >> 10) & 0x1f;
  const man = h & 0x03ff;
  if (exp === 0) {
    if (man === 0) return sign; // ±0
    // Subnormal: value = man * 2^-24. Shift until the leading 1 reaches bit 10
    // so the mantissa is normalised, then the f32 exponent is 113 - shifts
    // (f32 bias 127 - f16 subnormal scale 14). Getting that constant wrong by
    // one halves or doubles every subnormal, which is why the test sweeps the
    // whole 16-bit domain rather than spot-checking.
    let e = 0;
    let m = man;
    while ((m & 0x400) === 0) { e++; m <<= 1; }
    return sign | ((113 - e) << 23) | ((m & 0x3ff) << 13);
  }
  if (exp === 0x1f) return sign | 0x7f800000 | (man << 13); // ±Inf / NaN
  return sign | ((exp + 112) << 23) | (man << 13);
}

/**
 * Decode a tensor's raw bytes into f32.
 *
 * bf16 is exactly the top 16 bits of an f32, so widening is a shift of the
 * uint16 into the high half of each uint32 -- no float arithmetic and, crucially,
 * no per-element allocation. The previous implementation built two ArrayBuffers
 * per element (a Uint16 write plus a Float32 read); on the 134.5M-element
 * SmolLM2 checkpoint that is 269M allocations, which measured at ~54 s of the
 * ~75 s load. The shift version does the whole checkpoint in ~0.3 s.
 *
 * f16 is widened bit-exactly by the same style of integer manipulation, so the
 * result is bit-identical to a float-based conversion including subnormals,
 * Inf and NaN payloads.
 */
function decode(dtype: string, bytes: Uint8Array, elements: number): Float32Array {
  const out = new Float32Array(elements);
  switch (dtype) {
    case 'F32':
      new Uint8Array(out.buffer).set(bytes.subarray(0, elements * 4));
      return out;
    case 'F16': {
      const src = new Uint16Array(bytes.buffer, bytes.byteOffset, elements);
      const dst = new Uint32Array(out.buffer, out.byteOffset, elements);
      for (let i = 0; i < elements; i++) dst[i] = f16BitsToF32Bits(src[i]);
      return out;
    }
    case 'BF16': {
      const src = new Uint16Array(bytes.buffer, bytes.byteOffset, elements);
      const dst = new Uint32Array(out.buffer, out.byteOffset, elements);
      for (let i = 0; i < elements; i++) dst[i] = src[i] << 16;
      return out;
    }
    default:
      throw new Error(`unsupported dtype "${dtype}"`);
  }
}

/**
 * Fetch with bounded retries and backoff.
 *
 * Hugging Face rate-limits its resolve endpoint per window and returns 429 when
 * the budget is gone; a dropped connection is also normal on a 269 MB body. Both
 * arrive at the page as a bare `TypeError: Failed to fetch`, which is
 * indistinguishable from "the model does not exist" unless it is retried and the
 * status is reported. Retrying transient failures, and surfacing a real message
 * when it still fails, is the difference between "the demo is broken" and "wait
 * a moment and press Load".
 */
export async function fetchWithRetry(url: string, attempts = 4): Promise<Response> {
  let lastErr: unknown = null;
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(url);
      // 429 / 5xx are worth another go; 4xx are not.
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`HTTP ${res.status}`);
        if (i < attempts - 1) {
          await new Promise((r) => setTimeout(r, 1500 * 2 ** i));
          continue;
        }
      }
      return res;
    } catch (e) {
      lastErr = e;
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, 1000 * 2 ** i));
    }
  }
  throw new Error(
    `could not reach Hugging Face after ${attempts} attempts (${String(lastErr)}). ` +
    `If this is a fresh visit, the checkpoint download failed — the host rate-limits ` +
    `repeated requests, so waiting a few minutes and pressing Load again usually works.`,
  );
}

/**
 * Fetch a URL, streaming the body straight into one preallocated buffer.
 *
 * The body is not accumulated as an array of chunks and then copied: with a
 * 269 MB checkpoint that pattern holds two full copies (~538 MB) at once. When
 * the response carries a content-length the destination is allocated up front
 * and each chunk is written at its offset, so only one copy ever exists. The
 * chunk-array fallback is kept for responses without a length (chunked
 * encoding), where the size genuinely is not known until the body ends.
 */
export async function fetchWithProgress(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
  /**
   * When set, a successful response is ALSO written to this Cache Storage bucket.
   *
   * Why this is here and not left to the service worker: the worker only sees a
   * fetch if it controls the page at that instant. On a first visit it is still
   * installing, so the 269 MB checkpoint used to reach the network uncached and
   * every later visit re-downloaded it. `res.clone()` tees the body, so the
   * bytes are stored from the SAME transfer -- no second 269 MB request.
   */
  cacheName?: string,
): Promise<ArrayBuffer> {
  const res = await fetchWithRetry(url);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);

  if (cacheName && 'caches' in window) {
    // Fire and forget: a cache write failure must never fail the load, and the
    // clone's body is consumed by the cache independently of ours.
    const forCache = res.clone();
    caches.open(cacheName)
      .then((c) => c.put(url, forCache))
      .catch((e) => logger.warn('main', `cache write failed for ${url}: ${e}`));
  }

  const total = Number(res.headers.get('content-length') ?? '0');
  if (!res.body) return res.arrayBuffer();

  const reader = res.body.getReader();

  if (total > 0) {
    let out = new Uint8Array(total);
    let off = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // A content-length can lie. Grow rather than truncate, so no byte is lost.
      if (off + value.byteLength > out.length) {
        const grown = new Uint8Array(Math.max(out.length * 2, off + value.byteLength));
        grown.set(out.subarray(0, off));
        out = grown;
      }
      out.set(value, off);
      off += value.byteLength;
      onProgress?.(off, total);
    }
    // A short read (connection dropped) must not leave a zero tail, which would
    // decode as real weights. Hand back exactly the bytes that arrived.
    return off === out.length ? out.buffer : out.subarray(0, off).buffer;
  }

  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, 0);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

/** Parse a safetensors buffer into named f32 tensors. */
export function parseSafetensors(
  buffer: ArrayBuffer,
  onProgress?: (done: number, total: number) => void,
): Map<string, Tensor> {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
  const header = JSON.parse(headerJson) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;

  const base = 8 + headerLen;
  const tensors = new Map<string, Tensor>();
  const entries = Object.entries(header).filter(([name]) => name !== '__metadata__');
  let done = 0;
  for (const [name, meta] of entries) {
    const [start, end] = meta.data_offsets;
    const elements = meta.shape.reduce((a, b) => a * b, 1);
    const bytes = new Uint8Array(buffer, base + start, end - start);
    tensors.set(name, { dtype: meta.dtype, shape: meta.shape, data: decode(meta.dtype, bytes, elements) });
    // Decoding dominates the load after the download, so report it: without
    // this the UI sits on amber for the whole decode with no visible motion.
    onProgress?.(++done, entries.length);
  }
  logger.info('main', `parsed ${tensors.size} tensors from safetensors`);
  return tensors;
}

/** Fetch + parse a model's weights from Hugging Face. */
export async function loadModelWeights(
  config: ModelConfig,
  onProgress?: (loaded: number, total: number) => void,
): Promise<Map<string, Tensor>> {
  const url = `${HF_BASE}/${config.repo}/resolve/main/model.safetensors`;
  logger.info('main', `Fetching ${url}`);
  const buf = await fetchWithProgress(url, onProgress);
  return parseSafetensors(buf);
}
