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

/** IEEE-754 binary16 -> f32. */
export function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** bfloat16 -> f32 (bf16 is the top 16 bits of an f32). */
export function bf16ToF32(h: number): number {
  const buf = new ArrayBuffer(4);
  new Uint16Array(buf)[1] = h; // little-endian: high half
  return new Float32Array(buf)[0];
}

function decode(dtype: string, bytes: Uint8Array, elements: number): Float32Array {
  const out = new Float32Array(elements);
  switch (dtype) {
    case 'F32':
      new Uint8Array(out.buffer).set(bytes.subarray(0, elements * 4));
      return out;
    case 'F16': {
      const v = new Uint16Array(bytes.buffer, bytes.byteOffset, elements);
      for (let i = 0; i < elements; i++) out[i] = f16ToF32(v[i]);
      return out;
    }
    case 'BF16': {
      const v = new Uint16Array(bytes.buffer, bytes.byteOffset, elements);
      for (let i = 0; i < elements; i++) out[i] = bf16ToF32(v[i]);
      return out;
    }
    default:
      throw new Error(`unsupported dtype "${dtype}"`);
  }
}

/** Fetch a URL with byte-level progress reporting. */
export async function fetchWithProgress(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);

  const total = Number(res.headers.get('content-length') ?? '0');
  if (!res.body) return res.arrayBuffer();

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

/** Parse a safetensors buffer into named f32 tensors. */
export function parseSafetensors(buffer: ArrayBuffer): Map<string, Tensor> {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
  const header = JSON.parse(headerJson) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;

  const base = 8 + headerLen;
  const tensors = new Map<string, Tensor>();
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [start, end] = meta.data_offsets;
    const elements = meta.shape.reduce((a, b) => a * b, 1);
    const bytes = new Uint8Array(buffer, base + start, end - start);
    tensors.set(name, { dtype: meta.dtype, shape: meta.shape, data: decode(meta.dtype, bytes, elements) });
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
