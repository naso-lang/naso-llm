/**
 * Weight-only int8 quantisation, and the error bound that makes it a statement
 * rather than a hope.
 *
 * Each row of a weight matrix gets its own symmetric scale:
 *
 *   scale_o = max_j |W[o, j]| / 127
 *   q[o, j] = round(W[o, j] / scale_o)        -- clamped to [-127, 127]
 *   W'[o, j] = q[o, j] * scale_o              -- dequantised
 *
 * so the reconstruction error is bounded by half a quantisation step:
 *
 *   |W - W'| <= scale_o / 2      for every element
 *
 * `maxQuantError()` reports the worst observed value against that bound over a
 * real checkpoint, which is the checkable claim. (It is a property of the
 * rounding, so it holds for any input -- but asserting it on the actual weights
 * is what catches a scale applied to the wrong row, an off-by-one in the
 * packing, or a sign-extension bug.)
 *
 * Relationship to Naso:
 *   * The ELEMENT-WISE step (`round(x / scale)`, clamp, widening back) is what
 *     Naso's `quantize_int8_symmetric` kernel does, and `kernels/` contains the
 *     Naso source and the WGSL Naso emits for it.
 *   * The PER-ROW SCALE is a reduction (a max over a row). Naso's WGSL backend
 *     cannot express a reduction today, so the host computes it. That gap is
 *     exactly what README's "What Naso proves here" section describes, and the
 *     host-side implementation here is the honest placeholder for it.
 */

/**
 * int4 weights packed eight-per-u32 (little-endian nibble order).
 *
 * Same symmetric per-row quantisation as int8, but 4 bits per value:
 *
 *   scale_o = max_j |W[o, j]| / 7
 *   q[o, j] = round(W[o, j] / scale_o)   -- clamped to [-7, 7]
 *   W'[o, j] = q[o, j] * scale_o          -- dequantised
 *
 * Error bound: |W - W'| <= scale_o / 2  (half a step, same as int8).
 * The step is coarser (max/7 vs max/127), so the absolute error is larger,
 * but the packed representation is 8× smaller than f32 and the app verifies
 * that the resulting model still produces identical argmax / text.
 */
export interface QuantizedMatrix4 {
  /** Number of rows (output features). */
  n: number;
  /** Number of columns (input features); must be a multiple of 8. */
  k: number;
  /** n * k / 8 words; nibble j of word o*k/8+w holds q[o][8w+j]. */
  packed: Uint32Array;
  /** Per-row dequantisation scale, length n. */
  scales: Float32Array;
}

/** Compute symmetric per-row int4 scales (max/7). */
export function computeScales4(W: Float32Array, n: number, k: number): Float32Array {
  const scales = new Float32Array(n);
  for (let o = 0; o < n; o++) {
    let max = 0;
    const row = o * k;
    for (let j = 0; j < k; j++) {
      const a = Math.abs(W[row + j]);
      if (a > max) max = a;
    }
    // A zero row must not produce a zero scale (0/0); keep 1 so q stays 0.
    scales[o] = max === 0 ? 1 : max / 7;
  }
  return scales;
}

/** Quantise `W` ([n, k], row-major) to packed int4 with per-row scales. */
export function quantizeRows4(W: Float32Array, n: number, k: number): QuantizedMatrix4 {
  if (k % 8 !== 0) throw new Error(`quantizeRows4: k=${k} must be a multiple of 8 for u32 packing`);
  const scales = computeScales4(W, n, k);
  const kwords = k / 8;
  const packed = new Uint32Array(n * kwords);
  for (let o = 0; o < n; o++) {
    const row = o * k;
    const scale = scales[o];
    const base = o * kwords;
    for (let w = 0; w < kwords; w++) {
      let word = 0;
      for (let i = 0; i < 8; i++) {
        const v = W[row + w * 8 + i] / scale;
        // round-half-away-from-zero, then clamp to the symmetric range.
        let q = Math.round(v);
        if (q > 7) q = 7;
        else if (q < -7) q = -7;
        word |= (q & 0xf) << (i * 4);
      }
      packed[base + w] = word >>> 0;
    }
  }
  return { n, k, packed, scales };
}

/** Read one signed 4-bit value back out of a packed word. */
export function unpackNibble(word: number, i: number): number {
  const b = (word >>> (i * 4)) & 0xf;
  return b < 8 ? b : b - 16;
}

/** Dequantise int4 to f32 (allocates a full [n, k] matrix). For verification. */
export function dequantizeRows4(q: QuantizedMatrix4): Float32Array {
  const out = new Float32Array(q.n * q.k);
  const kwords = q.k / 8;
  for (let o = 0; o < q.n; o++) {
    const scale = q.scales[o];
    const row = o * q.k;
    for (let w = 0; w < kwords; w++) {
      const word = q.packed[o * kwords + w];
      for (let i = 0; i < 8; i++) out[row + w * 8 + i] = unpackNibble(word, i) * scale;
    }
  }
  return out;
}

/** Bytes of a packed int4 matrix (8x smaller than f32 for the weights). */
export function packedBytes4(q: QuantizedMatrix4): number {
  return q.packed.byteLength + q.scales.byteLength;
}

/** int8 weights packed four-per-u32 (little-endian byte order). */
export interface QuantizedMatrix {
  /** Number of rows (output features). */
  n: number;
  /** Number of columns (input features); must be a multiple of 4. */
  k: number;
  /** n * k / 4 words; byte i of word o*k/4+w is q[o][4w+i] as a signed byte. */
  packed: Uint32Array;
  /** Per-row dequantisation scale, length n. */
  scales: Float32Array;
}

/**
 * Compute the symmetric per-row scales. This is the reduction Naso's WGSL
 * backend cannot emit yet; it is pure arithmetic over the weight matrix.
 */
export function computeScales(W: Float32Array, n: number, k: number): Float32Array {
  const scales = new Float32Array(n);
  for (let o = 0; o < n; o++) {
    let max = 0;
    const row = o * k;
    for (let j = 0; j < k; j++) {
      const a = Math.abs(W[row + j]);
      if (a > max) max = a;
    }
    // A zero row must not produce a zero scale (0/0); keep 1 so q stays 0.
    scales[o] = max === 0 ? 1 : max / 127;
  }
  return scales;
}

/** Quantise `W` ([n, k], row-major) to packed int8 with per-row scales. */
export function quantizeRows(W: Float32Array, n: number, k: number): QuantizedMatrix {
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
        // round-half-away-from-zero, then clamp to the symmetric range.
        let q = Math.round(v);
        if (q > 127) q = 127;
        else if (q < -127) q = -127;
        word |= (q & 0xff) << (i * 8);
      }
      packed[base + w] = word >>> 0;
    }
  }
  return { n, k, packed, scales };
}

/** Read one signed byte back out of a packed word. */
export function unpackByte(word: number, i: number): number {
  const b = (word >>> (i * 8)) & 0xff;
  return b < 128 ? b : b - 256;
}

/** Dequantise to f32 (allocates a full [n, k] matrix). For verification. */
export function dequantizeRows(q: QuantizedMatrix): Float32Array {
  const out = new Float32Array(q.n * q.k);
  const kwords = q.k / 4;
  for (let o = 0; o < q.n; o++) {
    const scale = q.scales[o];
    const row = o * q.k;
    for (let w = 0; w < kwords; w++) {
      const word = q.packed[o * kwords + w];
      for (let i = 0; i < 4; i++) out[row + w * 4 + i] = unpackByte(word, i) * scale;
    }
  }
  return out;
}

export interface QuantErrorReport {
  maxError: number;
  maxBound: number;
  worstRow: number;
  violated: number;
  rows: number;
}

/**
 /** Measure the worst |W - W'| for int8 against the scale/2 bound. */
 export function quantError(W: Float32Array, q: QuantizedMatrix): QuantErrorReport {
   let maxError = 0;
   let maxBound = 0;
   let worstRow = -1;
   let violated = 0;

   for (let o = 0; o < q.n; o++) {
     const bound = q.scales[o] / 2;
     let rowViolated = false;
     const row = o * q.k;
     const kwords = q.k / 4;
     for (let w = 0; w < kwords; w++) {
       const word = q.packed[o * kwords + w];
       for (let i = 0; i < 4; i++) {
         const j = w * 4 + i;
         const err = Math.abs(W[row + j] - unpackByte(word, i) * q.scales[o]);
         if (err > maxError) { maxError = err; worstRow = o; }
         if (err > bound) rowViolated = true;
       }
     }
     if (rowViolated) violated++;
   }
   for (let o = 0; o < q.n; o++) maxBound = Math.max(maxBound, q.scales[o] / 2);
   return { maxError, maxBound, worstRow, violated, rows: q.n };
 }

 /** Measure the worst |W - W'| for int4 against the scale/2 bound. */
 export function quantError4(W: Float32Array, q: QuantizedMatrix4): QuantErrorReport {
   let maxError = 0;
   let maxBound = 0;
   let worstRow = -1;
   let violated = 0;

   for (let o = 0; o < q.n; o++) {
     const bound = q.scales[o] / 2;
     let rowViolated = false;
     const row = o * q.k;
     const kwords = q.k / 8;
     for (let w = 0; w < kwords; w++) {
       const word = q.packed[o * kwords + w];
       for (let i = 0; i < 8; i++) {
         const j = w * 8 + i;
         const err = Math.abs(W[row + j] - unpackNibble(word, i) * q.scales[o]);
         if (err > maxError) { maxError = err; worstRow = o; }
         if (err > bound) rowViolated = true;
       }
     }
     if (rowViolated) violated++;
   }
   for (let o = 0; o < q.n; o++) maxBound = Math.max(maxBound, q.scales[o] / 2);
   return { maxError, maxBound, worstRow, violated, rows: q.n };
 }

/** Bytes of a packed matrix (4x smaller than f32 for the weights). */
export function packedBytes(q: QuantizedMatrix): number {
  return q.packed.byteLength + q.scales.byteLength;
}
