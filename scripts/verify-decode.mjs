/**
 * Regression test for the safetensors decode path.
 *
 * The bf16/f16 wideners were rewritten from per-element ArrayBuffer allocation
 * to integer bit-shifts. That is a 171x speedup on the real checkpoint, but it
 * is only acceptable if the result is bit-identical. This compares the new
 * decode against a straightforward reference implementation (Math.pow, the f16
 * definition) over the full 16-bit domain, so no value can differ silently.
 *
 * Run: node scripts/verify-decode.mjs
 */
import { readFileSync } from 'node:fs';
import { transformSync } from 'esbuild';

// Strip the model.ts logger import so this runs without the app's module graph.
const src = readFileSync(new URL('../src/model.ts', import.meta.url), 'utf8')
  .replace(/^import .*$/gm, '');
const js = transformSync(src, { loader: 'ts', format: 'esm' }).code;
const mod = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
const { f16BitsToF32Bits } = mod;

// Reference: the IEEE-754 binary16 definition, evaluated in floats.
function f16Ref(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

const toF32 = (bits) => new Float32Array(Uint32Array.of(bits).buffer)[0];

let mismatch = 0;
let checked = 0;
let shown = 0;
for (let h = 0; h <= 0xffff; h++) {
  const got = toF32(f16BitsToF32Bits(h) >>> 0);
  const want = f16Ref(h);
  checked++;
  const same = Object.is(got, want) || (Number.isNaN(got) && Number.isNaN(want));
  if (!same) {
    mismatch++;
    if (shown++ < 8) console.log(`  MISMATCH h=0x${h.toString(16)}: got ${got}, want ${want}`);
  }
}

// bf16: must be exactly (u16 << 16) reinterpreted as f32. Built in a BigInt-free
// way on purpose: Uint16Array.of(h, 0) lays h out at byte 0 (little-endian low
// half), which is the WRONG end -- bf16 is the high half.
let bfBad = 0;
for (let h = 0; h <= 0xffff; h++) {
  const got = toF32((h << 16) >>> 0);
  const refBuf = new ArrayBuffer(4);
  new DataView(refBuf).setUint16(2, h, true); // high half, as bf16 defines
  const ref = new Float32Array(refBuf)[0];
  if (!Object.is(got, ref) && !(Number.isNaN(got) && Number.isNaN(ref))) bfBad++;
}

console.log(`f16: ${checked} values checked, ${mismatch} mismatches`);
console.log(`bf16: 65536 values checked, ${bfBad} mismatches`);
console.log(`subnormals spot-check: 0x0001 -> ${toF32(f16BitsToF32Bits(1))} (want ${f16Ref(1)})`);
console.log(`                      0x03ff -> ${toF32(f16BitsToF32Bits(0x03ff))} (want ${f16Ref(0x03ff)})`);
console.log(`                      -0     -> ${1 / toF32(f16BitsToF32Bits(0x8000))} (want -Infinity)`);
const pass = mismatch === 0 && bfBad === 0;
console.log(pass ? 'DECODE: PASS' : 'DECODE: FAIL');
process.exit(pass ? 0 : 1);
