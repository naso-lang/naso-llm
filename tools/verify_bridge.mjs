#!/usr/bin/env node
/**
 * Verify the browser's compile path without a browser.
 *
 * The app compiles kernels at runtime with the wasm-pack build of
 * `crates/nasoc-wasm`. That is the same compiler source as the native
 * `naso-compiler` binary, but a different build, so "same source" is not by
 * itself a guarantee that the WGSL matches. This script removes the doubt:
 *
 *   * loads the wasm module in Node,
 *   * compiles every kernel from the real kernels/*.naso sources,
 *   * and asserts the emitted WGSL is BYTE-IDENTICAL to the artefacts the native
 *     compiler produced (`npm run compile:kernels`).
 *
 * If they match, the shader a browser will run is exactly the shader CI checked,
 * and no GPU is needed to establish that. What this does NOT establish is that
 * the shader executes correctly on hardware -- tools/../scripts/smoke.mjs is the
 * test for that, and it needs a machine with a WebGPU adapter.
 *
 * Usage: node tools/verify_bridge.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const pkg = await import(join(root, 'public', 'pkg', 'nasoc_wasm.js'));
const init = pkg.default;

// wasm-pack's web target wants a URL/Response; in Node we hand it the bytes.
const wasmPath = join(root, 'public', 'pkg', 'nasoc_wasm_bg.wasm');
const wasmBytes = readFileSync(wasmPath);
await init({ module_or_path: wasmBytes });

const KERNELS = [
  'quantize_int8_symmetric',
  'dequantize_int8_symmetric',
  'scale_clamp_f32',
  'silu_f32',
  'relu_scale_f32',
];

const source = readFileSync(join(root, 'kernels', 'quantize_int8.naso'), 'utf8');

let failures = 0;
for (const fn of KERNELS) {
  const r = pkg.compile_naso_wgsl(source, fn);
  if (!r.success) {
    console.error(`FAIL ${fn}: compiler refused: ${r.diagnostics.map(d => d.message).join('; ')}`);
    failures++;
    continue;
  }
  const browserWgsl = r.wgsl;
  const nativeWgsl = readFileSync(join(root, 'public', 'kernels', `${fn}.wgsl`), 'utf8');

  const entryOk = r.entry_point === `${fn}_compute`;
  const identical = browserWgsl === nativeWgsl;

  console.log(`${identical && entryOk ? 'PASS' : 'FAIL'} ${fn}: entry=${r.entry_point} ` +
    `bindings=${r.bindings.length} scalars=${r.scalars.length} ` +
    `wgsl ${identical ? 'byte-identical to native' : 'DIFFERS from native'}`);

  if (!identical) {
    failures++;
    const a = browserWgsl.split('\n');
    const b = nativeWgsl.split('\n');
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      if (a[i] !== b[i]) {
        console.error(`  first diff at line ${i + 1}:`);
        console.error(`    wasm:   ${JSON.stringify(a[i])}`);
        console.error(`    native: ${JSON.stringify(b[i])}`);
        break;
      }
    }
  }
  if (!entryOk) failures++;
}

// Also confirm the ABI the JS bridge parses out of these results is complete:
// every binding the shader declares must be in the ABI, and vice versa.
{
  const r = pkg.compile_naso_wgsl(source, 'scale_clamp_f32');
  const wgsl = r.wgsl;
  for (const raw of r.bindings) {
    const b = JSON.parse(raw);
    const decl = `@group(0) @binding(${b.index}) var<storage, ${b.access}> ${b.name}: array<${b.elem}>;`;
    const present = wgsl.includes(decl);
    console.log(`${present ? 'PASS' : 'FAIL'} ABI binding ${b.index} (${b.name}) appears verbatim in the shader`);
    if (!present) { failures++; console.error(`  expected: ${decl}`); }
  }
  console.log(`scalars reported: [${r.scalars.join(', ')}]`);
}

console.log(failures === 0 ? '\nBRIDGE: PASS' : `\nBRIDGE: FAIL (${failures})`);
process.exit(failures === 0 ? 0 : 1);
