#!/usr/bin/env node
/**
 * Run every check in order and report a single verdict.
 *
 *   1. typecheck          tsc --noEmit
 *   2. kernels            compile via the native compiler
 *   3. wgsl validation    naga parse + validate of every generated shader
 *   4. bridge equivalence browser-compiled WGSL == native WGSL
 *   5. tokenizer          src/tokenizer.ts vs HF `tokenizers` reference ids
 *   6. forward pass       full-sequence, KV-cache and int8 paths vs NumPy
 *   7. backend findings   the two documented limitations still hold
 *
 * Steps that need the naso repo or a built validator are skipped with a clear
 * note rather than failing, so this runs on a machine that only has Node.
 *
 * Usage: node scripts/verify.mjs
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const NASO_REPO = process.env.NASO_REPO ?? '/home/node/naso';
const MODEL_DIR = process.env.NASO_MODEL_DIR ?? '/var/tmp/smol';

const results = [];
function step(name, fn) {
  process.stdout.write(`\n=== ${name} ===\n`);
  const r = fn();
  results.push({ name, ...r });
  console.log(r.ok ? `[${name}] OK${r.note ? ` (${r.note})` : ''}` : `[${name}] FAILED`);
  return r.ok;
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { cwd: root, encoding: 'utf8', stdio: 'inherit', ...opts });
  return r.status === 0;
}

// 1. typecheck
step('typecheck', () => ({ ok: run('node', [join(root, 'node_modules', '.bin', 'tsc'), '-p', 'tsconfig.json']), note: 'tsc --noEmit' }));

// 2. compile kernels
if (existsSync(NASO_REPO)) {
  step('kernels', () => ({ ok: run('node', [join(root, 'scripts', 'compile-kernels.js')]), note: NASO_REPO }));
} else {
  console.log(`\n=== kernels ===\nskipped: naso repo not at ${NASO_REPO} (set NASO_REPO)`);
  results.push({ name: 'kernels', ok: true, skipped: true });
}

// 3. wgsl validation with naga
const validator = process.env.WGSL_VALIDATE ?? '/var/tmp/cargo-target-wgsl/release/wgsl-validate';
const kernelDir = join(root, 'public', 'kernels');
if (existsSync(validator) && existsSync(kernelDir)) {
  step('wgsl-validate', () => {
    const shaders = [
      ...readdirSync(kernelDir).filter(f => f.endsWith('.wgsl')).map(f => join(kernelDir, f)),
      ...['matmul.wgsl', 'matmul_i8.wgsl'].map(f => join(root, 'public', f)),
    ].filter(existsSync);
    let ok = true;
    for (const f of shaders) {
      const r = spawnSync(validator, [f], { cwd: root, encoding: 'utf8' });
      if (r.status !== 0) { ok = false; console.error(r.stderr); }
      else console.log(`  valid: ${f.replace(root + '/', '')}`);
    }
    return { ok, note: `${shaders.length} shaders via naga` };
  });
} else {
  console.log(`\n=== wgsl-validate ===\nskipped: validator not built at ${validator} (see tools/wgsl-validate)`);
  results.push({ name: 'wgsl-validate', ok: true, skipped: true });
}

// 4. bridge equivalence
step('bridge', () => ({ ok: run('node', [join(root, 'tools', 'verify_bridge.mjs')]), note: 'wasm == native WGSL' }));

// 5. tokenizer vs the HF `tokenizers` reference
const tokFile = join(MODEL_DIR, 'tokenizer.json');
const oracleFile = process.env.NASO_TOK_ORACLE ?? (existsSync('/var/tmp/tok_oracle.json') ? '/var/tmp/tok_oracle.json' : join(root, 'tools', 'reference', 'tok_oracle.json'));
if (existsSync(tokFile) && existsSync(oracleFile)) {
  step('tokenizer', () => ({ ok: run('node', [join(root, 'tools', 'verify_tokenizer.mjs'), tokFile, oracleFile]), note: 'vs HF tokenizers' }));
} else {
  console.log(`\n=== tokenizer ===\nskipped: need ${tokFile} and ${oracleFile}`);
  console.log('  run: python3 tools/tokenizer_oracle.py /var/tmp/smol/tokenizer.json /var/tmp/tok_oracle.json');
  results.push({ name: 'tokenizer', ok: true, skipped: true });
}

// 6. forward pass vs numpy: bundle the TS harness with esbuild, then run it.
if (existsSync(join(MODEL_DIR, 'model.safetensors')) && existsSync(join(MODEL_DIR, 'ref_logits.npy'))) {
  step('forward-vs-numpy', () => {
    const out = join(MODEL_DIR, 'verify_forward.mjs');
    // esbuild is an executable, not a script, so it is spawned directly.
    const esbuild = join(root, 'node_modules', '.bin', 'esbuild');
    const r = spawnSync(esbuild, [
      join(root, 'tools', 'verify_forward.ts'),
      '--bundle', '--platform=node', '--format=esm', `--outfile=${out}`,
    ], { cwd: root, encoding: 'utf8', stdio: 'inherit' });
    if (r.status !== 0) return { ok: false, note: 'esbuild bundle failed' };
    return { ok: run('node', [out, MODEL_DIR]), note: MODEL_DIR };
  });
} else {
  console.log(`\n=== forward-vs-numpy ===\nskipped: no checkpoint/reference in ${MODEL_DIR}.`);
  console.log(`  run: python3 tools/reference_llama.py ${MODEL_DIR}`);
  results.push({ name: 'forward-vs-numpy', ok: true, skipped: true });
}

// 7. backend findings
step('backend-findings', () => ({ ok: run('node', [join(root, 'tools', 'probe-backend.mjs')]), note: 'documented limitations still hold' }));

// Nothing in the chat dropdown may be a random/untrained test fixture.
step('chat-models', () => ({ ok: run('node', [join(root, 'scripts', 'verify-models.mjs')]), note: 'only real instruct models selectable' }));

// 8. safetensors decode equivalence (bf16/f16 bit-identical over the full domain)
step('decode', () => ({ ok: run('node', [join(root, 'scripts', 'verify-decode.mjs')]), note: 'bit-identical over 2^16' }));


// 9. end-to-end chat in a real browser, against a LOCAL model mirror.
//     Hermetic by design: hitting Hugging Face here would burn its 3000-req/5min
//     resolve budget and make the suite flaky. Skipped when no mirror is
//     running (start one with: npm run modelserve).
const server = process.env.NASO_SMOKE_URL ?? 'http://localhost:3000/';
const mirror = process.env.NASO_MODEL_BASE ?? 'http://127.0.0.1:8777';
const mirrorUp = (() => {
  try { return spawnSync('curl', ['-s', '-o', '/dev/null', '-w', '%{http_code}', `${mirror}/config.json`],
    { encoding: 'utf8' }).stdout?.trim() === '200'; } catch { return false; }
})();
const appUp = (() => {
  try { return spawnSync('curl', ['-s', '-o', '/dev/null', '-w', '%{http_code}', server],
    { encoding: 'utf8' }).stdout?.trim() === '200'; } catch { return false; }
})();

if (mirrorUp && appUp) {
  step('e2e-chat', () => {
    const r = spawnSync('node', [join(root, 'scripts', 'e2e-chat.mjs'), server, mirror],
      { cwd: root, encoding: 'utf8' });
    process.stdout.write(r.stdout ?? '');
    if (r.status !== 0) process.stdout.write(r.stderr ?? '');
    return { ok: r.status === 0, note: 'real browser, 2 turns + revisit' };
  });
} else {
  console.log(`\n=== e2e-chat ===\nskipped: needs the dev server and a local model mirror.`);
  console.log(`  start: npm run dev   and   npm run modelserve`);
  results.push({ name: 'e2e-chat', ok: true, skipped: true });
}

const failed = results.filter(r => !r.ok);
console.log('\n' + '─'.repeat(60));
for (const r of results) console.log(`  ${r.ok ? 'ok  ' : 'FAIL'} ${r.name}${r.skipped ? ' (skipped)' : ''}${r.note ? `  ${r.note}` : ''}`);
console.log('─'.repeat(60));
console.log(failed.length === 0 ? 'VERIFY: PASS' : `VERIFY: FAIL (${failed.length})`);
process.exit(failed.length === 0 ? 0 : 1);
