#!/usr/bin/env node
/**
 * Reproduce the two Naso WGSL backend findings referenced in the README.
 *
 * These are properties of the compiler, not of this app, so they are pinned
 * here as an executable record: if a future compiler fixes either, this script
 * says so instead of leaving a stale claim in the docs.
 *
 * Usage: node tools/probe-backend.mjs [--repo /path/to/naso]
 * Exit 0 if the findings still hold, 1 if the backend has changed (review!).
 */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const repoIdx = process.argv.indexOf('--repo');
const NASO_REPO = repoIdx >= 0 ? process.argv[repoIdx + 1] : process.env.NASO_REPO ?? '/home/node/naso';

const dir = mkdtempSync(join(tmpdir(), 'naso-probe-'));

function compile(source, kernel) {
  const file = join(dir, 'probe.naso');
  writeFileSync(file, source);
  const res = spawnSync(
    'cargo',
    ['run', '--release', '-q', '-p', 'naso-compiler', '--', 'build', '--target', 'wgsl', '--kernel', kernel, file],
    { cwd: NASO_REPO, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024, env: { ...process.env, CARGO_TERM_COLOR: 'never' } },
  );
  return { code: res.status, stdout: res.stdout ?? '', stderr: res.stderr ?? '' };
}

const FINDINGS = [];

// 1. if/else is emitted as `/* unsupported */` with a zero exit code.
{
  const src = `fn probe_k(input: [1] Tensor[f32, 1024], output: inout [1] Tensor[f32, 1024]) {
    forall i in 0..1024 {
        let a = input[i];
        if (a > 0.0) { output[i] = a; } else { output[i] = 0.0; }
    }
}`;
  const r = compile(src, 'probe_k');
  const silentMiscompile = r.code === 0 && r.stdout.includes('/* unsupported */');
  FINDINGS.push({
    name: 'if/else inside a kernel -> `/* unsupported */` at exit 0',
    holds: silentMiscompile,
    detail: silentMiscompile
      ? 'still a silent miscompile: the compiler exits 0 and emits `/* unsupported */`'
      : `changed: exit=${r.code}, unsupported=${r.stdout.includes('/* unsupported */')}`,
  });
}

// 2. Reductions are refused: all tensor bindings must share one extent.
{
  const src = `fn probe_k(input: [*] Tensor[f32, 4], weight: [*] Tensor[f32, 16], output: inout [1] Tensor[f32, 4]) {
    forall i in 0..4 {
        let a = input[0] * weight[0];
        let b = input[1] * weight[1];
        let c = input[2] * weight[2];
        let d = input[3] * weight[3];
        output[i] = a + b + c + d;
    }
}`;
  const r = compile(src, 'probe_k');
  const refused = r.code !== 0 && /differing extents/.test(r.stderr + r.stdout);
  FINDINGS.push({
    name: 'mixed-extent reduction is refused at codegen',
    holds: refused,
    detail: refused
      ? 'still refused with the "differing extents" diagnostic (the honest behaviour)'
      : `changed: exit=${r.code}`,
  });
}

rmSync(dir, { recursive: true, force: true });

let allHold = true;
for (const f of FINDINGS) {
  if (!f.holds) allHold = false;
  console.log(`${f.holds ? 'HOLDS ' : 'CHANGED'} ${f.name}\n        ${f.detail}`);
}
console.log(allHold
  ? '\npass: both findings reproduced; README is accurate'
  : '\nreview: the backend changed; update README "Honest limitations"');
process.exit(allHold ? 0 : 1);
