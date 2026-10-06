#!/usr/bin/env node
/**
 * Guard: every model in the chat dropdown must be a real instruct checkpoint.
 *
 * This exists because of a reported bug. The app once defaulted to
 * `hf-internal-testing/tiny-random-LlamaForCausalLM` -- a tiny complete Llama
 * used as a kernel fixture, which has RANDOM weights. Selecting it (or hitting
 * that build through a stale service-worker cache) produced token soup like
 * "pelculasumssinglemehrere...userloyamsod...rodu surfaceosaur". The output
 * looked like a broken quantiser; it was an untrained model answering politely.
 *
 * The check imports the SAME MODELS array the app uses, so it cannot drift from
 * it. Teeth: re-adding a test fixture to MODELS fails this test.
 *
 * Usage: node scripts/verify-models.mjs
 */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

// Bundle the real module so we read the app's actual list, not a copy.
const out = join(mkdtempSync(join(tmpdir(), 'naso-models-')), 'models.mjs');
writeFileSync(join(root, 'src', '_models_probe.ts'),
  `import { MODELS, DEFAULT_MODEL } from './types.js';\nconsole.log(JSON.stringify({ models: MODELS, defaultId: DEFAULT_MODEL.id }));\n`);
const bundle = spawnSync(join(root, 'node_modules', '.bin', 'esbuild'),
  [join(root, 'src', '_models_probe.ts'), '--bundle', '--platform=node', '--format=esm',
   `--outfile=${out}`, '--log-level=error'], { cwd: root, encoding: 'utf8' });
rmSync(join(root, 'src', '_models_probe.ts'), { force: true });
if (bundle.status !== 0) {
  console.error(bundle.stderr);
  console.error('verify-models: could not bundle src/types.ts');
  process.exit(1);
}

const runOut = spawnSync(process.execPath, [out], { encoding: 'utf8' });
rmSync(dirname(out), { recursive: true, force: true });
if (runOut.status !== 0) {
  console.error(runOut.stderr);
  process.exit(1);
}
const { models, defaultId } = JSON.parse(runOut.stdout.trim());

console.log(`chat dropdown lists ${models.length} model(s):`);
for (const m of models) console.log(`  - ${m.id}  ${m.repo}  (${(m.weightsBytes / 1e6).toFixed(1)} MB)${m.id === defaultId ? '  [default]' : ''}`);

// Random/untrained test fixtures must never be chat-selectable. Match on the
// repo path, which is what identifies them.
const FIXTURE = /(tiny-random|random.*CausalLM|hf-internal-testing)/i;
const violations = models.filter((m) => FIXTURE.test(m.repo) || FIXTURE.test(m.id));

let ok = true;
if (violations.length) {
  ok = false;
  for (const v of violations) {
    console.error(`FAIL: "${v.id}" (${v.repo}) is a random/untrained fixture but is chat-selectable.`);
  }
}
if (!models.some((m) => m.id === defaultId)) {
  ok = false;
  console.error(`FAIL: DEFAULT_MODEL "${defaultId}" is not in MODELS.`);
}
// An instruct chat model needs both a tokenizer and enough weights to be real;
// a few-MB "model" in the chat list is a fixture in disguise.
for (const m of models) {
  if (m.weightsBytes < 50e6) {
    ok = false;
    console.error(`FAIL: "${m.id}" is only ${(m.weightsBytes / 1e6).toFixed(1)} MB -- too small to be an instruct chat model.`);
  }
}

console.log(ok ? 'models-are-chat-capable: OK' : 'models-are-chat-capable: FAILED');
process.exit(ok ? 0 : 1);
