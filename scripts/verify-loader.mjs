#!/usr/bin/env node
// Bundle tools/verify_loader.ts (which imports ../src/*.ts) into one node-runnable
// esm file with esbuild, then run it. Mirrors scripts/verify-forward.mjs.
// Usage: node scripts/verify-loader.mjs [modelDir] [npqArtifact] [prompt]
import { spawnSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const modelDir = process.argv[2] ?? process.env.NASO_MODEL_DIR ?? '/var/tmp/smol';
const npq = process.argv[3] ?? '/var/tmp/model.int8.npq';
const prompt = process.argv[4] ?? 'What is the capital of France?';

const out = join(modelDir, 'verify_loader.mjs');
const esbuild = join(root, 'node_modules', '.bin', 'esbuild');
const build = spawnSync(esbuild, [
  join(root, 'tools', 'verify_loader.ts'),
  '--bundle', '--platform=node', '--format=esm', `--outfile=${out}`,
], { cwd: root, stdio: 'inherit' });
if (build.status !== 0) process.exit(build.status ?? 1);

const run = spawnSync('node', [out, modelDir, npq, prompt], {
  cwd: root, stdio: 'inherit',
});
process.exit(run.status ?? 1);
