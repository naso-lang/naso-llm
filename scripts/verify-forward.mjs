#!/usr/bin/env node
// Bundle tools/verify_forward.ts with esbuild and run it against a model dir.
// Usage: node scripts/verify-forward.mjs [modelDir]
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const modelDir = process.argv[2] ?? process.env.NASO_MODEL_DIR ?? '/var/tmp/tiny';

if (!existsSync(join(modelDir, 'model.safetensors'))) {
  console.error(`no model.safetensors in ${modelDir}`);
  console.error(`prepare it with: python3 tools/reference_llama.py ${modelDir}`);
  process.exit(2);
}

const out = join(modelDir, 'verify_forward.mjs');
const esbuild = join(root, 'node_modules', '.bin', 'esbuild');
const build = spawnSync(esbuild, [
  join(root, 'tools', 'verify_forward.ts'),
  '--bundle', '--platform=node', '--format=esm', `--outfile=${out}`,
], { cwd: root, stdio: 'inherit' });
if (build.status !== 0) process.exit(build.status ?? 1);

const run = spawnSync('node', [out, modelDir, ...process.argv.slice(3)], { cwd: root, stdio: 'inherit' });
process.exit(run.status ?? 1);
