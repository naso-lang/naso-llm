#!/usr/bin/env node
// Bundle tools/chat_demo.ts with esbuild and run it. The browser runs the same
// modules; this is just a headless entry point.
// Usage: node scripts/chat.mjs [--tokens N] [--temp T] [--topk K] [--seed S] "question"
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const scratch = mkdtempSync(join(tmpdir(), 'chat-'));
const outfile = join(scratch, 'chat.mjs');

const build = spawnSync(
  join(root, 'node_modules/.bin/esbuild'),
  [join(root, 'tools/chat_demo.ts'), '--bundle', '--format=esm', '--platform=node', `--outfile=${outfile}`],
  { encoding: 'utf8' },
);
if (build.status !== 0) {
  console.error(build.stderr || build.stdout);
  process.exit(2);
}

const run = spawnSync('node', [outfile, ...process.argv.slice(2)], { encoding: 'utf8', stdio: 'inherit' });
rmSync(scratch, { recursive: true, force: true });
process.exit(run.status ?? 1);
