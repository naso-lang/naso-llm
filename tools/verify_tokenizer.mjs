#!/usr/bin/env node
// Verify the TypeScript BPE tokenizer in src/tokenizer.ts against reference ids
// produced by the real Hugging Face `tokenizers` (Rust) library.
//
// Reference data comes from tools/tokenizer_oracle.py -- run that first:
//   VIRTUAL_ENV=/var/tmp/tokvenv uv pip install tokenizers
//   /var/tmp/tokvenv/bin/python tools/tokenizer_oracle.py \
//       /var/tmp/smol/tokenizer.json /var/tmp/tok_oracle.json
//
// Usage: node tools/verify_tokenizer.mjs <tokenizer.json> <oracle.json>
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const tokPath = process.argv[2] ?? '/var/tmp/smol/tokenizer.json';
const oraclePath = process.argv[3] ?? '/var/tmp/tok_oracle.json';

for (const p of [tokPath, oraclePath]) {
  if (!existsSync(p)) {
    console.error(`missing input: ${p}`);
    process.exit(2);
  }
}

const root = new URL('..', import.meta.url).pathname;
const scratch = mkdtempSync(join(tmpdir(), 'tokverify-'));
const entry = join(scratch, 'entry.ts');
const outfile = join(scratch, 'bundle.mjs');

// A tiny entry point: read the tokenizer + oracle from disk, run the port, and
// print a JSON verdict. Bundling with esbuild means we exercise the real
// module the browser loads, not a re-implementation.
writeFileSync(
  entry,
  `
import { readFileSync } from 'node:fs';
import { BPETokenizer } from ${JSON.stringify(join(root, 'src/tokenizer.ts'))};

const tok = BPETokenizer.fromJSON(JSON.parse(readFileSync(${JSON.stringify(tokPath)}, 'utf8')));
const oracle = JSON.parse(readFileSync(${JSON.stringify(oraclePath)}, 'utf8'));

const failures = [];
let checked = 0;
for (const c of oracle.cases) {
  const mine = tok.encode(c.text, c.allowSpecial === true);
  const want = c.ids;
  checked++;
  if (JSON.stringify(mine) !== JSON.stringify(want)) {
    failures.push({ text: c.text, want, got: mine });
    continue;
  }
  if (c.decoded !== undefined) {
    const back = tok.decode(mine);
    if (back !== c.decoded) failures.push({ text: c.text, decodeWant: c.decoded, decodeGot: back });
  }
}
console.log(JSON.stringify({
  checked, failures: failures.slice(0, 10), nFailures: failures.length,
  eosId: tok.eosId, imStartId: tok.imStartId, size: tok.size,
}));
`,
);

const esbuild = join(root, 'node_modules/.bin/esbuild');
const build = spawnSync(
  esbuild,
  [entry, '--bundle', '--format=esm', '--platform=node', `--outfile=${outfile}`],
  { encoding: 'utf8' },
);
if (build.status !== 0) {
  console.error(build.stderr || build.stdout);
  process.exit(2);
}

const run = spawnSync('node', [outfile], { encoding: 'utf8' });
rmSync(scratch, { recursive: true, force: true });
if (run.status !== 0) {
  console.error(run.stderr || run.stdout);
  process.exit(2);
}

const r = JSON.parse(run.stdout.trim().split('\n').pop());
if (r.nFailures === 0) {
  console.log(`tokenizer: OK  ${r.checked}/${r.checked} cases match the Rust tokenizers reference`);
  console.log(`  vocab=${r.size}  eosId=${r.eosId}  imStartId=${r.imStartId}`);
  process.exit(0);
}
console.error(`tokenizer: FAIL  ${r.nFailures}/${r.checked} mismatched`);
for (const f of r.failures) {
  console.error(`  ${JSON.stringify(f.text)}`);
  console.error(`    want ${JSON.stringify(f.want ?? f.decodeWant)}`);
  console.error(`    got  ${JSON.stringify(f.got ?? f.decodeGot)}`);
}
process.exit(1);
