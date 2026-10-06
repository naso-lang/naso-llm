#!/usr/bin/env node
/**
 * End-to-end chat check in a real browser, against a LOCAL model mirror.
 *
 * Why a local mirror: the Hugging Face resolve endpoint rate-limits (3000
 * requests / 5 min) and each full test pays 269 MB, so iterating against the
 * real CDN burns the budget and makes tests flaky and non-hermetic. Serving the
 * same files from 127.0.0.1 exercises the identical app code path -- same
 * fetchWithProgress, same safetensors decode, same ChatML prompt, same sampler.
 *
 * Asserts:
 *   1. the model reaches green
 *   2. the first reply is coherent AND correct ("The capital of France is Paris.")
 *   3. a SECOND turn stays coherent (this is what the bare-text prompt broke)
 *   4. the checkpoint is in Cache Storage after visit 1
 *   5. a reload is served from cache (no re-download)
 *
 * Usage: node scripts/e2e-chat.mjs [appUrl] [modelBase]
 */
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const appUrl = process.argv[2] ?? 'http://localhost:3000/';
const modelBase = process.argv[3] ?? 'http://127.0.0.1:8777';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
if (!chromePath) { console.error('no chromium'); process.exit(2); }

const profile = '/var/tmp/e2e-profile';
rmSync(profile, { recursive: true, force: true });

const browser = await puppeteer.launch({
  executablePath: chromePath, headless: true, userDataDir: profile,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ' :: ' + detail : ''}`);
  if (!ok) failures++;
};

async function waitReady(page, ms) {
  const t0 = Date.now();
  let state = '';
  while (Date.now() - t0 < ms) {
    try {
      state = await page.evaluate(() => {
        const d = document.getElementById('dot-model');
        return `${d?.className} | ${document.getElementById('model-state')?.textContent}`;
      }, { timeout: 5000 });
    } catch { state = 'PROBE-FAIL'; }
    if (/dot ok|dot err/.test(state)) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  return state;
}

async function ask(page, text) {
  await page.type('#input', text);
  await page.keyboard.press('Enter');
  // Wait until the send button is re-enabled (generation finished).
  const t0 = Date.now();
  while (Date.now() - t0 < 90_000) {
    const busy = await page.evaluate(() => document.getElementById('send').disabled).catch(() => true);
    if (!busy) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  return page.evaluate(() => [...document.querySelectorAll('#chat .msg.assistant .body')].map((b) => b.textContent));
}

console.log(`e2e chat check: app=${appUrl} model=${modelBase}`);

// Visit 1.
const page = await browser.newPage();
// Redirect HF requests for the checkpoint to the local mirror, so the test is
// hermetic. Everything else in the app is untouched.
await page.setRequestInterception(true);
page.on('request', (req) => {
  const u = req.url();
  const m = u.match(/huggingface\.co\/[^/]+\/[^/]+\/resolve\/main\/(.+)$/);
  if (m) {
    req.continue({ url: `${modelBase}/${m[1]}` });
  } else {
    req.continue();
  }
});

await page.goto(appUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
const state = await waitReady(page, 150_000);
check('model reaches ready', /dot ok/.test(state), state);

const r1 = await ask(page, 'What is the capital of France?');
const a1 = r1[r1.length - 1] ?? '';
check('turn 1 answers correctly', /Paris/i.test(a1), JSON.stringify(a1));
check('turn 1 has no hallucinated turns', !/\n\s*(user|assistant)\s*\n/.test(a1), JSON.stringify(a1));

const r2 = await ask(page, 'And what is its population?');
const a2 = r2[r2.length - 1] ?? '';
check('turn 2 stays coherent', !/\n\s*(user|assistant)\s*\n/.test(a2), JSON.stringify(a2));

// Cache must hold the checkpoint after visit 1.
await new Promise((r) => setTimeout(r, 2500));
const cached = await page.evaluate(async () => {
  for (const n of await caches.keys()) {
    const c = await caches.open(n);
    for (const k of await c.keys()) {
      if (!k.url.includes('safetensors')) continue;
      const res = await c.match(k);
      return { cache: n, bytes: (await res.arrayBuffer()).byteLength };
    }
  }
  return null;
}, { timeout: 20_000 });
check('checkpoint persisted on visit 1', !!cached && cached.bytes > 200_000_000, JSON.stringify(cached));

// Visit 2: reload; the checkpoint must come from cache. Assert by counting
// network requests to the model base during the reload.
let netHits = 0;
const page2 = await browser.newPage();
await page2.setRequestInterception(true);
page2.on('request', (req) => {
  const u = req.url();
  const m = u.match(/huggingface\.co\/[^/]+\/[^/]+\/resolve\/main\/(.+)$/);
  if (m) { netHits++; req.continue({ url: `${modelBase}/${m[1]}` }); }
  else req.continue();
});
await page2.goto(appUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
const state2 = await waitReady(page2, 90_000);
check('reload reaches ready', /dot ok/.test(state2), state2);
check('reload did NOT re-download the checkpoint', netHits === 0, `network hits=${netHits}`);

await browser.close();
rmSync(profile, { recursive: true, force: true });
console.log(failures === 0 ? '\nE2E: PASS' : `\nE2E: FAIL (${failures})`);
process.exit(failures === 0 ? 0 : 1);
