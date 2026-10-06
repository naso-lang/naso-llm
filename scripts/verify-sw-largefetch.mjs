#!/usr/bin/env node
/**
 * Service worker must not break a LARGE cross-origin fetch.
 *
 * The bug this guards: the HF branch of the fetch handler did
 * `await cache.put(request, res.clone())`. Cloning and buffering a 269 MB
 * cross-origin body inside the handler made the browser reject the request with
 * `TypeError: Failed to fetch`, while the small config.json/tokenizer.json
 * served by the very same handler succeeded -- an extremely misleading signal
 * that pointed at rate limiting and the network for a long time.
 *
 * `verify-sw.mjs` and `e2e-chat.mjs` both missed it: the first used a local
 * mirror (so the HF branch never ran) and this one used localhost (same). This
 * test registers the app's real service worker, then fetches a large
 * cross-origin URL and asserts the response is readable.
 *
 * Usage: node scripts/verify-sw-largefetch.mjs [appUrl]
 */
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const appUrl = process.argv[2] ?? 'http://localhost:3000/';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
if (!chromePath) { console.error('no chromium'); process.exit(2); }

const profile = '/var/tmp/swlarge-profile';
rmSync(profile, { recursive: true, force: true });

const browser = await puppeteer.launch({
  executablePath: chromePath, headless: true, userDataDir: profile,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const page = await browser.newPage();
await page.goto(appUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });

// Wait for the worker to control the page -- the fetch below must go THROUGH it,
// which is the whole point.
const controlled = await page.evaluate(async () => {
  if (!navigator.serviceWorker.controller) {
    await Promise.race([
      new Promise((r) => navigator.serviceWorker.addEventListener('controllerchange', r, { once: true })),
      new Promise((r) => setTimeout(r, 15000)),
    ]);
  }
  return !!navigator.serviceWorker.controller;
}, { timeout: 20_000 });

console.log(`service-worker large-fetch check against ${appUrl}`);
console.log(`  worker controlling the page: ${controlled}`);

// The real 269 MB checkpoint. This is the size that triggers the bug: a 2 MB
// body passes through the same handler without error, which is exactly why the
// smaller tests missed it. It is one request against HF's 3000/5min resolve
// budget, and the body is drained and discarded (never cached), so it does not
// leave a large cache entry behind.
const result = await page.evaluate(async () => {
  const URL_BIG = 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors';
  try {
    const r = await fetch(URL_BIG);
    if (!r.ok) return { ok: false, why: `HTTP ${r.status}` };
    // Read the first few chunks: a response that fails mid-stream still throws
    // here, and stopping early keeps the test cheap.
    const reader = r.body.getReader();
    let got = 0;
    while (got < 8 * 1024 * 1024) {
      const { done, value } = await reader.read();
      if (done) break;
      got += value?.byteLength ?? 0;
    }
    await reader.cancel();
    return { ok: true, bytesRead: got, type: r.type, declared: Number(r.headers.get('content-length') ?? 0) };
  } catch (e) {
    return { ok: false, why: String(e) };
  }
}, { timeout: 120_000 });

console.log(`  cross-origin checkpoint fetch through the worker: ${JSON.stringify(result)}`);

const ok = controlled && result.ok && result.bytesRead > 1_000_000;
console.log(ok
  ? 'SW LARGE FETCH: PASS (worker does not break a large cross-origin body)'
  : 'SW LARGE FETCH: FAIL');

await browser.close();
rmSync(profile, { recursive: true, force: true });
process.exit(ok ? 0 : 1);
