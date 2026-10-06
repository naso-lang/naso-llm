#!/usr/bin/env node
/**
 * Service-worker regression: a cached model must still load on the SECOND visit.
 *
 * The bug this guards against: Hugging Face redirects to a CDN, so a response
 * stored in Cache Storage carries `redirected: true`. Handing that back to a
 * request that did not itself redirect makes the browser reject the fetch with a
 * bare "TypeError: Failed to fetch". The first visit loads (live network, no
 * cache); every visit after it fails. That is invisible to a single-visit test
 * and to the offline test in isolation -- which is why this visits twice, with
 * the same profile, and asserts the model reaches green both times.
 *
 * Usage: node scripts/verify-sw.mjs [url]
 */
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const url = process.argv.find((a) => a.startsWith('http')) ?? 'http://localhost:3000/';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
if (!chromePath) { console.error('no chromium'); process.exit(2); }

const profile = '/var/tmp/sw-verify-profile';
rmSync(profile, { recursive: true, force: true });

const browser = await puppeteer.launch({
  executablePath: chromePath, headless: true, userDataDir: profile,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

async function visit(label, ms) {
  const page = await browser.newPage();
  const failures = [];
  page.on('console', (m) => {
    if (m.type() === 'error' && /Failed to fetch|load failed/.test(m.text())) failures.push(m.text());
  });

  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });

  const t0 = Date.now();
  let state = '';
  while (Date.now() - t0 < ms) {
    try {
      state = await page.evaluate(() => {
        const d = document.getElementById('dot-model');
        return `${d?.className} | ${document.getElementById('model-state')?.textContent}`;
      }, { timeout: 5000 });
    } catch { state = 'POLL-PROBE-FAIL'; }
    if (/dot ok/.test(state) || /dot err/.test(state)) break;
    await new Promise((r) => setTimeout(r, 1000));
  }

  const hitCache = await page.evaluate(async () => {
    const names = await caches.keys();
    const parts = [];
    for (const n of names) {
      const c = await caches.open(n);
      const keys = await c.keys();
      if (keys.length === 0) continue;
      const st = await Promise.all(keys.map(async (k) => (await c.match(k))?.status ?? '?'));
      parts.push(`${n}: ${keys.length} [${st.join(',')}]`);
    }
    return parts.join('  |  ') || '(no caches)';
  }, { timeout: 8000 }).catch((e) => `cache query failed: ${e.message.slice(0, 50)}`);

  // The model response must be cached AND served back with a status, not an
  // opaque/redirected body the browser will refuse on the next visit.
  const modelCached = await page.evaluate(async () => {
    for (const n of await caches.keys()) {
      const c = await caches.open(n);
      for (const k of await c.keys()) {
        if (!k.url.includes('safetensors')) continue;
        const res = await c.match(k);
        return JSON.stringify({ cache: n, status: res?.status, redirected: res?.redirected, type: res?.type });
      }
    }
    return 'not cached';
  }, { timeout: 8000 }).catch((e) => `query failed: ${e.message.slice(0, 40)}`);

  const ok = /dot ok/.test(state);
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}: ${state}`);
  console.log(`         caches: ${hitCache}`);
  console.log(`         safetensors entry: ${modelCached}`);
  await page.close();
  return ok && failures.length === 0;
}

console.log(`service-worker double-visit check against ${url}`);
const a = await visit('visit 1 (cold, fills the cache)', 90_000);
const b = await visit('visit 2 (warm, served from cache)', 60_000);

await browser.close();
rmSync(profile, { recursive: true, force: true });

const pass = a && b;
console.log(pass ? 'SW: PASS (cached model loads on revisit)' : 'SW: FAIL');
process.exit(pass ? 0 : 1);
