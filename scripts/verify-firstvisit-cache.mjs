#!/usr/bin/env node
/**
 * Does the FIRST visit actually persist the checkpoint?
 *
 * This is the reported bug (issue 1): the page worked on the first visit, but
 * the 269 MB checkpoint was never written to Cache Storage, so every later visit
 * re-downloaded it. The service worker was not controlling the page yet when the
 * fetch fired. CDP byte counts for SW-mediated responses can read 0, so this
 * asserts on the CACHE CONTENTS directly, which is the actual claim.
 *
 * Usage: node scripts/verify-firstvisit-cache.mjs [url]
 */
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const url = process.argv.find((a) => a.startsWith('http')) ?? 'http://localhost:3000/';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
const profile = '/var/tmp/firstvisit-profile';
rmSync(profile, { recursive: true, force: true });

const browser = await puppeteer.launch({
  executablePath: chromePath, headless: true, userDataDir: profile,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

const page = await browser.newPage();
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });

const t0 = Date.now();
let state = '';
while (Date.now() - t0 < 120_000) {
  try {
    state = await page.evaluate(() => {
      const d = document.getElementById('dot-model');
      return `${d?.className} | ${document.getElementById('model-state')?.textContent}`;
    }, { timeout: 5000 });
  } catch { state = 'PROBE-FAIL'; }
  if (/dot ok|dot err/.test(state)) break;
  await new Promise((r) => setTimeout(r, 1000));
}

// Wait a moment for the fire-and-forget cache.put to settle.
await new Promise((r) => setTimeout(r, 3000));

const report = await page.evaluate(async () => {
  const out = { caches: [], safetensors: [] };
  for (const name of await caches.keys()) {
    const c = await caches.open(name);
    const keys = await c.keys();
    out.caches.push(`${name}: ${keys.length}`);
    for (const k of keys) {
      if (k.url.includes('safetensors')) {
        const res = await c.match(k);
        const buf = await res.arrayBuffer();
        out.safetensors.push({ cache: name, bytes: buf.byteLength, status: res.status });
      }
    }
  }
  return out;
}, { timeout: 20000 }).catch((e) => ({ error: String(e).slice(0, 120) }));

console.log(`first-visit cache check against ${url}`);
console.log(`  model state : ${state}`);
console.log(`  caches      : ${report.caches?.join('  |  ') ?? report.error}`);
console.log(`  safetensors : ${report.safetensors?.length ? JSON.stringify(report.safetensors) : 'NOT CACHED'}`);

const ok = /dot ok/.test(state)
  && Array.isArray(report.safetensors)
  && report.safetensors.some((s) => s.bytes > 200_000_000);

console.log(ok
  ? 'FIRST-VISIT CACHE: PASS (checkpoint persisted on visit 1)'
  : 'FIRST-VISIT CACHE: FAIL (revisit would re-download)');

await browser.close();
rmSync(profile, { recursive: true, force: true });
process.exit(ok ? 0 : 1);
