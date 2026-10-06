#!/usr/bin/env node
/**
 * Final live check: does the deployed app cache the checkpoint on visit 1, and
 * is visit 2 served from cache? Uses a fresh profile so nothing is pre-warmed.
 *
 * Usage: node scripts/_live-final.mjs
 */
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const appUrl = 'https://naso-lang.github.io/naso-llm/';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
const profile = '/var/tmp/livefinal-profile';
rmSync(profile, { recursive: true, force: true });

const browser = await puppeteer.launch({
  executablePath: chromePath, headless: true, userDataDir: profile,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

async function visit(label) {
  const page = await browser.newPage();
  await page.goto(appUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  const t0 = Date.now();
  let state = '';
  while (Date.now() - t0 < 150_000) {
    try { state = await page.evaluate(() => `${document.getElementById('dot-model')?.className} | ${document.getElementById('model-state')?.textContent}`); }
    catch { state = 'PROBE-FAIL'; }
    if (/dot ok|dot err/.test(state)) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
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
  }, { timeout: 30_000 }).catch((e) => 'ERR ' + String(e).slice(0, 60));
  console.log(`  ${label}: ${state}  (${secs}s)  cache=${JSON.stringify(cached)}`);
  await page.close();
  return { state, cached };
}

console.log(`final live check: ${appUrl}`);
const a = await visit('visit 1');
const b = await visit('visit 2');

const ok1 = /dot ok/.test(a.state) && a.cached && a.cached.bytes > 200_000_000;
const ok2 = /dot ok/.test(b.state);
console.log(ok1 && ok2 ? 'LIVE: PASS' : 'LIVE: FAIL');

await browser.close();
rmSync(profile, { recursive: true, force: true });
process.exit(ok1 && ok2 ? 0 : 1);
