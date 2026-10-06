#!/usr/bin/env node
/**
 * Diagnose the two reported problems:
 *   1. "slow to load" -- is the 269 MB checkpoint actually cached after visit 1?
 *   2. "nothing is green" -- what are the three dots doing, and does the model
 *      dot ever reach green, or does the tab die first?
 *
 * Run: node scripts/diag-load.mjs http://localhost:3000/
 */
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

const url = process.argv.find((a) => a.startsWith('http')) ?? 'http://localhost:3000/';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
if (!chromePath) { console.error('no chromium'); process.exit(2); }

const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: true,
  userDataDir: '/var/tmp/diag-profile',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-webgpu'],
});

async function visit(label) {
  const page = await browser.newPage();
  const t0 = Date.now();
  const events = [];
  page.on('console', (m) => {
    const t = m.text();
    if (/fetching|parsed|tokenizer ready|quantization:|positions used|model load failed|No WebGPU/.test(t)) {
      events.push(`${((Date.now() - t0) / 1000).toFixed(1)}s ${t.slice(0, 110)}`);
    }
  });
  page.on('requestfinished', async (r) => {
    const u = r.url();
    if (u.includes('safetensors') || u.includes('tokenizer.json')) {
      const res = r.response();
      try {
        const len = (await res.headerValue('content-length')) ?? (await res.headerValue('content-range')) ?? '?';
        const fromCache = res.fromCache?.() ?? (await res.fromServiceWorker?.()) ?? false;
        events.push(`${((Date.now() - t0) / 1000).toFixed(1)}s RESP ${u.split('/').pop()} len=${len} fromCache=${fromCache}`);
      } catch { events.push(`${((Date.now() - t0) / 1000).toFixed(1)}s RESP ${u.split('/').pop()}`); }
    }
  });

  let crashed = false;
  page.on('error', () => { crashed = true; });

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  } catch (e) { events.push(`goto failed: ${e.message}`); }

  // Poll the dots without blocking: use a short timeout so a wedged main thread
  // shows up as a poll failure rather than hanging the whole probe.
  const samples = [];
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    try {
      const s = await page.evaluate(() => ({
        naso: document.getElementById('dot-naso')?.className,
        model: document.getElementById('dot-model')?.className,
        gpu: document.getElementById('dot-gpu')?.className,
        state: document.getElementById('model-state')?.textContent,
        bar: document.querySelector('#progress > div')?.style.width,
        heap: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null,
      }), { timeout: 5000 });
      samples.push(`${((Date.now() - t0) / 1000).toFixed(1)}s ${JSON.stringify(s)}`);
      if (s.model?.includes('ok') || s.model?.includes('err') || crashed) break;
    } catch (e) {
      samples.push(`${((Date.now() - t0) / 1000).toFixed(1)}s POLL-FAIL ${e.message.slice(0, 60)}`);
      crashed = true;
      break;
    }
    await new Promise((r) => setTimeout(r, 2500));
  }

  // Cache contents, with a bounded timeout.
  let cacheInfo = 'unavailable';
  try {
    cacheInfo = await page.evaluate(async () => {
      const names = await caches.keys();
      const out = {};
      for (const n of names) {
        const c = await caches.open(n);
        out[n] = (await c.keys()).length;
      }
      return JSON.stringify(out);
    }, { timeout: 8000 });
  } catch (e) { cacheInfo = `cache query failed: ${e.message.slice(0, 60)}`; }

  console.log(`\n========== ${label} ==========`);
  console.log(events.join('\n'));
  console.log('--- dot samples ---');
  console.log(samples.join('\n'));
  console.log(`caches: ${cacheInfo}  crashed=${crashed}`);
  await page.close().catch(() => {});
}

await visit('visit 1 (cold)');
await visit('visit 2 (should be warm)');

await browser.close();
