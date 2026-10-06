#!/usr/bin/env node
/**
 * Authoritative second-visit diagnostic.
 *
 * `performance.getEntriesByType('resource')` reports transferSize=0 for the
 * cross-origin HF response (no Timing-Allow-Origin), so it cannot answer "did
 * the second visit re-download 269 MB?". The CDP Network domain can: it reports
 * the real encodedDataLength for every response, including cross-origin ones.
 *
 * Per visit this prints:
 *   - whether the page was SW-controlled from the start
 *   - bytes actually received from huggingface.co / *.hf.co
 *   - whether the safetensors response came from the SW cache ("fromServiceWorker")
 *   - time to green
 *
 * Usage: node scripts/diag-revisit.mjs [url]
 */
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const url = process.argv.find((a) => a.startsWith('http')) ?? 'https://naso-lang.github.io/naso-llm/';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
if (!chromePath) { console.error('no chromium'); process.exit(2); }

const profile = '/var/tmp/revisit-profile';
rmSync(profile, { recursive: true, force: true });

const browser = await puppeteer.launch({
  executablePath: chromePath, headless: true, userDataDir: profile,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

async function visit(label, budgetMs) {
  const page = await browser.newPage();
  const client = await page.createCDPSession();
  await client.send('Network.enable');

  const byReq = new Map();      // requestId -> {url, fromSW, bytes}
  const failures = [];
  client.on('Network.responseReceived', (e) => {
    byReq.set(e.requestId, {
      url: e.response.url,
      fromSW: !!e.response.fromServiceWorker,
      status: e.response.status,
      bytes: 0,
    });
  });
  client.on('Network.loadingFinished', (e) => {
    const r = byReq.get(e.requestId);
    if (r) r.bytes = e.encodedDataLength;
  });
  client.on('Network.loadingFailed', (e) => {
    const r = byReq.get(e.requestId);
    failures.push(`${(r?.url ?? e.requestId).slice(0, 70)} :: ${e.errorText}`);
  });

  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  const controlled = await page.evaluate(() => !!navigator.serviceWorker?.controller);

  const t0 = Date.now();
  let state = '';
  while (Date.now() - t0 < budgetMs) {
    try {
      state = await page.evaluate(() => {
        const d = document.getElementById('dot-model');
        return `${d?.className} | ${document.getElementById('model-state')?.textContent}`;
      }, { timeout: 5000 });
    } catch { state = 'PROBE-FAIL'; }
    if (/dot ok|dot err/.test(state)) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  await new Promise((r) => setTimeout(r, 1500)); // let trailing loadingFinished land

  let hfBytes = 0;
  const interesting = [];
  for (const r of byReq.values()) {
    const isHF = /huggingface\.co|\.hf\.co/.test(r.url);
    if (isHF) hfBytes += r.bytes;
    if (/safetensors|config\.json|tokenizer\.json/.test(r.url)) {
      interesting.push(`    ${r.url.split('/').pop().slice(0, 34).padEnd(34)} ${String(r.status).padEnd(4)} ${(r.bytes / 1e6).toFixed(1)} MB  sw=${r.fromSW}`);
    }
  }

  console.log(`\n${label}`);
  console.log(`  SW-controlled from the start : ${controlled}`);
  console.log(`  state after ${elapsed}s          : ${state}`);
  console.log(`  bytes received from HF       : ${(hfBytes / 1e6).toFixed(1)} MB`);
  console.log(`  model-related requests:`);
  for (const s of interesting) console.log(s);
  if (failures.length) for (const f of failures) console.log(`  FAILED: ${f}`);

  await page.close();
  return { ok: /dot ok/.test(state), hfBytes };
}

console.log(`revisit diagnostic (CDP byte accounting) against ${url}`);
const a = await visit('VISIT 1 (cold)', 120_000);
const b = await visit('VISIT 2 (revisit)', 120_000);

console.log(`\nSUMMARY`);
console.log(`  visit 1 : ok=${a.ok}  HF bytes=${(a.hfBytes / 1e6).toFixed(1)} MB`);
console.log(`  visit 2 : ok=${b.ok}  HF bytes=${(b.hfBytes / 1e6).toFixed(1)} MB`);
if (!b.ok) console.log('  => revisit FAILED');
else if (b.hfBytes < 5e6) console.log('  => revisit served from cache (no big HF transfer)');
else console.log('  => revisit RE-DOWNLOADED the checkpoint');

await browser.close();
rmSync(profile, { recursive: true, force: true });
process.exit(b.ok ? 0 : 1);
