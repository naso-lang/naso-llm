#!/usr/bin/env node
/**
 * Final live acceptance for the three reported issues, in one clean profile:
 *
 *   1. revisit does not re-download the checkpoint
 *   2. the chat dropdown offers no random-weights model
 *   3. the model answers a real question coherently, twice in a row
 *
 * Re-download is measured on the CDP Network domain: a response served by the
 * service worker from Cache Storage never reaches the network stack, so an HF
 * safetensors response recorded here means a genuine re-download. Counting
 * page-level `request` events does not work -- they fire for SW-served
 * responses too, which made an earlier version of this test report a false
 * failure.
 *
 * Usage: node scripts/verify-live.mjs
 */
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const APP = 'https://naso-lang.github.io/naso-llm/';
const CHROME = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
const PROFILE = '/var/tmp/accept-profile2';
rmSync(PROFILE, { recursive: true, force: true });

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage', `--user-data-dir=${PROFILE}`, '--disable-gpu'],
});

const netHf = [];
const page = await browser.newPage();
const client = await page.createCDPSession();
await client.send('Network.enable');
client.on('Network.responseReceived', (e) => {
  const u = e.response?.url ?? '';
  if (u.includes('huggingface.co') && u.includes('safetensors')) {
    netHf.push({ url: u.split('/').slice(-1)[0], len: e.response.encodedDataLength,
                   fromSW: e.response.fromServiceWorker === true, status: e.response.status });
  }
});

async function ready(p, label) {
  for (let i = 0; i < 120; i++) {
    const s = await p.evaluate(() => ({
      model: document.getElementById('model-state')?.textContent ?? '',
      dot: document.getElementById('dot-model')?.className ?? '',
    })).catch(() => ({ model: '', dot: '' }));
    if (/ready/i.test(s.model) && /ok/.test(s.dot)) return s;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`${label}: never reached ready`);
}

await page.goto(APP, { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.querySelectorAll('#model-select option').length > 0, { timeout: 20000 });
const options = await page.$$eval('#model-select option', (os) => os.map((o) => o.textContent));
const s1 = await ready(page, 'visit1');
const net1 = netHf.length;
const cache1 = await page.evaluate(async () => {
  const c = await caches.open('naso-llm-models-v1');
  const out = [];
  for (const k of await c.keys()) {
    const r = await c.match(k);
    const b = r ? await r.clone().blob() : null;
    out.push({ url: k.url.split('/').slice(-1)[0], bytes: b ? b.size : 0 });
  }
  return out;
});

await page.type('#input', 'What is the capital of France?');
await page.keyboard.press('Enter');
await page.waitForFunction(() => document.querySelectorAll('#chat .msg.assistant').length >= 1, { timeout: 120000 });
await page.type('#input', 'And what is its population?');
await page.keyboard.press('Enter');
await page.waitForFunction(() => document.querySelectorAll('#chat .msg.assistant').length >= 2, { timeout: 120000 });
const texts = await page.$$eval('#chat .msg', (ms) => ms.map((m) => m.querySelector('.body').textContent));
const roles = await page.$$eval('#chat .msg', (ms) => ms.map((m) => m.className.replace('msg ', '')));

const netBefore2 = netHf.length;
await page.goto(APP, { waitUntil: 'domcontentloaded' });
const s2 = await ready(page, 'visit2');
const net2 = netHf.length - netBefore2;

console.log(`\n[2] dropdown options (${options.length}):`);
for (const o of options) console.log(`      ${o}`);
const noFixture = options.length > 0 && options.every((o) => !/random/i.test(o));
console.log(`    no random-weights model selectable: ${noFixture ? 'YES' : 'NO'}`);

console.log(`\n[3] chat:`);
for (let i = 0; i < texts.length; i++) console.log(`      ${roles[i]}: ${texts[i]}`);
const a1 = texts[1] ?? '', a2 = texts[3] ?? '';
const coherent = texts.length === 4
  && /Paris/.test(a1) && /Paris/.test(a2)
  && !/^user:/im.test(a1) && !/^assistant:/im.test(a1)
  && !/^user:/im.test(a2) && !/^assistant:/im.test(a2);
console.log(`    coherent, no leaked turn labels: ${coherent ? 'YES' : 'NO'}`);

console.log(`\n[1] visit 1: ${s1.model}`);
for (const e of cache1) console.log(`      cached ${e.url}: ${e.bytes} bytes`);
const ck = cache1.find((e) => e.url.includes('safetensors'));
console.log(`    checkpoint cached on visit 1: ${ck && ck.bytes === 269060552 ? 'YES (269060552 B)' : 'NO'}`);
console.log(`    HF safetensors responses during visit 1: ${JSON.stringify(netHf.slice(0, net1))}`);
console.log(`[1] visit 2: ${s2.model}`);
const v2 = netHf.slice(netBefore2);
console.log(`    HF safetensors responses during visit 2: ${JSON.stringify(v2)}`);
const v2bytes = v2.filter(r => r.fromSW !== true).reduce((a, r) => a + (r.len || 0), 0);
console.log(`    bytes pulled from HF on visit 2 (excluding SW-served): ${v2bytes}`);
console.log(`    revisit did NOT re-download the checkpoint: ${v2bytes < 1e6 ? 'YES' : 'NO'}`);

const pass = noFixture && coherent && !!ck && ck.bytes === 269060552 && v2bytes < 1e6;
console.log(`\nLIVE ACCEPT: ${pass ? 'PASS' : 'FAIL'}`);
await browser.close();
process.exit(pass ? 0 : 1);
