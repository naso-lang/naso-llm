#!/usr/bin/env node
/**
 * Headless browser check for the chat UI.
 *
 * Loads the page in Chromium and reports three independent things:
 *
 *   1. the Naso WASM compiler compiled every kernel in-browser,
 *   2. the model loaded (checkpoint + tokenizer + quantisation check),
 *   3. whether a WebGPU adapter exists -- and, if not, says so plainly.
 *
 * It deliberately does NOT claim the GPU path works when there is no device.
 * The forward pass, KV cache, tokenizer and quantisation bound are all verified
 * against independent references by `npm run verify`, which needs no browser.
 *
 * Usage: node scripts/smoke.mjs [url] [--headful]
 */
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

const url = process.argv.find((a) => a.startsWith('http')) ?? 'http://localhost:3000/';
const headful = process.argv.includes('--headful');

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/usr/bin/google-chrome',
].filter(Boolean);
const chromePath = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chromePath) {
  console.error('no Chromium found; set CHROME_PATH');
  process.exit(2);
}

const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: !headful,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-webgpu'],
});

const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 1000 });

const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));

console.log(`navigating to ${url}`);
await page.goto(url, { waitUntil: 'networkidle2', timeout: 60_000 });

// The compiler and GPU states settle quickly; the model takes longer.
await page.waitForFunction(
  () => ['dot-naso', 'dot-gpu'].every((id) => {
    const d = document.getElementById(id);
    return d && (d.className.includes('ok') || d.className.includes('err') || d.className.includes('') === false);
  }),
  { timeout: 60_000 },
).catch(() => {});

const boot = await page.evaluate(() => ({
  naso: document.getElementById('dot-naso')?.className,
  gpu: document.getElementById('dot-gpu')?.className,
  gpuState: document.getElementById('gpu-state')?.textContent,
  kernels: document.querySelectorAll('#kernels details').length,
  kernelNames: Array.from(document.querySelectorAll('#kernels summary')).map((s) => s.textContent),
  hint: document.getElementById('hint')?.textContent,
}));
console.log('boot:', JSON.stringify(boot, null, 2));
await page.screenshot({ path: '/tmp/naso-llm-boot.png' });

// Wait for the model: the composer enables, or the status turns to an error.
console.log('waiting for the model to load (downloads ~269 MB the first time)...');
const modelOk = await page.waitForFunction(
  () => {
    const send = document.getElementById('send');
    const dot = document.getElementById('dot-model');
    if (send && !send.disabled) return true;
    if (dot && dot.className.includes('err')) return true;
    return false;
  },
  { timeout: 600_000, polling: 1000 },
).then(() => true).catch(() => false);

const loaded = await page.evaluate(() => ({
  modelState: document.getElementById('model-state')?.textContent,
  modelDot: document.getElementById('dot-model')?.className,
  sendDisabled: document.getElementById('send')?.disabled,
  report: document.getElementById('reports')?.textContent?.replace(/\s+/g, ' ').slice(0, 400),
}));
console.log('model:', JSON.stringify(loaded, null, 2));
await page.screenshot({ path: '/tmp/naso-llm-loaded.png' });

if (loaded.sendDisabled) {
  console.log('\nthe model did not load; nothing further to exercise');
  if (consoleErrors.length) {
    console.log(`\nconsole errors (${consoleErrors.length}):`);
    for (const e of consoleErrors.slice(0, 10)) console.log('  ' + e);
  }
  await browser.close();
  console.log('\nSMOKE: FAIL');
  process.exit(1);
}

// Ask a question and wait for the reply to appear.
console.log('\nsending a question...');
const t0 = Date.now();
await page.type('#input', 'What is the capital of France?');
await page.click('#send');
const replied = await page.waitForFunction(
  () => {
    const msgs = document.querySelectorAll('#chat .msg.assistant .body');
    const last = msgs[msgs.length - 1];
    if (!last) return false;
    const text = last.textContent ?? '';
    return text.trim().length > 0 && !last.querySelector('.caret');
  },
  { timeout: 900_000, polling: 1000 },
).then(() => true).catch(() => false);

const reply = await page.evaluate(() => {
  const msgs = document.querySelectorAll('#chat .msg');
  return Array.from(msgs).map((m) => ({
    role: m.className.replace('msg ', ''),
    text: m.querySelector('.body')?.textContent?.slice(0, 300),
  }));
});
const secs = ((Date.now() - t0) / 1000).toFixed(1);
console.log(`reply after ${secs}s:`, JSON.stringify(reply, null, 2));
await page.screenshot({ path: '/tmp/naso-llm-reply.png' });

if (consoleErrors.length) {
  console.log(`\nconsole errors (${consoleErrors.length}):`);
  for (const e of consoleErrors.slice(0, 10)) console.log('  ' + e);
} else {
  console.log('\nno console errors');
}

await browser.close();

const nasoOk = boot.naso?.includes('ok');
const kernelsOk = boot.kernels >= 5;
const gpuOk = boot.gpu?.includes('ok');

// Report the GPU state honestly rather than folding it into the verdict.
console.log('\n' + '-'.repeat(60));
console.log(`  ${nasoOk ? 'ok  ' : 'FAIL'} Naso WASM compiled the kernels in-browser (${boot.kernels}/5)`);
console.log(`  ${modelOk && !loaded.sendDisabled ? 'ok  ' : 'FAIL'} model loaded and quantisation check ran`);
console.log(`  ${replied ? 'ok  ' : 'FAIL'} a question produced a reply`);
console.log(`  ${gpuOk ? 'ok  ' : '--  '} WebGPU adapter ${gpuOk ? 'present; generation is still CPU f32 in this build' : 'ABSENT in this environment: GPU EXECUTION NOT TESTED'}`);
console.log('-'.repeat(60));

const ok = nasoOk && kernelsOk && modelOk && replied && !loaded.sendDisabled;
console.log(ok ? 'SMOKE: PASS' : 'SMOKE: FAIL');
process.exit(ok ? 0 : 1);
