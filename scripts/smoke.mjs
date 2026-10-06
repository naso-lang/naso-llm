#!/usr/bin/env node
/**
 * Headless browser check.
 *
 * Loads the built page in Chromium, waits for the Naso compiler and WebGPU to
 * come up, then runs the forward pass and reports the page's own results plus
 * any console error. Screenshots go to /tmp so the reviewer can see the page.
 *
 * This is the only check that exercises the WebGPU path: the CPU forward pass is
 * covered separately by tools/verify_forward.ts.
 *
 * Usage: node scripts/smoke.mjs [url] [--headful]
 */
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

const url = process.argv.find(a => a.startsWith('http')) ?? 'http://localhost:3000/';
const headful = process.argv.includes('--headful');

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/usr/bin/google-chrome',
].filter(Boolean);
const chromePath = CHROME_CANDIDATES.find(p => existsSync(p));
if (!chromePath) {
  console.error('no Chromium found; set CHROME_PATH');
  process.exit(2);
}

const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: !headful,
  args: [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    // Enable WebGPU in headless. `--enable-unsafe-swiftshader` is intentional:
    // it is the only way to get a device without a GPU, and it changes the
    // PERFORMANCE story, not the numerical one (the shaders still run).
    '--enable-unsafe-swiftshader',
    '--enable-features=Vulkan',
    '--use-angle=swiftshader',
    '--use-gl=angle',
  ],
});

const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 900 });

const consoleErrors = [];
const logs = [];
page.on('console', m => {
  const text = m.text();
  logs.push(`${m.type()}: ${text}`);
  if (m.type() === 'error') consoleErrors.push(text);
});
page.on('pageerror', e => consoleErrors.push(`pageerror: ${e.message}`));

console.log(`navigating to ${url}`);
await page.goto(url, { waitUntil: 'networkidle2', timeout: 60_000 });

// Report WebGPU availability directly from the page.
const gpuInfo = await page.evaluate(async () => {
  if (!('gpu' in navigator)) return { available: false, reason: 'navigator.gpu undefined' };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, reason: 'requestAdapter() returned null' };
    const device = await adapter.requestDevice();
    return { available: true, hasDevice: !!device };
  } catch (e) {
    return { available: false, reason: String(e) };
  }
});
console.log('WebGPU:', JSON.stringify(gpuInfo));

// Wait for the status dots to settle (Naso + WebGPU both resolved).
await page.waitForFunction(
  () => {
    const dots = ['dot-webgpu', 'dot-naso'].map(id => document.getElementById(id));
    return dots.every(d => d && (d.className.includes('ok') || d.className.includes('err')));
  },
  { timeout: 60_000 },
);

const status = await page.evaluate(() => ({
  webgpu: document.getElementById('dot-webgpu')?.className,
  naso: document.getElementById('dot-naso')?.className,
  hint: document.getElementById('hint')?.textContent,
  kernels: document.querySelectorAll('#kernels details').length,
  kernelHeaders: Array.from(document.querySelectorAll('#kernels summary')).map(s => s.textContent),
  runDisabled: (document.getElementById('run')).disabled,
}));
console.log('status:', JSON.stringify(status, null, 2));

await page.screenshot({ path: '/tmp/naso-llm-boot.png' });

// If the run button is enabled, exercise the actual forward pass.
let runResult = null;
if (!status.runDisabled) {
  console.log('clicking Run forward pass (downloads the checkpoint)...');
  await page.click('#run');
  try {
    await page.waitForFunction(
      () => document.getElementById('output-card')?.style.display === 'block'
        || (document.getElementById('run-hint')?.textContent ?? '').startsWith('Failed'),
      { timeout: 180_000 },
    );
    runResult = await page.evaluate(() => ({
      reports: Array.from(document.querySelectorAll('#reports tbody tr')).map(tr =>
        Array.from(tr.querySelectorAll('td')).map(td => td.textContent)),
      summary: document.querySelector('#reports p')?.textContent,
      topToken: document.querySelector('#output .out')?.textContent,
      top: Array.from(document.querySelectorAll('#output tbody tr')).map(tr =>
        Array.from(tr.querySelectorAll('td')).map(td => td.textContent)),
      hint: document.getElementById('run-hint')?.textContent,
    }));
  } catch (e) {
    runResult = { error: String(e) };
  }
  await page.screenshot({ path: '/tmp/naso-llm-run.png' });
}

console.log('runResult:', JSON.stringify(runResult, null, 2));
if (consoleErrors.length) {
  console.log(`\nconsole errors (${consoleErrors.length}):`);
  for (const e of consoleErrors.slice(0, 20)) console.log('  ' + e);
} else {
  console.log('\nno console errors');
}

await browser.close();

const nasoOk = status.naso?.includes('ok');
const kernelsOk = status.kernels >= 5;
const gpuOk = status.webgpu?.includes('ok');

if (!gpuOk) {
  // A headless container with no Vulkan driver cannot create a WebGPU adapter;
  // that is an environment gap, not an app failure. What is still checked is
  // the half that does not need a device: the WASM compiler ran in the browser
  // and produced all five kernels, which is the path CI cannot otherwise see.
  const ok = nasoOk && kernelsOk;
  console.log(`\nno WebGPU adapter in this environment: GPU EXECUTION NOT TESTED`);
  console.log(`browser WASM compile path: ${ok ? 'PASS' : 'FAIL'} (naso=${nasoOk}, kernels=${status.kernels})`);
  console.log(ok ? '\nSMOKE: PASS (compile-only)' : '\nSMOKE: FAIL');
  process.exit(ok ? 0 : 1);
}

const ok = nasoOk
  && kernelsOk
  && !status.runDisabled
  && runResult
  && !runResult.error
  && (runResult.summary ?? '').includes('satisfy');
console.log(ok ? '\nSMOKE: PASS' : '\nSMOKE: FAIL');
process.exit(ok ? 0 : 1);
