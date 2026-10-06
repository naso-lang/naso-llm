#!/usr/bin/env node
// Probe WebGPU flag combinations for this container's Chromium, with a hard
// per-candidate timeout so a hanging GPU process cannot stall the sweep.
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_PATH ?? '/usr/bin/chromium';
const ICD = '/usr/lib/chromium/vk_swiftshader_icd.json';
const PER_CANDIDATE_MS = 20_000;

const CANDIDATES = [
  { name: 'unsafe-webgpu+swiftshader', args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'] },
  { name: 'unsafe-webgpu only', args: ['--enable-unsafe-webgpu'] },
  { name: 'headless=new + swiftshader', args: ['--headless=new', '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'] },
  { name: 'vulkan+icd+swiftshader', args: ['--enable-features=Vulkan', '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'], env: { VK_ICD_FILENAMES: ICD } },
  { name: 'angle=swiftshader', args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader'] },
];

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`timeout ${ms}ms (${label})`)), ms)),
  ]);
}

for (const c of CANDIDATES) {
  let browser;
  try {
    browser = await withTimeout(puppeteer.launch({
      executablePath: CHROME,
      headless: true,
      args: ['--no-sandbox', '--disable-dev-shm-usage', ...c.args],
      env: { ...process.env, ...(c.env ?? {}) },
    }), PER_CANDIDATE_MS, 'launch');

    const page = await browser.newPage();
    await withTimeout(page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded' }), PER_CANDIDATE_MS, 'goto');
    const r = await withTimeout(page.evaluate(async () => {
      if (!('gpu' in navigator)) return { ok: false, why: 'navigator.gpu undefined' };
      try {
        const a = await navigator.gpu.requestAdapter();
        if (!a) return { ok: false, why: 'adapter null' };
        const d = await a.requestDevice();
        return { ok: !!d, why: d ? 'device ok' : 'device null' };
      } catch (e) { return { ok: false, why: String(e) }; }
    }), PER_CANDIDATE_MS, 'evaluate');
    console.log(`${r.ok ? 'PASS' : 'fail'}  ${c.name}  ${JSON.stringify(r)}`);
  } catch (e) {
    console.log(`ERROR ${c.name}: ${String(e).split('\n')[0]}`);
  } finally {
    try { await browser?.close(); } catch { /* already gone */ }
  }
}
process.exit(0);
