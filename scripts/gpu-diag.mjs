#!/usr/bin/env node
// Find why no WebGPU adapter appears: check Dawn/Vulkan init via GL/GPU info.
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_PATH ?? '/usr/bin/chromium';
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'],
});

const page = await browser.newPage();
await page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded', timeout: 30000 });

const info = await page.evaluate(async () => {
  const out = { ua: navigator.userAgent, gpuPresent: 'gpu' in navigator };
  try {
    const adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
    out.fallbackAdapter = !!adapter;
    if (adapter) {
      const d = await adapter.requestDevice();
      out.fallbackDevice = !!d;
      out.adapterInfo = adapter.info ? { ...adapter.info } : 'no .info';
    }
  } catch (e) { out.fallbackErr = String(e); }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    out.defaultAdapter = !!adapter;
    if (adapter) {
      const d = await adapter.requestDevice();
      out.defaultDevice = !!d;
      const features = [...adapter.features ?? []];
      out.features = features.slice(0, 10);
      out.limits = adapter.limits ? { maxStorage: adapter.limits.maxStorageBufferBindingSize, wg: adapter.limits.maxComputeWorkgroupSizeX } : 'none';
    }
  } catch (e) { out.defaultErr = String(e); }
  return out;
});
console.log(JSON.stringify(info, null, 2));

await browser.close();
process.exit(0);
