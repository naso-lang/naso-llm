#!/usr/bin/env node
// Flag/adapter matrix sweep for headless WebGPU. Each candidate runs in its own
// browser with a hard timeout, so one hanging combination cannot stall the run.
// Prints which, if any, yields a non-null adapter AND a correct compute result.
import puppeteer from 'puppeteer-core';

const EXES = [
  process.argv[2] ?? '/opt/ms-playwright/chromium-1243/chrome-linux64/chrome',
  '/usr/bin/chromium',
];

const BASE = [
  '--no-sandbox',
  '--disable-dev-shm-usage',
  '--enable-unsafe-webgpu',
  '--ignore-gpu-blocklist',
  '--enable-features=Vulkan',
];

const VARIANTS = [
  { name: 'vulkan+swiftshader-icd', args: [...BASE, '--use-angle=vulkan', '--use-vulkan=swiftshader', '--enable-vulkan', '--disable-vulkan-surface'] },
  { name: 'webgpu-adapter=swiftshader', args: [...BASE, '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader'] },
  { name: 'angle=swiftshader', args: [...BASE, '--use-angle=swiftshader', '--use-gl=angle'] },
  { name: 'dawn-swiftshader-vk', args: [...BASE, '--use-webgpu-adapter=swiftshader', '--use-vulkan=swiftshader', '--enable-dawn-features=allow_unsafe_apis'] },
  { name: 'plain-unsafe-webgpu', args: [...BASE] },
];

const ICD = '/usr/lib/chromium/vk_swiftshader_icd.json';
const ENV = { ...process.env, VK_ICD_FILENAMES: ICD, VK_DRIVER_FILES: ICD, LIBGL_ALWAYS_SOFTWARE: '1' };

async function tryOne(exe, variant) {
  let browser;
  try {
    browser = await puppeteer.launch({ executablePath: exe, headless: true, args: variant.args, env: ENV, protocolTimeout: 20000 });
    const page = await browser.newPage();
    await page.goto('about:blank');
    // about:blank is not a secure context for WebGPU in some builds.
    await page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded', timeout: 15000 });
    const r = await page.evaluate(async () => {
      if (!navigator.gpu) return { adapter: null, note: 'no navigator.gpu' };
      const a = await navigator.gpu.requestAdapter();
      if (!a) {
        const f = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
        return { adapter: null, fallback: !!f, note: 'requestAdapter null' };
      }
      const d = await a.requestDevice();
      const module = d.createShaderModule({ code: `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> b: array<f32>;
        @compute @workgroup_size(4)
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { b[g.x] = a[g.x] * 2.0 + 1.0; }` });
      const p = d.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const ab = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(ab, 0, new Float32Array([1, 2, 3, 4]));
      const bb = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const bg = d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ab } }, { binding: 1, resource: { buffer: bb } }] });
      const e = d.createCommandEncoder();
      const pass = e.beginComputePass();
      pass.setPipeline(p); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end();
      d.queue.submit([e.finish()]);
      const s = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e2 = d.createCommandEncoder();
      e2.copyBufferToBuffer(bb, 0, s, 0, 16);
      d.queue.submit([e2.finish()]);
      await s.mapAsync(GPUMapMode.READ);
      return { adapter: a.name ?? '(unnamed)', compute: Array.from(new Float32Array(s.getMappedRange())) };
    });
    return r;
  } catch (e) {
    return { error: String(e).slice(0, 120) };
  } finally {
    await browser?.close().catch(() => {});
  }
}

for (const exe of EXES) {
  for (const v of VARIANTS) {
    const r = await Promise.race([
      tryOne(exe, v),
      new Promise((res) => setTimeout(() => res({ error: 'timeout' }), 45000)),
    ]);
    const ok = r.compute && r.compute.join(',') === '3,5,7,9';
    console.log(`${ok ? 'OK  ' : '    '} ${exe.includes('playwright') ? 'pw ' : 'sys'} ${v.name.padEnd(28)} ${JSON.stringify(r).slice(0, 110)}`);
    if (ok) { console.log('\nEXECUTION POSSIBLE with the above configuration.'); process.exit(0); }
  }
}
console.log('\nNo configuration produced a working adapter.');
process.exit(1);
