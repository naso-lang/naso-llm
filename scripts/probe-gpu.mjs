#!/usr/bin/env node
// Probe: can this container actually execute a WebGPU compute shader?
//
// The earlier conclusion was "no Vulkan driver". But SwiftShader ships an ICD
// inside the Chromium bundle, so the real question is whether the loader can
// find it. This runs the browser with VK_ICD_FILENAMES pointed at that ICD and
// reports, in order: adapter, device, a trivial shader's output.
//
// Usage: node scripts/probe-gpu.mjs [executablePath]
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

const exe = process.argv[2] ?? '/usr/bin/chromium';
const ICD = '/usr/lib/chromium/vk_swiftshader_icd.json';

const browser = await puppeteer.launch({
  executablePath: exe,
  headless: true,
  args: [
    '--no-sandbox',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-angle=vulkan',
    '--use-vulkan=swiftshader',
    '--enable-vulkan',
    '--disable-vulkan-surface',
    '--ignore-gpu-blocklist',
  ],
  env: {
    ...process.env,
    VK_ICD_FILENAMES: ICD,
    VK_DRIVER_FILES: ICD,
    VK_LOADER_DEBUG: 'error',
    LIBGL_ALWAYS_SOFTWARE: '1',
  },
});

const page = await browser.newPage();
page.on('console', (m) => console.log(`  [page] ${m.text()}`));
await page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded' });

const result = await page.evaluate(async () => {
  const out = { hasGpu: !!navigator.gpu, adapter: null, device: null, compute: null, error: null };
  if (!navigator.gpu) return out;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return out;
    out.adapter = { name: adapter.name ?? '(unnamed)', limits: adapter.limits.maxStorageBufferBindingSize };
    const device = await adapter.requestDevice();
    out.device = 'ok';
    // A trivial compute pass: out[i] = in[i] * 2 + 1, to prove dispatch works.
    device.pushErrorScope('validation');
    const module = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> b: array<f32>;
        @compute @workgroup_size(4)
        fn main(@builtin(global_invocation_id) g: vec3<u32>) {
          b[g.x] = a[g.x] * 2.0 + 1.0;
        }`,
    });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const a = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(a, 0, new Float32Array([1, 2, 3, 4]));
    const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([enc.finish()]);

    const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(b, 0, staging, 0, 16);
    device.queue.submit([enc2.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    out.compute = Array.from(new Float32Array(staging.getMappedRange()));
    staging.unmap();
    const err = await device.popErrorScope();
    if (err) out.error = err.message;
  } catch (e) {
    out.error = String(e);
  }
  return out;
});

console.log(JSON.stringify(result, null, 2));
await browser.close();
process.exit(result.compute && result.compute.join(',') === '3,5,7,9' ? 0 : 1);
