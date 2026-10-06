import type { ComputeAbi, Kernel } from './types.js';
import { logger } from './logger.js';

/**
 * A thin WebGPU compute engine.
 *
 * The engine knows nothing about LLMs or quantization. It takes a `Kernel`
 * (WGSL text plus the ABI the verified compiler produced alongside it) and runs
 * it. Every buffer it allocates and every bind group it builds is derived from
 * the ABI; it never parses shader text to discover a binding. That is the point
 * of the ABI existing -- the host cannot drift from the shader, because neither
 * the binding order nor the dispatch count is typed by hand here.
 */

/** A compiled kernel plus its live GPU resources. */
export interface CompiledKernel {
  kernel: Kernel;
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  bindGroup: GPUBindGroup;
}

export class WebGPUEngine {
  adapter: GPUAdapter | null = null;
  device: GPUDevice | null = null;
  private initialized = false;

  async init(): Promise<boolean> {
    if (this.initialized) return this.device !== null;
    this.initialized = true;

    if (!('gpu' in navigator)) {
      logger.error('webgpu', 'navigator.gpu is undefined: this browser has no WebGPU. Use Chrome/Edge 113+ or a recent Firefox/Nightly.');
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    } catch (e) {
      logger.error('webgpu', `requestAdapter() threw: ${e}`);
      return false;
    }
    if (!this.adapter) {
      logger.error('webgpu', 'No WebGPU adapter. On Linux this usually means the browser has no Vulkan/Mesa driver.');
      return false;
    }

    // `GPUAdapter.name` is not in the shipping spec; read it defensively.
    const info = this.adapter as GPUAdapter & { name?: string };
    logger.info('webgpu', `Adapter: ${info.name ?? '(unnamed)'}`);
    logger.info('webgpu', `maxStorageBufferBindingSize = ${this.adapter.limits.maxStorageBufferBindingSize} bytes`);

    // Ask only for limits the adapter already advertises: requestDevice rejects
    // a requirement above what the adapter reports.
    const caps = this.adapter.limits;
    this.device = await this.adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: caps.maxStorageBufferBindingSize,
        maxBufferSize: caps.maxBufferSize,
      },
    });

    this.device.lost.then(info2 => {
      logger.error('webgpu', `Device lost: ${info2.reason} -- ${info2.message}`);
      this.device = null;
    });
    this.device.onuncapturederror = ev => {
      logger.error('webgpu', `Uncaptured WebGPU error: ${ev.error.message}`);
    };

    logger.success('webgpu', 'Device ready');
    return true;
  }

  deviceOrThrow(): GPUDevice {
    if (!this.device) throw new Error('WebGPU device not initialized');
    return this.device;
  }

  /**
   * Compile a kernel into a pipeline and bind group, reporting shader
   * compilation errors through the logger rather than throwing opaquely.
   */
  async compile(kernel: Kernel, bufferFor: (index: number) => GPUBufferBinding): Promise<CompiledKernel | null> {
    const device = this.deviceOrThrow();

    const module = device.createShaderModule({ label: `${kernel.name}.wgsl`, code: kernel.wgsl });

    // Push an error scope so a rejected shader is reported with its message
    // instead of surfacing as a generic pipeline-creation failure.
    device.pushErrorScope('validation');
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: kernel.abi.entryPoint },
      label: `${kernel.name}.pipeline`,
    });
    const error = await device.popErrorScope();
    if (error) {
      logger.error('webgpu', `Shader "${kernel.name}" rejected by the device: ${error.message}`);
      return null;
    }

    const needed = kernel.abi.bindings.length + kernel.abi.scalars.length;
    const entries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < needed; i++) {
      entries.push({ binding: i, resource: bufferFor(i) });
    }
    const bindGroupLayout = pipeline.getBindGroupLayout(0);
    const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries, label: `${kernel.name}.bindgroup` });

    logger.success('webgpu', `Pipeline ready: ${kernel.name} -> ${kernel.abi.entryPoint} (${kernel.abi.dispatchGroups} workgroup(s))`);
    return { kernel, pipeline, bindGroupLayout, bindGroup };
  }

  /** Allocate a storage buffer and upload `data` at offset 0. */
  createStorageBuffer(label: string, data: Float32Array | Int32Array | Uint32Array): GPUBuffer {
    const device = this.deviceOrThrow();
    // Storage bindings must be 4-byte aligned; every element type here is 4 bytes.
    const size = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
    const buffer = device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    const range = buffer.getMappedRange();
    // Preserve element interpretation: an f32 payload must be written as f32s,
    // not reinterpreted as raw bytes.
    if (data instanceof Float32Array) new Float32Array(range).set(data);
    else if (data instanceof Int32Array) new Int32Array(range).set(data);
    else new Uint32Array(range).set(data);
    buffer.unmap();
    return buffer;
  }

  /** An empty storage buffer of `bytes` (for a kernel output). */
  createEmptyStorageBuffer(label: string, bytes: number): GPUBuffer {
    const device = this.deviceOrThrow();
    return device.createBuffer({
      label,
      size: Math.max(4, Math.ceil(bytes / 4) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /**
   * A uniform buffer holding one scalar.
   *
   * The WGSL backend emits each scalar parameter as a `var<uniform>` binding
   * (a compute entry point may take only builtins), reported by the ABI as a
   * 16-byte binding. A bare f32/u32 needs 4 bytes of payload; the binding is 16.
   */
  createScalarUniform(label: string, value: number, type: string, abiBytes = 16): GPUBuffer {
    const device = this.deviceOrThrow();
    const size = Math.max(16, Math.ceil(abiBytes / 16) * 16);
    const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const staging = (type === 'u32' || type === 'i32')
      ? new Uint32Array([value >>> 0])
      : new Float32Array([value]);
    device.queue.writeBuffer(buffer, 0, staging.buffer, staging.byteOffset, staging.byteLength);
    return buffer;
  }

  /**
   * Encode and submit a dispatch of `count` workgroups, then resolve when the
   * GPU has finished (via the queue's onSubmittedWorkDone, not a timer).
   */
  async dispatch(compiled: CompiledKernel, count = compiled.kernel.abi.dispatchGroups): Promise<void> {
    const device = this.deviceOrThrow();
    const encoder = device.createCommandEncoder({ label: `${compiled.kernel.name}.encoder` });
    const pass = encoder.beginComputePass({ label: `${compiled.kernel.name}.pass` });
    pass.setPipeline(compiled.pipeline);
    pass.setBindGroup(0, compiled.bindGroup);
    pass.dispatchWorkgroups(count);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    logger.debug('webgpu', `dispatched ${compiled.kernel.abi.entryPoint}: ${count} x ${compiled.kernel.abi.workgroupSize}`);
  }

  /**
   * Read `elements` elements back from a storage buffer.
   * `elem` selects the view; the returned array is trimmed to `elements`.
   */
  async readBuffer(source: GPUBuffer, elements: number, elem: 'f32' | 'i32' | 'u32'): Promise<Float32Array | Int32Array | Uint32Array> {
    const device = this.deviceOrThrow();
    const bytes = Math.max(4, elements * 4);
    const staging = device.createBuffer({
      label: 'readback',
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder({ label: 'readback' });
    encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const out = elem === 'i32'
      ? new Int32Array(elements)
      : elem === 'u32'
        ? new Uint32Array(elements)
        : new Float32Array(elements);
    new Uint8Array(out.buffer).set(new Uint8Array(staging.getMappedRange(), 0, elements * 4));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Read floats and drop the int8 padding introduced by the 4-byte element. */
  async readAsFloat32(source: GPUBuffer, elements: number): Promise<Float32Array> {
    return (await this.readBuffer(source, elements, 'f32')) as Float32Array;
  }

  destroy(): void {
    this.device?.destroy();
    this.device = null;
    this.adapter = null;
    this.initialized = false;
  }
}

export const webgpuEngine = new WebGPUEngine();

/** Convenience re-export so callers need only one import. */
export type { ComputeAbi };
