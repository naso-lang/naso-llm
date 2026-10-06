import { compileKernel } from './naso.js';
import { loadKernelSource, KERNEL_SPECS } from './kernels.js';
import { webgpuEngine, type CompiledKernel } from './webgpu.js';
import { logger } from './logger.js';
import { forward } from './llama.js';
import { type Kernel, type ModelConfig } from './types.js';
import { type Tensor } from './model.js';

/**
 * The quantization pipeline: Naso kernels drive the GPU, and every number they
 * produce is checked against a plain-f32 CPU reference.
 *
 * WHAT IS REAL HERE
 *
 *   * The WGSL the GPU runs is compiled from `kernels/quantize_int8.naso` by
 *     the verified compiler, in the browser, at load time (see naso.ts). It is
 *     not a hand-written shader that happens to be committed next to a .naso
 *     file.
 *   * Quantization rounds-trips through the actual kernel: f32 -> int8 codes on
 *     the GPU -> back to f32 on the GPU, and the result is compared to the
 *     reference. Because the round-trip is executed, a mistake in either
 *     direction is caught by the comparison rather than assumed away.
 *
 * WHAT IS NOT CLAIMED
 *
 *   * The matmuls and attention still run on the CPU in f32 (see llama.ts). The
 *     Naso WGSL backend is elementwise-only -- a compute entry point indexes
 *     every binding with the same index, so a reduction is refused at codegen.
 *     Driving a real dot product would need the straight-line backend or a
 *     reduction the current backend does not emit, and pretending otherwise
 *     would make this demo a lie. The demo therefore quantizes the weights with
 *     the verified kernels and runs the model exactly; the *quantization* is
 *     the part the toolchain proves.
 *   * The error bound is the real one: per-element |x - dequant(q(x))| <= s/2,
 *     which is measurable here directly.
 */

export interface QuantizationReport {
  tensorName: string;
  elements: number;
  /** scale = absmax / 127, the symmetric int8 step. */
  scale: number;
  /** Max |x - dequant(quant(x))| over the tensor. */
  maxAbsError: number;
  /** The bound the quantiser promises: scale / 2. */
  promisedBound: number;
  boundHolds: boolean;
  /** Mean squared error of the round-trip. */
  mse: number;
}

export interface PipelineResult {
  /** The kernel objects compiled from Naso source, for display. */
  kernels: Map<string, Kernel>;
  reports: QuantizationReport[];
  /** f32 weights recovered from the int8 codes through the GPU kernels. */
  quantized: Map<string, Float32Array>;
  /** Last f32 hidden state from the exact reference forward pass. */
  hidden: Float32Array;
  logits: Float32Array;
}

/** Which checkpoint tensors get quantized (the projection matrices). */
function quantizableNames(tensors: Map<string, Tensor>, config: ModelConfig): string[] {
  const names: string[] = [];
  for (let l = 0; l < config.numLayers; l++) {
    const p = `model.layers.${l}`;
    names.push(
      `${p}.self_attn.q_proj.weight`,
      `${p}.self_attn.k_proj.weight`,
      `${p}.self_attn.v_proj.weight`,
      `${p}.self_attn.o_proj.weight`,
      `${p}.mlp.gate_proj.weight`,
      `${p}.mlp.up_proj.weight`,
      `${p}.mlp.down_proj.weight`,
    );
  }
  return names.filter(n => tensors.has(n));
}

export class QuantizationPipeline {
  /** Compiled kernels keyed by Naso function name. */
  kernels = new Map<string, Kernel>();
  private gpu = new Map<string, CompiledKernel>();

  /** Compile the Naso kernels to WGSL + ABI. No GPU needed. */
  async compile(): Promise<void> {
    // Compile each distinct (file, function) once.
    const seen = new Set<string>();
    for (const spec of KERNEL_SPECS) {
      if (seen.has(spec.name)) continue;
      seen.add(spec.name);
      const source = await loadKernelSource(spec.file);
      const { kernel, diagnostics } = await compileKernel(source, spec.name);
      if (!kernel) {
        throw new Error(`Naso refused kernel "${spec.name}": ${diagnostics.map(d => d.message).join('; ')}`);
      }
      this.kernels.set(spec.name, kernel);
    }
  }

  /** Compile (if needed) and build the GPU pipelines and buffers. */
  async prepare(): Promise<void> {
    if (this.gpu.size > 0) return;
    await this.compile();

    // Build a GPU pipeline for the kernels that are actually dispatched. Each
    // gets its own buffers, sized from the ABI (never hand-typed).
    await this.buildKernel('quantize_int8_symmetric');
    await this.buildKernel('dequantize_int8_symmetric');
    await this.buildKernel('scale_clamp_f32');
    await this.buildKernel('silu_f32');
  }

  private async buildKernel(name: string): Promise<void> {
    const kernel = this.kernels.get(name);
    if (!kernel) throw new Error(`kernel "${name}" was not compiled`);

    const abi = kernel.abi;
    // Storage bindings 0 (input) and 1 (output) share the extent; the scalar
    // uniforms follow. Sizes come from the ABI, in bytes.
    const buffers: GPUBuffer[] = [];
    for (const b of abi.bindings) {
      if (b.access === 'read') {
        // Allocate by element type so a zero-filled i32 input is not created as
        // an f32 view whose bytes happen to coincide.
        const zero = b.elem === 'i32' ? new Int32Array(abi.elements)
          : b.elem === 'u32' ? new Uint32Array(abi.elements)
            : new Float32Array(abi.elements);
        buffers[b.index] = webgpuEngine.createStorageBuffer(`${name}.${b.name}`, zero);
      } else {
        buffers[b.index] = webgpuEngine.createEmptyStorageBuffer(`${name}.${b.name}`, b.bytes);
      }
    }
    const uniforms: GPUBuffer[] = abi.scalars.map(s => webgpuEngine.createScalarUniform(`${name}.${s.name}`, 0, s.type));

    const compiled = await webgpuEngine.compile(kernel, i => {
      const buf = i < abi.bindings.length ? buffers[i] : uniforms[i - abi.bindings.length];
      if (!buf) throw new Error(`no buffer for binding ${i} of "${name}"`);
      return { buffer: buf };
    });
    if (!compiled) throw new Error(`failed to build GPU pipeline for "${name}"`);

    // Stash the buffers on the compiled record for reuse in the dispatch below.
    (compiled as CompiledKernel & { _storage?: GPUBuffer[]; _uniforms?: GPUBuffer[] })._storage = buffers;
    (compiled as CompiledKernel & { _storage?: GPUBuffer[]; _uniforms?: GPUBuffer[] })._uniforms = uniforms;
    this.gpu.set(name, compiled);
  }

  private storage(name: string, index: number): GPUBuffer {
    const c = this.gpu.get(name) as (CompiledKernel & { _storage?: GPUBuffer[] }) | undefined;
    if (!c?._storage?.[index]) throw new Error(`no storage binding ${index} for "${name}"`);
    return c._storage[index];
  }

  /** Overwrite the scalar uniform at `index` (relative to the ABI's scalars). */
  private setScalar(name: string, index: number, value: number): void {
    const c = this.gpu.get(name) as (CompiledKernel & { _uniforms?: GPUBuffer[] }) | undefined;
    const kernel = this.kernels.get(name)!;
    const buf = c?._uniforms?.[index];
    if (!buf) throw new Error(`no scalar ${index} for "${name}"`);
    const type = kernel.abi.scalars[index]?.type ?? 'f32';
    const staging = type === 'u32' || type === 'i32'
      ? new Uint32Array([value >>> 0])
      : new Float32Array([value]);
    webgpuEngine.deviceOrThrow().queue.writeBuffer(buf, 0, staging.buffer, staging.byteOffset, staging.byteLength);
  }

  /**
   * Round-trip a tensor through the GPU: f32 -> int8 codes -> f32.
   * Returns the recovered f32 values and the error report.
   */
  async quantizeRoundTrip(name: string, values: Float32Array): Promise<{ recovered: Float32Array; scale: number; maxAbsError: number; mse: number }> {
    const n = values.length;
    const qk = this.kernels.get('quantize_int8_symmetric')!;
    const dqk = this.kernels.get('dequantize_int8_symmetric')!;

    // The compiled shaders are fixed to a 1024-element extent; process in tiles.
    const tile = qk.abi.elements;
    const recovered = new Float32Array(n);
    const dequantBuf = this.storage('dequantize_int8_symmetric', 1);

    let scale = 0;
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(values[i]));
    scale = scale / 127 || 1 / 127;

    for (let off = 0; off < n; off += tile) {
      const chunk = new Float32Array(tile);
      const count = Math.min(tile, n - off);
      chunk.set(values.subarray(off, off + count));

      // quantize: input (f32) -> output (i32 codes)
      const inBuf = this.storage('quantize_int8_symmetric', 0);
      const outBuf = this.storage('quantize_int8_symmetric', 1);
      webgpuEngine.deviceOrThrow().queue.writeBuffer(inBuf, 0, chunk.buffer, 0, chunk.byteLength);
      this.setScalar('quantize_int8_symmetric', 0, scale);
      await webgpuEngine.dispatch(this.gpu.get('quantize_int8_symmetric')!);

      // dequantize: input (i32 codes) <- quantize output, output -> f32
      const codes = await webgpuEngine.readBuffer(outBuf, tile, 'i32');
      webgpuEngine.deviceOrThrow().queue.writeBuffer(this.storage('dequantize_int8_symmetric', 0), 0, codes.buffer, 0, codes.byteLength);
      this.setScalar('dequantize_int8_symmetric', 0, scale);
      await webgpuEngine.dispatch(this.gpu.get('dequantize_int8_symmetric')!);

      const back = await webgpuEngine.readAsFloat32(dequantBuf, tile);
      recovered.set(back.subarray(0, count), off);
      void dqk;
    }

    let maxAbsError = 0;
    let se = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(values[i] - recovered[i]);
      if (d > maxAbsError) maxAbsError = d;
      se += d * d;
    }
    return { recovered, scale, maxAbsError, mse: se / n };
  }

  /**
   * Run the demo end to end on a loaded model: compile the kernels, quantize
   * every projection matrix through the GPU, and run the exact forward pass.
   */
  async run(config: ModelConfig, tensors: Map<string, Tensor>, inputIds: number[]): Promise<PipelineResult> {
    await this.prepare();

    const reports: QuantizationReport[] = [];
    const quantized = new Map<string, Float32Array>();

    for (const name of quantizableNames(tensors, config)) {
      const src = tensors.get(name)!.data;
      const { recovered, scale, maxAbsError, mse } = await this.quantizeRoundTrip(name, src);
      quantized.set(name, recovered);
      const promisedBound = scale / 2;
      const boundHolds = maxAbsError <= promisedBound + 1e-6;
      reports.push({ tensorName: name, elements: src.length, scale, maxAbsError, promisedBound, boundHolds, mse });
      logger.info('main', `${name}: scale=${scale.toExponential(3)} maxErr=${maxAbsError.toExponential(3)} bound=${promisedBound.toExponential(3)} ${boundHolds ? 'OK' : 'VIOLATED'}`);
    }

    // Exact f32 forward pass for the reference hidden state and logits. (A
    // quantized GPU forward pass would need reduction kernels the current WGSL
    // backend refuses; see the file header.)
    const ref = forward(Object.fromEntries(tensors), config, inputIds);

    return { kernels: this.kernels, reports, quantized, hidden: ref.hidden, logits: ref.logits };
  }

  /** Free GPU resources. */
  destroy(): void {
    for (const c of this.gpu.values()) {
      for (const b of (c as CompiledKernel & { _storage?: GPUBuffer[] })._storage ?? []) b.destroy?.();
      for (const b of (c as CompiledKernel & { _uniforms?: GPUBuffer[] })._uniforms ?? []) b.destroy?.();
    }
    this.gpu.clear();
  }
}

export const quantizationPipeline = new QuantizationPipeline();
