/**
 * Kernel registry.
 *
 * The canonical Naso sources live in `kernels/*.naso` at the repo root.
 * `scripts/compile-kernels.js` copies them to `public/kernels/` so they are
 * served (and cached by the service worker), and this module fetches them at
 * runtime. That keeps the .naso file the single source of truth: the shader the
 * browser runs is compiled from the same text a reviewer reads.
 */

export interface KernelSpec {
  /** Function name inside the source file; the compiler is asked for this fn. */
  name: string;
  /** File under public/kernels/, relative to that directory. */
  file: string;
  /** What it computes, for the UI. */
  description: string;
}

export const KERNEL_SPECS: KernelSpec[] = [
  {
    name: 'quantize_int8_symmetric',
    file: 'quantize_int8.naso',
    description: 'Symmetric int8 quantisation: q = clamp(round(x / s), -127, 127)',
  },
  {
    name: 'dequantize_int8_symmetric',
    file: 'quantize_int8.naso',
    description: 'Dequantisation: x = q * s',
  },
  {
    name: 'scale_clamp_f32',
    file: 'quantize_int8.naso',
    description: 'Elementwise scale and clamp (the reference kernel)',
  },
  {
    name: 'silu_f32',
    file: 'quantize_int8.naso',
    description: 'SiLU / Swish activation x / (1 + exp(-x))',
  },
  {
    name: 'relu_scale_f32',
    file: 'quantize_int8.naso',
    description: 'Fused scale + ReLU, via the exact identity (v + |v|) / 2',
  },
  {
    name: 'quantize_int4_symmetric',
    file: 'quantize_int4.naso',
    description: 'Symmetric int4 quantisation: q = clamp(round(x / s), -7, 7)',
  },
  {
    name: 'dequantize_int4_symmetric',
    file: 'quantize_int4.naso',
    description: 'Dequantisation: x = q * s',
  },
  {
    name: 'dequant_scale_clamp_int4',
    file: 'quantize_int4.naso',
    description: 'Fused int4 dequant + scale + clamp',
  },
];

const sourceCache = new Map<string, string>();

/** Fetch a kernel source file, caching the text in memory. */
export async function loadKernelSource(spec: KernelSpec): Promise<string> {
  const cached = sourceCache.get(spec.file);
  if (cached !== undefined) return cached;

  // The service worker serves this from cache when offline; the network is only
  // hit on the first load.
  const url = new URL(`kernels/${spec.file}`, document.baseURI).href;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: HTTP ${res.status}`);
  const text = await res.text();
  sourceCache.set(spec.file, text);
  return text;
}
