import type { Binding, ComputeAbi, Diagnostic, Kernel, ScalarParam } from './types.js';
import { logger } from './logger.js';

// The wasm-pack build of crates/nasoc-wasm. `init` is the module initializer;
// the rest are the #[wasm_bindgen] exports. Types come from src/nasoc-wasm.d.ts
// (the generated .d.ts is excluded from tsc: wasm-bindgen emits qualified names
// like `qir.apply1` in type positions, which is a TypeScript syntax error, not
// a type error, so skipLibCheck cannot suppress it).
import init, { compile_naso_wgsl, compile_naso_wasm } from 'nasoc-wasm';

let ready = false;
let initPromise: Promise<void> | null = null;

/** Load and initialize the Naso WASM module. Idempotent. */
export async function initNaso(): Promise<void> {
  if (ready) return;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    logger.info('naso', 'Loading Naso compiler (WebAssembly)...');
    const t0 = performance.now();
    await init();
    ready = true;
    logger.success('naso', `Naso compiler ready in ${(performance.now() - t0).toFixed(0)} ms`);
  })();
  return initPromise;
}

export function isNasoReady(): boolean {
  return ready;
}

function toDiagnostics(raw: readonly { severity: string; message: string; line: number; column: number; code?: string | undefined }[]): Diagnostic[] {
  return raw.map(d => ({
    severity: d.severity,
    message: d.message,
    line: d.line,
    column: d.column,
    code: d.code ?? null,
  }));
}

/**
 * Compile one Naso kernel to a WGSL compute shader and its host ABI.
 *
 * Both come out of the same compiler analysis, so the buffer layout the host
 * is handed cannot disagree with the shader. If the kernel is refused (a
 * feature the WGSL backend does not support, e.g. an i8 tensor), `success` is
 * false and no shader is produced -- the refusal is explicit, never a silently
 * widened shader.
 */
export async function compileKernel(source: string, kernel: string): Promise<{ kernel: Kernel | null; diagnostics: Diagnostic[] }> {
  await initNaso();
  const r = compile_naso_wgsl(source, kernel);
  const diagnostics = toDiagnostics(r.diagnostics);

  if (!r.success || !r.wgsl) {
    logger.error('naso', `Refused to compile "${kernel}": ${diagnostics.map(d => d.message).join('; ') || 'unknown error'}`);
    return { kernel: null, diagnostics };
  }

  // Each binding is a JSON object: {"name","elem","access","index","bytes"}.
  const bindings: Binding[] = r.bindings.map(raw => JSON.parse(raw) as Binding);

  // Scalars are reported as uniform bindings named `<name>_u`, indexed after
  // the storage bindings. The strings arrive as "name: type".
  const scalars: ScalarParam[] = r.scalars.map((s, i) => {
    const sep = s.indexOf(': ');
    const name = sep < 0 ? s.trim() : s.slice(0, sep).trim();
    const type = sep < 0 ? 'f32' : s.slice(sep + 2).trim();
    return { name, type, index: bindings.length + i };
  });

  // The dispatch covers `extent` elements at `workgroupSize` per group.
  const elements = bindings.length > 0 ? Math.floor(bindings[0].bytes / 4) : r.dispatch_groups * r.workgroup_size;

  const abi: ComputeAbi = {
    entryPoint: r.entry_point,
    workgroupSize: r.workgroup_size,
    dispatchGroups: r.dispatch_groups,
    elements,
    bindings,
    scalars,
  };

  logger.success('naso', `Compiled "${kernel}" -> ${abi.entryPoint}: ${bindings.length} storage binding(s), ${scalars.length} scalar uniform(s), ${abi.dispatchGroups} workgroup(s)`);

  return { kernel: { name: kernel, wgsl: r.wgsl, abi }, diagnostics };
}

/** Parse + typecheck Naso source without generating a shader. Used by the UI. */
export async function checkNaso(source: string): Promise<{ ok: boolean; diagnostics: Diagnostic[] }> {
  await initNaso();
  const r = compile_naso_wasm(source);
  return { ok: r.success, diagnostics: toDiagnostics(r.diagnostics) };
}
