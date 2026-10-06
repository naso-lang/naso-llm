/* tslint:disable */
/* eslint-disable */

/**
 * Complete compilation result
 */
export class CompileResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly ast_json: string | undefined;
    readonly diagnostics: Diagnostic[];
    readonly inverse_dag_json: string | undefined;
    readonly success: boolean;
}

/**
 * Diagnostic information for a single error/warning
 */
export class Diagnostic {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly code: string | undefined;
    readonly column: number;
    readonly end_column: number;
    readonly end_line: number;
    readonly line: number;
    readonly message: string;
    readonly severity: string;
}

/**
 * Result of parsing Naso source code
 */
export class ParseResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly ast_json: string | undefined;
    readonly error: string | undefined;
    readonly success: boolean;
}

/**
 * Token for syntax highlighting
 */
export class WasmToken {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly column: number;
    readonly end: number;
    readonly kind: string;
    readonly line: number;
    readonly start: number;
    readonly text: string;
}

/**
 * Result of [`compile_naso_wgsl`].
 *
 * Private fields with explicit getters, matching `ParseResult`. wasm-bindgen's
 * derived getters require `Copy`, which `String` is not, so the getters are
 * written out rather than derived.
 */
export class WgslResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The whole result as one JSON object.
     *
     * One call for the common case, and the shape is produced by `serde` from the
     * same fields the getters read -- so it cannot describe a different result.
     */
    toJson(): string;
    /**
     * Storage bindings in binding-index order, each a JSON object:
     * `{"name","elem","access","index","bytes"}`.
     *
     * JSON strings rather than exported struct instances: wasm-bindgen cannot put a
     * struct with `String` fields into a `Vec`, and a hand-maintained getter list
     * is exactly the kind of thing that drifts from the field list.
     */
    readonly bindings: string[];
    readonly diagnostics: Diagnostic[];
    /**
     * `dispatchWorkgroups` count for a full run: ceil(elements / workgroup_size).
     */
    readonly dispatch_groups: number;
    /**
     * Name of the `@compute` entry point to create the pipeline with.
     */
    readonly entry_point: string;
    /**
     * Entry-point scalar arguments as `name: type`, in parameter order.
     */
    readonly scalars: string[];
    /**
     * Whether a shader was produced.
     */
    readonly success: boolean;
    /**
     * The shader source, or `None` on failure.
     */
    readonly wgsl: string | undefined;
    /**
     * x-dimension of `@workgroup_size`.
     */
    readonly workgroup_size: number;
}

/**
 * Compile Naso source code (parse + typecheck + lower)
 */
export function compile_naso_wasm(source: string): CompileResult;

/**
 * Compile one function to a WGSL compute shader, and describe its host ABI.
 *
 * This is the browser's path from Naso source to a GPU kernel. Until now the
 * WASM bridge stopped at PIR, so a page could parse and typecheck Naso but had no
 * way to obtain a shader -- which is why the shipped NasoChat client carried
 * hand-written WGSL that could drift from the language without anything noticing.
 *
 * The shader and the ABI come from ONE analysis (`wgsl_compute::analyze`), so the
 * host cannot be told to allocate buffers that disagree with the shader it will
 * run. It gets `binding(n)` element types, access modes, buffer sizes, entry-point
 * scalar argument types, workgroup size, and the dispatch count -- everything
 * `createBindGroup` / `dispatchWorkgroups` need, without parsing shader text.
 *
 * Returns the same diagnostics as `compile_naso_wasm` for a program that does not
 * typecheck, so the UI has one error path rather than two.
 */
export function compile_naso_wgsl(source: string, kernel: string): WgslResult;

/**
 * Create a default Naso program for the playground
 */
export function default_naso_program(): string;

export function init(): void;

/**
 * Parse Naso source code to AST
 */
export function parse_naso(source: string): ParseResult;

/**
 * Tokenize Naso source code
 */
export function tokenize_naso(source: string): WasmToken[];

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_compileresult_free: (a: number, b: number) => void;
    readonly __wbg_diagnostic_free: (a: number, b: number) => void;
    readonly __wbg_parseresult_free: (a: number, b: number) => void;
    readonly __wbg_wasmtoken_free: (a: number, b: number) => void;
    readonly __wbg_wgslresult_free: (a: number, b: number) => void;
    readonly compile_naso_wasm: (a: number, b: number) => number;
    readonly compile_naso_wgsl: (a: number, b: number, c: number, d: number) => number;
    readonly compileresult_ast_json: (a: number) => [number, number];
    readonly compileresult_diagnostics: (a: number) => [number, number];
    readonly compileresult_inverse_dag_json: (a: number) => [number, number];
    readonly compileresult_success: (a: number) => number;
    readonly default_naso_program: () => [number, number];
    readonly diagnostic_code: (a: number) => [number, number];
    readonly diagnostic_column: (a: number) => number;
    readonly diagnostic_end_column: (a: number) => number;
    readonly diagnostic_end_line: (a: number) => number;
    readonly diagnostic_line: (a: number) => number;
    readonly diagnostic_message: (a: number) => [number, number];
    readonly diagnostic_severity: (a: number) => [number, number];
    readonly init: () => void;
    readonly parse_naso: (a: number, b: number) => number;
    readonly parseresult_ast_json: (a: number) => [number, number];
    readonly parseresult_error: (a: number) => [number, number];
    readonly parseresult_success: (a: number) => number;
    readonly tokenize_naso: (a: number, b: number) => [number, number];
    readonly wasmtoken_column: (a: number) => number;
    readonly wasmtoken_end: (a: number) => number;
    readonly wasmtoken_kind: (a: number) => [number, number];
    readonly wasmtoken_line: (a: number) => number;
    readonly wasmtoken_start: (a: number) => number;
    readonly wasmtoken_text: (a: number) => [number, number];
    readonly wgslresult_bindings: (a: number) => [number, number];
    readonly wgslresult_diagnostics: (a: number) => [number, number];
    readonly wgslresult_dispatch_groups: (a: number) => number;
    readonly wgslresult_entry_point: (a: number) => [number, number];
    readonly wgslresult_scalars: (a: number) => [number, number];
    readonly wgslresult_success: (a: number) => number;
    readonly wgslresult_toJson: (a: number) => [number, number];
    readonly wgslresult_wgsl: (a: number) => [number, number];
    readonly wgslresult_workgroup_size: (a: number) => number;
    readonly "qir.apply1": (a: bigint, b: number) => void;
    readonly "qir.apply2": (a: bigint, b: bigint, c: number) => void;
    readonly "qir.apply3": (a: bigint, b: bigint, c: bigint) => void;
    readonly "qir.ccx": (a: bigint, b: bigint, c: bigint) => void;
    readonly "qir.cx": (a: bigint, b: bigint) => void;
    readonly "qir.cy": (a: bigint, b: bigint) => void;
    readonly "qir.cz": (a: bigint, b: bigint) => void;
    readonly "qir.h": (a: bigint) => void;
    readonly "qir.live_qubit_count": () => bigint;
    readonly "qir.mz": (a: bigint) => number;
    readonly "qir.probability_of_one": (a: bigint) => number;
    readonly "qir.qubit_alloc": () => bigint;
    readonly "qir.qubit_release": (a: bigint) => void;
    readonly "qir.r1": (a: number, b: bigint) => void;
    readonly "qir.s": (a: bigint) => void;
    readonly "qir.swap": (a: bigint, b: bigint) => void;
    readonly "qir.t": (a: bigint) => void;
    readonly "qir.x": (a: bigint) => void;
    readonly "qir.y": (a: bigint) => void;
    readonly "qir.z": (a: bigint) => void;
    readonly qir_amplitude: (a: bigint, b: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
