/**
 * Ambient types for the wasm-pack generated `nasoc-wasm` bindings.
 *
 * The generated `.d.ts` emits qualified names like `qir.apply1` in type
 * positions, which is not valid TypeScript (`error TS1131: Property or
 * signature expected`). `skipLibCheck` does not help — it suppresses TYPE
 * errors, not syntax errors. So the package is excluded from the tsc program
 * and its public surface is declared here instead, matching the real
 * exports exactly (see public/pkg/nasoc_wasm.d.ts).
 */
declare module 'nasoc-wasm' {
  export class Diagnostic {
    readonly code: string | undefined;
    readonly column: number;
    readonly end_column: number;
    readonly end_line: number;
    readonly line: number;
    readonly message: string;
    readonly severity: string;
    free(): void;
  }

  export class WgslResult {
    readonly success: boolean;
    readonly wgsl: string | undefined;
    readonly entry_point: string;
    readonly workgroup_size: number;
    readonly dispatch_groups: number;
    readonly bindings: string[];
    readonly scalars: string[];
    readonly diagnostics: Diagnostic[];
    toJson(): string;
    free(): void;
  }

  export class CompileResult {
    readonly success: boolean;
    readonly ast_json: string | undefined;
    readonly inverse_dag_json: string | undefined;
    readonly diagnostics: Diagnostic[];
    free(): void;
  }

  export class ParseResult {
    readonly success: boolean;
    readonly ast_json: string | undefined;
    readonly error: string | undefined;
    free(): void;
  }

  export class WasmToken {
    readonly kind: string;
    readonly text: string;
    readonly start: number;
    readonly end: number;
    readonly line: number;
    readonly column: number;
    free(): void;
  }

  export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

  export function compile_naso_wgsl(source: string, kernel: string): WgslResult;
  export function compile_naso_wasm(source: string): CompileResult;
  export function default_naso_program(): string;
  export function parse_naso(source: string): ParseResult;
  export function tokenize_naso(source: string): WasmToken[];
  export function init(): void;
  export function initSync(module: { module: InitInput } | InitInput): unknown;

  export default function __wbg_init(
    module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>,
  ): Promise<unknown>;
}
