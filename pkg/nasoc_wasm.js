/* @ts-self-types="./nasoc_wasm.d.ts" */

/**
 * Complete compilation result
 */
export class CompileResult {
    static __wrap(ptr) {
        const obj = Object.create(CompileResult.prototype);
        obj.__wbg_ptr = ptr;
        CompileResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CompileResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_compileresult_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get ast_json() {
        const ret = wasm.compileresult_ast_json(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {Diagnostic[]}
     */
    get diagnostics() {
        const ret = wasm.compileresult_diagnostics(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get inverse_dag_json() {
        const ret = wasm.compileresult_inverse_dag_json(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {boolean}
     */
    get success() {
        const ret = wasm.compileresult_success(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) CompileResult.prototype[Symbol.dispose] = CompileResult.prototype.free;

/**
 * Diagnostic information for a single error/warning
 */
export class Diagnostic {
    static __wrap(ptr) {
        const obj = Object.create(Diagnostic.prototype);
        obj.__wbg_ptr = ptr;
        DiagnosticFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DiagnosticFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_diagnostic_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get code() {
        const ret = wasm.diagnostic_code(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {number}
     */
    get column() {
        const ret = wasm.diagnostic_column(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get end_column() {
        const ret = wasm.diagnostic_end_column(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get end_line() {
        const ret = wasm.diagnostic_end_line(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get line() {
        const ret = wasm.diagnostic_line(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get message() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.diagnostic_message(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get severity() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.diagnostic_severity(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) Diagnostic.prototype[Symbol.dispose] = Diagnostic.prototype.free;

/**
 * Result of parsing Naso source code
 */
export class ParseResult {
    static __wrap(ptr) {
        const obj = Object.create(ParseResult.prototype);
        obj.__wbg_ptr = ptr;
        ParseResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ParseResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_parseresult_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get ast_json() {
        const ret = wasm.parseresult_ast_json(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get error() {
        const ret = wasm.parseresult_error(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {boolean}
     */
    get success() {
        const ret = wasm.parseresult_success(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) ParseResult.prototype[Symbol.dispose] = ParseResult.prototype.free;

/**
 * Token for syntax highlighting
 */
export class WasmToken {
    static __wrap(ptr) {
        const obj = Object.create(WasmToken.prototype);
        obj.__wbg_ptr = ptr;
        WasmTokenFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTokenFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtoken_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get column() {
        const ret = wasm.wasmtoken_column(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get end() {
        const ret = wasm.wasmtoken_end(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get kind() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmtoken_kind(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get line() {
        const ret = wasm.wasmtoken_line(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start() {
        const ret = wasm.wasmtoken_start(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get text() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmtoken_text(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) WasmToken.prototype[Symbol.dispose] = WasmToken.prototype.free;

/**
 * Result of [`compile_naso_wgsl`].
 *
 * Private fields with explicit getters, matching `ParseResult`. wasm-bindgen's
 * derived getters require `Copy`, which `String` is not, so the getters are
 * written out rather than derived.
 */
export class WgslResult {
    static __wrap(ptr) {
        const obj = Object.create(WgslResult.prototype);
        obj.__wbg_ptr = ptr;
        WgslResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WgslResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wgslresult_free(ptr, 0);
    }
    /**
     * Storage bindings in binding-index order, each a JSON object:
     * `{"name","elem","access","index","bytes"}`.
     *
     * JSON strings rather than exported struct instances: wasm-bindgen cannot put a
     * struct with `String` fields into a `Vec`, and a hand-maintained getter list
     * is exactly the kind of thing that drifts from the field list.
     * @returns {string[]}
     */
    get bindings() {
        const ret = wasm.wgslresult_bindings(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Diagnostic[]}
     */
    get diagnostics() {
        const ret = wasm.wgslresult_diagnostics(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * `dispatchWorkgroups` count for a full run: ceil(elements / workgroup_size).
     * @returns {number}
     */
    get dispatch_groups() {
        const ret = wasm.wgslresult_dispatch_groups(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Name of the `@compute` entry point to create the pipeline with.
     * @returns {string}
     */
    get entry_point() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wgslresult_entry_point(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Entry-point scalar arguments as `name: type`, in parameter order.
     * @returns {string[]}
     */
    get scalars() {
        const ret = wasm.wgslresult_scalars(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Whether a shader was produced.
     * @returns {boolean}
     */
    get success() {
        const ret = wasm.wgslresult_success(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The whole result as one JSON object.
     *
     * One call for the common case, and the shape is produced by `serde` from the
     * same fields the getters read -- so it cannot describe a different result.
     * @returns {string}
     */
    toJson() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wgslresult_toJson(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The shader source, or `None` on failure.
     * @returns {string | undefined}
     */
    get wgsl() {
        const ret = wasm.wgslresult_wgsl(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * x-dimension of `@workgroup_size`.
     * @returns {number}
     */
    get workgroup_size() {
        const ret = wasm.wgslresult_workgroup_size(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WgslResult.prototype[Symbol.dispose] = WgslResult.prototype.free;

/**
 * Compile Naso source code (parse + typecheck + lower)
 * @param {string} source
 * @returns {CompileResult}
 */
export function compile_naso_wasm(source) {
    const ptr0 = passStringToWasm0(source, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compile_naso_wasm(ptr0, len0);
    return CompileResult.__wrap(ret);
}

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
 * @param {string} source
 * @param {string} kernel
 * @returns {WgslResult}
 */
export function compile_naso_wgsl(source, kernel) {
    const ptr0 = passStringToWasm0(source, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(kernel, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.compile_naso_wgsl(ptr0, len0, ptr1, len1);
    return WgslResult.__wrap(ret);
}

/**
 * Create a default Naso program for the playground
 * @returns {string}
 */
export function default_naso_program() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.default_naso_program();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

export function init() {
    wasm.init();
}

/**
 * Parse Naso source code to AST
 * @param {string} source
 * @returns {ParseResult}
 */
export function parse_naso(source) {
    const ptr0 = passStringToWasm0(source, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.parse_naso(ptr0, len0);
    return ParseResult.__wrap(ret);
}

/**
 * Tokenize Naso source code
 * @param {string} source
 * @returns {WasmToken[]}
 */
export function tokenize_naso(source) {
    const ptr0 = passStringToWasm0(source, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tokenize_naso(ptr0, len0);
    var v2 = getArrayJsValueFromWasm0(ret[0], ret[1]);
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_5d9e815e6fdf150f: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_diagnostic_new: function(arg0) {
            const ret = Diagnostic.__wrap(arg0);
            return ret;
        },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_wasmtoken_new: function(arg0) {
            const ret = WasmToken.__wrap(arg0);
            return ret;
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./nasoc_wasm_bg.js": import0,
    };
}

const CompileResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_compileresult_free(ptr, 1));
const DiagnosticFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_diagnostic_free(ptr, 1));
const ParseResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_parseresult_free(ptr, 1));
const WasmTokenFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtoken_free(ptr, 1));
const WgslResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wgslresult_free(ptr, 1));

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('nasoc_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
