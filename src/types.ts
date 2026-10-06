/**
 * Shared types for the Naso LLM demo.
 *
 * The important one is `ComputeAbi`: it is the contract between the host
 * (this app) and the shader (emitted by the verified compiler). The host
 * allocates its buffers ONLY from the ABI -- it never parses WGSL text -- so
 * the ABI and the shader must describe the same memory layout. The compiler
 * generates both from the same analysis, which is what makes that true.
 */

/** One storage binding, as serialized by the wasm bridge. */
export interface Binding {
  name: string;
  /** Element type: 'f32' | 'i32' | 'u32'. */
  elem: string;
  /** 'read' for [1] linear input, 'read_write' for inout [1]. */
  access: string;
  /** @binding index within @group(0). */
  index: number;
  /** Buffer size in bytes for this binding's extent. */
  bytes: number;
}

export interface ScalarParam {
  name: string;
  type: string;
  /** Binding index assigned to this uniform by the backend. */
  index: number;
}

/**
 * The host-side view of a compiled compute kernel.
 *
 * `scalars` are UNIFORM BINDINGS, not entry-point arguments. A WGSL compute
 * entry point may take only builtin values, so the backend emits scalars as
 * `@group(0) @binding(n) var<uniform> <name>_u: T`. The ABI reports the index
 * so the host can build one bind group without reading the shader.
 */
export interface ComputeAbi {
  entryPoint: string;
  workgroupSize: number;
  dispatchGroups: number;
  /** Total elements the dispatch covers. */
  elements: number;
  bindings: Binding[];
  scalars: ScalarParam[];
}

/** A compiled kernel: shader text plus the ABI that describes its buffers. */
export interface Kernel {
  name: string;
  wgsl: string;
  abi: ComputeAbi;
}

export interface Diagnostic {
  severity: string;
  message: string;
  line: number;
  column: number;
  code: string | null;
}

export interface LogEntry {
  timestamp: number;
  level: 'info' | 'success' | 'warn' | 'error' | 'debug';
  source: 'main' | 'webgpu' | 'naso';
  message: string;
}

/**
 * A model whose config.json / safetensors this app can fetch.
 *
 * `tiny-random-LlamaForCausalLM` is the default: a real, complete Llama
 * architecture (RMSNorm, QKV projection, RoPE, GQA, SwiGLU MLP, tied logits)
 * with tiny dimensions, so the whole forward pass runs in a browser tab. A
 * 2.7B model is not the right first target for a correctness demo -- the
 * maths is identical, and the tiny one can actually be executed and checked.
 */
export interface ModelConfig {
  id: string;
  name: string;
  /** Hugging Face repo id; files are fetched from resolve/main/. */
  repo: string;
  hiddenSize: number;
  intermediateSize: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  vocabSize: number;
  rmsNormEps: number;
  ropeTheta: number;
  /** Bytes of model.safetensors, for a progress bar. */
  weightsBytes: number;
}

export const MODELS: ModelConfig[] = [
  /**
   * The chat model. 135M params, real instruct tuning (SmolLM2's SFT+DPO mix),
   * ChatML template, tied embeddings. 269 MB in bf16, which is small enough to
   * cache for offline use and small enough to run client-side. GQA (9 query
   * heads, 3 KV heads) and rope_theta 100000 are both non-default, so this
   * exercises the same code paths a larger Llama would.
   */
  {
    id: 'smollm2-135m-instruct',
    name: 'SmolLM2-135M-Instruct',
    repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    hiddenSize: 576,
    intermediateSize: 1536,
    numLayers: 30,
    numHeads: 9,
    numKvHeads: 3,
    vocabSize: 49152,
    rmsNormEps: 1e-5,
    ropeTheta: 100000.0,
    weightsBytes: 269060552,
  },
  /**
   * The quantisation-kernel fixture: a complete Llama architecture at tiny
   * dimensions. Used to exercise Naso's WGSL kernels and the byte-exact
   * browser-vs-native compile check without a 269 MB download.
   */
  {
    id: 'tiny-random-llama',
    name: 'tiny-random-LlamaForCausalLM (2 layers, hidden 16)',
    repo: 'hf-internal-testing/tiny-random-LlamaForCausalLM',
    hiddenSize: 16,
    intermediateSize: 64,
    numLayers: 2,
    numHeads: 4,
    numKvHeads: 4,
    vocabSize: 32000,
    rmsNormEps: 1e-6,
    ropeTheta: 10000.0,
    weightsBytes: 4131280,
  },
];

/** The model the chat UI loads by default. */
export const DEFAULT_MODEL = MODELS[0];

/** Context window used by the chat demo (tokens kept in the KV cache). */
export const CHAT_MAX_SEQ = 1024;

export const HF_BASE = 'https://huggingface.co';
