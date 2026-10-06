# Naso LLM

**Browser LLM inference whose quantization kernels are compiled by Naso's formally verified compiler.**

A vanilla PWA — plain HTML, CSS and TypeScript, no framework — that loads a real
Llama checkpoint from Hugging Face, compiles Naso kernels to WGSL in the browser,
runs them on the GPU via WebGPU, and checks every result against an independent
reference. It works offline after the first visit.

```
kernels/quantize_int8.naso  ──►  nasoc-wasm (in the browser)  ──►  WGSL  ──►  WebGPU
                                        │
                                        └──►  host ABI (bindings, scalars, dispatch)
```

---

## Why this exists

Quantization kernels are numerically delicate. The whole point of int8
quantization is a *bound* — `|x − dequant(quant(x))| ≤ scale/2` — and the only
way to get a bound you can rely on is to know exactly what arithmetic ran.

Naso is a formally verified language (QTT, linear `[1]` types, uncomputation,
polyhedral codegen). Its compiler emits WGSL and has its output validated by
[naga](https://github.com/gfx-rs/wgpu), the same WGSL front-end wgpu and the
browsers use. This project is the end-to-end demonstration: the shader your GPU
executes is generated from a `.naso` source that has been parsed, typechecked and
lowered by that compiler — not a hand-written WGSL file committed next to it.

Three things make the claim checkable rather than aspirational:

1. **The WGSL is generated at runtime.** `kernels/quantize_int8.naso` is fetched
   and compiled by the WASM build of the compiler *in the page*. If you change
   the kernel, the shader changes; there is no copy to drift.
2. **The browser and native compilers agree byte-for-byte.**
   `tools/verify_bridge.mjs` compiles each kernel through both and diffs the
   output. `PASS … wgsl byte-identical to native`.
3. **The numbers are compared to a second implementation.** The forward pass in
   `src/llama.ts` is diffed against `tools/reference_llama.py`, an independent
   NumPy implementation, and the page re-checks its own logits against
   `golden.json` at runtime.

---

## Quick start

```bash
npm install
npm run compile:kernels     # native naso-compiler -> public/kernels/*.wgsl
npm run dev                 # http://localhost:3000
```

`npm run dev` serves the app; open it in a WebGPU-capable browser
(Chrome/Edge 113+, or a recent Firefox).

### Scripts

| command | what it does |
| --- | --- |
| `npm run dev` | Vite dev server |
| `npm run build` | typecheck + production build to `dist/` |
| `npm run compile:kernels` | compile every kernel to WGSL + ABI via the native compiler |
| `npm run verify` | **run all checks**: typecheck, bridge equivalence, forward-pass vs. NumPy |
| `npm run verify:forward` | diff `src/llama.ts` against the NumPy reference |
| `npm run verify:bridge` | assert browser-compiled WGSL == native WGSL |
| `npm run validate:wgsl` | validate every generated shader with naga |
| `npm run smoke` | headless-browser check of the served app |

### Requirements

- Node 18+
- The [naso compiler](https://github.com/naso-lang/naso) checked out, by default
  at `/home/node/naso` (override with `NASO_REPO=/path/to/naso` or
  `npm run compile:kernels -- --repo /path/to/naso`).
- A WebGPU-capable browser for the GPU path. Everything else runs anywhere.

---

## What actually runs where

This is the part worth being precise about, because it is the part a demo is
tempted to blur.

**Runs on the GPU, through Naso-generated shaders:**

- the int8 **quantize** and **dequantize** round-trip applied to every
  projection matrix in the checkpoint,
- the reference **scale + clamp** and **SiLU** kernels.

For each of these the WGSL is compiled from `.naso` in the page, dispatched, and
the result read back and measured.

**Does not run on the GPU (and is not claimed to):**

- the matmuls and attention of the forward pass themselves. They run on the CPU
  in f32 (`src/llama.ts`).

The reason is a real limitation of the current Naso WGSL backend, not a
shortcut: a compute entry point indexes *every* binding with the same index, so a
reduction (a dot product) is **refused at codegen** with

```
kernel `probe_k` has tensor parameters with differing extents [4, 16, 4].
One compute entry point indexes every binding with the same index, so a single
guard cannot be correct for all of them.
```

That refusal is the honest behaviour — better a refusal than a shader that
silently computes a wrong index. Fusing a real GEMM is future work (the
straight-line backend, or a reduction the compute backend grows support for).

A second limitation found while building this, and documented because it is a
soundness bug rather than a missing feature:

> **`if` / `else` inside a kernel body emits `/* unsupported */;` in place of
> both branches, and the compiler still exits 0** (`--target wgsl`). No
> diagnostic is raised. Until that is fixed, kernels here avoid conditionals.

Both findings are reproduced by `npm run probe` (see `tools/probe-backend.mjs`).

---

## Repository layout

```
kernels/quantize_int8.naso   the canonical kernels (source of truth)
public/kernels/              .naso copies + generated .wgsl / .abi.json
public/pkg/                  wasm-pack build of crates/nasoc-wasm
src/
  naso.ts        WASM bridge: Naso source -> { wgsl, ABI }
  webgpu.ts      engine: buffers and pipelines from the ABI, no shader parsing
  pipeline.ts    quantize every weight through the GPU, run the forward pass
  llama.ts       CPU f32 forward pass (RMSNorm, RoPE, GQA, SwiGLU)
  model.ts       safetensors reader (f32/f16/bf16)
  tokenizer.ts   tokenizer.json reader (vocabulary lookup; see the caveat below)
  kernels.ts     kernel registry
  types.ts       shared types, incl. ComputeAbi
  main.ts        UI wiring
tools/
  reference_llama.py   independent NumPy forward pass (the oracle)
  verify_forward.ts    diff src/llama.ts against the oracle
  verify_bridge.mjs    browser-compiled WGSL vs native WGSL
  wgsl-validate/       naga-backed WGSL validator
  probe-backend.mjs    reproduce the backend findings above
  golden.json          reference logits used by the in-page check
scripts/
  compile-kernels.js   native kernel compilation
  build.mjs / smoke.mjs / ...
```

## The kernel ABI

The compiler emits a shader *and* a description of its buffers, from the same
analysis, so the host never parses WGSL to discover a binding:

```json
{
  "entryPoint": "quantize_int8_symmetric_compute",
  "workgroupSize": 64,
  "dispatchGroups": 16,
  "bindings": [
    { "name": "input",  "elem": "f32", "access": "read",       "index": 0, "bytes": 4096 },
    { "name": "output", "elem": "i32", "access": "read_write", "index": 1, "bytes": 4096 }
  ],
  "scalars": [ { "name": "scale", "type": "f32", "index": 2 } ]
}
```

Two details the compiler decides, which the host must not assume:

- **Scalars are uniform bindings**, not entry-point arguments. A WGSL compute
  entry point may take only builtins, so `scale: f32` becomes
  `@group(0) @binding(2) var<uniform> scale_u: f32`. The ABI reports the index.
- **Integer storage is `i32`, not `i8`.** WGSL has no i8 storage class, so an
  `i8` tensor is refused rather than silently widened; int8 codes are carried as
  `i32` clamped to `[-127, 127]`.

## Verification

```
npm run verify
```

runs, in order:

1. `tsc --noEmit` — the app typechecks.
2. `verify_bridge.mjs` — for each kernel, compile with the WASM build and the
   native build and assert the WGSL is identical; assert each ABI binding
   appears verbatim in the shader.
3. `reference_llama.py` then `verify_forward.ts` — compute the forward pass twice
   (NumPy, TypeScript) and report `max |ts − numpy|` and whether the argmax
   matches. Currently **5.96e-8** and a matching argmax on the bundled model.
4. `wgsl-validate` — parse and validate every generated shader with naga.

`npm run smoke` additionally loads the served app in Chromium, confirms the
in-browser compiler produced all five kernels, and (where a WebGPU adapter
exists) runs the full quantization + forward pass and checks the half-step bound.

**The container this was developed in has no WebGPU adapter** (no Vulkan driver
for Dawn/SwiftShader), so `smoke` reports `SMOKE: PASS (compile-only)` and the
GPU dispatch is untested there — by the environment, not the code. Everything
that does not need a device is covered.

## Offline

The service worker (`public/sw.js`) uses three caches: the app shell, the model
weights (Hugging Face, immutable, cache-first), and the compiler artefacts. After
one online visit the app runs with the network off — including a second forward
pass, since the checkpoint is cached.

## Honest limitations

- **The tokenizer is a vocabulary lookup, not BPE.** `src/tokenizer.ts` maps
  word pieces to ids directly and emits `<unk>` for anything not a literal vocab
  entry. It is enough to run a forward pass and read the next token; it is not a
  general tokenizer. Stated in the module, not hidden.
- **The default model is tiny** (`hf-internal-testing/tiny-random-LlamaForCausalLM`,
  2 layers, hidden 16, ~4 MB) so the whole thing runs and can be checked in a tab.
  The architecture is a complete Llama — RMSNorm, RoPE, GQA, SwiGLU, tied LM
  head — so the same code path holds for a larger checkpoint; the model is
  selectable in `src/types.ts`. Phi-2 / TinyLlama are not wired up because their
  weights do not fit the browser here and cannot be executed in this
  environment, so shipping a path that had never run would be exactly the kind of
  unverified claim this project is meant to avoid.
- **No KV cache.** One forward pass over the prompt, as specified.

## License

Apache-2.0 (matching the naso compiler).
