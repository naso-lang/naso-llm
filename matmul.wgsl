// Baseline matrix-vector product for the Llama forward pass.
//
// This shader is deliberately HAND-WRITTEN and is labelled as such in the UI.
// It is NOT produced by Naso. Naso's WGSL backend emits element-wise kernels
// only (no reductions, all tensor bindings share one extent), so a dot-product
// is not expressible today -- see README, "What Naso proves here, and what it
// does not". The Naso-generated kernels in public/kernels/ do the quantisation
// work; this file exists so the chat can run at a usable speed, and so that the
// boundary between the two is explicit rather than blurred.
//
// Layout: W is row-major [N, K] (the safetensors layout for `*.weight`), and
//   out[o] = sum_k W[o, k] * x[k]
// which is exactly what src/llama.ts's `linear()` computes on the CPU. Each
// thread produces FOUR consecutive outputs so the `x` load is reused four
// times from registers.
//
// Rotating the row base into a bitmask at the end of the sum is unnecessary
// here: the row is read sequentially, so the loop is bandwidth-friendly without
// staging x in workgroup memory.

struct Params {
  n: u32,   // output rows
  k: u32,   // input width
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.n;
  let k = params.k;
  let o0 = gid.x * 4u;
  if (o0 >= n) {
    return;
  }

  var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  if (o0 + 3u < n) {
    // Fast path: all four rows exist, so the inner loop has no per-row branch.
    let w0 = o0 * k;
    let w1 = w0 + k;
    let w2 = w1 + k;
    let w3 = w2 + k;
    for (var j: u32 = 0u; j < k; j = j + 1u) {
      let xv = x[j];
      acc.x = acc.x + W[w0 + j] * xv;
      acc.y = acc.y + W[w1 + j] * xv;
      acc.z = acc.z + W[w2 + j] * xv;
      acc.w = acc.w + W[w3 + j] * xv;
    }
  } else {
    // Tail path: fewer than four rows remain, so guard each one.
    for (var j: u32 = 0u; j < k; j = j + 1u) {
      let xv = x[j];
      if (o0 < n) { acc.x = acc.x + W[o0 * k + j] * xv; }
      if (o0 + 1u < n) { acc.y = acc.y + W[(o0 + 1u) * k + j] * xv; }
      if (o0 + 2u < n) { acc.z = acc.z + W[(o0 + 2u) * k + j] * xv; }
      if (o0 + 3u < n) { acc.w = acc.w + W[(o0 + 3u) * k + j] * xv; }
    }
  }

  if (o0 < n) { out[o0] = acc.x; }
  if (o0 + 1u < n) { out[o0 + 1u] = acc.y; }
  if (o0 + 2u < n) { out[o0 + 2u] = acc.z; }
  if (o0 + 3u < n) { out[o0 + 3u] = acc.w; }
}
