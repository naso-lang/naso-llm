// Int8 weight-only matrix-vector product for the Llama forward pass.
//
// HAND-WRITTEN, not Naso-generated -- see README, "What Naso proves here, and
// what it does not". Naso's WGSL backend emits element-wise kernels only (no
// reductions, and every tensor binding must share one extent), so a dot product
// is not expressible in Naso today. The Naso-generated kernels in
// public/kernels/ produce the int8 weights and their scales that this shader
// consumes; the reduction itself is plain WGSL.
//
// Layout:
//   * `Wq` packs four int8 weights per u32, little-endian:
//     byte i of word w of row o is the quantised weight W[o, 4*w + i].
//     Row o therefore starts at word o * (k/4).
//   * `scales[o]` is the per-row dequantisation scale: W[o, j] ~= q * scale[o].
//   * `x` is f32 activations (not quantised -- this is weight-only int8, the
//     same trade GPTQ/AWQ make: activations stay in float).
//
// Storage is 4x smaller than f32, which is the whole point: 28 MB of packed
// weights for the 49,152 x 576 output projection instead of 113 MB.
//
// Cost note: the inner loop does one u32 load and four unpack+dequant+MAC steps
// per four weights, which is instruction-heavy compared with a plain f32 row.
// For a 135M model that is still far more bandwidth-efficient than f32, but at
// larger sizes a packed-dot-product submission (DP4A-style) would be the next
// step; WGSL has no integer dot intrinsic, so the unpack below is manual.

struct Params {
  n: u32,      // output rows
  k: u32,      // input width (a multiple of 4)
  kwords: u32, // k / 4
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> Wq: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

/// Sign-extend int8 stored in byte `i` of a packed word.
///
/// `b ^ 0x80` flips the sign bit, so the byte becomes an unsigned value offset
/// by +128; reinterpreting as i32 and subtracting 128 recovers the signed value
/// without relying on a signed shift intrinsic.
fn unpack_i8(packed: u32, byte_index: u32) -> f32 {
  let b = (packed >> (byte_index * 8u)) & 0xffu;
  return f32(bitcast<i32>(b ^ 0x80u) - 128);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= params.n) {
    return;
  }

  let kwords = params.kwords;
  let row_base = o * kwords;
  let scale = scales[o];

  var acc = 0.0;
  for (var w: u32 = 0u; w < kwords; w = w + 1u) {
    let packed = Wq[row_base + w];
    let j = w * 4u;
    acc = acc + unpack_i8(packed, 0u) * x[j];
    acc = acc + unpack_i8(packed, 1u) * x[j + 1u];
    acc = acc + unpack_i8(packed, 2u) * x[j + 2u];
    acc = acc + unpack_i8(packed, 3u) * x[j + 3u];
  }

  out[o] = acc * scale;
}
