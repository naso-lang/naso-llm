#!/usr/bin/env python3
"""Build a pre-quantised int8 checkpoint for SmolLM2-135M-Instruct.

Purpose: stop downloading 269 MB of bf16 and quantising it in the browser. This
produces the artifact the browser would otherwise compute, so it can be fetched
directly.

The quantisation here MUST equal src/quantize.ts bit for bit. In particular the
rounding is HALF-UP (floor(x+0.5), what `Math.round` does), NOT numpy's default
half-to-even: on this checkpoint 88,677 elements (of 106,168,320) sit exactly on
a .5 tie and quantise differently under the two rules, so a numpy-default
builder would ship a checkpoint the app does not reproduce. tools/verify_quant.py
asserts the packed words are identical to the TypeScript implementation.

Format NPQ1 (self-describing, little-endian):
    magic     4s   "NPQ1"
    version   u32  1
    count     u32  number of tensors
    per tensor:
      name_len u32
      name     bytes (utf8)
      dtype    u8   (0 = f32, 1 = int8 packed, 2 = bf16)
      rank     u32
      dims     rank * u32   (the tensor's real shape, so a loader can rebuild it)
      data     f32  -> product(dims)*4 raw little-endian floats
               bf16 -> product(dims)*2 raw little-endian half-precision (a bf16 is
                       the top 16 bits of its f32, so this round-trips exactly)
               int8 -> (n*k/4)*4 packed u32, then n*4 f32 per-row scales, where
                       [n, k] = dims for a 2-D weight

Pass-through tensors are stored in their ORIGINAL dtype. Writing them as f32
would double the embedding (113 MB instead of 57 MB) and turn a 1.65x saving into
1.22x -- the first version of this builder did exactly that.

Usage: python3 tools/build_quantized.py /var/tmp/smol /var/tmp/model.int8.npq
"""
import json
import struct
import sys

import numpy as np

# Tensors quantised to int8: the projection matrices, which is what the app
# quantises today. The tied embedding stays f32 (it is also the output head, and
# it dominates the file -- see tools/quant_plan.py).
PROJ_SUFFIX = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
               "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
               "mlp.down_proj.weight")


def load_safetensors(path):
    """Return {name: (f32 array, original bf16 bit pattern or None)}."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dt = meta["dtype"]
            if dt == "BF16":
                bits = np.frombuffer(raw, dtype="<u2").copy()
                arr = (bits.astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                bits = None
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            else:
                bits = None
                arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            out[name] = (arr.reshape(meta["shape"]).astype(np.float32), dt, bits)
    return out


def quantise_like_ts(W):
    """Symmetric per-row int8 exactly as src/quantize.ts: Math.round = half-up.

    The division is done in float64 on purpose. JavaScript numbers are doubles,
    so `W[i] / scale` in quantize.ts (both f32 values widened to double) is an
    exact double division. numpy would compute it in float32 and round the
    quotient first, which flips the result at a handful of near-tie elements --
    it produced one wrong packed word in 6.7M before this was fixed.
    """
    n, k = W.shape
    mx = np.abs(W).max(axis=1)
    scales = np.where(mx == 0, 1.0, mx / 127.0).astype(np.float32)
    v = W.astype(np.float64) / scales.astype(np.float64)[:, None]
    q = np.floor(v + 0.5)                       # Math.round semantics (double)
    q = np.clip(q, -127, 127).astype(np.int64)
    # pack four signed bytes per u32, little-endian
    words = (q.reshape(n, k // 4, 4).astype(np.uint32) & 0xFF) << (8 * np.arange(4, dtype=np.uint32))
    packed = words.sum(axis=2, dtype=np.uint32).astype("<u4")
    return packed, scales


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/smol"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/var/tmp/model.int8.npq"
    t = load_safetensors(f"{src}/model.safetensors")

    entries = []
    for name in sorted(t):
        W, dt, bits = t[name]
        quantise = W.ndim == 2 and W.shape[1] % 4 == 0 and W.shape[0] % 4 == 0 \
            and name.endswith(PROJ_SUFFIX)
        entries.append((name, W, dt, bits, quantise))

    body = bytearray()
    n_int8 = n_pass = 0
    for name, W, dt, bits, quantise in entries:
        name_b = name.encode()
        dims = [int(d) for d in W.shape]
        head = struct.pack("<I", len(name_b)) + name_b + struct.pack("<BI", (1 if quantise else (2 if dt == "BF16" else 0)), len(dims))
        head += b"".join(struct.pack("<I", d) for d in dims)
        if quantise:
            packed, scales = quantise_like_ts(W)
            n, k = W.shape
            body += head + packed.tobytes() + scales.astype("<f4").tobytes()
            n_int8 += 1
        elif dt == "BF16":
            body += head + bits.astype("<u2").tobytes()     # original bits, bit-exact
            n_pass += 1
        else:
            body += head + W.reshape(-1).astype("<f4").tobytes()
            n_pass += 1

    with open(dst, "wb") as f:
        f.write(b"NPQ1" + struct.pack("<II", 1, len(entries)) + body)

    import os
    sz = os.path.getsize(dst)
    orig = os.path.getsize(f"{src}/model.safetensors")
    print(f"wrote {dst}")
    print(f"  tensors: {n_int8} int8, {n_pass} pass-through (original dtype, of {len(entries)})")
    print(f"  size   : {sz/1e6:.1f} MB  vs original bf16 {orig/1e6:.1f} MB  => {orig/sz:.2f}x smaller")


if __name__ == "__main__":
    main()
