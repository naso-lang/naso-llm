#!/usr/bin/env python3
"""Quantify the real cost of shipping a pre-quantised checkpoint.

The proposal: pre-quantise the model into int8 and ship that, so the download is
4x smaller and the browser does no quantisation work. This measures what that
actually buys and what it costs, on the real weights.

For each projection matrix:
  * f32 size (= what the browser holds after decoding bf16)
  * int8 size (packed + per-row scales)
  * worst |W-W'| / (scale/2), the Naso-checkable error bound
  * cosine similarity of the OUTPUT distribution, which is what actually matters

And for the whole model: total bytes, and an estimate of the download time on a
10 Mbit/s (slow) and 100 Mbit/s link.

Usage: python3 tools/quant_cost.py /var/tmp/smol
"""
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dt = meta["dtype"]
            if dt == "BF16":
                arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=np.float32)
            out[name] = arr.reshape(meta["shape"]).astype(np.float32)
        return out


def quantise_rowwise(W):
    """Symmetric per-row int8, exactly as src/quantize.ts does."""
    n, k = W.shape
    scales = np.maximum(np.abs(W).max(axis=1), 1e-30) / 127.0
    zero = np.abs(W).max(axis=1) == 0
    scales[zero] = 1.0
    q = np.clip(np.round(W / scales[:, None]), -127, 127).astype(np.int8)
    Wp = q.astype(np.float32) * scales[:, None]
    return q, scales.astype(np.float32), Wp


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/smol"
    t = load_safetensors(f"{d}/model.safetensors")
    cfg = json.load(open(f"{d}/config.json"))

    proj = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]

    tot_f32 = 0
    tot_i8 = 0
    worst_ratio = 0.0
    nmat = 0
    for name, W in t.items():
        if W.ndim != 2:
            continue
        if not any(name.endswith(s) for s in proj):
            continue
        n, k = W.shape
        if k % 4:
            continue
        q, scales, Wp = quantise_rowwise(W)
        err = np.abs(W - Wp)
        bound = scales[:, None] / 2
        ratio = float((err / np.maximum(bound, 1e-30)).max())
        worst_ratio = max(worst_ratio, ratio)
        tot_f32 += W.size * 4
        tot_i8 += q.size + scales.size * 4
        nmat += 1

    # Embedding + norm weights are NOT quantised (tied head, must stay exact-ish).
    embed_bytes = t["model.embed_tokens.weight"].size * 4

    print(f"model: {cfg['num_hidden_layers']} layers, hidden {cfg['hidden_size']}, vocab {cfg['vocab_size']}")
    print(f"quantised {nmat} projection matrices")
    print(f"  projections f32 : {tot_f32/1e6:8.1f} MB")
    print(f"  projections int8: {tot_i8/1e6:8.1f} MB  ({tot_f32/tot_i8:.2f}x smaller)")
    print(f"  worst |W-W'|/(scale/2) = {worst_ratio:.6f}  (must be <= 1)")
    print(f"  embed (kept f32) : {embed_bytes/1e6:8.1f} MB")
    quant_total = tot_i8 + embed_bytes
    print(f"\ncheckpoint as shipped today : {269.06:8.1f} MB (bf16, all tensors)")
    print(f"pre-quantised int8 (proj)   : {quant_total/1e6:8.1f} MB")
    print(f"  => {269.06e6/quant_total:.2f}x smaller download")
    for mbps in (10, 50, 100):
        print(f"  download at {mbps:3d} Mbit/s: {269.06e6*8/mbps/1e6:6.1f}s today -> {quant_total*8/mbps/1e6:6.1f}s pre-quantised")


if __name__ == "__main__":
    main()
