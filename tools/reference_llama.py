#!/usr/bin/env python3
"""Independent NumPy reference forward pass for tiny-random-LlamaForCausalLM.

Written from the Llama paper / HF modeling_llama conventions, deliberately NOT
by translating the TypeScript. The point is to have a second, independent
implementation to check src/llama.ts against: if both agree to f32 round-off,
the TypeScript is right; if they disagree, one of them is wrong and the diff
localises it.

The container has no torch and no onnxruntime, so this is the oracle.

Usage:
  python3 tools/reference_llama.py <model_dir> [token_id ...]
Prints the top-10 next-token logits for the last position.
"""
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        f.seek(base)
        raw = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dtype = np.float32 if meta["dtype"] == "F32" else None
        assert dtype is not None, f"unsupported dtype {meta['dtype']}"
        arr = np.frombuffer(raw, dtype=dtype, count=(end - start) // 4, offset=start)
        out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


def rms_norm(x, w, eps):
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(x, positions, head_dim, theta):
    # x: [T, heads, head_dim]; rotate pairs (i, i+half).
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
    freqs = positions[:, None] * inv_freq[None, :]  # [T, half]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    a = x[..., :half]
    b = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = a * cos[:, None, :] - b * sin[:, None, :]
    out[..., half:] = a * sin[:, None, :] + b * cos[:, None, :]
    return out


def forward(tensors, cfg, input_ids):
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    eps = cfg["rms_norm_eps"]
    theta = cfg.get("rope_theta", 10000.0)
    vocab = cfg["vocab_size"]
    T = len(input_ids)
    head_dim = H // NH
    kv_repeat = NH // NKV

    h = tensors["model.embed_tokens.weight"][input_ids].astype(np.float32)  # [T,H]

    for layer in range(L):
        p = f"model.layers.{layer}"
        residual = h
        normed = rms_norm(h, tensors[f"{p}.input_layernorm.weight"], eps)

        q = normed @ tensors[f"{p}.self_attn.q_proj.weight"].T   # [T,H]
        k = normed @ tensors[f"{p}.self_attn.k_proj.weight"].T   # [T,kvH*hd]
        v = normed @ tensors[f"{p}.self_attn.v_proj.weight"].T

        q = q.reshape(T, NH, head_dim)
        k = k.reshape(T, NKV, head_dim)
        v = v.reshape(T, NKV, head_dim)
        pos = np.arange(T, dtype=np.float32)
        q = rope(q, pos, head_dim, theta)
        k = rope(k, pos, head_dim, theta)

        # causal attention
        attn = np.zeros((T, NH, head_dim), dtype=np.float32)
        scale = 1.0 / np.sqrt(head_dim)
        for s in range(T):
            for head in range(NH):
                kvh = head // kv_repeat
                # scores vs positions 0..s
                qs = q[s, head]                       # [hd]
                ks = k[: s + 1, kvh]                  # [s+1, hd]
                sc = (ks @ qs) * scale                # [s+1]
                sc = sc - sc.max()
                w = np.exp(sc)
                w = w / w.sum()
                attn[s, head] = w @ v[: s + 1, kvh]
        attn = attn.reshape(T, H)
        proj = attn @ tensors[f"{p}.self_attn.o_proj.weight"].T
        h = residual + proj

        residual = h
        normed = rms_norm(h, tensors[f"{p}.post_attention_layernorm.weight"], eps)
        gate = normed @ tensors[f"{p}.mlp.gate_proj.weight"].T
        up = normed @ tensors[f"{p}.mlp.up_proj.weight"].T
        mlp = silu(gate) * up
        down = mlp @ tensors[f"{p}.mlp.down_proj.weight"].T
        h = residual + down

    hidden = rms_norm(h, tensors["model.norm.weight"], eps)
    lm_head = tensors.get("lm_head.weight", tensors["model.embed_tokens.weight"])
    logits = hidden @ lm_head.T  # [T, vocab]
    return logits


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/tiny"
    cfg = json.load(open(f"{model_dir}/config.json"))
    tensors = load_safetensors(f"{model_dir}/model.safetensors")
    print(f"loaded {len(tensors)} tensors; hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']}")

    ids = [int(x) for x in sys.argv[2:]] or [128000, 1229, 3000]
    logits = forward(tensors, cfg, ids)
    last = logits[-1]
    order = np.argsort(-last)[:10]
    print(f"input ids: {ids}")
    print("top-10 next-token logits (last position):")
    for rank, i in enumerate(order):
        print(f"  {rank+1:2d}. id={i:6d}  logit={last[i]:+.6f}")

    # Emit the full last-position logits so the TypeScript can be diffed
    # against them directly.
    np.save(f"{model_dir}/ref_logits.npy", last.astype(np.float32))
    np.save(f"{model_dir}/ref_hidden.npy", logits)
    print(f"wrote {model_dir}/ref_logits.npy ({last.shape[0]} values)")

    # A quantisation round-trip check, the same thing the browser does, in
    # NumPy: |x - dequant(quant(x))| <= scale/2.
    w = tensors["model.layers.0.self_attn.q_proj.weight"].ravel()
    scale = np.abs(w).max() / 127.0
    q = np.clip(np.round(w / scale), -127, 127)
    deq = q * scale
    err = np.abs(w - deq).max()
    print(f"quant round-trip: scale={scale:.6e} max_err={err:.6e} bound={scale/2:.6e} "
          f"{'OK' if err <= scale/2 + 1e-6 else 'VIOLATED'}")


if __name__ == "__main__":
    main()
