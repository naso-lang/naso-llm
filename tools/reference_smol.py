#!/usr/bin/env python3
"""Independent NumPy reference for SmolLM2-135M-Instruct.

Written from the Llama architecture and Hugging Face's published conventions
(RMSNorm, RoPE with rotate-half, GQA with repeat_kv, SwiGLU, tied embeddings),
NOT translated from the TypeScript. That is the point: the TypeScript forward
pass in src/llama.ts is diffed against THIS, so a shared misconception cannot
cancel out.

Reads the bf16 safetensors directly (no torch): bfloat16 is the top 16 bits of
an f32, so widening is a left shift.

Usage:
  python3 tools/reference_smol.py <model_dir> <out.npy> <token_id>...
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
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dt = meta["dtype"]
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                raise ValueError(f"unsupported dtype {dt}")
            tensors[name] = arr.reshape(meta["shape"]).astype(np.float32)
        return tensors


def rms_norm(x, w, eps):
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def rope(x, positions, head_dim, theta):
    """x: [T, heads, head_dim]; rotate-half, matching HF's Llama convention."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    freqs = positions[:, None].astype(np.float32) * inv_freq[None, :]  # [T, half]
    cos = np.cos(freqs)[:, None, :]  # [T, 1, half]
    sin = np.sin(freqs)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def forward(t, cfg, ids):
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    eps = cfg["rms_norm_eps"]
    theta = cfg.get("rope_theta", 10000.0)
    head_dim = H // NH
    kv_dim = NKV * head_dim
    repeat = NH // NKV

    T = len(ids)
    embed = t["model.embed_tokens.weight"]
    h = embed[np.array(ids)].astype(np.float32)  # [T, H]

    positions = np.arange(T)
    scale = 1.0 / np.sqrt(head_dim)

    for l in range(L):
        p = f"model.layers.{l}"
        Wq = t[f"{p}.self_attn.q_proj.weight"]
        Wk = t[f"{p}.self_attn.k_proj.weight"]
        Wv = t[f"{p}.self_attn.v_proj.weight"]
        Wo = t[f"{p}.self_attn.o_proj.weight"]
        Wg = t[f"{p}.mlp.gate_proj.weight"]
        Wu = t[f"{p}.mlp.up_proj.weight"]
        Wd = t[f"{p}.mlp.down_proj.weight"]
        ln1 = t[f"{p}.input_layernorm.weight"]
        ln2 = t[f"{p}.post_attention_layernorm.weight"]

        residual = h
        normed = rms_norm(h, ln1, eps)
        # Row-major weight [out, in]: y = x @ W.T
        q = normed @ Wq.T
        k = normed @ Wk.T
        v = normed @ Wv.T

        q = q.reshape(T, NH, head_dim)
        k = k.reshape(T, NKV, head_dim)
        v = v.reshape(T, NKV, head_dim)
        q = rope(q, positions, head_dim, theta)
        k = rope(k, positions, head_dim, theta)

        # GQA: broadcast each KV head to `repeat` query heads.
        k_rep = np.repeat(k, repeat, axis=1)  # [T, NH, head_dim]
        v_rep = np.repeat(v, repeat, axis=1)

        scores = np.einsum("thd,shd->hts", q, k_rep) * scale  # [NH, T, T]
        mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
        scores = scores + mask[None, :, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        attn = np.einsum("hts,shd->thd", probs, v_rep).reshape(T, H)

        h = residual + attn @ Wo.T

        residual2 = h
        normed2 = rms_norm(h, ln2, eps)
        gate = normed2 @ Wg.T
        up = normed2 @ Wu.T
        silu = gate / (1.0 + np.exp(-gate))
        h = residual2 + (silu * up) @ Wd.T

    hidden = rms_norm(h, t["model.norm.weight"], eps)
    lm = t.get("lm_head.weight", embed)
    logits = hidden @ lm.T  # [T, vocab]
    return logits


def main():
    model_dir = sys.argv[1]
    out_path = sys.argv[2]
    ids = [int(x) for x in sys.argv[3:]]

    cfg = json.load(open(f"{model_dir}/config.json"))
    t = load_safetensors(f"{model_dir}/model.safetensors")
    print(f"loaded {len(t)} tensors; hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']}")
    print(f"architectures: {cfg.get('architectures')}")

    logits = forward(t, cfg, ids)
    last = logits[-1].astype(np.float32)
    np.save(out_path, last)
    print(f"wrote {out_path}: vocab={last.shape[0]} argmax={int(np.argmax(last))}")
    print(f"logit[argmax]={float(last.max()):.6f}")
    top5 = np.argsort(-last)[:5]
    print("top5:", [(int(i), round(float(last[i]), 4)) for i in top5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
