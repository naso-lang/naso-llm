#!/usr/bin/env python3
"""Tensor-level budget for shipping SmolLM2-135M-Instruct pre-quantised.

Before building an artifact, establish exactly what a downloadable int8
checkpoint would contain and what it would weigh. Every number here is measured
from the real checkpoint, not assumed.

Rules that make the file self-describing (all measured, not chosen blindly):
  * a matrix is quantised only if BOTH dims are multiples of 4, because
    src/quantize.ts packs four int8 per u32;
  * int8 storage = packed bytes (n*k/4 words) + per-row f32 scales (n*4);
  * other 2-D weights (e.g. the tied embedding) are OPTIONAL: the quantiser and
    the loader must agree, so the plan states which set it covers.

Usage: python3 tools/quant_plan.py /var/tmp/smol
"""
import collections
import json
import struct
import sys


def load_header_and_sizes(path):
    """Return {name: (dtype, shape, nbytes)} without materialising the data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = (meta["dtype"], meta["shape"], end - start)
    return out


def classify(name):
    if name.endswith(".self_attn.q_proj.weight"):  return "attn.q_proj"
    if name.endswith(".self_attn.k_proj.weight"):  return "attn.k_proj"
    if name.endswith(".self_attn.v_proj.weight"):  return "attn.v_proj"
    if name.endswith(".self_attn.o_proj.weight"):  return "attn.o_proj"
    if name.endswith(".mlp.gate_proj.weight"):     return "mlp.gate_proj"
    if name.endswith(".mlp.up_proj.weight"):       return "mlp.up_proj"
    if name.endswith(".mlp.down_proj.weight"):     return "mlp.down_proj"
    if name == "model.embed_tokens.weight":        return "embed"
    if name.endswith(".input_layernorm.weight"):   return "norm.input"
    if name.endswith(".post_attention_layernorm.weight"): return "norm.post"
    if name == "model.norm.weight":                return "norm.final"
    return "other"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/smol"
    t = load_header_and_sizes(f"{d}/model.safetensors")
    cfg = json.load(open(f"{d}/config.json"))
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]

    print(f"model : {cfg['num_hidden_layers']} layers, hidden {hidden}, "
          f"intermediate {cfg['intermediate_size']}, vocab {vocab}")
    print(f"file  : {sum(v[2] for v in t.values())/1e6:.1f} MB, {len(t)} tensors\n")

    by = collections.defaultdict(lambda: [0, 0, 0])  # count, bf16 bytes, int8 bytes
    unquantisable = []
    for name, (dt, shape, nb) in t.items():
        c = classify(name)
        by[c][0] += 1
        by[c][1] += nb
        is2d = len(shape) == 2
        if is2d and shape[0] % 4 == 0 and shape[1] % 4 == 0:
            n, k = shape
            by[c][2] += (n * k // 4) * 4 + n * 4        # packed u32 + f32 scales
        elif is2d:
            by[c][2] += nb                               # too small to pack; kept as-is
            unquantisable.append((name, shape))

    print(f"{'group':<16}{'#':>4}{'bf16 MB':>10}{'int8 MB':>10}{'ratio':>8}")
    for c, (cnt, bf, i8) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        r = f"{bf/i8:.2f}x" if i8 else "-"
        print(f"{c:<16}{cnt:>4}{bf/1e6:>10.1f}{i8/1e6:>10.1f}{r:>8}")

    tot_bf = sum(v[1] for v in by.values())
    # Plan A: quantise projections only, keep embed + norms at bf16 (current behaviour)
    proj = {"attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    a = sum(by[c][2] for c in proj) + sum(by[c][1] for c in by if c not in proj)
    # Plan B: also quantise the embedding
    b = sum(by[c][2] for c in proj) + by["embed"][2] + sum(by[c][1] for c in by if c not in proj and c != "embed")

    print(f"\ntotal as shipped today (bf16)        : {tot_bf/1e6:8.1f} MB")
    print(f"Plan A  int8 projections, f32 embed  : {a/1e6:8.1f} MB   ({tot_bf/a:.2f}x smaller)")
    print(f"Plan B  int8 projections + int8 embed: {b/1e6:8.1f} MB   ({tot_bf/b:.2f}x smaller)")
    print(f"embed is {by['embed'][1]/1e6:.1f} MB bf16 -> {by['embed'][2]/1e6:.1f} MB int8 "
          f"(it dominates: {by['embed'][1]/tot_bf*100:.0f}% of the file)")

    # Sanity: does the geometry our tests use match the checkpoint?
    for name, shape in [("model.layers.0.self_attn.q_proj.weight", [hidden, hidden]),
                        ("model.layers.0.mlp.gate_proj.weight", [cfg["intermediate_size"], hidden]),
                        ("model.embed_tokens.weight", [vocab, hidden])]:
        got = t.get(name, (None, None, None))[1]
        print(f"  {name:<46} header={got} expected={shape} {'OK' if got == shape else 'MISMATCH'}")

    if unquantisable:
        print("\n2-D tensors kept at full precision (dim not a multiple of 4):")
        for name, shape in unquantisable:
            print(f"  {name} {shape}")


if __name__ == "__main__":
    main()
