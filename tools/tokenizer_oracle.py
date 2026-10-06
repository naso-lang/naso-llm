#!/usr/bin/env python3
"""Tokenization oracle.

Uses Hugging Face's `tokenizers` (the Rust implementation behind transformers)
to produce reference token ids for a fixed corpus. The TypeScript port in
src/tokenizer.ts is checked against these; a byte-level BPE that is *almost*
right produces plausible-looking but wrong ids, so agreeing with the reference
on non-trivial inputs (unicode, digits, punctuation, whitespace runs, the chat
control tokens) is the only real evidence the port is correct.

Each case records `allowSpecial`, which tells the harness whether the input
should be encoded with the special-token matcher enabled:

  * False -- plain text. Control strings are not expected, and the browser port
    must reproduce ordinary byte-level BPE.
  * True  -- a chat template. `<|im_start|>` / `<|im_end|>` are added tokens
    with `normalized: false`, so the reference `tokenizers` matches them
    literally in *all* input; the port must do the same when asked.

Usage:
  python3 tools/tokenizer_oracle.py <tokenizer.json> <out.json>
"""
import json
import sys

from tokenizers import Tokenizer

# Exercise every branch of the GPT-2 pre-tokenizer regex plus the byte-level
# fallback: letters with/without a leading space, digits (which the `Digits`
# pre-tokenizer splits individually), punctuation, contractions, multi-space
# runs, newlines, tabs, and non-ASCII (where byte-level BPE and a naive UTF-8
# split diverge). `(text, allowSpecial)` pairs.
CASES = [
    ("", False),
    ("hello", False),
    ("Hello, world!", False),
    ("The capital of France is", False),
    (" Paris", False),
    ("2 + 2 = 4", False),
    ("1234567890", False),
    ("don't can't won't I'm you're we've they'll", False),
    ("    leading spaces", False),
    ("trailing spaces    ", False),
    ("multiple     inner     spaces", False),
    ("line one\nline two\n\nline three", False),
    ("\ttab\tseparated", False),
    ("café naïve résumé", False),
    ("日本語のテキスト", False),
    ("emoji: 🚀🌍💡", False),
    ("München — Straße", False),
    ("a1b2c3d4", False),
    ("Punctuation!!! ??? ... ;: []{}() <> @#$%^&*", False),
    ("def f(x):\n    return x * 2", False),
    ("What is 17 * 23?", False),
    ("the quick brown fox jumps over the lazy dog", False),
    ("THE QUICK BROWN FOX", False),
    ("MiXeD CaSe WoRdS", False),
    # The exact string the chat UI feeds the model, with the control tokens.
    ("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", True),
    ("<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n", True),
    ("<|im_end|>", True),
]


def main() -> int:
    tok_path, out_path = sys.argv[1], sys.argv[2]
    tok = Tokenizer.from_file(tok_path)

    results = []
    for text, allow_special in CASES:
        enc = tok.encode(text, add_special_tokens=False)
        results.append({
            "text": text,
            "allowSpecial": allow_special,
            "ids": enc.ids,
            "tokens": enc.tokens,
            "decoded": tok.decode(enc.ids, skip_special_tokens=False),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_path}: {len(results)} cases")
    for r in results[:4]:
        print(f"  {r['text']!r} -> {r['ids']}")
    print("  ...")
    for r in results[-3:]:
        print(f"  special={r['allowSpecial']} -> {r['ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
