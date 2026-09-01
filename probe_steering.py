"""
probe_steering.py
=================

Finds prompts where the trained assistant SLM has *genuine choice*, so the
Simplex Thought Field can be seen steering it.

Two probes:
  1. Greedy: scans one- and two-word corpus prefixes; reports where the
     thought field (gain G, seed 42) flips the greedy completion.
  2. Sampling: on the four canonical prompts, temperature 1.0 with a fixed
     RNG seed — compares vanilla vs steered (seeds 42 / 7) samples.

Run::

    python probe_steering.py --gain 2.0 --device cuda
"""

from __future__ import annotations

import argparse
import re

import torch

from llm_thought import LMConfig, TinyThoughtLM
from steer_assistant import build_model
from tokenizer import CharTokenizer

CANONICAL = [
    "The core idea is that ",
    "Let me be honest about ",
    "One thing to flag. ",
    "The result is ",
]


def candidate_prompts(corpus: str, limit: int = 300):
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", corpus)
    seen, out = set(), []
    for i in range(len(words) - 1):
        for span in (1, 2):
            if i + span > len(words):
                break
            p = " ".join(words[i : i + span]) + " "
            if p not in seen:
                seen.add(p)
                out.append(p)
            if len(out) >= limit:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="assistant_weights.pt")
    ap.add_argument("--tokenizer", default="assistant_tokenizer.json")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = CharTokenizer.load(args.tokenizer)
    corpus = open("assistant_corpus.txt", encoding="utf-8").read()
    vanilla = build_model(
        None, args.weights, tok, args.hidden, args.layers, args.device
    )
    steered = build_model(
        {
            "seed": 42,
            "gain_b": args.gain,
            "gain_a": args.gain,
            "drift": (0.0, 0.0, 0.02),
        },
        args.weights,
        tok,
        args.hidden,
        args.layers,
        args.device,
    )

    # -- probe 1: greedy flips across corpus prefixes -------------------------
    print(
        f"--- probe 1: greedy steering at gain {args.gain} "
        f"({len(candidate_prompts(corpus))} candidate prefixes) ---"
    )
    flips = []
    prompts = candidate_prompts(corpus)
    for p in prompts:
        v = vanilla.generate_text(p, tok, max_new_tokens=args.max_tokens)
        steered.set_thought_seed(42)
        s = steered.generate_text(p, tok, max_new_tokens=args.max_tokens)
        if s != v:
            flips.append((p, v, s))
    print(f"flipped: {len(flips)}/{len(prompts)} prompts\n")
    for p, v, s in flips[:8]:
        print(f"  {p!r}\n    vanilla: {v!r}\n    seed42 : {s!r}\n")
    if not flips:
        print("  (no greedy flips at this gain — try --gain 5 or higher)")

    # -- probe 2: temperature-1.0 sampling on the canonical prompts -----------
    print("--- probe 2: sampling, temperature 1.0, rng_seed 0 ---")
    for p in CANONICAL:
        v = vanilla.generate_text(
            p, tok, max_new_tokens=args.max_tokens, temperature=1.0, rng_seed=0
        )
        rows = []
        for s in (42, 7):
            steered.set_thought_seed(s)
            rows.append(
                (
                    s,
                    steered.generate_text(
                        p,
                        tok,
                        max_new_tokens=args.max_tokens,
                        temperature=1.0,
                        rng_seed=0,
                    ),
                )
            )
        diff = any(o != v for _, o in rows)
        print(f"  {p!r}")
        print(f"    vanilla: {v!r}")
        for s, o in rows:
            print(f"    seed {s:>2}{'->' if o != v else '  '}: {o!r}")
        if diff:
            print("    ^^ steered")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
