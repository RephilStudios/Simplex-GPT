"""
confidence_steering_corr.py
===========================

Does the thought field's steering effect really scale inversely with the
model's confidence?

For every candidate prompt:
  * measure the model's *confidence* = top1_logit - top2_logit at the first
    completion position (vanilla model),
  * check whether the thought field (gain G, seed 42) flips the greedy
    completion,
then report the flip rate bucketed by confidence, plus a Pearson r between
margin and (flipped).

Run::

    python confidence_steering_corr.py --gain 2.0 --device cuda
"""

from __future__ import annotations

import argparse
import math

import torch

import probe_steering
from steer_assistant import build_model
from tokenizer import CharTokenizer


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
    prompts = probe_steering.candidate_prompts(corpus, limit=300)

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

    rows = []
    for p in prompts:
        ids = tok.encode(p)
        x = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            logits = vanilla(x)[0, -1]  # (V,)
        vals, _ = logits.sort(descending=True)
        margin = float(vals[0] - vals[1])  # confidence of the first step
        v = vanilla.generate_text(p, tok, max_new_tokens=args.max_tokens)
        steered.set_thought_seed(42)
        s = steered.generate_text(p, tok, max_new_tokens=args.max_tokens)
        rows.append((margin, s != v))

    # bucketed flip rates
    buckets = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
    print(
        f"--- steering vs. model confidence (gain {args.gain}, seed 42, "
        f"{len(rows)} prompts) ---"
    )
    print(f"{'margin (top1-top2)':<22} {'n':>4} {'flips':>6} {'flip rate':>10}")
    for lo, hi in buckets:
        inb = [(m, f) for m, f in rows if lo <= m < hi]
        if not inb:
            continue
        n = len(inb)
        flips = sum(f for _, f in inb)
        label = f"[{lo:g}, {hi:g})" if hi < 1e8 else f"[{lo:g}, inf)"
        print(f"{label:<22} {n:>4} {flips:>6} {flips / n:>9.0%}")

    # Pearson r between margin and flipped (1.0 = flipped, 0.0 = not)
    xs = [m for m, _ in rows]
    ys = [1.0 if f else 0.0 for _, f in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    r = cov / (sx * sy) if sx * sy > 0 else float("nan")
    print(
        f"\nPearson r (margin, flipped) = {r:+.3f}   (negative = steering "
        f"anti-correlated with confidence)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
