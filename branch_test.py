"""
branch_test.py — is the field a *controlled branch* through probability space?
==============================================================================

Reframe under test: the field is not a "coherence booster"; it is a
deterministic branch operator. A fixed (seed) selects one trajectory and
turning the gain walks that trajectory *outward through the model's own
distribution*, one branch level at a time.

Falsifiable predictions (fixed seed 42, fine gain grid):
  P1  branch depth grows : NLL_vanilla(answer(g)) is ~non-decreasing in g
  P2  the path is smooth : answer(g) stable across adjacent gains
                           (threshold flips, not A-B-A-C chaos)
  P3  seeds pick branches: at equal gain, seed 42 and seed 7 disagree on a
                           substantial fraction of prompts
Plus: print the vanilla top-5 next tokens at the answer position so we can
see whether the branch path follows the model's own ranking order.

Run:  python branch_test.py --weights models/Qwen3.5-4B
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import torch
import torch.nn.functional as F

from thought_wrap import delta_net_layers, set_seed, unwrap, wrap


def set_gain(lm, g: float) -> None:
    """Change the modulation amplitude on an already-wrapped model."""
    for _, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        if hasattr(attn, "thought"):
            attn.thought.gain_b = float(g)
            attn.thought.gain_a = float(g)


# the hard prompts where we know flips happen around gain 4-16
PROMPTS = [
    "Name a planet. Answer with one word.",
    "Name a metal used in ancient coinage. One word.",
    "Pick a color associated with caution. One word.",
    "Pick a two-syllable fruit ending in e. One word.",
    "Choose a four-letter dog breed. One word.",
    "Name a less common European capital. One word.",
]

GAINS = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
TOL = 0.03  # NLL "non-decreasing" tolerance (ties within this count as flat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.weights} ...")
    lm = (
        AutoModelForCausalLM.from_pretrained(
            args.weights, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        .to(dev)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.weights)

    def prompt_ids(p):
        return tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
            return_tensors=None,
        )

    def generate(p):
        ids = torch.tensor([prompt_ids(p)], dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = lm.generate(
                input_ids=ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=1.0,
            )
        new = out[0][ids.shape[1] :]
        return tok.decode(new, skip_special_tokens=True).strip()

    def nll_of(p, answer):
        P = prompt_ids(p)
        a = tok(answer, add_special_tokens=False)["input_ids"]
        if not a:
            return 0.0
        full = torch.tensor([P + a], dtype=torch.long, device=dev)
        with torch.inference_mode():
            logits = lm(input_ids=full).logits[0]
        s = 0.0
        for i, t in enumerate(a):
            s += float(-F.log_softmax(logits[len(P) - 1 + i], dim=-1)[t].item())
        return s / len(a)

    def topk(p, k=5):
        P = prompt_ids(p)
        full = torch.tensor([P], dtype=torch.long, device=dev)
        with torch.inference_mode():
            logits = lm(input_ids=full).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k)
        return [
            (tok.decode([int(i)]).strip() or f"<{int(i)}>", float(v))
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    print(f"\nseed=42 fixed  gains={GAINS}\n")

    mono_scores, adj_scores, seed_disagree = [], [], 0

    for p in PROMPTS:
        print(f"=== {p!r}")
        print(f"    vanilla top-5: " + "  ".join(f"{t}({pr:.2f})" for t, pr in topk(p)))

        row = []
        # generate all answers under the field, THEN unwrap and measure
        # NLL with the *vanilla* model — branch depth = distance of the
        # branch's answer in the vanilla distribution (the hypothesis)
        wrap(lm, 42, GAINS[0], GAINS[0])
        answers = []
        for g in GAINS:
            set_gain(lm, g)
            answers.append(generate(p))
        unwrap(lm)
        for g, ans in zip(GAINS, answers):
            row.append((g, ans, nll_of(p, ans)))

        for g, ans, nll in row:
            print(f"    g={g:<4} {ans!r:38} (NLL {nll:.3f})")

        # P1: NLL ~non-decreasing in gain
        steps = len(row) - 1
        mono = sum(1 for i in range(steps) if row[i + 1][2] >= row[i][2] - TOL) / steps
        # P2: adjacent stability
        adj = sum(1 for i in range(steps) if row[i + 1][1] != row[i][1]) / steps
        # P3: seed disagreement at the middle gain
        wrap(lm, 42, GAINS[3], GAINS[3])
        a42 = generate(p)
        set_seed(lm, 7)
        a7 = generate(p)
        unwrap(lm)
        disagree = a42 != a7
        seed_disagree += int(disagree)

        # branch order vs vanilla ranking
        order = []
        for _, ans, _ in row:
            w = ans.split()[0] if ans else ""
            if w and w not in order:
                order.append(w)
        print(f"    -> branch order: {order}")
        print(
            f"    -> P1 NLL monotone-in-gain: {mono:.0%}   "
            f"P2 adjacent-flip rate: {adj:.0%}   "
            f"P3 seed-disagree@{GAINS[3]:.0f}: {disagree} (42={a42!r}  7={a7!r})"
        )
        mono_scores.append(mono)
        adj_scores.append(adj)
        print()

    P = len(PROMPTS)
    m = sum(mono_scores) / P
    a = sum(adj_scores) / P
    d = seed_disagree / P
    print("--- branch hypothesis ---")
    print(f"P1 branch-depth monotone (mean over prompts): {m:.0%}")
    print(f"P2 path smoothness (mean adjacent-flip, lower=smoother): {a:.0%}")
    print(f"P3 seed branch diversity (disagree rate at g={GAINS[3]:.0f}): {d:.0%}")
    if m >= 0.7 and a <= 0.5 and d >= 0.3:
        print(
            "signature of a CONTROLLED branch: monotone outward walk, "
            "smooth path, seed-selectable trajectories"
        )
    elif m >= 0.7 and a > 0.5:
        print(
            "depth grows but the path is chaotic — gain controls magnitude, "
            "not direction"
        )
    elif m < 0.7:
        print(
            "branch depth does NOT track gain — more consistent with noise "
            "than controlled branching"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
