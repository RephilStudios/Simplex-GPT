"""
mix_sweep_test.py — is branch blending a smooth dial between two code paths?
=============================================================================

Branch blending (new):
    bias = (1-α) · field(branch A) + α · field(branch B)

α = 0 is exactly branch A's code, α = 1 exactly branch B's. The question:
does the generated code change GRADUALLY with α (a real dial), or jump
like the seed/offset experiments showed (any change = full jump)?

Hypothesis (from the mechanism): the logit perturbation at sensitive
positions grows LINEARLY with α, so the set of flipped decisions grows
with α -> code similarity to A should decay smoothly from 1.0 to the
A↔B endpoint distance, and similarity to B should rise smoothly.

Setup: branch A = seed 42, branch B = seed 7 (known to differ on these
tasks), gain 4.0, the 3 mid-difficulty code tasks.

Run:  python mix_sweep_test.py --weights models/Qwen3.5-4B --gain 4
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import torch

from code_branch_test import TASKS, eval_code, extract_code
from seed_locality_test import make_offset, tok_sim
from thought_wrap import clear_mix, set_mix, set_seed, wrap

TASK_IDX = [3, 4, 5]  # merge_intervals, flatten, longest_palindromic_substring
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED_A, SEED_B = 42, 7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--max-tokens", type=int, default=400)
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

    def generate(p, max_new_tokens):
        ids = torch.tensor([prompt_ids(p)], dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = lm.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        new = out[0][ids.shape[1] :]
        return tok.decode(new, skip_special_tokens=True).strip()

    tasks = [TASKS[i] for i in TASK_IDX]

    wrap(lm, SEED_A, args.gain, args.gain)

    # endpoints first (α=0 and α=1 are just the two plain branches)
    baseA = {}
    for prompt, fname, tests in tasks:
        baseA[fname] = extract_code(generate(prompt, args.max_tokens))
    set_seed(lm, SEED_B)
    baseB = {}
    for prompt, fname, tests in tasks:
        baseB[fname] = extract_code(generate(prompt, args.max_tokens))
    set_seed(lm, SEED_A)

    b_off = make_offset(SEED_B)
    endpoint_sim = {f: tok_sim(baseA[f], baseB[f]) for f in baseA}
    print(
        f"\nendpoint A↔B token similarity: "
        + "  ".join(f"{f}:{s:.2f}" for f, s in endpoint_sim.items())
    )

    # α sweep (skip the endpoints — already generated)
    print("\nα     " + " | ".join(f"{tasks[i][1][:14]:>14}" for i in range(len(tasks))))
    for label, codes in (("A(α=0)", baseA), ("B(α=1)", baseB)):
        sims = {f: tok_sim(baseA[f], c) for f, c in codes.items()}
        print(
            f"{label:>5} "
            + " | ".join(f"{sims[f]:>14.2f}" for f in sims)
            + "   "
            + " ".join(
                f"{f}:{'ok' if eval_code(codes[f], TASKS[TASK_IDX[i]][2])[0] else 'FAIL'}"
                for i, f in enumerate(sims)
            )
        )
    ROWS = {}
    for a in ALPHAS:
        if a in (0.0, 1.0):
            continue
        set_mix(lm, b_off, a)
        row = {}
        for prompt, fname, tests in tasks:
            code = extract_code(generate(prompt, args.max_tokens))
            row[fname] = code
        simsA = {f: tok_sim(baseA[f], row[f]) for f in row}
        simsB = {f: tok_sim(baseB[f], row[f]) for f in row}
        ROWS[a] = (row, simsA, simsB)
        print(
            f"{a:<5} "
            + " | ".join(f"{simsA[f]:>14.2f}" for f in simsA)
            + "   "
            + " ".join(
                f"{f}:{'ok' if eval_code(row[f], TASKS[TASK_IDX[i]][2])[0] else 'FAIL'}"
                for i, f in enumerate(simsA)
            )
        )
        print(f"{'':>5} sim-to-B:" + " | ".join(f"{simsB[f]:>12.2f}" for f in simsB))
    clear_mix(lm)  # blending disabled; model left wrapped (harmless)

    # ---- verdict ------------------------------------------------------------
    import statistics

    mid_a = [0.25, 0.5, 0.75]
    simA = {a: statistics.mean(ROWS[a][1].values()) for a in mid_a}
    simB = {a: statistics.mean(ROWS[a][2].values()) for a in mid_a}
    base_simA = statistics.mean(endpoint_sim.values())  # A↔B distance at α=1

    # monotonicity across the full 0..1 range (endpoints as anchors)
    fullA = [1.0] + [simA[a] for a in mid_a] + [base_simA]
    decays = all(fullA[i + 1] <= fullA[i] + 0.15 for i in range(len(fullA) - 1))

    xs = [0.0] + mid_a + [1.0]
    ys = [1.0] + [simA[a] for a in mid_a] + [base_simA]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    vary = sum((y - my) ** 2 for y in ys)
    corr = (
        sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (varx * vary) ** 0.5
        if varx * vary > 0
        else 0.0
    )
    print("\n--- verdict ---")
    print(
        f"sim-to-A: α=0:1.00  "
        + "  ".join(f"α={a}:{simA[a]:.2f}" for a in mid_a)
        + f"  α=1:{base_simA:.2f}"
    )
    print(
        f"sim-to-B: α=0:{base_simA:.2f}  "
        + "  ".join(f"α={a}:{simB[a]:.2f}" for a in mid_a)
        + "  α=1:1.00"
    )
    print(f"corr(α, sim-to-A) = {corr:+.2f}  (strong negative = smooth dial)")
    if corr < -0.7 and decays:
        print(
            "SMOOTH DIAL CONFIRMED: blending walks gradually between the two branches"
        )
    elif corr < -0.3:
        print("weakly graded: the dial exists but decoding quantizes it")
    else:
        print("no smooth dial: blending behaves like another global jump")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
