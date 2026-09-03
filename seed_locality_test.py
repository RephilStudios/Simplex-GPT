"""
seed_locality_test.py — is the field's seed space a *local* neighborhood?
=========================================================================

Key fact (from simplex_thought_field.py): a seed maps to a coordinate
offset via ``torch.Generator(seed) -> rand(dim)*64-32`` — a PRNG hash.
Seeds 42 and 43 are therefore typically ~18 units apart in offset space,
i.e. "nearby seeds" is a hash, NOT a coordinate. Locality, if it exists,
must live in OFFSET space: small offset perturbation -> nearby field ->
nearby code; large perturbation -> far code. A flat similarity-vs-delta
curve (no decay) means the neighborhood is as random as any other pair.

Part A — offset perturbation sweep (the core test)
  base = seed 42's offset. For delta in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
  (uniform random direction, fixed), set every layer's field offset to
  base + delta*u and generate code for each task.
  Report: token similarity + gold-test behavioral agreement vs delta.
  Signature of locality: similarity DECAYS with delta.
  Signature of "just another random pair": similarity FLAT with delta.

Part B — seed pool geometry (which integer seeds ARE nearby)
  Compute offsets for 60 candidate seeds, find the pool member geometrically
  closest to seed 42 (nearest neighbor), and compare:
    pair NN : (42, nearest)   vs   pair FAR : (42, 7)
  If NN >= FAR on both metrics, offset-geometry transfers to seeds.

Metrics per code pair:
  tok   : token Jaccard similarity of the generated code
  beh   : Jaccard agreement on which gold test cases each code passes

Run:  python seed_locality_test.py --weights models/Qwen3.5-4B --gain 4
"""

import argparse
import os
import random
import textwrap

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import torch

from code_branch_test import TASKS, eval_code, extract_code
from thought_wrap import delta_net_layers, unwrap, wrap

# use the mid-difficulty tasks (branching happens; decode_str fails for all)
TASK_IDX = [3, 4, 5]  # merge_intervals, flatten, longest_palindromic_substring

DELTA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEED_POOL = [
    42,
    7,
    13,
    21,
    33,
    55,
    64,
    99,
    101,
    128,
    200,
    233,
    256,
    333,
    377,
    400,
    421,
    512,
    600,
    641,
    700,
    777,
    800,
    888,
    900,
    1000,
    1024,
    1111,
    1200,
    1234,
    1300,
    1400,
    1500,
    1600,
    1729,
    1800,
    1944,
    2000,
    2048,
    2222,
    2400,
    2500,
    2600,
    2718,
    2800,
    2900,
    3000,
    3141,
    3200,
    3333,
    3500,
    3600,
    3700,
    3800,
    3900,
    4000,
    4042,
    4096,
    4242,
]


def make_offset(seed: int, dim: int = 3) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return torch.rand(dim, generator=gen) * 64.0 - 32.0


def set_field_offset(lm, offset: torch.Tensor) -> None:
    """Set every wrapped layer's field coordinate offset in place.

    NOTE: slot_bias always calls field(coords, seed=<int>), and SimplexField
    recomputes the offset from that seed, ignoring the buffer. So we also
    patch each field's forward to take the seed=None (buffer) path, which is
    the only way an *arbitrary* offset (not seed-derivable) can take effect.
    """
    import types

    for _, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        fld = getattr(attn.thought, "field", None)
        if fld is None:
            continue
        fld.seed_offset.copy_(offset.to(fld.seed_offset.dtype))
        if not getattr(fld, "_offset_forced", False):
            orig = fld.forward  # bound original
            fld.forward = types.MethodType(
                lambda self, coords, seed=None: orig(coords), fld
            )
            fld._offset_forced = True


def tok_sim(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0


def pass_vector(code: str, tests: str) -> set:
    """Which gold cases does this code pass? (runs each case in isolation)"""
    src = code + "\n"
    out = set()
    for i, line in enumerate(tests.strip().splitlines()):
        line = line.strip()
        if not line.startswith("assert "):
            continue
        probe = (
            src
            + "\ntry:\n    "
            + line
            + "\n    print('P')\nexcept Exception:\n    pass\n"
        )
        import subprocess
        import sys
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(probe)
            r = subprocess.run(
                [sys.executable, path], capture_output=True, text=True, timeout=20
            )
            if "P" in (r.stdout or ""):
                out.add(i)
        except Exception:
            pass
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return out


def beh_sim(a: str, b: str, tests: str) -> float:
    pa, pb = pass_vector(a, tests), pass_vector(b, tests)
    return len(pa & pb) / len(pa | pb) if (pa | pb) else 1.0


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

    wrap(lm, 42, args.gain, args.gain)
    base_off = make_offset(42).clone()

    # fixed random perturbation direction (deterministic)
    rng = random.Random(0)
    u = torch.tensor([rng.random() for _ in range(3)])
    u = u / u.norm()

    tasks = [TASKS[i] for i in TASK_IDX]

    # ---- Part A: delta sweep ------------------------------------------------
    print(
        f"\n=== Part A: offset perturbation from seed-42 base "
        f"(offset {base_off.tolist()}) ==="
    )
    # base code at seed 42's own offset (generated once, not per delta)
    base_codes = {}
    for prompt, fname, tests in tasks:
        base_codes[fname] = extract_code(generate(prompt, args.max_tokens))
    RESULTS = {}
    for d in DELTA_GRID:
        set_field_offset(lm, base_off + d * u)
        row, row_ok = {}, {}
        for prompt, fname, tests in tasks:
            code = extract_code(generate(prompt, args.max_tokens))
            ok, _ = eval_code(code, tests)
            row[fname] = code
            row_ok[fname] = ok
        sims = {f: tok_sim(base_codes[f], row[f]) for f in base_codes}
        RESULTS[d] = (row, sims)
        print(
            f"d={d:<5} "
            + " | ".join(f"{sims[f]:>6.2f}" for f in base_codes)
            + "   "
            + " ".join(f"{f}:{'ok' if row_ok[f] else 'FAIL'}" for f in base_codes)
        )
    set_field_offset(lm, base_off)  # restore base before Part B
    unwrap(lm)

    # decay check: corr(delta, 1 - similarity)
    import statistics

    ds = [d for d in DELTA_GRID if d > 0]
    mean_sim = {d: sum(RESULTS[d][1].values()) / len(RESULTS[d][1]) for d in ds}
    xs = ds
    ys = [1.0 - mean_sim[d] for d in ds]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    corr = cov / (
        (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        if (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) > 0
        else 1.0
    )
    print("\nmean token similarity vs delta:")
    for d in ds:
        print(f"  d={d:<5} sim={mean_sim[d]:.2f}")
    print(f"Pearson corr(delta, distance) = {corr:+.2f}")

    # ---- Part B: seed geometry ---------------------------------------------
    print("\n=== Part B: seed pool geometry (distance in offset space) ===")
    offs = {s: make_offset(s) for s in SEED_POOL}
    dists = sorted(
        (float((offs[s] - base_off).norm()), s) for s in SEED_POOL if s != 42
    )
    nn_dist, nn_seed = dists[0]  # dists is sorted (dist, seed) tuples
    far_seed = 7
    far_dist = float((offs[far_seed] - base_off).norm())
    print(f"nearest neighbor of 42 in pool: seed {nn_seed} (offset dist {nn_dist:.1f})")
    print(f"far reference: seed {far_seed} (offset dist {far_dist:.1f})")

    def gen_pair(seed):
        wrap(lm, seed, args.gain, args.gain)
        codes = {}
        for prompt, fname, tests in tasks:
            codes[fname] = extract_code(generate(prompt, args.max_tokens))
        unwrap(lm)
        return codes

    wrap(lm, 42, args.gain, args.gain)
    base2 = {}
    for prompt, fname, tests in tasks:
        base2[fname] = extract_code(generate(prompt, args.max_tokens))
    unwrap(lm)
    nn_codes = gen_pair(nn_seed)
    far_codes = gen_pair(far_seed)

    print("\npair            tok-sim (mean)   behavioral (mean)   passes (mean)")
    for label, codes in (
        ("42 vs NN(%d)" % nn_seed, nn_codes),
        ("42 vs FAR(7)", far_codes),
    ):
        ts = [tok_sim(base2[f], codes[f]) for f in base2]
        bs = [
            beh_sim(base2[f], codes[f], TASKS[TASK_IDX[i]][2])
            for i, f in enumerate(base2)
        ]
        ps = [
            eval_code(codes[f], TASKS[TASK_IDX[i]][2])[0] for i, f in enumerate(base2)
        ]
        print(
            f"{label:15} {sum(ts) / len(ts):.2f}               "
            f"{sum(bs) / len(bs):.2f}              {sum(ps) / len(ps):.2f}"
        )

    # ---- verdict -------------------------------------------------------------
    print("\n--- verdict ---")
    if corr > 0.5:
        print(
            "strong LOCALITY: code distance grows with offset distance -> "
            "the field is a local neighborhood, hill-climbable"
        )
    elif corr > 0.2:
        print("weak locality: some structure, but greedy decoding flattens it")
    else:
        print(
            "NO locality: any offset perturbation is as disruptive as any "
            "other -> seed space is a flat hash, not a coordinate"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
