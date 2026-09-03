"""
repair_alpha_sweep.py — minimal-α repair via the live endpoint
==============================================================

Find the SMALLEST α such that the blend of branch A (seed) and branch B
(seed) passes tests that BOTH pure branches fail. That α is the smallest
conceptual change (in thought-space) that fixes the bug — the "minimal
repair" demo.

Per (task, branch pair):
  1. endpoints A and B (plain branches) -> run the tests
  2. if BOTH fail: coarse α sweep 0.05..1.00, then 3 bisection
     refinement steps below the first passing α
  3. report the curve, minimal passing α, its code, and a
     determinism check (same α twice -> byte-identical)
Falls through to the next branch pair until a both-fail pair is found.

Requires:  serve_real_endpoint.py running with --thought-enabled
           (tested at gain 4.0; restart the server to pick up the
           thought_mix_seed / thought_mix_alpha request fields).
Run:       python repair_alpha_sweep.py [--url http://127.0.0.1:8100]
"""

import argparse
import json
import urllib.request

from code_branch_test import TASKS, eval_code, extract_code

# tasks where the 4B is at/near its ceiling (repair regime candidates)
TASK_IDX = [5, 7]  # longest_palindromic_substring, decode_str
PAIRS = [(42, 7), (42, 13), (7, 13), (42, 1024)]
COARSE = [round(0.05 * i, 2) for i in range(1, 21)]  # 0.05 .. 1.00


def chat(url, prompt, seed, mix_seed=None, alpha=None, max_tokens=300):
    extra = {}
    if mix_seed is not None and alpha is not None:
        extra = {"thought_mix_seed": mix_seed, "thought_mix_alpha": alpha}
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "thought_seed": seed,
            **extra,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"], d.get("thought", {})


def gen_code(url, prompt, seed, mix_seed=None, alpha=None):
    text, thought = chat(url, prompt, seed, mix_seed, alpha)
    return extract_code(text), thought


def try_pair(url, prompt, fname, tests, A, B):
    codeA, _ = gen_code(url, prompt, A)
    codeB, _ = gen_code(url, prompt, B)
    okA = eval_code(codeA, tests)[0]
    okB = eval_code(codeB, tests)[0]
    print(
        f"  pair ({A},{B}): A {'PASS' if okA else 'FAIL'}  "
        f"B {'PASS' if okB else 'FAIL'}"
    )
    if okA or okB:
        return None  # not in the repair regime for this pair

    curve = []
    first_pass = None
    for a in COARSE:
        code, _ = gen_code(url, prompt, A, B, a)
        ok = eval_code(code, tests)[0]
        curve.append((a, ok))
        print(f"  α={a:<5} {'PASS' if ok else 'fail'}")
        if ok and first_pass is None:
            first_pass = a
    if first_pass is None:
        print("  no passing blend in [0.05, 1.0]")
        return None

    # bisection: smallest passing α in (0, first_pass]
    lo, hi = 0.0, first_pass
    for _ in range(3):
        mid = round((lo + hi) / 2, 3)
        code, _ = gen_code(url, prompt, A, B, mid)
        ok = eval_code(code, tests)[0]
        print(f"  refine α={mid:<7} {'PASS' if ok else 'fail'}")
        if ok:
            hi = mid
        else:
            lo = mid

    # determinism at the minimal α
    code1, t1 = gen_code(url, prompt, A, B, hi)
    code2, _ = gen_code(url, prompt, A, B, hi)
    deterministic = code1 == code2
    ok_final = eval_code(code1, tests)[0]

    print(
        f"  -> minimal passing α = {hi}  (deterministic: {deterministic}, "
        f"final: {'PASS' if ok_final else 'FAIL'})"
    )
    print("  --- the repaired code ---")
    print("\n".join("    " + ln for ln in code1.splitlines()[:25]))
    return {
        "task": fname,
        "pair": (A, B),
        "alpha": hi,
        "curve": curve,
        "deterministic": deterministic,
        "passes": ok_final,
        "code": code1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    print(f"endpoint: {url}")
    results = []
    for idx in TASK_IDX:
        prompt, fname, tests = TASKS[idx]
        print(f"\n=== {fname} ===")
        result = None
        for A, B in PAIRS:
            result = try_pair(url, prompt, fname, tests, A, B)
            if result:
                break
            print("  -> trying next branch pair")
        if result:
            results.append(result)

    print("\n--- summary ---")
    if not results:
        print("no repair regime found for these tasks/pairs on this server")
        return 0
    for r in results:
        passing = [a for a, ok in r["curve"] if ok]
        print(
            f"{r['task']}: branches {r['pair']} both FAIL -> minimal "
            f"passing α = {r['alpha']}  (first coarse pass: {passing[0]}, "
            f"deterministic: {r['deterministic']})"
        )
    print("\naddress of the minimal repair (re-runnable forever):")
    for r in results:
        A, B = r["pair"]
        print(
            f"  task={r['task']}  seed={A}  mix_seed={B}  "
            f"alpha={r['alpha']}  (server gain as configured)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
