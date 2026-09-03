"""
repair_alpha_sweep_cont.py — finish the decode_str coarse sweep
===============================================================

The full sweep (repair_alpha_sweep.py) covered alpha 0.05..0.80 for
decode_str / pair (42,7) — all FAIL — before the terminal session timed
out. This runs the four remaining points 0.85..1.00 and reports the
verdict. (alpha=1.00 is pure branch B, which already FAILs.)

Run with the endpoint up:
  python repair_alpha_sweep_cont.py
"""

import json
import urllib.request

from code_branch_test import TASKS, eval_code, extract_code

URL = "http://127.0.0.1:8100/v1/chat/completions"
IDX = 7  # decode_str
PROMPT, FNAME, TESTS = TASKS[IDX]
A, B = 42, 7
TAIL = [0.85, 0.90, 0.95, 1.00]


def gen(seed, mix_seed, alpha):
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 300,
            "temperature": 0,
            "thought_seed": seed,
            "thought_mix_seed": mix_seed,
            "thought_mix_alpha": alpha,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return extract_code(d["choices"][0]["message"]["content"])


def main() -> int:
    print(f"continuing sweep: {FNAME}, pair ({A},{B}), alpha {TAIL}")
    any_pass = False
    for a in TAIL:
        code = gen(A, B, a)
        ok, detail = eval_code(code, TESTS)
        any_pass = any_pass or ok
        print(f"  α={a:<5} {'PASS' if ok else 'fail'}   [{detail}]")

    if any_pass:
        print("\nVERDICT: passing blend found in (0.80, 1.00] — needs bisection")
    else:
        print(
            "\nVERDICT: full curve α=0.05..1.00 is FAIL for every blend."
            "\nNo minimal passing α exists for this task on this server."
            "\nInterpretation: both endpoints and the entire blend path stay"
            "\nbelow the pass line — decode_str is at/below the 4B ceiling,"
            "\nand blending (as established) cannot lift the ceiling."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
