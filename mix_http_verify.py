"""
mix_http_verify.py — end-to-end verification of branch blending over HTTP
=========================================================================

Verifies the handoff checklist item:
  1. plain (no thought fields) x2          -> byte-identical
  2. mix (seed 42 + mix_seed 7, alpha .5) x2 -> byte-identical AND
     different from plain
  3. response `thought` block carries mix_seed / mix_alpha
  4. bonus: does the mixed code actually pass the task's test cases?

Runs against http://127.0.0.1:8100 (serve_real_endpoint, field ON).
Run:  python mix_http_verify.py
"""

import json
import urllib.request

from code_branch_test import TASKS, eval_code, extract_code

URL = "http://127.0.0.1:8100/v1/chat/completions"
PROMPT = TASKS[5][0]  # longest_palindromic_substring
TESTS = TASKS[5][2]


def chat(extra: dict, max_tokens: int = 300) -> dict:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            **extra,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main() -> int:
    print("verifying branch-blending mix over HTTP at", URL)
    print(f"prompt (TASKS[5]): {PROMPT[:70]}...")

    plain_a = chat({})
    plain_b = chat({})
    mix_args = {"thought_seed": 42, "thought_mix_seed": 7, "thought_mix_alpha": 0.5}
    mix_a = chat(mix_args)
    mix_b = chat(mix_args)

    p1 = plain_a["choices"][0]["message"]["content"]
    p2 = plain_b["choices"][0]["message"]["content"]
    m1 = mix_a["choices"][0]["message"]["content"]
    m2 = mix_b["choices"][0]["message"]["content"]

    checks = [
        ("plain x2 byte-identical", p1 == p2),
        ("mix   x2 byte-identical", m1 == m2),
        ("mix   differs from plain", m1 != p1),
    ]

    thought = mix_a.get("thought") or {}
    checks.append(("thought.mix_seed == 7", thought.get("mix_seed") == 7))
    checks.append(
        (
            "thought.mix_alpha == 0.5",
            thought.get("mix_alpha") is not None
            and abs(thought["mix_alpha"] - 0.5) < 1e-9,
        )
    )
    checks.append(("thought.seed == 42 (base branch)", thought.get("seed") == 42))

    # plain request must NOT report an active mix
    p_thought = plain_a.get("thought") or {}
    checks.append(("plain thought.mix_seed is None", p_thought.get("mix_seed") is None))

    ok, detail = eval_code(extract_code(m1), TESTS)
    checks.append(
        (f"mixed code {'passes' if ok else 'FAILS'} test cases [{detail}]", ok)
    )

    print("\n--- checks ---")
    all_ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_ok = all_ok and passed

    print("\n--- plain thought block ---")
    print(
        json.dumps({k: v for k, v in p_thought.items() if k != "trajectory"}, indent=2)
    )
    print("\n--- mix thought block ---")
    print(json.dumps({k: v for k, v in thought.items() if k != "trajectory"}, indent=2))
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
