"""
code_branch_http_test.py — coding-branch test against the LIVE endpoint
=======================================================================

Runs the same 5 tasks + test cases as code_branch_test.py, but against the
already-running serve_real_endpoint (field ON, gain 4.0) so we don't need a
second model copy on the GPU:

  field42 / field7 : per-request thought_seed (two deterministic branches)
  retrace          : seed-42 request twice -> byte-identical code
  (vanilla and random baselines need a local process or a field-OFF server)

Requires:  python serve_real_endpoint.py --thought-enabled --gain 4.0
Run:       python code_branch_http_test.py
"""

import json
import urllib.request

from code_branch_test import TASKS, eval_code, extract_code

URL = "http://127.0.0.1:8100/v1/chat/completions"


def chat(task: str, seed: int, max_tokens: int = 300) -> str:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": task}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "thought_seed": seed,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def main() -> int:
    print("using live endpoint at", URL)
    RESULTS = {}
    RETRACE = False
    for prompt, fname, tests in TASKS:
        RESULTS[fname] = {}
        for s in (42, 7):
            code = extract_code(chat(prompt, s))
            ok, detail = eval_code(code, tests)
            RESULTS[fname][f"field{s}"] = (ok, detail, code)
        if fname == TASKS[0][1]:
            code2 = extract_code(chat(prompt, 42))
            RETRACE = code2 == RESULTS[fname]["field42"][2]

        marks = "  ".join(
            f"{c}:{'PASS' if RESULTS[fname][c][0] else 'FAIL'}"
            for c in ("field42", "field7")
        )
        print(f"{fname:20} {marks}")

    n = len(TASKS)

    def rate(cond):
        return sum(RESULTS[t][cond][0] for t in RESULTS) / n

    diversity = sum(
        1 for t in RESULTS if RESULTS[t]["field42"][2] != RESULTS[t]["field7"][2]
    )

    print("\n--- summary (live endpoint, gain 4.0) ---")
    print(f"pass rate: field42 {rate('field42'):.0%}   field7 {rate('field7'):.0%}")
    print(f"branch diversity (field42 vs field7 differ): {diversity}/{n}")
    print(f"retrace (field42 twice -> byte-identical): {'PASS' if RETRACE else 'FAIL'}")

    t = TASKS[0][1]
    print(f"\n--- example: {t} ---")
    for c in ("field42", "field7"):
        code = RESULTS[t][c][2]
        head = "\n".join(code.splitlines()[:8])
        print(f"[{c}] {'PASS' if RESULTS[t][c][0] else 'FAIL'}\n{head}\n" + "-" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
