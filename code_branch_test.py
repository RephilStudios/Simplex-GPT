"""
code_branch_test.py — does the thought field's controlled branching help coding?
================================================================================

Objective test on the 4B. Each task is a small function with real test cases,
executed in a subprocess (timeout-guarded). Conditions at gain 4 (field RMS
matched for the random baseline):

  vanilla : greedy, no field
  field42 / field7 : the field, two seeds -> two deterministic branches
  rand42  : per-position noise, RMS-matched to the field

Metrics:
  pass/fail per condition
  rescue     : vanilla FAILs but at least one field branch PASSes
  diversity  : field42 vs field7 code differs (real alternative implementation?)
  retrace    : field42 run twice -> byte-identical code

Run:  python code_branch_test.py --weights models/Qwen3.5-4B --gain 4
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import torch

from steer_ablation import _restore, _wrap_bias, field_rms, random_factory
from thought_wrap import set_seed, unwrap, wrap

# (task prompt, function name, test source)
TASKS = [
    (
        "Write a Python function parse_seconds(s) that converts a duration "
        "string like '1h 20m' to total seconds (4800). It may contain 'h', "
        "'m' and 's' parts in any order, separated by spaces. "
        "Return only the code of the function, no explanation.",
        "parse_seconds",
        textwrap.dedent(
            """
            try:
                assert parse_seconds("1h 20m") == 4800
                assert parse_seconds("45m") == 2700
                assert parse_seconds("3h") == 10800
                assert parse_seconds("10m 30s") == 630
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    (
        "Write a Python function valid_parentheses(s) that returns True if the "
        "string of brackets '([{])}' is properly matched and nested. "
        "Return only the code of the function, no explanation.",
        "valid_parentheses",
        textwrap.dedent(
            """
            try:
                assert valid_parentheses("([{}])") == True
                assert valid_parentheses("([})") == False
                assert valid_parentheses("") == True
                assert valid_parentheses("(())(())") == True
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    (
        "Write a Python function most_frequent(nums) that returns the most "
        "common integer in the list. Return only the code of the function, "
        "no explanation.",
        "most_frequent",
        textwrap.dedent(
            """
            try:
                assert most_frequent([1, 2, 2, 3]) == 2
                assert most_frequent([5, 5, 5, 1]) == 5
                assert most_frequent([7]) == 7
                assert most_frequent([3, 1, 3, 1, 3]) == 3
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    (
        "Write a Python function merge_intervals(ivs) that takes a list of "
        "[start, end] pairs and merges all overlapping or touching intervals. "
        "Example: [[1,3],[2,6],[8,10]] -> [[1,6],[8,10]]. "
        "Return only the code of the function, no explanation.",
        "merge_intervals",
        textwrap.dedent(
            """
            try:
                assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]
                assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
                assert merge_intervals([[5, 6]]) == [[5, 6]]
                assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    (
        "Write a Python function flatten(nested) that flattens an arbitrarily "
        "nested list of integers into a single list. "
        "Example: [1, [2, [3, 4]]] -> [1, 2, 3, 4]. "
        "Return only the code of the function, no explanation.",
        "flatten",
        textwrap.dedent(
            """
            try:
                assert flatten([1, [2, [3, 4]]]) == [1, 2, 3, 4]
                assert flatten([]) == []
                assert flatten([[[]]]) == []
                assert flatten([[[1]], [2], 3]) == [1, 2, 3]
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    # ---- edge-of-ability (vanilla should fail on some) ----
    (
        "Write a Python function longest_palindromic_substring(s) that returns "
        "the longest substring of s which is a palindrome. "
        "Return only the code of the function, no explanation.",
        "longest_palindromic_substring",
        textwrap.dedent(
            """
            try:
                assert longest_palindromic_substring("babad") in ("bab", "aba")
                assert longest_palindromic_substring("cbbd") == "bb"
                assert longest_palindromic_substring("a") == "a"
                assert longest_palindromic_substring("forgeeksskeegfor") == "geeksskeeg"
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    (
        "Write a Python function num_islands(grid) where grid is a list of "
        "lists of the characters '1' (land) and '0' (water). Count the number "
        "of connected land regions (4-directional). "
        "Return only the code of the function, no explanation.",
        "num_islands",
        textwrap.dedent(
            """
            try:
                assert num_islands([["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]) == 3
                assert num_islands([["1","1","1"],["0","1","0"],["1","1","1"]]) == 1
                assert num_islands([["0"]]) == 0
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    (
        "Write a Python function decode_str(s) that decodes an encoded string "
        "where k[x] means repeat the inside x exactly k times. "
        "Example: '3[a2[b]]' -> 'aaabaaab'. "
        "Return only the code of the function, no explanation.",
        "decode_str",
        textwrap.dedent(
            """
            try:
                assert decode_str("3[a2[b]]") == "abbabbabb"
                assert decode_str("2[ab]3[cd]") == "ababcdcd"
                assert decode_str("abc3[def]") == "abcdefdefdef"
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
    (
        "Write a Python function min_coin_change(coins, amount) that returns "
        "the minimum number of coins needed to make the amount, or -1 if it "
        "is impossible. coins is a list of positive integers. "
        "Return only the code of the function, no explanation.",
        "min_coin_change",
        textwrap.dedent(
            """
            try:
                assert min_coin_change([1, 2, 5], 11) == 3
                assert min_coin_change([2], 3) == -1
                assert min_coin_change([1], 0) == 0
                assert min_coin_change([186, 419, 83, 408], 6249) == 20
                print("ALL_PASS")
            except Exception as e:
                print("FAIL", type(e).__name__, e)
            """
        ),
    ),
]

# thinking-end tag, built at runtime so the literal tag sequence never appears in source
_THINK_END_RE = re.compile(r"\n?\s*" + "<" + "/think" + ">" + r"\s*\n?")


def extract_code(txt: str) -> str:
    txt = _THINK_END_RE.sub(" ", txt)  # strip thinking-end tokens if present
    m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)
    if m:
        return m.group(1).strip()
    i = txt.find("def ")
    return txt[i:].strip() if i != -1 else txt.strip()


def eval_code(code: str, tests: str):
    fd, path = tempfile.mkstemp(suffix=".py", dir=".")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code + "\n\n" + tests)
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return ("ALL_PASS" in out), out.strip().splitlines()[-1] if out.strip() else ""
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--max-tokens", type=int, default=300)
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

    rms = field_rms(lm, args.gain)
    print(f"\ngain={args.gain} (field RMS {rms:.3f}, random matched)\n")

    RETRACE = False
    RESULTS = {}  # name -> {cond: (ok, detail, code)}
    for prompt, fname, tests in TASKS:
        RESULTS[fname] = {}

        # vanilla
        code = extract_code(generate(prompt, args.max_tokens))
        ok, detail = eval_code(code, tests)
        RESULTS[fname]["vanilla"] = (ok, detail, code)

        # field branches (retrace on the first task)
        wrap(lm, 42, args.gain, args.gain)
        for s in (42, 7):
            set_seed(lm, s)
            code = extract_code(generate(prompt, args.max_tokens))
            ok, detail = eval_code(code, tests)
            RESULTS[fname][f"field{s}"] = (ok, detail, code)
        if fname == TASKS[0][1]:
            set_seed(lm, 42)
            code2 = extract_code(generate(prompt, args.max_tokens))
            RETRACE = code2 == RESULTS[fname]["field42"][2]
        unwrap(lm)

        # random baseline
        st = _wrap_bias(lm, random_factory(42, rms))
        code = extract_code(generate(prompt, args.max_tokens))
        ok, detail = eval_code(code, tests)
        RESULTS[fname]["rand42"] = (ok, detail, code)
        _restore(st)

        marks = "  ".join(
            f"{c}:{'PASS' if RESULTS[fname][c][0] else 'FAIL'}"
            for c in ("vanilla", "field42", "field7", "rand42")
        )
        print(f"{fname:20} {marks}")

    n = len(TASKS)

    def rate(cond):
        return sum(RESULTS[t][cond][0] for t in RESULTS) / n

    rescue = sum(
        1
        for t in RESULTS
        if not RESULTS[t]["vanilla"][0]
        and (RESULTS[t]["field42"][0] or RESULTS[t]["field7"][0])
    )
    vanilla_fail = sum(1 for t in RESULTS if not RESULTS[t]["vanilla"][0])
    diversity = sum(
        1 for t in RESULTS if RESULTS[t]["field42"][2] != RESULTS[t]["field7"][2]
    )

    print("\n--- summary ---")
    print(
        f"pass rate: vanilla {rate('vanilla'):.0%}   field42 {rate('field42'):.0%}   "
        f"field7 {rate('field7'):.0%}   rand42 {rate('rand42'):.0%}"
    )
    print(
        f"rescue (vanilla FAIL -> a field branch PASS): {rescue}/{vanilla_fail} "
        f"of the vanilla failures"
    )
    print(f"branch diversity (field42 vs field7 differ): {diversity}/{n}")
    print(f"retrace (field42 twice -> byte-identical): {'PASS' if RETRACE else 'FAIL'}")

    # show one example: first task
    t = TASKS[0][1]
    print(f"\n--- example: {t} ---")
    for c in ("vanilla", "field42", "field7", "rand42"):
        code = RESULTS[t][c][2]
        head = "\n".join(code.splitlines()[:6])
        print(f"[{c}] {'PASS' if RESULTS[t][c][0] else 'FAIL'}\n{head}\n" + "-" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
