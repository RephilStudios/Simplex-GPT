"""
python_corpus.py
================

Generates a synthetic Python coding corpus for the tiny SLM.

Design goals
------------
* **Correct, consistent Python** — a tiny model learns best from a clean,
  repetitive distribution, so every snippet is valid and uses a limited,
  regular style (4-space indent, double quotes, simple statements).
* **Strong hello-world signal** — greeting / ``print`` programs are heavily
  weighted so the model reliably learns to produce ``print("hello world")``,
  including as the body of ``def main():`` / ``def hello():``.
* **Deterministic** — a seed gives a reproducible corpus.
"""

from __future__ import annotations

import random
from typing import List, Tuple

# -- small vocabularies the snippets draw from ------------------------------

GREETINGS = [
    "hello world",
    "hello",
    "hi there",
    "hey",
    "good morning",
    "good evening",
    "hello python",
    "welcome",
]
NAMES = ["alice", "bob", "carol", "dave", "eve", "frank", "grace"]
COLORS = ["red", "blue", "green", "yellow", "purple"]
ANIMALS = ["cat", "dog", "bird", "fish", "fox"]
ADJECTIVES = ["happy", "sad", "fast", "slow", "big", "small"]
VARS = ["x", "y", "n", "count", "value", "name", "age", "total"]
FUNCS = ["greet", "hello", "say", "show", "run", "main", "start"]
ARGS = ["name", "x", "n", "msg", "text", "word"]


def _pick(rng: random.Random, seq) -> object:
    return seq[rng.randrange(len(seq))]


# -- snippet generators ------------------------------------------------------


def gen_hello_world(rng: random.Random) -> str:
    """The canonical target. Several common shapes, all print hello world."""
    q = _pick(rng, ['"', "'"])
    style = rng.randrange(5)
    if style == 0:
        return f"print({q}hello world{q})\n"
    if style == 1:
        return f"def hello():\n    print({q}hello world{q})\n"
    if style == 2:
        return f"def main():\n    print({q}hello world{q})\n"
    if style == 3:
        return f"# say hello to the world\nprint({q}hello world{q})\n"
    return f"message = {q}hello world{q}\nprint(message)\n"


def gen_greeting(rng: random.Random) -> str:
    s = _pick(rng, GREETINGS)
    q = _pick(rng, ['"', "'"])
    return f"print({q}{s}{q})\n"


def gen_named(rng: random.Random) -> str:
    name = _pick(rng, NAMES)
    q = _pick(rng, ['"', "'"])
    return f"print({q}hello {name}{q})\n"


def gen_assign(rng: random.Random) -> str:
    var = _pick(rng, VARS)
    kind = rng.randrange(3)
    if kind == 0:
        val = str(rng.randrange(0, 51))
    elif kind == 1:
        val = f'"{_pick(rng, NAMES)}"'
    else:
        val = f'"{_pick(rng, COLORS)}"'
    return f"{var} = {val}\n"


def gen_arith(rng: random.Random) -> str:
    a, b = rng.randrange(1, 10), rng.randrange(1, 10)
    op = _pick(rng, ["+", "-", "*", "//"])
    return f"print({a} {op} {b})\n"


def gen_function(rng: random.Random) -> str:
    name = _pick(rng, FUNCS)
    arg = _pick(rng, ARGS)
    ret = _pick(
        rng,
        [
            f'"hello {arg}"',
            f'"{arg}"',
            f"{arg} + 1",
            f'"hi, {arg}"',
            f'"{arg} * 2"',
        ],
    )
    return f"def {name}({arg}):\n    return {ret}\n"


def gen_for(rng: random.Random) -> str:
    n = rng.randrange(1, 11)
    var = _pick(rng, ["i", "j", "k"])
    body = _pick(
        rng,
        [
            f"print({var})",
            'print("hello world")',
            f"print({var} * 2)",
            f"{var} = {var} + 1",
        ],
    )
    return f"for {var} in range({n}):\n    {body}\n"


def gen_if(rng: random.Random) -> str:
    var = _pick(rng, VARS)
    val = rng.randrange(1, 11)
    cond = _pick(
        rng, [f"{var} > {val}", f"{var} < {val}", f"{var} == {val}", f"{var} != {val}"]
    )
    then = _pick(rng, ['print("yes")', 'print("hello world")', f"{var} = {var} + 1"])
    els = _pick(rng, ['print("no")', f"{var} = {var} - 1"])
    return f"if {cond}:\n    {then}\nelse:\n    {els}\n"


# Weighted mix: hello-world and greeting/print dominate so the target is
# strongly reinforced, with the rest providing general coding syntax.
GENERATORS: List[Tuple[object, int]] = [
    (gen_hello_world, 20),
    (gen_greeting, 10),
    (gen_named, 8),
    (gen_assign, 8),
    (gen_arith, 8),
    (gen_function, 8),
    (gen_for, 6),
    (gen_if, 4),
]


def _weighted(rng: random.Random):
    total = sum(w for _, w in GENERATORS)
    r = rng.randrange(total)
    acc = 0
    for fn, w in GENERATORS:
        acc += w
        if r < acc:
            return fn
    return GENERATORS[0][0]


def build_corpus(n_snippets: int = 40000, seed: int = 7) -> Tuple[str, int]:
    """Return ``(corpus_text, n_snippets)``.

    Snippets are joined with a blank line so line/program boundaries are
    learned (the model sees ``\\n\\n`` as a statement/program separator).
    """
    rng = random.Random(seed)
    parts: List[str] = []
    for _ in range(n_snippets):
        fn = _weighted(rng)
        parts.append(fn(rng))
    text = "\n\n".join(parts) + "\n\n"
    return text, len(parts)


if __name__ == "__main__":
    text, n = build_corpus(n_snippets=200, seed=1)
    print(f"generated {n} snippets, {len(text)} chars")
    print("--- sample ---")
    print(text[:800])
