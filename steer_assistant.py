"""
steer_assistant.py
==================

A/B steering demo for the trained assistant-prose SLM.

Loads the same vanilla checkpoint twice:
  * once as a plain ``TinyThoughtLM`` (thought off),
  * once wrapped with the Simplex Thought Field (bit-exact same weights via
    ``load_vanilla_state_dict``; the only new parameters are ``thought.*``).

For each prompt it then prints:
  * the vanilla greedy completion,
  * the thought-field completion for each seed (default 42 and 7),
  * a retrace check: the same seed run twice must be bit-identical.

Run::

    python steer_assistant.py --device cuda
    python steer_assistant.py --gain 2.5 --seeds 1 2 3 --max-tokens 60
"""

from __future__ import annotations

import argparse

import torch

from llm_thought import LMConfig, TinyThoughtLM
from tokenizer import CharTokenizer

PROMPTS = [
    "The core idea is that ",
    "Let me be honest about ",
    "One thing to flag. ",
    "The result is ",
]


def build_model(
    thought: dict | None,
    weights: str,
    tok: CharTokenizer,
    hidden: int,
    layers: int,
    device: str,
) -> TinyThoughtLM:
    cfg = LMConfig
    cfg.hidden_size = hidden
    cfg.num_layers = layers
    cfg.intermediate_size = hidden * 3
    model = TinyThoughtLM(cfg, thought, vocab_size=tok.vocab_size).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    if thought is not None:
        model.load_vanilla_state_dict(state)
    else:
        model.load_state_dict(state)
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="assistant_weights.pt")
    ap.add_argument("--tokenizer", default="assistant_tokenizer.json")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = CharTokenizer.load(args.tokenizer)
    print(f"loaded {args.tokenizer} (vocab {tok.vocab_size}) + {args.weights}\n")

    vanilla = build_model(
        None, args.weights, tok, args.hidden, args.layers, args.device
    )
    thought_cfg = {
        "seed": args.seeds[0],
        "gain_b": args.gain,
        "gain_a": args.gain,
        "drift": (0.0, 0.0, 0.02),
    }
    steered = build_model(
        thought_cfg, args.weights, tok, args.hidden, args.layers, args.device
    )

    changed = 0
    retrace_ok = True
    for p in PROMPTS:
        v = vanilla.generate_text(p, tok, max_new_tokens=args.max_tokens)
        outs = {}
        for s in args.seeds:
            steered.set_thought_seed(s)
            outs[s] = steered.generate_text(p, tok, max_new_tokens=args.max_tokens)
        # retrace: re-run the first seed, must be identical
        steered.set_thought_seed(args.seeds[0])
        re = steered.generate_text(p, tok, max_new_tokens=args.max_tokens)
        retrace_ok = retrace_ok and (re == outs[args.seeds[0]])

        hit = any(o != v for o in outs.values())
        changed += int(hit)
        print(f"prompt: {p!r}")
        print(f"  vanilla     : {v!r}")
        for s in args.seeds:
            mark = "  " if outs[s] == v else "->"
            print(f"  thought {s:>3}{mark}: {outs[s]!r}")
        print()

    print(f"steering changed the output on {changed}/{len(PROMPTS)} prompts")
    print(
        f"retraceability (same seed twice => identical): {'PASS' if retrace_ok else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
