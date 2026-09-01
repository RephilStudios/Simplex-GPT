"""
real_model_steering.py
======================

A/B steering demo on the **real Qwen3.5-4B** checkpoint.

Flow:
  1. load ``Qwen3_5ForCausalLM`` (bf16) + tokenizer,
  2. run the VANILLA greedy baseline on a few short, choice-rich prompts,
  3. wrap every ``linear_attn`` (GatedDeltaNet) layer **in place** with the
     Simplex Thought Field (shared base weights — no parameter duplication),
  4. re-run the same prompts for each thought seed + a retrace check,
  5. unwrap and verify the baseline is bit-exactly restored.

The in-place wrap reuses each layer's ``in_proj_a/b`` Linear objects, so the
only new parameters are the small ``thought.*`` modules (~15 per layer).

Run::

    python real_model_steering.py --device cuda
    python real_model_steering.py --gain 1.0 --seeds 42 7 99 --max-tokens 24
"""

from __future__ import annotations

import argparse
import os
import time

# keep large CUDA blocks unsplit (mitigates the Windows allocator
# "N free but can't place M" fragmentation on 12 GB cards)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import torch

from thought_wrap import delta_net_layers, set_seed, unwrap, wrap

PROMPTS = [
    "Pick a color. Answer with one word.",
    "Name a capital city in Europe. One word.",
    "What sound does a bee make? One word.",
    "Choose a four-letter dog breed. One word.",
]


def get_text_model(weights: str):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

    tok = AutoTokenizer.from_pretrained(weights)
    lm = Qwen3_5ForCausalLM.from_pretrained(weights, torch_dtype=torch.bfloat16)
    lm = lm.to("cuda").eval()
    return lm, tok


@torch.no_grad()
def generate(lm, tok, prompt: str, max_tokens: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    kwargs = {
        "add_generation_prompt": True,
        "return_dict": False,
        "return_tensors": None,
    }
    try:
        out = tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:  # template without enable_thinking
        out = tok.apply_chat_template(msgs, **kwargs)
    # this transformers version returns token IDs (list) or a BatchEncoding
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    else:
        ids = out
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device="cuda")
    out_ids = lm.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
    )
    new = out_ids[0][input_ids.shape[1] :]
    return tok.decode(new, skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    lm, tok = get_text_model(args.weights)
    n_lin = len(delta_net_layers(lm))
    print(
        f"loaded {args.weights} ({time.time() - t0:.0f}s) | "
        f"{n_lin} GatedDeltaNet layers to wrap | "
        f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB\n"
    )

    # 1. vanilla baseline ----------------------------------------------------
    print("--- vanilla (greedy) ---")
    base = {}
    for p in PROMPTS:
        base[p] = generate(lm, tok, p, args.max_tokens)
        print(f"  {p!r}\n    -> {base[p]!r}")
    print()

    # 2. thought field on ----------------------------------------------------
    wrap(lm, args.seeds[0], args.gain, args.gain)
    print(f"wrapped {n_lin} layers (gain {args.gain})\n")
    outs: dict[int, dict[str, str]] = {}
    for s in args.seeds:
        set_seed(lm, s)
        outs[s] = {}
        print(f"--- thought seed {s} ---")
        for p in PROMPTS:
            outs[s][p] = generate(lm, tok, p, args.max_tokens)
            print(f"  {p!r}\n    -> {outs[s][p]!r}")
        print()

    # 3. retrace: first seed again -------------------------------------------
    set_seed(lm, args.seeds[0])
    re = {p: generate(lm, tok, p, args.max_tokens) for p in PROMPTS}
    retrace_ok = all(re[p] == outs[args.seeds[0]][p] for p in PROMPTS)

    # 4. unwrap + verify restoration ------------------------------------------
    unwrap(lm)
    restored = all(generate(lm, tok, p, args.max_tokens) == base[p] for p in PROMPTS)

    changed = sum(1 for p in PROMPTS if any(outs[s][p] != base[p] for s in args.seeds))
    print(f"steering changed the answer on {changed}/{len(PROMPTS)} prompts")
    print(
        f"retraceability (same seed twice => identical): "
        f"{'PASS' if retrace_ok else 'FAIL'}"
    )
    print(f"unwrap restores vanilla bit-exactly: {'PASS' if restored else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
