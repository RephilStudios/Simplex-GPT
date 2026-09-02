"""
probe_model.py
==============

Run this ON THE DGX SPARK (any box with the checkpoint + a matching
``transformers``) to confirm the Qwen3.5-35B-A3B loads through the project's
real code path *before* you start the endpoint.

    # fast: architecture/config only (no model load)
    python probe_model.py --weights /models/Qwen3.5-35B-A3B --config-only

    # full: load the model, wrap the thought field, generate a few tokens
    python probe_model.py --weights /models/Qwen3.5-35B-A3B --device cuda

Exits non-zero if no GatedDeltaNet (``linear_attn``) layers are found, because
that's what the thought field hooks.
"""

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config-only", action="store_true")
    ap.add_argument("--gain", type=float, default=2.0)
    args = ap.parse_args()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.weights)
    arch = (getattr(cfg, "architectures", None) or ["?"])[0]
    tc = getattr(cfg, "text_config", cfg)
    print("architectures :", arch)
    print("model_type    :", getattr(cfg, "model_type", None))
    lt = getattr(tc, "layer_types", None)
    if lt:
        lin = sum(1 for x in lt if x == "linear_attention")
        full = sum(1 for x in lt if x == "full_attention")
        print(
            f"layers        : {len(lt)}  (linear_attention={lin}, full_attention={full})"
        )
    for k in (
        "hidden_size",
        "linear_num_value_heads",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ):
        if hasattr(tc, k):
            print(f"  {k:<24}: {getattr(tc, k)}")

    if args.config_only:
        print("\n(config-only; re-run without the flag to load + generate)")
        return 0

    from serve_real_endpoint import load_real
    from thought_wrap import delta_net_layers, is_wrapped, n_wrapped, wrap

    lm, tok = load_real(args.weights, args.device)
    layers = delta_net_layers(lm)
    print("\nGatedDeltaNet (linear_attn) layers found:", len(layers))
    if not layers:
        print("!! none found — the thought field would have nothing to hook")
        return 1

    sample = layers[-1][1].linear_attn
    has_ab = hasattr(sample, "in_proj_a") and hasattr(sample, "in_proj_b")
    print("sample linear_attn has in_proj_a/in_proj_b:", has_ab)

    if has_ab:
        n = wrap(lm, 42, args.gain, args.gain)
        print(
            f"wrapped {n} layers | is_wrapped={is_wrapped(lm)} | n_wrapped={n_wrapped(lm)}"
        )

    # tiny text generate to confirm the whole path works end-to-end
    import torch

    msgs = [{"role": "user", "content": "Say hello in exactly one word."}]
    kwargs = {
        "add_generation_prompt": True,
        "return_dict": False,
        "return_tensors": None,
    }
    try:
        ids = tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tok.apply_chat_template(msgs, **kwargs)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    it = torch.tensor([list(ids)], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        out = lm.generate(
            input_ids=it,
            max_new_tokens=8,
            do_sample=False,
            temperature=1.0,
            pad_token_id=getattr(tok, "eos_token_id", None),
        )
    new = out[0][it.shape[1] :]
    print("generated   :", repr(tok.decode(new, skip_special_tokens=True)))
    print("\nPROBE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
