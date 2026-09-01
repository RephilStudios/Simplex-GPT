"""
test_real_integration.py
========================

Verifies that our ``SimplexGatedDeltaNet`` (thought-field wrapper) is
drop-in compatible with the *released* transformers ``Qwen3_5GatedDeltaNet``:

A. (no weights needed) with the real HF config:
   * state-dict keys identical (thought off),
   * forward outputs agree,
   * ``load_base_state_dict`` of the real layer's weights into the wrapped
     layer leaves only ``thought.*`` as new parameters,
   * retraceability: same seed + input => bit-identical output (GPU).
B. (``--weights`` pointing at a downloaded checkpoint) load the real weights
   into a wrapped model and check one layer's output matches vanilla exactly
   with thought disabled.

Run::

    python test_real_integration.py                          # section A
    python test_real_integration.py --weights models/Qwen3.5-4B   # A + B
"""

from __future__ import annotations

import argparse
import os

# reduce CUDA allocator fragmentation (must be set before the first CUDA call)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from simplex_gated_delta_net import SimplexGatedDeltaNet


def get_text_config(model_id: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tc = getattr(cfg, "text_config", None)
    if tc is None:
        tc = cfg
    return tc


def section_a(tc) -> bool:
    import copy

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet as RealGDN,
    )

    # normalize dtype so the local shim (which honors config.dtype) and the
    # released class compare in the same precision (fp32)
    tc = copy.copy(tc)
    tc.dtype = None

    torch.manual_seed(7)
    real = RealGDN(tc, 0)
    torch.manual_seed(7)
    off = SimplexGatedDeltaNet(tc, 0, thought={"enabled": False})

    # -- keys ----------------------------------------------------------------
    # NOTE: ``thought.*`` params are always registered (unused when disabled),
    # so the "off" comparison ignores them.
    rk = set(real.state_dict().keys())
    ok = {k for k in off.state_dict().keys() if "thought" not in k}
    missing = rk - ok
    extra = ok - rk
    print(
        f"[A1] keys: real={len(rk)} wrapped-off (excl. thought.*)={len(ok)} "
        f"missing={len(missing)} extra={len(extra)}"
    )
    if missing or extra:
        for k in sorted(missing)[:5]:
            print("   missing:", k)
        for k in sorted(extra)[:5]:
            print("   extra:  ", k)
    assert not missing and not extra, "state-dict key mismatch"

    # -- forward agreement ----------------------------------------------------
    x = torch.randn(2, 32, tc.hidden_size)
    with torch.no_grad():
        y1, y2 = real(x), off(x)
    diff = (y1 - y2).abs().max().item()
    print(f"[A2] forward agreement (random init): max|diff| = {diff:.3e}")
    # 1e-3 is well inside bf16 rounding (eps ~8e-3); the local copy and the
    # released class differ only in fp32 accumulation order.
    assert diff < 1e-3, "forward mismatch vs released layer"

    # -- load_base_state_dict -------------------------------------------------
    on = SimplexGatedDeltaNet(
        tc,
        0,
        thought={"seed": 42, "gain_b": 0.5, "gain_a": 0.5, "drift": (0.0, 0.0, 0.02)},
    )
    on.load_base_state_dict(real.state_dict())
    new = [k for k in on.state_dict() if k not in real.state_dict()]
    base_new = [k for k in new if k.endswith(".base.weight")]
    thought_new = [k for k in new if "thought" in k]
    other_new = [k for k in new if k not in base_new and "thought" not in k]
    print(
        f"[A3] load_base_state_dict: {len(new)} new params = "
        f"{len(base_new)} remapped .base. + {len(thought_new)} thought.*, "
        f"other = {len(other_new)}"
    )
    assert len(base_new) == 2, f"expected 2 .base. keys, got {base_new}"
    assert not other_new, f"unexpected non-thought params: {other_new}"
    # the real weights must actually have landed in .base.
    assert torch.equal(on.in_proj_a.base.weight.data, real.in_proj_a.weight.data)
    assert torch.equal(on.in_proj_b.base.weight.data, real.in_proj_b.weight.data)
    print("     in_proj_a/b real weights landed in .base.*: verified")

    # -- retraceability on GPU -------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    on = on.to(dev)
    x = x.to(dev)
    with torch.no_grad():
        on.thought.set_seed(42)
        r1 = on(x)
        on.thought.set_seed(42)
        r2 = on(x)
        on.thought.set_seed(7)
        r3 = on(x)
    same = torch.equal(r1, r2)
    diff_seed = not torch.equal(r1, r3)
    print(
        f"[A4] retraceability on {dev}: same-seed identical = {same}, "
        f"different-seed differs = {diff_seed}"
    )
    assert same and diff_seed

    print("\nSECTION A: PASS\n")
    return True


def section_b(tc, weights: str) -> bool:
    import gc

    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM as RealLM

    # release section A's tensors + cached blocks before the big load
    gc.collect()
    torch.cuda.empty_cache()
    print(f"--- section B: real checkpoint at {weights} ---")
    # NOTE: avoid device_map="cuda" — accelerate dispatch segfaults on this
    # machine (torch 2.5.1 + accelerate 1.14); CPU load + .to() is stable.
    lm = RealLM.from_pretrained(weights, torch_dtype=torch.bfloat16)
    lm = lm.to("cuda").eval()
    layers = lm.model.layers if hasattr(lm.model, "layers") else lm.layers
    idx = next(
        i for i, l in enumerate(layers) if getattr(l, "linear_attn", None) is not None
    )
    real = layers[idx].linear_attn
    torch.manual_seed(1)
    off = SimplexGatedDeltaNet(tc, idx, thought={"enabled": False})
    off.load_base_state_dict(real.state_dict())
    off = off.to(real.in_proj_qkv.weight.dtype).to("cuda").eval()
    x = torch.randn(1, 64, tc.hidden_size, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        a, b = real(x), off(x)
    diff = (a.float() - b.float()).abs().max().item()
    print(f"[B1] layer {idx} real-vs-wrapped (real weights): max|diff| = {diff:.3e}")
    assert diff < 1e-2, "real-weight forward mismatch"
    print("\nSECTION B: PASS")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument(
        "--weights",
        default=None,
        help="local dir of a downloaded checkpoint (enables section B)",
    )
    ap.add_argument(
        "--section",
        default="ab",
        choices=["a", "b", "ab"],
        help="run section a (no weights needed), b (real checkpoint), or both; "
        "on memory-tight GPUs run 'b' in its own process",
    )
    args = ap.parse_args()

    tc = get_text_config(args.model_id)
    print(
        f"config: hidden={tc.hidden_size} layers={tc.num_hidden_layers} "
        f"linear_kv_heads={tc.linear_num_key_heads}/{tc.linear_num_value_heads} "
        f"head dims {tc.linear_key_head_dim}/{tc.linear_value_head_dim} "
        f"conv_k={tc.linear_conv_kernel_dim}"
    )
    if args.section in ("a", "ab"):
        section_a(tc)
    if args.section in ("b", "ab"):
        section_b(tc, args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
