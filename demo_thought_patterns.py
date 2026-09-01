"""
demo_thought_patterns.py
========================

End-to-end demonstration of the Simplex Thought Field on the
Qwen3.5 GatedDeltaNet layer.

Run::

    python demo_thought_patterns.py
"""

import os

import torch

from modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from simplex_gated_delta_net import SimplexGatedDeltaNet
from simplex_thought_field import ThoughtTrace


class DemoConfig:
    hidden_size = 128
    linear_num_value_heads = 8
    linear_num_key_heads = 2
    linear_key_head_dim = 16
    linear_value_head_dim = 16
    linear_conv_kernel_dim = 4
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    dtype = None
    # stub: the released layer reads config.layer_types[layer_idx]
    layer_types = ["linear_attention"] * 64


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def ascii_ridge(values, width=64, height=10) -> str:
    """Tiny 'ridge plot' of a 1-D thought pattern as ASCII art."""
    T = len(values)
    idx = [round(i * (T - 1) / (width - 1)) for i in range(width)]
    ds = [values[i] for i in idx]
    vmin, vmax = min(ds), max(ds)
    span = (vmax - vmin) or 1.0
    step = span / (height - 1)
    lines = []
    for r in range(height):
        level = vmin + step * r
        lines.append("".join("#" if d >= level else " " for d in ds))
    footer = f"  t=0 {' ' * max(0, width - 6)} t={T - 1}"
    return "\n".join(reversed(lines)) + "\n" + footer


def main() -> None:
    torch.manual_seed(1234)
    x = torch.randn(1, 64, DemoConfig.hidden_size)

    # 1 -------------------------------------------------------------------
    banner("1. Passthrough check: thought={'enabled': False} == vanilla GatedDeltaNet")
    torch.manual_seed(7)
    base = Qwen3_5GatedDeltaNet(DemoConfig, 0)
    off = SimplexGatedDeltaNet(DemoConfig, 0, thought={"enabled": False})
    off.load_state_dict(base.state_dict(), strict=False)
    out_base, out_off = base(x), off(x)
    assert torch.equal(out_base, out_off)
    print("identical outputs: True")

    # 2 -------------------------------------------------------------------
    banner("2. Retraceability: same seed + input => bit-identical thought field")
    torch.manual_seed(7)
    layer = SimplexGatedDeltaNet(DemoConfig, 0, thought={"seed": 42})
    o1, o2 = layer(x), layer(x)
    assert torch.equal(o1, o2)
    tr = layer.thought.last_trace()
    assert tr is not None
    print(f"layer output identical across runs: True")
    print(f"thought fingerprint : {tr.fingerprint}")
    print(
        "coords shape        : "
        f"{tuple(tr.coords.shape)}   slots: "
        + ", ".join(f"{k}={tuple(v.shape)}" for k, v in tr.slot_values.items())
    )

    # 3 -------------------------------------------------------------------
    banner("3. Seed dependence: same weights, different seed => different pattern")
    torch.manual_seed(7)
    layer_alt = SimplexGatedDeltaNet(DemoConfig, 0, thought={"seed": 7})
    sd = {k: v for k, v in layer.thought.state_dict().items() if "seed_offset" not in k}
    layer_alt.thought.load_state_dict(sd, strict=False)
    o_alt = layer_alt(x)
    assert not torch.equal(o1, o_alt)
    print("outputs differ across seeds: True")

    # 4 -------------------------------------------------------------------
    banner("4. Coherence: the temporal drift sweeps field-space smoothly")
    # Fix the hidden state so the ONLY motion is the per-token drift. The
    # per-token bias is then a 1-D curve S(M h0 + drift*t + off): a smooth
    # 'thought wave' rather than jittery per-token noise. We measure the
    # step-to-step change relative to the signal amplitude; a coherent wave
    # has a small ratio, white noise would sit near ~1.
    h0 = torch.randn(1, 1, DemoConfig.hidden_size)
    hfix = h0.expand(1, 64, DemoConfig.hidden_size).contiguous()
    curve = layer.thought.slot_bias(hfix, "b")[0, :, 0]  # (64,)
    step = (curve[1:] - curve[:-1]).abs().mean().item()
    amp = (curve - curve.mean()).abs().mean().item()
    ratio = step / (amp or 1.0)
    print(f"mean |step|   (drift sweep) : {step:.5f}")
    print(f"mean amplitude             : {amp:.5f}")
    print(f"smoothness  (step/amplitude): {ratio:.3f}   (<< 1 -> coherent wave)")
    assert ratio < 0.2, f"drift sweep not smooth (ratio={ratio:.3f})"

    # 5 -------------------------------------------------------------------
    banner("5. Trace round-trip (JSON) + fingerprint verification")
    payload = tr.to_json()
    tr2 = ThoughtTrace.from_json(payload)
    assert tr2.fingerprint == tr.fingerprint
    assert tr2.verify_fingerprint()
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "thought_trace.json"
    )
    with open(path, "w") as f:
        f.write(payload)
    print(
        f"saved {os.path.basename(path)} ({len(payload)} bytes), fingerprint verified"
    )

    # 6 -------------------------------------------------------------------
    banner("6. Replay: trace coords + weights reproduce the modulation exactly")
    assert torch.equal(layer.thought.replay_slot(tr2.coords, "b"), tr2.slot_values["b"])
    assert torch.equal(layer.thought.replay_slot(tr2.coords, "a"), tr2.slot_values["a"])
    print("replay matches recorded trace: True")

    # 7 -------------------------------------------------------------------
    banner("7. What the model feels: slot 'b' bias, head 0, 64 tokens (top=max)")
    print(ascii_ridge(tr.slot_values["b"][0, :, 0].tolist()))

    # 8 -------------------------------------------------------------------
    banner("8. Gradient flow: the latent -> field map is learnable")
    torch.manual_seed(0)
    glayer = SimplexGatedDeltaNet(DemoConfig, 0, thought={"seed": 1})
    gx = torch.randn(2, 16, DemoConfig.hidden_size)
    glayer(gx).sum().backward()
    gw = glayer.thought.proj.weight.grad
    assert gw is not None and torch.isfinite(gw).all() and gw.abs().sum().item() > 0
    print(f"thought.proj.weight.grad: finite, ||g||_1 = {gw.abs().sum().item():.6f}")

    print("\nAll demo checks passed.")


if __name__ == "__main__":
    main()
