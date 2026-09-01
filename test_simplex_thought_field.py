"""
test_simplex_thought_field.py
=============================

Standalone test suite (no pytest required). Run::

    python test_simplex_thought_field.py
"""

import sys

import torch

from modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from simplex_gated_delta_net import SimplexGatedDeltaNet
from simplex_thought_field import (
    SimplexField,
    ThoughtModulator,
    ThoughtTrace,
    clear_request_seed,
    request_seed,
    set_request_seed,
    snoise2,
    snoise3,
)


class DemoConfig:
    hidden_size = 64
    linear_num_value_heads = 4
    linear_num_key_heads = 2
    linear_key_head_dim = 8
    linear_value_head_dim = 8
    linear_conv_kernel_dim = 3
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    dtype = None
    # stub: the released layer reads config.layer_types[layer_idx]
    layer_types = ["linear_attention"] * 64


# -- noise ------------------------------------------------------------------


def _grid2(n=41, lo=-3.0, hi=3.0):
    xs = torch.linspace(lo, hi, n)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([gx, gy], -1).reshape(-1, 2)


def _grid3(n=13, lo=-2.0, hi=2.0):
    xs = torch.linspace(lo, hi, n)
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    return torch.stack([gx, gy, gz], -1).reshape(-1, 3)


def test_snoise2_range_mean_smooth():
    s = snoise2(_grid2())
    assert torch.isfinite(s).all()
    assert s.abs().max() <= 2.0, f"range: {s.abs().max().item()}"
    assert abs(s.mean().item()) < 0.15, f"mean: {s.mean().item()}"
    pts = torch.randn(2000, 2, generator=torch.Generator().manual_seed(3)) * 2.0
    d = (snoise2(pts + 1e-3) - snoise2(pts)).abs().max().item()
    assert d < 0.1, f"not smooth: {d}"


def test_snoise3_range_mean_smooth():
    s = snoise3(_grid3())
    assert torch.isfinite(s).all()
    assert s.abs().max() <= 2.0, f"range: {s.abs().max().item()}"
    assert abs(s.mean().item()) < 0.15, f"mean: {s.mean().item()}"
    pts = torch.randn(1500, 3, generator=torch.Generator().manual_seed(4)) * 1.5
    d = (snoise3(pts + 1e-3) - snoise3(pts)).abs().max().item()
    assert d < 0.1, f"not smooth: {d}"


def test_noise_differentiability():
    v = torch.randn(64, 3, requires_grad=True)
    snoise3(v).sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert v.grad.abs().sum().item() > 0

    v2 = torch.randn(64, 2, requires_grad=True)
    snoise2(v2).sum().backward()
    assert v2.grad.abs().sum().item() > 0


def test_seed_shifts_the_field():
    c = torch.randn(256, 3, generator=torch.Generator().manual_seed(5))
    f1 = SimplexField(seed=1)
    f2 = SimplexField(seed=2)
    assert not torch.allclose(f1(c), f2(c))


def test_fbm_octaves():
    f = SimplexField(seed=5, dim=3, octaves=3, gain=0.5)
    out = f(torch.randn(64, 3, generator=torch.Generator().manual_seed(6)))
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 2.0


# -- modulator --------------------------------------------------------------


def test_modulator_shapes_and_determinism():
    torch.manual_seed(0)
    m = ThoughtModulator(hidden_size=32, num_heads=4, seed=11)
    x = torch.randn(2, 7, 32)
    b = m.slot_bias(x, "b")
    a = m.slot_bias(x, "a")
    assert b.shape == (2, 7, 4) and a.shape == (2, 7, 4)
    assert torch.equal(b, m.slot_bias(x, "b"))
    assert torch.equal(a, m.slot_bias(x, "a"))


def test_modulator_replay_and_trace():
    torch.manual_seed(0)
    m = ThoughtModulator(hidden_size=32, num_heads=4, seed=21)
    x = torch.randn(1, 9, 32)
    _ = m.slot_bias(x, "b")
    _ = m.slot_bias(x, "a")
    tr = m.last_trace()
    assert tr is not None

    # replay: trace coords + weights reproduce the modulation exactly
    assert torch.equal(m.replay_slot(tr.coords, "b"), tr.slot_values["b"])
    assert torch.equal(m.replay_slot(tr.coords, "a"), tr.slot_values["a"])

    # JSON round-trip preserves everything
    tr2 = ThoughtTrace.from_json(tr.to_json())
    assert tr2.fingerprint == tr.fingerprint
    assert tr2.verify_fingerprint()
    assert torch.equal(tr2.coords, tr.coords)
    for k, v in tr.slot_values.items():
        assert torch.equal(tr2.slot_values[k], v)


# -- per-request (thread-local) seed context --------------------------------


def test_field_explicit_seed_matches_buffer():
    """``field(x, seed=S)`` must equal the buffer-based evaluation for seed S."""
    torch.manual_seed(3)
    f = SimplexField(seed=0, dim=3)
    x = torch.randn(4, 11, 3)
    for s in (0, 1, 42, 98765):
        explicit = f(x, seed=s)
        f.seed_offset.copy_(SimplexField._make_seed_offset(s, 3))
        buffer = f(x)
        assert torch.equal(explicit, buffer), f"seed {s} mismatch"


def test_modulator_thread_local_seed_isolation():
    """Two threads each get their own seed's field, concurrently, exactly."""
    import threading

    torch.manual_seed(0)
    m = ThoughtModulator(hidden_size=32, num_heads=4, seed=0)
    x = torch.randn(1, 8, 32)

    # reference: single-threaded evaluation with each seed set globally
    ref = {}
    for s in (42, 7):
        m.set_seed(s)
        ref[s] = {slot: m.slot_bias(x, slot) for slot in ("b", "a")}

    results: dict = {}

    def worker(s):
        set_request_seed(s)
        try:
            results[s] = {slot: m.slot_bias(x, slot).clone() for slot in ("b", "a")}
        finally:
            clear_request_seed()

    barrier = threading.Barrier(2)

    def thread(s):
        worker(s)
        barrier.wait()  # force the two evaluations to actually overlap

    ts = [threading.Thread(target=thread, args=(s,)) for s in (42, 7)]
    [t.start() for t in ts]
    [t.join() for t in ts]

    for s in (42, 7):
        for slot in ("b", "a"):
            assert torch.equal(results[s][slot], ref[s][slot]), (
                f"thread-local seed {s} slot {slot!r} mismatched reference"
            )


def test_modulator_stateless_interleaved():
    """Interleaved slot_bias calls on different inputs never return stale data."""
    torch.manual_seed(0)
    m = ThoughtModulator(hidden_size=32, num_heads=4, seed=11)
    x1 = torch.randn(2, 7, 32)
    x2 = torch.randn(2, 7, 32)
    # interleave, exactly as concurrent requests would
    a = m.slot_bias(x1, "b")
    b = m.slot_bias(x2, "b")
    c = m.slot_bias(x1, "b")
    assert torch.equal(a, c), "re-evaluation of the same input drifted (stale cache)"
    assert not torch.equal(a, b), "two different inputs gave identical bias"


# -- layer integration ------------------------------------------------------


def test_layer_disabled_passthrough():
    torch.manual_seed(7)
    base = Qwen3_5GatedDeltaNet(DemoConfig, 0)
    off = SimplexGatedDeltaNet(DemoConfig, 0, thought={"enabled": False})
    # 'thought.*' keys are present but unused when enabled=False
    missing, unexpected = off.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    x = torch.randn(1, 24, DemoConfig.hidden_size)
    assert torch.equal(base(x), off(x))


def test_layer_base_state_dict_remap():
    torch.manual_seed(7)
    base = Qwen3_5GatedDeltaNet(DemoConfig, 0)
    wrapped = SimplexGatedDeltaNet(DemoConfig, 0, thought={"seed": 3})
    wrapped.load_base_state_dict(base.state_dict())
    # the wrapped in_proj weights must now match the base ones
    assert torch.equal(wrapped.in_proj_b.base.weight, base.in_proj_b.weight)
    assert torch.equal(wrapped.in_proj_a.base.weight, base.in_proj_a.weight)


def test_layer_determinism_and_seed():
    torch.manual_seed(7)
    l1 = SimplexGatedDeltaNet(DemoConfig, 0, thought={"seed": 42})
    x = torch.randn(
        1, 24, DemoConfig.hidden_size, generator=torch.Generator().manual_seed(99)
    )
    assert torch.equal(l1(x), l1(x))

    # same weights, different field seed -> different thought pattern
    l2 = SimplexGatedDeltaNet(DemoConfig, 0, thought={"seed": 7})
    sd = {k: v for k, v in l1.thought.state_dict().items() if "seed_offset" not in k}
    l2.thought.load_state_dict(sd, strict=False)
    assert not torch.equal(l1(x), l2(x))


def test_layer_grad_flow():
    torch.manual_seed(0)
    l = SimplexGatedDeltaNet(DemoConfig, 0, thought={"seed": 1})
    x = torch.randn(1, 16, DemoConfig.hidden_size)
    l(x).sum().backward()
    g = l.thought.proj.weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g.abs().sum().item() > 0


# -- runner -----------------------------------------------------------------

if __name__ == "__main__":
    fns = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback

            print(f"  FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
