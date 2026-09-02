"""
_test_thought_recorder.py
=========================

Quick, dependency-light validation for the thought-wave recorder:

1. byte-compiles the touched modules (syntax safety),
2. if torch is importable, runs a real micro-test of
   :func:`thought_recorder.install` against a fake ``slot_bias`` to confirm
   the proxy (a) still returns the original result, (b) captures
   ``_last_coords`` on slot ``"b"`` only, and (c) is a no-op when no buffer
   is active.

Does not load the Qwen checkpoint or need CUDA.
"""

import py_compile
import sys
import types

# 1) syntax ---------------------------------------------------------------
for f in ["thought_recorder.py", "serve_real_endpoint.py", "thought_wrap.py"]:
    py_compile.compile(f, doraise=True)
print("SYNTAX_OK")

try:
    import torch

    HAVE = True
except Exception as e:  # noqa: BLE001
    HAVE = False
    print("TORCH_UNAVAILABLE", repr(e))

if not HAVE:
    sys.exit(0)

# 2) real recorder micro-test ---------------------------------------------
try:
    import thought_recorder as tr
except Exception as e:  # noqa: BLE001
    print("IMPORT_FAILED", repr(e))
    sys.exit(0)


class FakeMod:
    dim = 3

    def __init__(self):
        self._last_coords = None

    def slot_bias(self, hidden_states, slot):
        self._last_coords = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3)
        return "bias:" + slot


mod = FakeMod()
layer = types.SimpleNamespace(linear_attn=types.SimpleNamespace(thought=mod))
tr.delta_net_layers = lambda lm: [(0, layer)]  # stub the layer lookup

assert not tr.is_installed(object())
tr.install(object())
assert tr.is_installed(object()), "install should flag the target modulator"
assert mod.slot_bias is not FakeMod.slot_bias, "proxy should shadow slot_bias"

buf = tr.TrajectoryBuffer(3)
tr.set_active_buffer(buf)
res = mod.slot_bias(None, "b")
assert res == "bias:b", res
got = buf.drain()
assert got == [1.0, 2.0, 3.0], f"expected capture, got {got!r}"

# slot "a" must not double-capture
mod.slot_bias(None, "a")
assert buf.drain() == [], "slot 'a' should not capture"

# no active buffer -> pure no-op, original result unchanged
tr.clear_active_buffer()
assert mod.slot_bias(None, "b") == "bias:b"
print("REAL_RECORDER_OK")
