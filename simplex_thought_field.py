"""
simplex_thought_field.py
========================

A differentiable simplex-noise "thought field" that turns an LLM's latent
states into coherent, retraceable thought patterns.

Design
------
1. **Latent-space coordinates.** Each token's hidden state ``h_t`` is mapped
   by a small learnable projection ``M`` into 2D/3D field-space, plus a
   per-token temporal drift:

       p_t = M h_t + drift * t

   The field then applies its own scale (``freq``) and a seed-derived
   offset, so the full sampling point is a pure function of
   ``(h_t, t, seed, weights)``.

2. **Simplex field.** A differentiable simplex-noise ``S`` (Ashima Arts /
   IQ formulation, vectorized in PyTorch) evaluated at ``p_t``. Simplex
   noise is *smooth*: neighbouring coordinates yield neighbouring values,
   so the modulation the model feels is a continuous landscape, not
   independent per-token randomness. That is what makes the patterns
   **coherent**.

3. **Slots.** The field is sampled at per-slot, per-head phase offsets.
   In the GatedDeltaNet layer there are two slots:

   * slot ``"b"`` biases ``in_proj_b`` output -> modulates
     ``beta = sigmoid(b)`` (the delta-rule **write gain**),
   * slot ``"a"`` biases ``in_proj_a`` output -> modulates
     ``g = -exp(A) * softplus(a + dt_bias)`` (the **log decay**).

   Every head samples the *same* field at a different phase, so the whole
   layer moves coherently without all heads doing exactly the same thing.

4. **Retraceability.** There is *no RNG in the forward pass*. Each forward
   records a :class:`ThoughtTrace` (coordinates, per-slot field values,
   and a SHA-256 fingerprint). The trace can be serialized to JSON,
   reloaded, verified against the fingerprint, and — combined with the
   model weights — replayed exactly via :meth:`ThoughtModulator.replay_slot`.

All noise functions are differentiable with respect to the coordinates,
so ``M`` (the latent -> field map) is learnable end to end.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = [
    "snoise2",
    "snoise3",
    "SimplexField",
    "ThoughtModulator",
    "ThoughtTrace",
    "set_request_seed",
    "clear_request_seed",
    "request_seed",
]


# ---------------------------------------------------------------------------
# Per-request (thread-local) seed context
# ---------------------------------------------------------------------------
#
# Historically the active seed was shared mutable module state, so a server
# had to serialize every request behind one lock.  The effective field seed
# is now resolved *per thread* at evaluation time: each request (or the
# generation thread of a streamed response) carries its own seed in this
# context, so concurrent requests can each be steered independently without
# touching the shared buffers.
_seed_tls = threading.local()


def set_request_seed(seed: int) -> None:
    """Set the thought-field seed for the current thread (per-request)."""
    _seed_tls.seed = int(seed)


def clear_request_seed() -> None:
    """Drop the thread-local seed so the modulator's own seed is used."""
    _seed_tls.__dict__.pop("seed", None)


def request_seed(default: Optional[int] = None) -> Optional[int]:
    """The current thread's request seed, or ``default`` if none is set."""
    s = getattr(_seed_tls, "seed", None)
    return int(s) if s is not None else default


# ---------------------------------------------------------------------------
# Differentiable simplex noise (Ashima Arts / IQ formulation, vectorized)
# ---------------------------------------------------------------------------


def _permute(x: torch.Tensor) -> torch.Tensor:
    """IQ hash permutation: ``((x * 34) + 1) * x  (mod 289)``.

    The polynomial is periodic with period 289, so it is well defined on
    un-modded integer coordinates (including negatives, via the
    mathematical modulo of :func:`torch.remainder`).
    """
    xi = x.to(torch.int64)
    r = ((xi * 34) + 1) * xi
    return torch.remainder(r, 289).to(torch.float32)


def _taylor_inv_sqrt(r: torch.Tensor) -> torch.Tensor:
    """Polynomial approximation of ``1/sqrt(r)`` for small ``r``."""
    return 1.79284291400159 - 0.85373472095314 * r


def _fract(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


def snoise2(v: torch.Tensor) -> torch.Tensor:
    """
    2D simplex noise (Gustavson's original formulation), differentiable w.r.t. ``v``.

    The triangle *surrounding the sample point* is selected per point, so the
    field is continuous everywhere (unlike the Ashima/IQ cell-index variant,
    which has value jumps across some cell edges).

    Args:
        v: tensor of shape ``(..., 2)``.

    Returns:
        tensor of shape ``(...)`` with values in roughly ``[-1, 1]``.
    """
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)  # 0.366025403784439
    G2 = (3.0 - math.sqrt(3.0)) / 6.0  # 0.211324865405187

    v0, v1 = v[..., 0], v[..., 1]
    s = (v0 + v1) * F2
    i = torch.floor(v0 + s)
    j = torch.floor(v1 + s)
    t = (i + j) * G2
    x0 = v0 - i + t
    y0 = v1 - j + t

    # Select the triangle surrounding the point
    i1 = (x0 > y0).to(v.dtype)
    j1 = 1.0 - i1
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    # Hash the three corners of that triangle
    ii = torch.remainder(i, 289)
    jj = torch.remainder(j, 289)
    h0 = _permute(_permute(jj) + ii)
    h1 = _permute(_permute(jj + j1) + ii + i1)
    h2 = _permute(_permute(jj + 1.0) + ii + 1.0)

    # Eight symmetric unit gradients
    inv = 1.0 / (2.0**0.5)
    grads = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [inv, inv],
            [inv, -inv],
            [-inv, inv],
            [-inv, -inv],
        ],
        device=v.device,
        dtype=v.dtype,
    )

    def corner(x, y, h):
        n = torch.clamp(0.5 - (x * x + y * y), min=0.0)
        n = n * n * n * n
        g = grads[(h % 8).to(torch.int64)]
        return n * (g[..., 0] * x + g[..., 1] * y)

    return 70.0 * (corner(x0, y0, h0) + corner(x1, y1, h1) + corner(x2, y2, h2))


def snoise3(v: torch.Tensor) -> torch.Tensor:
    """
    3D simplex noise (Gustavson's original formulation), differentiable w.r.t. ``v``.

    The tetrahedron *surrounding the sample point* is selected per point, so
    the field is continuous everywhere.

    Args:
        v: tensor of shape ``(..., 3)``.

    Returns:
        tensor of shape ``(...)`` with values in roughly ``[-1, 1]``.
    """
    # Skew/unskew must satisfy G3 = F3 / (1 + 3*F3) for the six tetrahedra to
    # tile the cell correctly (otherwise non-shared corners leak into the
    # falloff on selection boundaries, producing value jumps).
    # Canonical 3D simplex constants (Gustavson): F3 = 1/3, G3 = 1/6.
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0

    v0, v1, v2 = v.unbind(-1)

    # First corner (offset coordinates)
    s = (v0 + v1 + v2) * F3
    i = torch.floor(v0 + s)
    j = torch.floor(v1 + s)
    k = torch.floor(v2 + s)
    t = (i + j + k) * G3
    x0 = v0 - i + t
    y0 = v1 - j + t
    z0 = v2 - k + t

    # Select the tetrahedron surrounding the point (Gustavson ordering).
    # The six cases partition the cell by the coordinate rank ordering;
    # non-empty cases below are mutually exclusive and exhaustive.
    case_xyz = (x0 >= y0) & (y0 >= z0)
    case_xzy = (x0 >= y0) & (y0 < z0) & (x0 >= z0)
    case_zxy = (x0 >= y0) & (y0 < z0) & (x0 < z0)
    case_zyx = (y0 > x0) & (y0 < z0)
    case_yzx = (y0 > x0) & (y0 >= z0) & (x0 < z0)
    case_yxz = (y0 > x0) & (y0 >= z0) & (x0 >= z0)

    i1x = (case_xyz | case_xzy).to(v.dtype)
    i1y = (case_yzx | case_yxz).to(v.dtype)
    i1z = (case_zxy | case_zyx).to(v.dtype)
    i2x = (case_xyz | case_xzy | case_zxy | case_yxz).to(v.dtype)
    i2y = (case_xyz | case_zyx | case_yzx | case_yxz).to(v.dtype)
    i2z = (case_xzy | case_zxy | case_zyx | case_yzx).to(v.dtype)

    # The four corners of that tetrahedron (offset from v)
    corners = [
        (x0, y0, z0),
        (x0 - i1x + G3, y0 - i1y + G3, z0 - i1z + G3),
        (x0 - i2x + 2.0 * G3, y0 - i2y + 2.0 * G3, z0 - i2z + 2.0 * G3),
        (x0 - 1.0 + 3.0 * G3, y0 - 1.0 + 3.0 * G3, z0 - 1.0 + 3.0 * G3),
    ]
    lattice = [
        (i, j, k),
        (i + i1x, j + i1y, k + i1z),
        (i + i2x, j + i2y, k + i2z),
        (i + 1.0, j + 1.0, k + 1.0),
    ]

    # 12 symmetric gradients (Gustavson's table)
    grads = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
        ],
        device=v.device,
        dtype=v.dtype,
    )

    total = None
    for (cx, cy, cz), (ix, iy, iz) in zip(corners, lattice):
        h = torch.remainder(_permute(_permute(_permute(iz) + iy) + ix), 12)
        g = grads[h.to(torch.int64)]
        n = torch.clamp(0.6 - (cx * cx + cy * cy + cz * cz), min=0.0)
        n = n * n * n * n
        contrib = n * (g[..., 0] * cx + g[..., 1] * cy + g[..., 2] * cz)
        total = contrib if total is None else total + contrib

    return 32.0 * total


# ---------------------------------------------------------------------------
# Seeded field
# ---------------------------------------------------------------------------


class SimplexField(nn.Module):
    """A seeded, differentiable simplex-noise field (optionally fbm).

    Args:
        seed: integer seed. Derives a deterministic offset vector that
            translates the field, so different seeds traverse different
            (but structurally identical) regions of the same noise.
        dim: noise dimensionality, 2 or 3.
        octaves: number of fbm octaves (1 = plain simplex).
        freq: coordinate frequency (noise scale).
        gain: fbm amplitude gain between octaves.
    """

    def __init__(
        self,
        seed: int = 0,
        dim: int = 3,
        octaves: int = 1,
        freq: float = 1.0,
        gain: float = 0.5,
    ):
        super().__init__()
        assert dim in (2, 3), "dim must be 2 or 3"
        self.dim = dim
        self.octaves = max(1, int(octaves))
        self.freq = float(freq)
        self.gain = float(gain)
        self.noise_fn = snoise3 if dim == 3 else snoise2
        self.register_buffer("seed_offset", self._make_seed_offset(seed, dim))
        norm = sum(self.gain**i for i in range(self.octaves))
        self.register_buffer("fbm_norm", torch.tensor(norm, dtype=torch.float32))

    @staticmethod
    def _make_seed_offset(seed: int, dim: int) -> torch.Tensor:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        return torch.rand(dim, generator=gen) * 64.0 - 32.0

    def forward(self, coords: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """``coords: (..., dim) -> (...,)``.

        ``seed`` optionally overrides the field's own seed for this
        evaluation only (per-request steering without mutating shared
        state). With ``seed=None`` the registered ``seed_offset`` buffer
        is used, exactly as before.
        """
        offset = (
            self.seed_offset
            if seed is None
            else self._make_seed_offset(seed, self.dim).to(
                self.seed_offset.device, self.seed_offset.dtype
            )
        )
        out: Optional[torch.Tensor] = None
        amp = 1.0
        for o in range(self.octaves):
            scale = 2.0**o
            c = coords * (self.freq * scale) + offset * scale
            val = self.noise_fn(c)
            out = val if out is None else out + amp * val
            amp *= self.gain
        assert out is not None
        return out / self.fbm_norm


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def _tensor_to_b64(t: torch.Tensor) -> Dict[str, object]:
    t = t.detach().to(torch.float32).contiguous().cpu()
    return {
        "shape": list(t.shape),
        "b64": base64.b64encode(t.numpy().tobytes()).decode("ascii"),
    }


def _b64_to_tensor(d: Dict[str, object]) -> torch.Tensor:
    raw = base64.b64decode(d["b64"])  # type: ignore[arg-type]
    shape = tuple(d["shape"])  # type: ignore[arg-type]
    return torch.frombuffer(raw, dtype=torch.float32).view(shape).clone()


def _fingerprint(coords: torch.Tensor, slot_values: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    h.update(coords.detach().contiguous().cpu().numpy().tobytes())
    for k in sorted(slot_values):
        v = slot_values[k]
        h.update(k.encode("utf-8"))
        h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


@dataclass
class ThoughtTrace:
    """The exact thought pattern of one forward pass.

    A trace plus the model weights fully determines the modulation, so
    traces are *retraceable*: replay with
    :meth:`ThoughtModulator.replay_slot` and verify with
    :meth:`verify_fingerprint`.
    """

    seed: int
    batch: int
    seq_len: int
    num_heads: int
    coords: torch.Tensor  # (B, T, dim)
    slot_values: Dict[str, torch.Tensor]  # slot name -> (B, T, num_heads)
    fingerprint: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": "simplex-thought-trace/v1",
                "seed": self.seed,
                "batch": self.batch,
                "seq_len": self.seq_len,
                "num_heads": self.num_heads,
                "fingerprint": self.fingerprint,
                "coords": _tensor_to_b64(self.coords),
                "slot_values": {
                    k: _tensor_to_b64(v) for k, v in self.slot_values.items()
                },
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "ThoughtTrace":
        d = json.loads(s)
        return cls(
            seed=d["seed"],
            batch=d["batch"],
            seq_len=d["seq_len"],
            num_heads=d["num_heads"],
            coords=_b64_to_tensor(d["coords"]),
            slot_values={k: _b64_to_tensor(v) for k, v in d["slot_values"].items()},
            fingerprint=d["fingerprint"],
        )

    def verify_fingerprint(self) -> bool:
        return _fingerprint(self.coords, self.slot_values) == self.fingerprint

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        slots = ", ".join(f"{k}: {tuple(v.shape)}" for k, v in self.slot_values.items())
        return (
            f"ThoughtTrace(seed={self.seed}, coords={tuple(self.coords.shape)}, "
            f"slots=[{slots}], fingerprint={self.fingerprint})"
        )


# ---------------------------------------------------------------------------
# Modulator
# ---------------------------------------------------------------------------


class ThoughtModulator(nn.Module):
    """Maps latent states + time -> smooth per-slot thought-field modulation.

    Coordinate model (per token ``t``)::

        p_t = M @ h_t + drift * t

    where ``M`` is a small learnable ``(hidden_size -> dim)`` projection,
    ``drift`` advances the sample point along the sequence so the field
    evolves into a *thought wave* rather than a static bump. The
    :class:`SimplexField` then applies its frequency scale and seed offset.

    Args:
        hidden_size: model hidden dimension of the input states.
        num_heads: number of value heads of the layer being modulated.
        seed: field seed (the reproducibility knob).
        dim: 2 or 3.
        octaves: fbm octaves.
        freq: field frequency (noise scale).
        drift: per-token coordinate drift; pad/truncate to ``dim``.
        slot_spread: phase offset between adjacent heads/slots.
        gain_b: bias amplitude applied to slot ``"b"`` (write gain).
        gain_a: bias amplitude applied to slot ``"a"`` (decay).
    """

    SLOTS = ("b", "a")

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        seed: int = 0,
        dim: int = 3,
        octaves: int = 1,
        freq: float = 0.75,
        drift: Sequence[float] = (0.0, 0.0, 0.02),
        slot_spread: float = 1.0,
        gain_b: float = 0.75,
        gain_a: float = 0.75,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.seed = int(seed)
        self.dim = dim
        self.slot_spread = float(slot_spread)
        self.gain_b = float(gain_b)
        self.gain_a = float(gain_a)

        # Learnable latent -> field-space map (small init so it doesn't
        # dominate the field at initialization).
        self.proj = nn.Linear(hidden_size, dim, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / (hidden_size**0.5))

        d = list(drift) + [0.0] * dim
        self.register_buffer("drift", torch.tensor(d[:dim], dtype=torch.float32))

        self.field = SimplexField(seed=self.seed, dim=dim, octaves=octaves, freq=freq)

        # Slot offsets: slot "b" occupies phases 0..H-1, slot "a" the next
        # H phases, so the two dynamics are correlated but distinct.
        offs: List[List[float]] = []
        for s_idx in range(len(self.SLOTS)):
            for h in range(num_heads):
                phase = (s_idx * num_heads + h) * float(slot_spread)
                if dim == 3:
                    offs.append([0.0, 0.0, phase])
                else:
                    offs.append([phase, phase * 0.5])
        self.register_buffer("slot_offsets", torch.tensor(offs, dtype=torch.float32))

        # Trace bookkeeping (plain attributes, not parameters).  Written on
        # every ``slot_bias`` call and read by ``last_trace``.
        self._last_coords: Optional[torch.Tensor] = None
        self._last_slot_values: Dict[str, torch.Tensor] = {}

    # -- internals ----------------------------------------------------------

    def _coords_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Field-space coordinates for a ``(B, T, D)`` hidden state.

        Stateless — a pure function of the input tensor and the projection
        weights — so it is safe to evaluate concurrently from multiple
        request threads (and free of any cross-step cache staleness).
        """
        T = hidden_states.shape[1]
        t = torch.arange(T, device=hidden_states.device, dtype=hidden_states.dtype)
        return self.proj(hidden_states) + t.view(1, T, 1) * self.drift  # (B, T, dim)

    # -- public API ---------------------------------------------------------

    def slot_bias(self, hidden_states: torch.Tensor, slot: str) -> torch.Tensor:
        """Per-head additive bias for an ``in_proj`` output.

        Args:
            hidden_states: ``(B, T, hidden_size)``.
            slot: ``"b"`` (write-gain path) or ``"a"`` (decay path).

        Returns:
            ``gain * S(p_t + slot_offset)`` of shape ``(B, T, num_heads)``.
        """
        assert slot in self.SLOTS, f"unknown slot {slot!r}"
        p = self._coords_for(hidden_states)
        H = self.num_heads
        s = self.SLOTS.index(slot)
        off = self.slot_offsets[s * H : (s + 1) * H]  # (H, dim)
        # The current thread's request seed (per-request A/B steering) wins;
        # otherwise fall back to this modulator's own seed.  Stateless:
        # no shared buffer is mutated, so concurrent requests are isolated.
        seed = request_seed(self.seed)
        vals = self.field(p.unsqueeze(2) + off.unsqueeze(0).unsqueeze(0), seed=seed)
        self._last_coords = p.detach()
        self._last_slot_values[slot] = vals.detach()
        gain = self.gain_b if slot == "b" else self.gain_a
        return gain * vals

    def set_seed(self, seed: int) -> None:
        """Change the modulator's default field seed (global fallback).

        Any per-thread request seed (see :func:`set_request_seed`) still
        takes precedence while it is set, so this no longer needs to worry
        about concurrent requests.
        """
        self.seed = int(seed)
        self.field.seed_offset.copy_(SimplexField._make_seed_offset(seed, self.dim))

    def replay_slot(
        self, coords: torch.Tensor, slot: str, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Re-evaluate a slot from recorded coordinates (no hidden state needed).

        ``coords`` is the ``(B, T, dim)`` tensor from a
        :class:`ThoughtTrace`. Together with the current weights this
        reproduces the recorded modulation exactly. Pass ``seed`` to pin the
        field seed explicitly (e.g. the trace's recorded seed); otherwise
        the modulator's current default seed is used.
        """
        assert slot in self.SLOTS, f"unknown slot {slot!r}"
        H = self.num_heads
        s = self.SLOTS.index(slot)
        off = self.slot_offsets[s * H : (s + 1) * H]  # (H, dim)
        return self.field(
            coords.unsqueeze(2) + off.unsqueeze(0).unsqueeze(0), seed=seed
        )

    def last_trace(self) -> Optional[ThoughtTrace]:
        """The :class:`ThoughtTrace` of the most recent forward pass, if any."""
        if self._last_coords is None or not self._last_slot_values:
            return None
        return ThoughtTrace(
            seed=self.seed,
            batch=self._last_coords.shape[0],
            seq_len=self._last_coords.shape[1],
            num_heads=self.num_heads,
            coords=self._last_coords.clone(),
            slot_values={k: v.clone() for k, v in self._last_slot_values.items()},
            fingerprint=_fingerprint(self._last_coords, self._last_slot_values),
        )
