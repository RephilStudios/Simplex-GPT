"""
thought_recorder.py
===================

Captures the Simplex Thought Field's latent->field coordinates
(``p_t = M @ h_t + drift * t``) as they are computed during ``lm.generate``
and hands them to the UI so it can plot the 3D "thought wave" in real time,
right alongside the token stream.

How it works
------------
* :func:`install` wraps the ``slot_bias`` of ONE target layer (default: the
  last GatedDeltaNet layer) with a shared, thread-safe proxy. It is a
  one-time, idempotent install — the same proxy serves every request.
* The proxy is a **no-op unless the calling thread has an active
  :class:`TrajectoryBuffer`** (stored thread-locally). So concurrent
  requests on different worker threads each record into their own buffer
  and never cross-contaminate — the same isolation model as ``request_seed``
  in :mod:`simplex_thought_field`.
* On each capture (slot ``"b"``) it appends the full coordinate row —
  ``(B, T, dim)`` flattened to ``T * dim`` floats, in sequence order.
  During the prefill that contributes the whole prompt-length wave at once;
  during decode it contributes exactly one point per generated token. So the
  buffer is precisely the running ``p_t`` trajectory, in generation order.

No new parameters, no effect on the math: the proxy runs the original
``slot_bias`` first and only *reads* the ``_last_coords`` it already
recorded for its trace, so a thought-disabled or un-instrumented model is
unaffected.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from thought_wrap import delta_net_layers

__all__ = [
    "TrajectoryBuffer",
    "install",
    "is_installed",
    "target_dim",
    "set_active_buffer",
    "clear_active_buffer",
    "active_buffer",
]

_tls = threading.local()


class TrajectoryBuffer:
    """A lock-protected, order-preserving flat list of coordinate floats.

    Producers (the ``slot_bias`` proxy, running on the generation thread)
    call :meth:`extend`; the request-serving thread calls :meth:`drain` to
    pull out whatever has accumulated since the last call. The buffer is a
    plain shared object — the lock is what makes cross-thread hand-off safe.
    """

    def __init__(self, dim: int = 3):
        self.dim = int(dim)
        self._lock = threading.Lock()
        self._data: List[float] = []

    def extend(self, points: List[float]) -> None:
        with self._lock:
            self._data.extend(points)

    def drain(self) -> List[float]:
        """Return and remove all points accumulated so far."""
        with self._lock:
            out, self._data = self._data, []
            return out

    def snapshot(self) -> List[float]:
        """Return a copy without removing the points."""
        with self._lock:
            return list(self._data)

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._data)


def set_active_buffer(buf: Optional[TrajectoryBuffer]) -> None:
    """Point the *current thread's* capture proxy at ``buf``.

    Pass ``None`` to disable capture for this thread. Thread-local by design,
    so setting it here never affects another worker thread.
    """
    _tls.buffer = buf


def clear_active_buffer() -> None:
    _tls.buffer = None


def active_buffer() -> Optional[TrajectoryBuffer]:
    return getattr(_tls, "buffer", None)


def _target_layer(lm, layer_index: Optional[int]):
    layers = delta_net_layers(lm)
    if not layers:
        return None
    if layer_index is None:
        return layers[-1][1]
    for idx, layer in layers:
        if idx == layer_index:
            return layer
    return layers[-1][1]


def install(lm, layer_index: Optional[int] = None):
    """Install the shared capture proxy on the target layer. Idempotent.

    Returns the target :class:`ThoughtModulator`, or ``None`` if there is no
    wrapped GatedDeltaNet layer to instrument.
    """
    layer = _target_layer(lm, layer_index)
    if layer is None:
        return None
    mod = getattr(getattr(layer, "linear_attn", None), "thought", None)
    if mod is None or getattr(mod, "_recorder_installed", False):
        return mod

    orig = mod.slot_bias  # bound method, captured before we shadow it

    def proxy(hidden_states, slot):
        out = orig(hidden_states, slot)
        buf = active_buffer()
        if buf is not None and slot == "b":
            coords = getattr(mod, "_last_coords", None)
            if coords is not None:
                buf.extend(coords.reshape(-1).tolist())
        return out

    mod._recorder_installed = True
    mod._orig_slot_bias = orig
    mod.slot_bias = proxy  # instance attr shadows the class method
    return mod


def is_installed(lm) -> bool:
    layer = _target_layer(lm, None)
    mod = (
        getattr(getattr(layer, "linear_attn", None), "thought", None) if layer else None
    )
    return bool(mod is not None and getattr(mod, "_recorder_installed", False))


def target_dim(lm) -> int:
    layer = _target_layer(lm, None)
    mod = (
        getattr(getattr(layer, "linear_attn", None), "thought", None) if layer else None
    )
    return int(getattr(mod, "dim", 3)) if mod is not None else 3
