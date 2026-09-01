"""
simplex_gated_delta_net.py
==========================

Drop-in upgrade of :class:`Qwen3_5GatedDeltaNet` that threads a
Simplex Thought Field through the delta rule.

Mechanism (no duplication of the parent ``forward``)
-----------------------------------------------------
The parent computes::

    beta = in_proj_b(h).sigmoid()                 # memory write gain
    g    = -exp(A) * softplus(in_proj_a(h) + dt)  # log decay

We wrap ``in_proj_b`` / ``in_proj_a`` with :class:`ThoughtBiasWrapper`,
which adds a per-head bias sampled from the thought field::

    in_proj_b(h)  += gain_b * S(p_t + offset_b(h))
    in_proj_a(h)  += gain_a * S(p_t + offset_a(h))

    p_t = M h_t + drift * t   (+ field frequency scale and seed offset)

Because ``S`` is a pure function of ``(h_t, t, seed, weights)``:

* the modulation is **smooth** (simplex noise is continuous) ->
  coherent, non-jittery thought patterns;
* the forward pass is **deterministic** -> fully retraceable via
  :class:`~simplex_thought_field.ThoughtTrace`;
* the map ``M`` from latent space into field-space is
  **differentiable** -> learnable end to end.

State-dict note
---------------
With the thought field enabled, the projection weights live under
``in_proj_b.base.*`` / ``in_proj_a.base.*``. Use
:meth:`SimplexGatedDeltaNet.load_base_state_dict` to load a checkpoint
trained on the vanilla layer.

Base-class note
---------------
When ``transformers`` is installed with the ``qwen3_5`` module, this class
subclasses the **released** ``Qwen3_5GatedDeltaNet`` (so a thought-disabled
layer is bit-exact with the official model by construction). The local
``modeling_qwen3_5.py`` copy is the fallback for environments without it.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

try:  # prefer the released implementation when available
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
    )
except Exception:  # noqa: BLE001 - local fallback (transformers too old / absent)
    from modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from simplex_thought_field import ThoughtModulator

__all__ = ["ThoughtBiasWrapper", "SimplexGatedDeltaNet"]


class ThoughtBiasWrapper(nn.Module):
    """Wraps an ``in_proj`` linear and adds the thought-field slot bias."""

    def __init__(self, base: nn.Module, thought: ThoughtModulator, slot: str):
        super().__init__()
        self.base = base
        self.thought = thought
        self.slot = slot

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        y = self.base(hidden_states)
        return y + self.thought.slot_bias(hidden_states, self.slot)


class SimplexGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """GatedDeltaNet with a Simplex Thought Field wired into the delta rule.

    Args:
        config: same config object as the vanilla layer.
        layer_idx: layer index (as before).
        thought: optional dict of :class:`ThoughtModulator` kwargs, e.g.
            ``{"seed": 42, "gain_b": 0.5, "drift": (0, 0, 0.02)}``.
            Use ``{"enabled": False}`` for a strict vanilla passthrough.
    """

    def __init__(self, config, layer_idx, thought: Optional[Dict] = None):
        super().__init__(config, layer_idx)
        cfg = dict(thought or {})
        enabled = bool(cfg.pop("enabled", True))
        self.thought = ThoughtModulator(
            hidden_size=self.hidden_size,
            num_heads=self.num_v_heads,
            **cfg,
        )
        if enabled:
            self.in_proj_b = ThoughtBiasWrapper(self.in_proj_b, self.thought, "b")
            self.in_proj_a = ThoughtBiasWrapper(self.in_proj_a, self.thought, "a")

    # -- checkpoint bridging ------------------------------------------------

    @staticmethod
    def remap_base_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Map vanilla ``in_proj_{a,b}.*`` keys onto the wrapped layout."""
        out = {}
        for k, v in state_dict.items():
            if k.startswith("in_proj_b."):
                k = "in_proj_b.base." + k[len("in_proj_b.") :]
            elif k.startswith("in_proj_a."):
                k = "in_proj_a.base." + k[len("in_proj_a.") :]
            out[k] = v
        return out

    def load_base_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights from a vanilla :class:`Qwen3_5GatedDeltaNet` checkpoint.

        Handles both layouts: the wrapped one (``in_proj_{a,b}.base.*``) and
        the disabled passthrough (plain ``in_proj_{a,b}.*``). Missing
        thought-field keys (``thought.*``) keep their initialization.
        """
        wrapped = hasattr(self.in_proj_b, "base") or hasattr(self.in_proj_a, "base")
        remapped = (
            self.remap_base_state_dict(state_dict) if wrapped else dict(state_dict)
        )
        self.load_state_dict(remapped, strict=False)
