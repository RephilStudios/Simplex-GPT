"""
thought_wrap.py
===============

In-place Simplex Thought Field wrapping for a *real* transformers model
(e.g. ``Qwen3_5ForCausalLM``).

The wrap reuses each GatedDeltaNet layer's existing ``in_proj_a/b`` Linear
objects — **shared weights, no parameter duplication**. The only new
parameters are the small ``thought.*`` modules (one per wrapped layer).
:func:`unwrap` restores the original Linears exactly, so a thought-disabled
model is bit-exact with the vanilla checkpoint.

This is the single source of truth shared by ``real_model_steering.py`` and
``serve_real_endpoint.py``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from simplex_gated_delta_net import ThoughtBiasWrapper
from simplex_thought_field import (
    ThoughtModulator,
    clear_request_mix,
    clear_request_seed,
    request_mix,
    set_request_mix,
    set_request_seed,
)

__all__ = [
    "delta_net_layers",
    "wrap",
    "unwrap",
    "set_seed",
    "is_wrapped",
    "n_wrapped",
    "set_mix",
    "clear_mix",
    "set_request_mix",
    "clear_request_mix",
    "request_mix",
    "set_request_seed",
    "clear_request_seed",
    "request_seed",
]


def _decoder_layer_stack(lm) -> list:
    """Locate the text decoder ``layers`` list across model wrappers.

    Works for both a plain CausalLM (``lm.model.layers``) and a VL/MoE
    conditional-generation model where the text stack sits one level down
    (``lm.model.language_model.layers``). Returns the first ``layers`` list
    that actually contains GatedDeltaNet (``linear_attn``) layers.
    """
    roots = [
        lm,
        getattr(lm, "model", None),
        getattr(getattr(lm, "model", None), "language_model", None),
        getattr(lm, "language_model", None),
    ]
    for root in roots:
        if root is None:
            continue
        layers = getattr(root, "layers", None)
        if layers is None:
            continue
        try:
            if any(getattr(l, "linear_attn", None) is not None for l in layers):
                return list(layers)
        except TypeError:
            continue
    return []


def delta_net_layers(lm) -> List[Tuple[int, object]]:
    """``(layer_index, decoder_layer)`` for every GatedDeltaNet layer."""
    return [
        (i, l)
        for i, l in enumerate(_decoder_layer_stack(lm))
        if getattr(l, "linear_attn", None) is not None
    ]


def is_wrapped(lm) -> bool:
    for _, layer in delta_net_layers(lm):
        if hasattr(layer.linear_attn, "_orig_in_proj_b"):
            return True
    return False


def n_wrapped(lm) -> int:
    return sum(
        1
        for _, layer in delta_net_layers(lm)
        if hasattr(layer.linear_attn, "_orig_in_proj_b")
    )


def wrap(
    lm,
    seed: int,
    gain_b: float,
    gain_a: float,
    drift: Tuple[float, float, float] = (0.0, 0.0, 0.02),
) -> int:
    """Wrap every GatedDeltaNet layer in place. Returns the number wrapped."""
    count = 0
    for _, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        if hasattr(attn, "_orig_in_proj_b"):
            continue  # already wrapped
        thought = ThoughtModulator(
            hidden_size=attn.hidden_size,
            num_heads=attn.num_v_heads,
            drift=drift,
            gain_b=gain_b,
            gain_a=gain_a,
        )
        thought.set_seed(seed)
        w = attn.in_proj_qkv.weight
        thought = thought.to(dtype=w.dtype, device=w.device)
        attn.thought = thought
        attn._orig_in_proj_b = attn.in_proj_b
        attn._orig_in_proj_a = attn.in_proj_a
        attn.in_proj_b = ThoughtBiasWrapper(attn.in_proj_b, thought, "b")
        attn.in_proj_a = ThoughtBiasWrapper(attn.in_proj_a, thought, "a")
        count += 1
    return count


def unwrap(lm) -> None:
    """Restore the original ``in_proj_a/b`` Linears (bit-exact vanilla)."""
    for _, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        if hasattr(attn, "_orig_in_proj_b"):
            attn.in_proj_b = attn._orig_in_proj_b
            attn.in_proj_a = attn._orig_in_proj_a
            del attn._orig_in_proj_b, attn._orig_in_proj_a


def set_seed(lm, seed: int) -> None:
    """Re-seed every wrapped layer's thought field (per-request A/B)."""
    for _, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        if hasattr(attn, "thought"):
            attn.thought.set_seed(seed)


def set_mix(lm, offset, alpha: float) -> None:
    """Blend a second branch pattern (raw offset) into every wrapped layer.

    ``alpha = 0`` turns blending off; ``alpha = 1`` is fully the other
    branch's pattern. The primary (seed) pattern stays in effect, so this
    is a smooth α-dial between the two branches.
    """
    for _, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        if hasattr(attn, "thought"):
            attn.thought.set_mix(offset, alpha)


def clear_mix(lm) -> None:
    """Disable branch blending on every wrapped layer."""
    for _, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        if hasattr(attn, "thought"):
            attn.thought.clear_mix()
