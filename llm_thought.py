"""
llm_thought.py
==============

A complete tiny causal LLM built around :class:`SimplexGatedDeltaNet`, for
testing the thought field at the *generation* level (i.e. as an LLM endpoint).

The decoder mimics the linear-attention layers of Qwen3.5::

    h = x + GatedDeltaNet(rms(x))
    h = h + MLP(rms(h))

plus an embedding and a linear LM head. It is tiny on purpose: the goal is to
exercise the real :class:`Qwen3_5GatedDeltaNet` forward (causal conv + gated
delta rule + conv/recurrent cache) with the thought field injected, inside a
real autoregressive generate loop — not to produce meaningful text (the weights
are random unless you load a checkpoint).

The generate loop uses the same cache protocol the layer implements
(``conv_states`` / ``recurrent_states`` per layer), so the incremental-decode
path of the delta rule is genuinely exercised, token by token.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from simplex_gated_delta_net import SimplexGatedDeltaNet


class LMConfig:
    """Plain config object (no HF dependency)."""

    vocab_size = 512
    hidden_size = 256
    num_layers = 3
    intermediate_size = 768

    # GatedDeltaNet hyper-parameters (same names the real layer reads)
    linear_num_value_heads = 8
    linear_num_key_heads = 2
    linear_key_head_dim = 16
    linear_value_head_dim = 16
    linear_conv_kernel_dim = 4
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    dtype = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))


class MLP(nn.Module):
    """SwiGLU-style MLP (gate * up -> down)."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeltaCache:
    """Minimal cache protocol expected by :class:`Qwen3_5GatedDeltaNet`."""

    def __init__(self, num_layers: int):
        self.conv_states: list = [None] * num_layers
        self.recurrent_states: list = [None] * num_layers

    @property
    def has_previous_state(self) -> bool:
        return any(s is not None for s in self.recurrent_states)


class TinyDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, thought: Optional[dict] = None):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if thought is not None:
            self.linear_attn = SimplexGatedDeltaNet(config, layer_idx, thought=thought)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self, hidden_states, cache_params=None, cache_position=None, attention_mask=None
    ):
        hidden_states = hidden_states + self.linear_attn(
            self.input_layernorm(hidden_states),
            cache_params,
            cache_position,
            attention_mask,
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class TinyThoughtLM(nn.Module):
    """A tiny autoregressive LM whose decoder layers carry the thought field."""

    def __init__(
        self,
        config=LMConfig,
        thought: Optional[dict] = None,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.thought_enabled = thought is not None
        self.vocab_size = (
            int(vocab_size) if vocab_size is not None else int(config.vocab_size)
        )
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyDecoderLayer(config, i, thought) for i in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids, cache_params=None, cache_position=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h, cache_params, cache_position)
        return self.lm_head(self.norm(h))

    # -- generation ---------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> List[int]:
        """Autoregressive generation. Returns prompt ids + generated ids.

        ``temperature=0`` (default) is greedy and fully deterministic. With
        ``temperature>0`` and ``rng_seed`` set, sampling is deterministic
        (seeded multinomial); without ``rng_seed`` it draws from the global RNG.
        """
        assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "batch size must be 1"
        cache = DeltaCache(len(self.layers))
        device = input_ids.device
        history = input_ids[0].tolist()
        cur = input_ids
        for step in range(max_new_tokens):
            if step == 0:
                pos = torch.arange(input_ids.shape[1], device=device)
            else:
                pos = torch.tensor([len(history) - 1], device=device)
            logits = self.forward(cur, cache, pos)[0, -1].float()
            tok = self._sample(logits, temperature, top_k, rng_seed, step)
            history.append(tok)
            cur = torch.tensor([[tok]], device=device)
        return history

    @staticmethod
    def _sample(logits, temperature, top_k, rng_seed, step):
        z = logits
        if top_k is not None and top_k > 0:
            keep = z.topk(min(top_k, z.numel())).values[-1]
            z = z.masked_fill(z < keep, float("-inf"))
        if temperature and temperature > 0:
            probs = F.softmax(z / temperature, dim=-1).cpu()
            if rng_seed is None:
                return int(torch.multinomial(probs, 1).item())
            gen = torch.Generator(device="cpu").manual_seed(int(rng_seed) + step)
            return int(torch.multinomial(probs, 1, generator=gen).item())
        return int(z.argmax().item())

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        rng_seed: Optional[int] = None,
        stop: Optional[str] = None,
    ) -> str:
        """Tokenize ``prompt``, generate, and return the generated text.

        ``stop`` (default ``"\\n\\n"``) ends generation at a program/statement
        boundary, which is what we want for snippet-style code output.
        """
        stop = "\\n\\n" if stop is None else stop
        ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )
        out = self.generate(ids, max_new_tokens, temperature, top_k, rng_seed)
        text = tokenizer.decode(out[len(prompt) :])
        if stop is not None and stop in text:
            text = text.split(stop, 1)[0]
        return text

    # -- thought-field API ---------------------------------------------------

    def set_thought_seed(self, seed: int) -> None:
        """Re-seed every layer's thought field at runtime (A/B steering)."""
        for layer in self.layers:
            attn = layer.linear_attn
            if hasattr(attn, "thought"):
                attn.thought.set_seed(seed)

    def last_thought_trace(self):
        """The :class:`ThoughtTrace` of the final layer's most recent forward."""
        attn = self.layers[-1].linear_attn
        if hasattr(attn, "thought"):
            return attn.thought.last_trace()
        return None

    # -- checkpoint loading ---------------------------------------------------

    def load_vanilla_state_dict(self, state_dict):
        """Load a vanilla (non-thought) checkpoint into this model.

        Remaps each layer's ``linear_attn.in_proj_{a,b}.*`` keys onto the
        wrapped ``...in_proj_{a,b}.base.*`` layout and loads with
        ``strict=False`` so the new ``thought.*`` parameters keep their init.
        """
        remapped = {}
        for k, v in state_dict.items():
            nk = k
            if ".linear_attn.in_proj_b." in k:
                nk = nk.replace(
                    ".linear_attn.in_proj_b.", ".linear_attn.in_proj_b.base."
                )
            elif ".linear_attn.in_proj_a." in k:
                nk = nk.replace(
                    ".linear_attn.in_proj_a.", ".linear_attn.in_proj_a.base."
                )
            remapped[nk] = v
        return self.load_state_dict(remapped, strict=False)
