"""
steer_ablation.py — is the thought field "important" or just "clever"?
=====================================================================

GAIN-SWEEP EDITION. All conditions perturb the SAME surface the field uses
(an additive bias on every GatedDeltaNet layer's ``in_proj_a`` / ``in_proj_b``):

  vanilla : no bias (deterministic greedy) — the reference answer
  field   : the real smooth simplex field at gain g (seeded)
  random  : per-position deterministic pseudorandom bias, RMS-matched to the field
  fixed   : a single constant per-head bias (repeng-style), RMS-matched

Prompts are a difficulty gradient: 8 near-deterministic anchors plus 16
high-entropy / multi-candidate / long-tail questions (where vanilla NLL is
high and greedy sits near a decision boundary — the only place a
perturbation of ~0.8-3.5 RMS bias can be expected to steer anything).

Per gain we report:
  flip : fraction of prompts whose answer differs from vanilla
  NLL  : the vanilla model's mean per-token NLL of the chosen answer
         damage = NLL - vanilla NLL
  eff  : flip / max(damage, eps) — behavior change per unit of coherence lost

Signatures:
  REAL KNOB  : flip grows with gain, and at matched gain the field flips
               >= random while losing <= NLL (cheaper, monotone steering)
  CLEVER ONLY: the field's flip/NLL curve tracks the random curve
  RETRACE    : same seed twice -> identical text + identical trace fingerprint

Run:  python steer_ablation.py --weights models/Qwen3.5-4B --gains 1 2 4 8 16
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import torch
import torch.nn as nn
import torch.nn.functional as F

from thought_wrap import delta_net_layers, set_seed, unwrap, wrap

# 8 anchors (low vanilla NLL) + 16 hard (ambiguous / multi-candidate / long tail)
PROMPTS = [
    "Pick a color. Answer with one word.",
    "Name a capital city in Europe. One word.",
    "What sound does a bee make? One word.",
    "Name a fruit. One word.",
    "Name a planet. One word.",
    "Pick a season. One word.",
    "Name an ocean. One word.",
    "Pick a weekday. One word.",
    # ---- hard ----
    "Name a planet with rings. One word.",
    "Choose a season that is not winter. One word.",
    "Name a metal used in ancient coinage. One word.",
    "What do you call the young of a goat? One word.",
    "Name a river in Africa. One word.",
    "Pick a color associated with caution. One word.",
    "Name a composer of the Baroque era. One word.",
    "Choose a five-letter bird. One word.",
    "Name a US state on the East Coast. One word.",
    "Pick a spice that is also a flower. One word.",
    "Name a Greek god of the sea. One word.",
    "Which fruit is associated with Eve? One word.",
    "Name a less common European capital. One word.",
    "Pick a two-syllable fruit ending in e. One word.",
    "Choose a four-letter dog breed. One word.",
    "Name a desert in Asia. One word.",
]


class _AddBias(nn.Module):
    def __init__(self, lin, bias_fn):
        super().__init__()
        self.lin = lin
        self.bias_fn = bias_fn

    def forward(self, h):
        b = self.bias_fn(h)
        return self.lin(h) + b if b is not None else self.lin(h)


def _wrap_bias(lm, factory):
    """Replace in_proj_a/b on every GatedDeltaNet layer with ``factory(i,slot,H)(h)``."""
    stored = []
    for i, layer in delta_net_layers(lm):
        attn = layer.linear_attn
        H = attn.num_v_heads
        for slot in ("a", "b"):
            name = f"in_proj_{slot}"
            lin = getattr(attn, name)
            setattr(attn, name, _AddBias(lin, factory(i, slot, H)))
            stored.append((attn, name, lin))
    return stored


def _restore(stored):
    for attn, name, lin in stored:
        setattr(attn, name, lin)


def random_factory(seed, rms):
    def mk(i, slot, H):
        base = seed * 100000 + i * 1000 + (0 if slot == "a" else 1)

        def bf(h):
            B, T, _ = h.shape
            g = torch.Generator(device=h.device)
            g.manual_seed(base)
            x = torch.randn((B, T, H), generator=g, device=h.device, dtype=h.dtype)
            return x * (rms / x.pow(2).mean().clamp_min(1e-9).sqrt())

        return bf

    return mk


def fixed_factory(rms):
    def mk(i, slot, H):
        sign = torch.where(torch.arange(H) % 2 == 0, 1.0, -1.0)

        def bf(h):
            B, T, _ = h.shape
            return (
                sign.view(1, 1, H).to(dtype=h.dtype, device=h.device).expand(B, T, H)
                * rms
            )

        return bf

    return mk


def field_rms(lm, gain):
    """RMS of the real field's per-element bias at this gain (on one layer)."""
    wrap(lm, 42, gain, gain)
    try:
        _, layer = delta_net_layers(lm)[0]
        attn = layer.linear_attn
        w = attn.in_proj_b.base.weight
        hs = torch.randn((1, 8, attn.hidden_size), dtype=w.dtype, device=w.device)
        b = attn.thought.slot_bias(hs, "b")
        return float(b.pow(2).mean().sqrt())
    finally:
        unwrap(lm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--gains", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.weights} ...")
    lm = (
        AutoModelForCausalLM.from_pretrained(
            args.weights, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        .to(dev)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.weights)

    def prompt_ids(p):
        return tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            enable_thinking=False,  # one-word answers, no thinking preamble
            return_dict=False,
            return_tensors=None,
        )

    def generate(p, max_new_tokens):
        ids = torch.tensor([prompt_ids(p)], dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = lm.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        new = out[0][ids.shape[1] :]
        return tok.decode(new, skip_special_tokens=True).strip()

    def nll_of(p, answer):
        P = prompt_ids(p)
        a = tok(answer, add_special_tokens=False)["input_ids"]
        if not a:
            return 0.0
        full = torch.tensor([P + a], dtype=torch.long, device=dev)
        with torch.inference_mode():
            logits = lm(input_ids=full).logits[0]
        s = 0.0
        for i, t in enumerate(a):
            s += float(-F.log_softmax(logits[len(P) - 1 + i], dim=-1)[t].item())
        return s / len(a)

    P = len(PROMPTS)
    print(f"\nprompts={P}  gains={args.gains}  seeds={args.seeds}\n")

    # ---- vanilla baseline -------------------------------------------------
    VAN = {}
    for p in PROMPTS:
        ans = generate(p, args.max_tokens)
        VAN[p] = {"ans": ans, "nll": nll_of(p, ans)}
    van_nll_mean = sum(r["nll"] for r in VAN.values()) / P
    print("vanilla (greedy):")
    for p, r in VAN.items():
        print(f"  {p!r:52} -> {r['ans']!r}  (NLL {r['nll']:.3f})")
    print(f"  vanilla mean answer-NLL = {van_nll_mean:.3f}\n")

    # ---- sweep ------------------------------------------------------------
    RES = {
        g: {"rms": None, "field": ([], []), "random": ([], []), "fixed": ([], [])}
        for g in args.gains
    }
    FLIPS_MAX = {}  # prompt -> {cond: [answers that flipped]} at the top gain

    for g in args.gains:
        rms = field_rms(lm, g)
        RES[g]["rms"] = rms
        row = RES[g]
        is_top = g == max(args.gains)
        if is_top:
            FLIPS_MAX = {p: {"field": [], "random": [], "fixed": []} for p in PROMPTS}

        # field (seeded; retrace on first prompt of first seed at top gain)
        retrace = None
        for s in args.seeds:
            wrap(lm, s, g, g)
            for p in PROMPTS:
                ans = generate(p, args.max_tokens)
                if is_top and s == args.seeds[0] and p == PROMPTS[0]:
                    tr1 = delta_net_layers(lm)[-1][1].linear_attn.thought.last_trace()
                    ans2 = generate(p, args.max_tokens)
                    tr2 = delta_net_layers(lm)[-1][1].linear_attn.thought.last_trace()
                    retrace = (ans == ans2) and (
                        tr1 is not None and tr1.fingerprint == tr2.fingerprint
                    )
                row["field"][0].append(ans != VAN[p]["ans"])
                row["field"][1].append(nll_of(p, ans))
                if is_top and ans != VAN[p]["ans"]:
                    FLIPS_MAX[p]["field"].append(f"seed{s}:{ans!r}")
            unwrap(lm)

        # random (RMS-matched, per-position noise)
        for s in args.seeds:
            st = _wrap_bias(lm, random_factory(s, rms))
            for p in PROMPTS:
                ans = generate(p, args.max_tokens)
                row["random"][0].append(ans != VAN[p]["ans"])
                row["random"][1].append(nll_of(p, ans))
                if is_top and ans != VAN[p]["ans"]:
                    FLIPS_MAX[p]["random"].append(f"seed{s}:{ans!r}")
            _restore(st)

        # fixed (RMS-matched constant per-head direction)
        st = _wrap_bias(lm, fixed_factory(rms))
        for p in PROMPTS:
            ans = generate(p, args.max_tokens)
            row["fixed"][0].append(ans != VAN[p]["ans"])
            row["fixed"][1].append(nll_of(p, ans))
            if is_top and ans != VAN[p]["ans"]:
                FLIPS_MAX[p]["fixed"].append(ans)
        _restore(st)
        print(f"  gain {g:<4} done  (field RMS {rms:.3f})")
    RETRACE_OK = bool(retrace)

    def agg(row, cond):
        flips, nlls = row[cond]
        n = len(flips)
        f = sum(flips) / n
        nll = sum(nlls) / n
        damage = max(nll - van_nll_mean, 1e-6)
        return f, nll, f / damage

    # ---- table ------------------------------------------------------------
    print("\n--- gain sweep ---")
    hdr = (
        f"{'gain':>5} | {'field flip':>10} {'NLL':>6} {'eff':>7} | "
        f"{'rand flip':>9} {'NLL':>6} {'eff':>7} | "
        f"{'fixed flip':>10} {'NLL':>6} {'eff':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for g in args.gains:
        rf, rfnll, rf_eff = agg(RES[g], "field")
        rn, r_nll, rn_eff = agg(RES[g], "random")
        fx, fx_nll, fx_eff = agg(RES[g], "fixed")
        print(
            f"{g:>5.0f} | {rf:>10.0%} {rfnll:>6.3f} {rf_eff:>7.3f} | "
            f"{rn:>9.0%} {r_nll:>6.3f} {rn_eff:>7.3f} | "
            f"{fx:>10.0%} {fx_nll:>6.3f} {fx_eff:>7.3f}"
        )
    print(
        f"retrace (top gain, seed {args.seeds[0]} twice -> identical text+trace): "
        f"{'PASS' if RETRACE_OK else 'FAIL'}"
    )

    # ---- flips at top gain (qualitative) ---------------------------------
    print(f"\n--- prompts that flipped at gain {max(args.gains)} ---")
    any_flip = False
    for p in PROMPTS:
        e = FLIPS_MAX[p]
        if e["field"] or e["random"] or e["fixed"]:
            any_flip = True
            print(f"  {p!r}  (vanilla {VAN[p]['ans']!r})")
            for c in ("field", "random", "fixed"):
                if e[c]:
                    print(f"    {c:6} -> {', '.join(map(str, e[c]))}")
    if not any_flip:
        print("  (none — the field and both baselines barely moved answers)")

    # ---- verdict ----------------------------------------------------------
    g_min, g_max = min(args.gains), max(args.gains)
    f_lo, _, _ = agg(RES[g_min], "field")
    f_hi = agg(RES[g_max], "field")[0]
    mono = f_hi > f_lo
    # honest summary numbers across all gains
    mean_adv_nll = sum(
        agg(RES[g], "random")[1] - agg(RES[g], "field")[1] for g in args.gains
    ) / len(args.gains)
    mean_adv_nll = sum(
        agg(RES[g], "random")[1] - agg(RES[g], "field")[1] for g in args.gains
    ) / len(args.gains)
    mean_flip_gap = sum(
        agg(RES[g], "field")[0] - agg(RES[g], "random")[0] for g in args.gains
    ) / len(args.gains)
    print("\n--- verdict ---")
    print(
        f"monotone gain response (flip {f_lo:.0%}@g{g_min:.0f} -> {f_hi:.0%}@g{g_max:.0f}): "
        f"{'yes' if mono else 'NO'}"
    )
    print(
        f"mean NLL advantage over random (+ = field loses less coherence): {mean_adv_nll:+.3f}"
    )
    print(f"mean flip gap vs random (+ = field steers more): {mean_flip_gap:+.0%}")
    if mono and mean_adv_nll > 0.05 and mean_flip_gap >= -0.05:
        print(
            "field is monotone AND at-least-as-coherent as matched noise -> "
            "behaves like a real control knob"
        )
    elif not mono and abs(mean_adv_nll) < 0.05:
        print("field does not respond to gain and matches noise -> inert decoration")
    elif mono and mean_adv_nll <= 0.05 and mean_flip_gap <= 0.05:
        print(
            "field responds to gain but with NO advantage over matched noise -> "
            "'clever' more than 'controllable'"
        )
    else:
        print("mixed signal — inspect the sweep table and flip list above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
