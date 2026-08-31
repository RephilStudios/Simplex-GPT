# Simplex GPT — a Simplex-Noise Thought Field for the Gated DeltaNet

A novel way of threading **simplex noise sampled from latent space** through the
Qwen3.5 Gated DeltaNet layer, producing **coherent, retraceable, learnable
"thought patterns"** that modulate the delta rule's memory-write gain and decay.

The core idea: instead of per-token randomness, a single **seeded,
differentiable noise field** is sampled at coordinates that are a _smooth,
learnable function of the hidden state and time_. Because the field is a pure
function of `(hidden_states, t, seed, weights)` and contains **no RNG in the
forward pass**, every thought pattern is fully reproducible — and can be
recorded as a compact, verifiable, replayable _trace_.

---

## Why it's coherent and retraceable

| Property        | Mechanism                                                                                                                                                                                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Coherent**    | The field is simplex noise — a _continuous_ function of its coordinates (no Perlin-style grid kinks, no per-token jitter). Coordinates `p_t = M·h_t + drift·t` move smoothly through field-space, so the modulation is a smooth "thought wave" along the sequence. |
| **Retraceable** | The forward pass is deterministic: same `(seed, input, weights)` ⇒ bit-identical modulation. No `randn`/`rand` in `forward`.                                                                                                                                       |
| **Learnable**   | The latent→field map `M` and the field scale are ordinary differentiable parameters, so the pattern can be trained end-to-end.                                                                                                                                     |
| **Verifiable**  | Each forward pass records a `ThoughtTrace` (coordinates + per-slot values + a SHA-256 fingerprint). It round-trips through JSON and can be replayed exactly.                                                                                                       |

---

## File map

| File                                                      | Purpose                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modeling_qwen3_5.py`                                     | The base `Qwen3_5GatedDeltaNet` layer (copied from Hugging Face Transformers, self-contained — no `transformers` import needed).                                                                                                                                            |
| `simplex_thought_field.py`                                | `snoise2` / `snoise3` (vectorized, differentiable Gustavson simplex), `SimplexField` (seeded + optional fbm), `ThoughtModulator` (latent→field map, slot injection, traces), `ThoughtTrace` (record / JSON / fingerprint / replay).                                         |
| `simplex_gated_delta_net.py`                              | `SimplexGatedDeltaNet` — drop-in upgrade of the base layer that injects the field into `in_proj_b` (write gain) and `in_proj_a` (decay) via `ThoughtBiasWrapper`, **without duplicating the parent `forward`**. Includes `load_base_state_dict` for pretrained checkpoints. |
| `test_simplex_thought_field.py`                           | Standalone test suite (no pytest). 11 tests.                                                                                                                                                                                                                                |
| `demo_thought_patterns.py`                                | End-to-end demonstration with 8 checks + ASCII "ridge" of a thought wave. Writes `thought_trace.json`.                                                                                                                                                                      |
| `llm_thought.py`                                          | A complete tiny causal LM (embedding → 3 × GatedDeltaNet decoder layers → LM head) with an autoregressive `generate()` loop that exercises the real conv/recurrent cache — the vehicle for LLM-level testing.                                                               |
| `serve_endpoint.py`                                       | OpenAI-compatible endpoint (FastAPI) serving the Tiny Thought LM: `/v1/completions`, `/v1/models`, `/health`, `/thought/last`. Per-request `thought_seed` override.                                                                                                         |
| `test_endpoint.py`                                        | LLM-endpoint test suite: spawns thought-on and vanilla servers, runs 7 behavioral tests over HTTP, verifies trace fingerprints client-side.                                                                                                                                 |
| `tokenizer.py`                                            | Char-level tokenizer for Python source (vocab built from the corpus; encode/decode; JSON save/load).                                                                                                                                                                        |
| `python_corpus.py`                                        | Synthetic Python coding corpus generator (valid, consistent snippets, heavily hello-world-biased; deterministic).                                                                                                                                                           |
| `train_slm.py`                                            | Trains a tiny Python coding model (SLM) from scratch on the corpus; saves `slm_weights.pt` + `tokenizer.json`; prints hello-world generation tests.                                                                                                                         |
| `test_coding_endpoint.py`                                 | Coding-endpoint test suite: serves the trained model, shows it writing real Python and the thought field steering a genuine code choice (6 checks).                                                                                                                         |
| `slm_weights.pt` / `tokenizer.json` / `python_corpus.txt` | Artifacts produced by `train_slm.py`: trained weights, char vocabulary, and the training corpus.                                                                                                                                                                            |
| `_diag_noise.py`                                          | Reference-comparison utility: scalar float64 `snoise2`/`snoise3` vs. the vectorized version, plus smoothness stats.                                                                                                                                                         |

---

## How to run

Requires Python 3.12 + PyTorch (CPU is fine for these sizes).

```bash
# test suite (11/11)
python test_simplex_thought_field.py

# end-to-end demo (8 checks; writes thought_trace.json)
python demo_thought_patterns.py

# LLM endpoint tests (spawns 2 servers; 7 checks over HTTP)
python test_endpoint.py

# or run the server manually and poke it with curl
python serve_endpoint.py --port 8101 --model-seed 1234 \
    --thought-enabled --thought-seed 42 --gain-b 0.6 --gain-a 0.6
curl -s http://127.0.0.1:8101/v1/completions -H 'Content-Type: application/json' \
    -d '{"prompt": "hello world", "max_tokens": 16, "thought_seed": 42}'

# (optional) reference / smoothness diagnostics
python _diag_noise.py
```

> The base layer prints a one-line warning that the fast `fla`/`causal-conv1d`
> kernels are unavailable and it falls back to the pure-torch implementation.
> That is expected and harmless here.

---

## Training a tiny Python coding model (SLM)

The tiny LM in `llm_thought.py` is randomly initialized, so its completions are
valid tokens but meaningless. To make it a _real_ (if tiny) coding model, train
it from scratch on a synthetic Python corpus — fully offline, no checkpoint:

```bash
# 1) train (defaults: 40k snippets, 4000 steps, 3 layers x 256 hidden)
python train_slm.py --device cuda

#    -> slm_weights.pt, tokenizer.json, python_corpus.txt

# 2) serve it as an OpenAI-compatible coding endpoint
python serve_endpoint.py --port 8101 --coding --device cuda \
    --thought-enabled --thought-seed 42 --gain-b 1.0 --gain-a 1.0
curl -s http://127.0.0.1:8101/v1/completions -H 'Content-Type: application/json' \
    -d '{"prompt": "def main():\n    ", "max_tokens": 16}'
#    -> {"text": "print(\"hello world\")", ...}

# 3) test it (spawns vanilla + thought coding servers; 6 checks over HTTP)
python test_coding_endpoint.py
```

The result is a ~2.2M-parameter model that reliably writes simple, correct
Python — including `print("hello world")` — and that the Simplex Thought Field
can steer. `serve_endpoint.py --coding` loads the char tokenizer and the
checkpoint; the thought-wrapped variant loads the same vanilla weights via
`load_vanilla_state_dict` (remapping `in_proj_{a,b}.*` onto `.base.*`), so the
only new parameters are the thought field's.

### What the thought field does to a _trained_ model (an honest finding)

A well-trained model is confident, so the thought field's effect depends on
how sharp the model's distribution is at each step:

- **Confident / memorized** — e.g. `def main():` → the model almost certainly
  completes `print("hello world")`. Even a large gain does **not** override it.
  This is the _desired_ behavior: the field should not break correct, confident
  answers.
- **Genuine choice** — e.g. `x = ` (many valid values). Here the thought seed
  steers the model to a different valid value: seed 42 → `x = "green"`,
  seed 7 → `x = "bob"`. The field acts as a **seedable choice knob** on the
  model's true uncertainties — a coherent, reproducible, steerable "creativity"
  dimension on a real model.

In short: the thought field is a smooth, seedable modulation of the model's
decisions, strongest where the model has real uncertainty, and harmless where
the model is confidently correct.

---

## The design, concretely

### 1. The noise field (`SimplexField`)

`snoise2` / `snoise3` are vectorized, differentiable implementations of
Gustavson's simplex noise. `SimplexField` wraps them with:

- a **seed** that derives a deterministic offset vector (`seed_offset`), so
  different seeds traverse different — but structurally identical — regions of
  the same noise. This is the _reproducibility knob_; it is a buffer, not an RNG.
- optional **fbm** (`octaves`, `gain`) and a global **frequency** scale.

### 2. Latent → field coordinates (`ThoughtModulator`)

Per token `t`, the field-space coordinate is

```
p_t = M · h_t + drift · t        # M: hidden_size -> dim (learnable), dim ∈ {2, 3}
```

`drift` advances the sample point along the sequence so the field _evolves_
into a wave rather than a static bump. Each modulated slot ("`b`" and "`a`",
see below) and each head gets a fixed **phase offset**, so all heads share one
field but read it at slightly different points.

### 3. Injection into the delta rule (`SimplexGatedDeltaNet`)

The base layer computes the memory dynamics as

```
beta = in_proj_b(h).sigmoid()                  # write gain
g    = -exp(A) · softplus(in_proj_a(h) + dt)   # log decay
```

We wrap `in_proj_b` / `in_proj_a` with `ThoughtBiasWrapper`, which _adds_ the
field's per-head bias:

```
in_proj_b(h) += gain_b · S(p_t + offset_b)
in_proj_a(h) += gain_a · S(p_t + offset_a)
```

No re-implementation of the parent `forward` — the parent's forward calls the
wrapped projections transparently. With `thought={"enabled": False}` the
projections are _not_ wrapped and the layer is a **bit-exact vanilla
passthrough**.

### 4. Traces, replay, verification (`ThoughtTrace`)

After a forward pass, `modulator.last_trace()` returns the exact thought
pattern: the coordinates `p_t`, the per-slot field values, and a SHA-256
fingerprint. A trace **plus the model weights** fully determines the
modulation, so:

```python
tr   = layer.thought.last_trace()
json = tr.to_json()                                  # compact, portable
tr2  = ThoughtTrace.from_json(json)
tr2.verify_fingerprint()                             # -> True
# replay with no input hidden state at all:
modulator.replay_slot(tr2.coords, "b") == tr2.slot_values["b"]   # -> True
```

---

## Usage with a real Hugging Face config

```python
from transformers import AutoConfig
from simplex_gated_delta_net import SimplexGatedDeltaNet

config = AutoConfig.from_pretrained("your/qwen3.5-model")  # any config with the
# ... linear-attention fields (hidden_size, linear_num_value_heads, ...)

layer = SimplexGatedDeltaNet(
    config,
    layer_idx=0,
    thought={
        "seed": 42,            # reproducibility knob
        "dim": 3,              # 2 or 3
        "freq": 0.75,          # noise scale
        "drift": (0.0, 0.0, 0.02),  # per-token drift (the "thought wave")
        "gain_b": 0.75,        # amplitude on the write-gain slot
        "gain_a": 0.75,        # amplitude on the decay slot
        "octaves": 1,
    },
)

# Load a checkpoint trained on the vanilla layer (remaps in_proj_a/b.* ->
# in_proj_a/b.base.*); the new thought.* weights keep their initialization.
layer.load_base_state_dict(vanilla_layer.state_dict())

out = layer(hidden_states)                 # forward as before
trace = layer.thought.last_trace()         # record the thought pattern
```

`load_base_state_dict` is the bridge for pretrained weights: it remaps the
vanilla `in_proj_a.*` / `in_proj_b.*` keys onto the wrapped `...base.*` layout
and loads with `strict=False`, leaving `thought.*` at their init values.

---

## A note on the simplex constants (and a subtle bug this project caught)

For a simplex cell to tile space correctly — so that the six tetrahedra (3D) /
two triangles (2D) share faces cleanly and **non-shared corners never leak into
the falloff** — the skew and unskew factors must satisfy the consistency
relation

```
G = F / (1 + d·F)        (d = dimension)
```

- **2D** (this repo): `F2 = 0.5(√3−1) ≈ 0.366`, `G2 = (3−√3)/6 ≈ 0.2113`.
  Check: `0.366 / (1 + 2·0.366) = 0.2113` ✓ — consistent, fully continuous.
- **3D** (this repo): `F3 = 1/3`, `G3 = 1/6` (the canonical Gustavson pair).
  Check: `(1/3) / (1 + 3·(1/3)) = 1/6` ✓ — consistent, fully continuous.

A common transcription sets `F3 = G3 = 1/6`. That pair is **inconsistent**
(`(1/6)/(1+3·(1/6)) = 1/9 ≠ 1/6`), so the tetrahedra no longer tile the cell
and non-shared corners bleed into the falloff on selection boundaries —
producing isolated O(0.1–0.2) value jumps in an otherwise smooth field. We
verified this numerically: with `F3 = 1/6` the max 1e-3 perturbation delta was
~0.21; with the correct `F3 = 1/3` it drops to ~0.008, on par with 2D.

Two related choices worth knowing:

- **2D triangle selection** uses Gustavson's _per-point_ rule (`x0 > y0`),
  which is continuous everywhere. The Ashima/IQ _cell-index_ variant
  (`i.x >= i.y`) has genuine value jumps across some cell edges — avoid it if
  continuity matters.
- The **IQ 3D "j-trick"** gradient selection is a transcription minefield and
  is not used here; we use the plain 12-gradient table per corner.

---

## Testing it as an LLM endpoint

`llm_thought.py` wraps the real `Qwen3_5GatedDeltaNet` into a complete tiny
casual LM (embedding → 3 decoder layers → LM head) with an autoregressive
`generate()` loop that drives the layer's genuine conv/recurrent cache, token
by token. `serve_endpoint.py` serves that LM behind an OpenAI-compatible
endpoint, and `test_endpoint.py` exercises it over HTTP — all offline, no
checkpoint needed:

```
GET  /health              liveness + thought config
GET  /v1/models           model list
POST /v1/completions      OpenAI-style completion (greedy by default)
GET  /thought/last        last recorded ThoughtTrace (JSON + fingerprint)
```

The request body accepts `prompt` (or explicit `prompt_ids`), `max_tokens`,
`temperature`, `top_k`, `rng_seed`, and a per-request `thought_seed` override
for A/B steering. Each response also returns the recorded thought
fingerprint, and `/thought/last` returns the full trace.

`test_endpoint.py` spawns two servers — one with the thought field enabled,
one vanilla (same model seed) — and checks, over HTTP:

| #   | Test                                                               | What it proves                                |
| --- | ------------------------------------------------------------------ | --------------------------------------------- |
| 1   | `/health` on both                                                  | correct thought config is served              |
| 2   | same prompt + same seed → identical tokens                         | **retraceable generation**                    |
| 3   | `thought_seed` 42 vs 7 → different tokens                          | **seed steers the thought pattern**           |
| 4   | vanilla server deterministic                                       | baseline sanity                               |
| 5   | thought-on ≠ vanilla (same weights)                                | **the field actually changes generation**     |
| 6   | `/thought/last` fingerprint verifies **client-side** (pure stdlib) | the recorded trace is real and tamper-evident |
| 7   | seeded sampling deterministic                                      | sampling path is reproducible too             |

Tokenization is a toy (`ord(c) % vocab`) — there is no tokenizer offline. The
point is the behavioral contract (determinism, seed steering, trace
recording) at the generation level, not meaningful text. With a real
checkpoint the same `SimplexGatedDeltaNet` swap + `generate`/serve stack
applies directly (see below).

---

## Using it with a real Qwen3.5 checkpoint

The local environment currently has `transformers==4.44` (pre-Qwen3.5) and no
cached checkpoint, so the endpoint demo uses the tiny LM above. To run it on
the real model once network + a newer `transformers` are available, the
pattern is a post-load surgery (no retraining, no architecture changes):

```python
from transformers import AutoConfig, AutoModelForCausalLM
from modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from simplex_gated_delta_net import SimplexGatedDeltaNet

config = AutoConfig.from_pretrained("your/qwen3.5")
model = AutoModelForCausalLM.from_pretrained("your/qwen3.5")

# find every GatedDeltaNet in the decoder and swap it for the thought version
for name, mod in list(model.named_modules()):
    if type(mod) is Qwen3_5GatedDeltaNet:
        parent, attr = _parent_of(model, name)
        new = SimplexGatedDeltaNet(config, mod.layer_idx,
                                   thought={"seed": 42, "gain_b": 0.25, "gain_a": 0.25})
        new.load_base_state_dict(mod.state_dict())   # remaps in_proj_a/b.* -> .base.*
        setattr(parent, attr, new)
```

`load_base_state_dict` bridges the pretrained weights onto the wrapped layout;
the `thought.*` parameters keep their (small) initialization. Because the
forward pass adds a bounded, smooth bias to the two delta-rule dynamics, a
modest gain is all that's needed to make a measurable difference without
distorting the model — and `thought={"enabled": False}` is still a bit-exact
vanilla fallback.

---

## Scope & limitations

- This is a **single-layer** upgrade of the Gated DeltaNet. Wiring it through
  a full model (all linear-attention layers, shared seed, etc.) is a
  straightforward extension.
- The modulation is an _additive bias_ on the two dynamics of the delta rule.
  That is intentionally minimal and safe (a strict vanilla passthrough is one
  flag away); richer couplings (e.g. gating the decay rate multiplicatively)
  are possible but change the dynamics more.
- The base layer uses the pure-torch recurrence; on GPU with `fla`/
  `causal-conv1d` installed it will use the fast kernels automatically.
