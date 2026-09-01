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

| File                                                                         | Purpose                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modeling_qwen3_5.py`                                                        | The base `Qwen3_5GatedDeltaNet` layer (copied from Hugging Face Transformers, self-contained — no `transformers` import needed).                                                                                                                                                                                                                                                                    |
| `simplex_thought_field.py`                                                   | `snoise2` / `snoise3` (vectorized, differentiable Gustavson simplex), `SimplexField` (seeded + optional fbm, per-evaluation `seed=` override), `ThoughtModulator` (latent→field map, slot injection, traces, **stateless per-thread seed**), `ThoughtTrace` (record / JSON / fingerprint / replay), plus the per-request seed context (`set_request_seed` / `clear_request_seed` / `request_seed`). |
| `simplex_gated_delta_net.py`                                                 | `SimplexGatedDeltaNet` — drop-in upgrade of the base layer that injects the field into `in_proj_b` (write gain) and `in_proj_a` (decay) via `ThoughtBiasWrapper`, **without duplicating the parent `forward`**. Includes `load_base_state_dict` for pretrained checkpoints.                                                                                                                         |
| `test_simplex_thought_field.py`                                              | Standalone test suite (no pytest). 14 tests, incl. per-thread seed isolation under parallel threads and stateless interleaving.                                                                                                                                                                                                                                                                     |
| `demo_thought_patterns.py`                                                   | End-to-end demonstration with 8 checks + ASCII "ridge" of a thought wave. Writes `thought_trace.json`.                                                                                                                                                                                                                                                                                              |
| `llm_thought.py`                                                             | A complete tiny causal LM (embedding → 3 × GatedDeltaNet decoder layers → LM head) with an autoregressive `generate()` loop that exercises the real conv/recurrent cache — the vehicle for LLM-level testing.                                                                                                                                                                                       |
| `serve_endpoint.py`                                                          | OpenAI-compatible endpoint (FastAPI) serving the Tiny Thought LM: `/v1/completions`, `/v1/models`, `/health`, `/thought/last`. Per-request `thought_seed` override.                                                                                                                                                                                                                                 |
| `test_endpoint.py`                                                           | LLM-endpoint test suite: spawns thought-on and vanilla servers, runs 7 behavioral tests over HTTP, verifies trace fingerprints client-side.                                                                                                                                                                                                                                                         |
| `tokenizer.py`                                                               | Char-level tokenizer for Python source (vocab built from the corpus; encode/decode; JSON save/load).                                                                                                                                                                                                                                                                                                |
| `python_corpus.py`                                                           | Synthetic Python coding corpus generator (valid, consistent snippets, heavily hello-world-biased; deterministic).                                                                                                                                                                                                                                                                                   |
| `train_slm.py`                                                               | Trains a tiny SLM from scratch; `--corpus python` (coding model, saves `slm_weights.pt` + `tokenizer.json`, runs hello-world tests) or `--corpus assistant` (prose model, saves `assistant_weights.pt` + `assistant_tokenizer.json`, runs prose tests).                                                                                                                                             |
| `test_coding_endpoint.py`                                                    | Coding-endpoint test suite: serves the trained model, shows it writing real Python and the thought field steering a genuine code choice (6 checks).                                                                                                                                                                                                                                                 |
| `slm_weights.pt` / `tokenizer.json` / `python_corpus.txt`                    | Artifacts produced by `train_slm.py --corpus python`: trained weights, char vocabulary, and the training corpus.                                                                                                                                                                                                                                                                                    |
| `assistant_corpus.py`                                                        | The assistant's own prose, 41 passages (~16k chars), deterministic — the training corpus for the assistant-style SLM.                                                                                                                                                                                                                                                                               |
| `steer_assistant.py`                                                         | A/B steering demo for the assistant SLM: vanilla vs. thought seeds (42 / 7) on canonical prompts + retraceability check.                                                                                                                                                                                                                                                                            |
| `probe_steering.py`                                                          | Finds prompts with _genuine choice_: scans corpus prefixes for greedy flips at a given gain, and compares temperature-1.0 samples (vanilla vs. steered).                                                                                                                                                                                                                                            |
| `assistant_weights.pt` / `assistant_tokenizer.json` / `assistant_corpus.txt` | Artifacts produced by `train_slm.py --corpus assistant` (6.1M params, 4 × 384).                                                                                                                                                                                                                                                                                                                     |
| `test_real_integration.py`                                                   | Verifies `SimplexGatedDeltaNet` against the _released_ transformers `Qwen3_5GatedDeltaNet`: state-dict keys, forward agreement, `load_base_state_dict` onto real weights, retraceability. `--section a                                                                                                                                                                                              | b`(run`b` in its own process on tight VRAM). |
| `real_model_steering.py`                                                     | A/B steering demo on the real `Qwen3.5-4B`: wraps all 24 GatedDeltaNet layers in place, compares vanilla vs. per-seed completions, verifies bit-exact unwrap.                                                                                                                                                                                                                                       |
| `thought_wrap.py`                                                            | Shared in-place Thought Field wrap/unwrap/set_seed for a real transformers model (shared base weights, no duplication).                                                                                                                                                                                                                                                                             |
| `serve_real_endpoint.py`                                                     | OpenAI-compatible **chat** endpoint for the real model (`/`, `/v1/chat/completions` with `stream` SSE, `/v1/completions`, `/health`, `/thought`); optional `--thought-enabled` with per-request `thought_seed` steering; **concurrent requests** (seed applied thread-locally, no global lock).                                                                                                     |
| `chat_demo.py`                                                               | Demo client: multi-turn conversation + A/B steering against a running endpoint (or `--spawn` to boot the server itself).                                                                                                                                                                                                                                                                            |
| `openai_client_demo.py`                                                      | Official `openai` SDK drop-in demo: `models.list()`, chat, streaming, A/B steering via `extra_body={"thought_seed": N}`.                                                                                                                                                                                                                                                                            |
| `chat_ui.html`                                                               | Tiny web chat page served at `GET /`: live SSE streaming, A/B compare (both panes stream in parallel) with steered/identical verdict.                                                                                                                                                                                                                                                               |
| `test_real_endpoint.py`                                                      | Spawns the real endpoint (thought on) and checks over HTTP: health, models, chat, retrace (same seed identical), seed routing, state.                                                                                                                                                                                                                                                               |
| `test_concurrency.py`                                                        | Spawns the real endpoint and proves concurrency: parallel distinct seeds stay isolated, parallel same-seed is retraceable, parallel streams agree with non-stream, and parallel wall time genuinely overlaps (7 checks).                                                                                                                                                                            |
| `_diag_noise.py`                                                             | Reference-comparison utility: scalar float64 `snoise2`/`snoise3` vs. the vectorized version, plus smoothness stats.                                                                                                                                                                                                                                                                                 |

---

## How to run

Requires Python 3.12 + PyTorch (CPU is fine for these sizes).

```bash
# test suite (14/14)
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

## Training the assistant-prose SLM (natural-language test)

The same pipeline can train a tiny model **on the assistant's own prose**
(`assistant_corpus.py`, 41 passages, ~16k chars) — a higher-entropy,
natural-language test of the architecture:

```bash
# train (6.1M params, 4 layers x 384 hidden)
python train_slm.py --corpus assistant --steps 6000 --hidden 384 --layers 4 --device cuda
#    -> assistant_weights.pt, assistant_tokenizer.json, assistant_corpus.txt
#    best loss 0.2077 nats/token (perplexity ~1.23)

# A/B steering demo (vanilla vs. thought seeds 42 / 7 + retrace check)
python steer_assistant.py --device cuda

# find where the model has genuine choice (greedy flips + temperature-1.0 samples)
python probe_steering.py --gain 2.0 --device cuda
```

### What the steering demo shows (gain 2.0)

- **Memorized continuations resist the field.** The canonical prompts
  complete to exact corpus sentences (e.g. `The core idea is that ` →
  `instead of per-token randomness, a single seeded`) — greedy output is
  identical for seed 42, seed 7, and vanilla. Retraceability (same seed twice
  → bit-identical output) holds everywhere.
- **Genuine choices steer.** Scanning 300 corpus prefixes, the thought field
  flips the greedy completion on **42/300**, changing specific word choices:
  `a small project` → `a small corpus`, `layer updates` → `layer, and fin…`,
  and even fixing a memorized typo: `whes l` → `whose`.
- **Sampling makes it more visible.** At temperature 1.0 the field shifts the
  distribution enough that 2/4 canonical prompts sample differently per seed.

Same honest conclusion as the coding model: the field is a seedable,
retraceable choice knob — strongest where the model has real uncertainty,
harmless where the model is confidently correct.

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

This is now **implemented and verified** against `Qwen/Qwen3.5-4B` (a 4B
hybrid: 24 GatedDeltaNet layers + 8 full-attention layers, bf16).

### Environment (Windows / RTX 3060 12 GB)

`transformers==5.16` (has `models.qwen3_5`), `accelerate==1.14`, and
`torchvision==0.20.1` + `torchaudio==2.5.1` matched to `torch 2.5.1`. Two
Windows quirks hit and worked around:

- `device_map="cuda"` **segfaults** (accelerate dispatch on this stack) — load
  on CPU then `.to("cuda")` instead.
- the CUDA allocator intermittently reports "N free, can't place M" (fragmentation);
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` (set in the scripts) fixes it.

### Integration test (`test_real_integration.py`)

| Check                                                 | Result                                                |
| ----------------------------------------------------- | ----------------------------------------------------- |
| state-dict keys, released vs wrapped                  | identical (9/9)                                       |
| forward, released vs wrapped (random init)            | max diff **0.0** (bit-exact)                          |
| `load_base_state_dict` onto real weights              | 2 remapped `.base.` + 15 `thought.*`, values verified |
| wrapped-off layer vs released layer, **real weights** | max diff **0.0** (bit-exact)                          |
| retraceability on GPU                                 | same seed identical, different seed differs           |

`SimplexGatedDeltaNet` subclasses the **released** `Qwen3_5GatedDeltaNet`
(local `modeling_qwen3_5.py` is the fallback when `transformers` is absent), so
a thought-disabled layer is bit-exact with the official model by construction.

### Steering the real model (`real_model_steering.py`)

Wraps all 24 GatedDeltaNet layers **in place** (shared base weights — only the
small `thought.*` modules are added), runs vanilla + per-seed greedy
completions, then unwraps and verifies the baseline is restored bit-exactly.

| Gain | prompts steered (4 choice prompts)                                    |
| ---- | --------------------------------------------------------------------- |
| 1.0  | 0/4                                                                   |
| 2.0  | 0/4                                                                   |
| 4.0  | 1/4 — `Beagle` → `Poodle` (seed 7)                                    |
| 8.0  | 2/4 — `Beagle` → `Poodle`/`Labrad` (seed-dependent); artifacts appear |

On a confident 4B model the field needs more gain than on the tiny SLM, and it
still lands **only** on the prompt with genuine choice (dog breed), while the
near-deterministic color/city/sound prompts resist even gain 8 — the same
"inversely proportional to confidence" behavior, confirmed on a real
checkpoint. Retraceability and clean unwrap **PASS** at every gain.

Run:

```bash
python test_real_integration.py --weights models/Qwen3.5-4B   # or --section a|b
python real_model_steering.py --gain 4.0 --seeds 42 7
```

### Chat endpoint (`serve_real_endpoint.py`)

OpenAI-compatible chat over the real model, with optional thought-field
steering (per-request `thought_seed`) and **SSE streaming** ("stream": true).

**Concurrency:** requests are not serialized. Each request's seed is applied
in a thread-local context (`set_request_seed`), and the thought modulator is
stateless per evaluation, so overlapping chats each get their own seed —
including streamed ones, whose `generate()` runs on its own worker thread.
Retraceability holds under concurrency: two simultaneous same-seed requests
return identical text (verified by `test_concurrency.py`).

```bash
# thought field on (A/B steering available per request)
python serve_real_endpoint.py --port 8100 --weights models/Qwen3.5-4B \
    --thought-enabled --gain 4.0

# chat (OpenAI format)
curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Pick a color. One word."}],
         "max_tokens": 16, "thought_seed": 42}'

# same message, different seed -> different retraceable choice
... -d '{..., "thought_seed": 7}'

# demo client (multi-turn + A/B; --spawn boots the server itself)
python chat_demo.py --port 8100
```

### Web UI

The endpoint also serves a tiny self-contained chat page at **`/`** (same
origin, no CORS). Start the server, then open `http://127.0.0.1:8100/` in a
browser. It has a normal **Send** (plain reply, **streams tokens live** via
SSE) and an **A/B** button that answers the same message under two
configurable thought seeds — **both panes stream in parallel** — flagging
when the seed steered the choice. Verified: `GET /` serves the page; A/B
through the UI path returns seed-dependent replies (seed 42 `Beagle` vs
seed 7 `Poodle`).

Verified: multi-turn context carries over; same seed + same messages =>
bit-identical reply; `seed 42 -> Poodle`, `seed 7 -> Labrad` on the
genuine-choice prompt.

### Concurrency (`test_concurrency.py`)

Requests are **not** serialized: each one's `thought_seed` is applied in a
thread-local context and the thought modulator keeps no cross-request state,
so overlapping chats (streamed or not) are isolated per thread. The suite
proves it over HTTP against the real endpoint:

| Check                                                                                | Result                        |
| ------------------------------------------------------------------------------------ | ----------------------------- |
| parallel chats with seeds 42 / 7 / 123 — all non-empty                               | PASS                          |
| parallel chats: each request's seed echoed back                                      | PASS (no cross-contamination) |
| parallel chats with the _same_ seed → identical text                                 | PASS (retraceable in flight)  |
| parallel _streamed_ chats, same seed → identical, and equal to the non-stream result | PASS                          |
| parallel wall time < sequential sum (ratio 0.97)                                     | PASS (genuinely overlapping)  |

```bash
python test_concurrency.py --weights models/Qwen3.5-4B --gain 4.0
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
