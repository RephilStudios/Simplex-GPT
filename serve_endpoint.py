"""
serve_endpoint.py
=================

An OpenAI-compatible LLM endpoint serving the Tiny Thought LM — a complete
autoregressive model whose decoder layers are the real Qwen3.5 GatedDeltaNet
with the Simplex Thought Field injected.

Two modes
---------
* **Coding model** (recommended): ``--coding`` loads a trained checkpoint
  (``slm_weights.pt`` + ``tokenizer.json`` from :mod:`train_slm`) and a real
  char tokenizer, so completions are actual Python source.
* **Legacy random model**: no ``--coding`` — a randomly-initialized LM with a
  toy ``ord(c) % vocab`` tokenizer. Used by the behavioral test-suite
  (determinism / steering) where real text is not needed.

Endpoints
---------
GET  /health             liveness + config
GET  /v1/models          model list
POST /v1/completions     OpenAI-style completion (greedy by default)
GET  /thought/last       last recorded ThoughtTrace (JSON + fingerprint)

Run
---
    # coding model, thought field enabled (A/B steering via thought_seed)
    python serve_endpoint.py --port 8101 --coding --device cuda \\
        --thought-enabled --thought-seed 42 --gain-b 1.0 --gain-a 1.0

    # coding model, vanilla (thought off)
    python serve_endpoint.py --port 8102 --coding --device cuda

    # legacy random model (behavioral tests)
    python serve_endpoint.py --port 8103 --model-seed 1234

Example
-------
    curl -s http://127.0.0.1:8101/v1/completions -H 'Content-Type: application/json' \\
        -d '{"prompt": "def main():\\n    ", "max_tokens": 16}'
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_thought import LMConfig, TinyThoughtLM
from tokenizer import CharTokenizer

MODEL_ID = "simplex-thought-llm"


class CompletionRequest(BaseModel):
    prompt: str = ""
    prompt_ids: Optional[List[int]] = None
    max_tokens: int = 32
    temperature: float = 0.0
    top_k: Optional[int] = None
    rng_seed: Optional[int] = None
    thought_seed: Optional[int] = None


def build(args) -> "tuple":
    """Return ``(model, thought_cfg, tokenizer_or_None)``."""
    torch.manual_seed(args.model_seed)
    thought = None
    if args.thought_enabled:
        thought = {
            "seed": args.thought_seed,
            "gain_b": args.gain_b,
            "gain_a": args.gain_a,
            "drift": (0.0, 0.0, 0.02),
        }

    if args.coding:
        tok = CharTokenizer.load(args.tokenizer)
        cfg = LMConfig
        cfg.hidden_size = args.hidden
        cfg.num_layers = args.layers
        cfg.intermediate_size = args.hidden * 3
        model = TinyThoughtLM(cfg, thought, vocab_size=tok.vocab_size).to(args.device)
        state = torch.load(args.weights, map_location=args.device)
        if thought is not None:
            model.load_vanilla_state_dict(state)
        else:
            model.load_state_dict(state)
        return model.eval(), thought, tok

    # legacy: randomly-initialized LM, toy tokenizer
    model = TinyThoughtLM(LMConfig, thought).to(args.device).eval()
    return model, thought, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--model-seed", type=int, default=1234)
    ap.add_argument("--thought-enabled", action="store_true")
    ap.add_argument("--thought-seed", type=int, default=42)
    ap.add_argument("--gain-b", type=float, default=1.0)
    ap.add_argument("--gain-a", type=float, default=1.0)
    # coding model
    ap.add_argument("--coding", action="store_true")
    ap.add_argument("--weights", default="slm_weights.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    model, thought, tok = build(args)
    vocab = tok.vocab_size if tok else LMConfig.vocab_size
    meta = {
        "model": MODEL_ID,
        "mode": "coding" if tok else "random",
        "device": args.device,
        "vocab_size": vocab,
        "num_layers": args.layers if tok else LMConfig.num_layers,
        "thought_enabled": thought is not None,
        "thought_seed": args.thought_seed if thought else None,
    }

    app = FastAPI(title="Simplex Thought LLM")

    @app.get("/health")
    def health():
        return {"status": "ok", **meta}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    def resolve_ids(req: CompletionRequest) -> List[int]:
        if req.prompt_ids is not None:
            return list(req.prompt_ids)
        if tok is not None:
            return tok.encode(req.prompt) or [0]
        return [ord(c) % vocab for c in req.prompt] or [0]

    def render_text(ids: List[int]) -> str:
        if tok is not None:
            return tok.decode(ids)
        return " ".join(f"<{t}>" for t in ids)

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        active_seed = None
        if thought is not None:
            if req.thought_seed is not None:
                model.set_thought_seed(req.thought_seed)
            active_seed = (
                req.thought_seed if req.thought_seed is not None else thought["seed"]
            )

        ids = resolve_ids(req)
        input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                rng_seed=req.rng_seed,
            )
        tr = model.last_thought_trace()
        completion = out_ids[len(ids) :]
        return {
            "id": "cmpl-" + uuid.uuid4().hex[:12],
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "logprobs": None,
                    "prompt_ids": ids,
                    "completion_ids": completion,
                    "text": render_text(completion),
                }
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(completion),
                "total_tokens": len(ids) + len(completion),
            },
            "thought": {
                "enabled": thought is not None,
                "seed": active_seed,
                "fingerprint": tr.fingerprint if tr is not None else None,
            },
        }

    @app.get("/thought/last")
    def thought_last():
        tr = model.last_thought_trace()
        if tr is None:
            raise HTTPException(
                404, "no thought recorded (thought disabled or no completions yet)"
            )
        return json.loads(tr.to_json())

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
