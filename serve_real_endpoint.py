"""
serve_real_endpoint.py
======================

An OpenAI-compatible **chat** endpoint serving the real **Qwen3.5-4B**
checkpoint, with the Simplex Thought Field optionally wrapped in place across
all 24 GatedDeltaNet layers.

Endpoints
---------
GET  /health                 liveness + thought config
GET  /v1/models              model list
POST /v1/chat/completions    OpenAI chat-completions (multi-turn)
POST /v1/completions         raw-prompt completion
GET  /thought                current thought-field state (seed, gain, wrapped)

Thought field
-------------
* ``--thought-enabled`` wraps all GatedDeltaNet layers **once at startup**
  (shared base weights — no parameter duplication).
* per-request ``thought_seed`` overrides the seed for that request
  (A/B steering / retrace: same seed + same messages => identical reply).
* **concurrent requests are safe**: each request's seed is applied
  thread-locally, so overlapping A/B chats don't cross-contaminate and
  nothing is serialized behind a global lock.
* without ``--thought-enabled`` the model is vanilla (bit-exact checkpoint).

Run
---
    # vanilla chat
    python serve_real_endpoint.py --port 8100

    # thought field on (A/B steering via per-request thought_seed)
    python serve_real_endpoint.py --port 8100 --thought-enabled --gain 2.0

Example
-------
    curl -s http://127.0.0.1:8100/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{
               "messages": [{"role": "user", "content": "Pick a color. One word."}],
               "max_tokens": 16,
               "thought_seed": 42
           }'

    # same request, different seed -> different (retraceable) choice
    ... "thought_seed": 7
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from typing import List, Optional

# keep large CUDA blocks unsplit (Windows allocator fragmentation workaround)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from thought_recorder import (
    TrajectoryBuffer,
    clear_active_buffer,
    set_active_buffer,
    target_dim,
)
from thought_recorder import (
    install as install_recorder,
)
from thought_recorder import (
    is_installed as is_recorder_installed,
)
from thought_wrap import (
    clear_request_seed,
    delta_net_layers,
    is_wrapped,
    n_wrapped,
    set_request_seed,
    wrap,
)

MODEL_ID = "qwen3.5-4b-simplex"

# NOTE: there is deliberately **no global generation lock** here.  Each
# request carries its thought seed in a thread-local context
# (``set_request_seed``), and the thought modulator keeps no cross-request
# state, so concurrent requests (including streamed ones, whose generate()
# runs on their own worker thread) are isolated per thread.


# --------------------------------------------------------------------------- #
# request / response models
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False
    thought_seed: Optional[int] = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    thought_seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_real(weights: str, device: str):
    """Load the checkpoint, auto-detecting the model family.

    Prefer the **text CausalLM** class so we don't load a vision tower we
    don't need — this is what kept the 4B inside a 12 GB card. Fall back to
    the multimodal ``…ForConditionalGeneration`` class only when the CausalLM
    class can't load that checkpoint (e.g. the VL-MoE Qwen3.5-35B-A3B, which
    has no standalone CausalLM entry).

    Always returns ``(model, tokenizer)`` where ``tokenizer`` supports
    ``apply_chat_template``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:  # older transformers
        AutoModelForImageTextToText = AutoProcessor = None

    kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    errors = []

    # 1) text CausalLM — the 4B path (and the 35B-A3B text stack if that
    #    model_type is registered here). No vision tower.
    try:
        lm = AutoModelForCausalLM.from_pretrained(weights, **kw)
        return lm.to(device).eval(), AutoTokenizer.from_pretrained(weights)
    except Exception as e:  # noqa: BLE001 - try the next class
        errors.append(("CausalLM", repr(e)))

    # 2) multimodal VL-MoE (Qwen3.5-35B-A3B is Qwen3_5MoeForConditionalGeneration)
    if AutoModelForImageTextToText is not None:
        try:
            lm = AutoModelForImageTextToText.from_pretrained(weights, **kw)
            return lm.to(device).eval(), AutoProcessor.from_pretrained(
                weights
            ).tokenizer
        except Exception as e:  # noqa: BLE001
            errors.append(("image-text-to-text", repr(e)))

    # 3) last resort: the original 4B class, exactly as before
    try:
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

        lm = Qwen3_5ForCausalLM.from_pretrained(weights, **kw)
        return lm.to(device).eval(), AutoTokenizer.from_pretrained(weights)
    except Exception as e:  # noqa: BLE001
        errors.append(("Qwen3_5ForCausalLM", repr(e)))

    raise RuntimeError(f"could not load {weights}: {errors}")


def safe_pad_id(tok):
    """A usable pad id (VL tokenizers may not define one)."""
    pid = getattr(tok, "pad_token_id", None)
    if pid is None:
        pid = getattr(tok, "eos_token_id", None)
    return pid


def strip_thinking(text: str) -> str:
    """Drop a leading thinking block (if present) so chat shows the answer."""
    close_tag = "<" + "/think" + ">"
    if close_tag in text:
        text = text.split(close_tag, 1)[1]
    return text.lstrip()


def make_chat_fn(lm, tok, enable_thinking: bool):
    def _chat(messages: List[ChatMessage]) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {
            "add_generation_prompt": True,
            "return_dict": False,
            "return_tensors": None,
        }
        try:
            out = tok.apply_chat_template(
                payload, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:  # template without enable_thinking
            out = tok.apply_chat_template(payload, **kwargs)
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
        else:
            ids = out
        if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return ids  # caller generates

    return _chat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--model-id", default="qwen3.5-4b-simplex")
    ap.add_argument("--max-tokens-default", type=int, default=256)
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="keep the  block in replies (default: off)",
    )
    # thought field
    ap.add_argument("--thought-enabled", action="store_true")
    ap.add_argument("--thought-seed", type=int, default=42)
    ap.add_argument("--gain", type=float, default=1.0)
    args = ap.parse_args()
    global MODEL_ID
    MODEL_ID = args.model_id

    t0 = time.time()
    lm, tok = load_real(args.weights, args.device)
    n_layers = len(delta_net_layers(lm))
    if args.thought_enabled:
        wrap(lm, args.thought_seed, args.gain, args.gain)
        install_recorder(lm)
    print(
        f"loaded {args.weights} ({time.time() - t0:.0f}s) | {n_layers} GatedDeltaNet layers | "
        f"thought={'ON (gain %.1f, seed %d)' % (args.gain, args.thought_seed) if args.thought_enabled else 'OFF (vanilla)'} | "
        f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
    )

    app = FastAPI(title="Qwen3.5-4B Simplex Chat")

    # allow the chat page to be opened straight from disk (file:// origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    CHAT_UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_ui.html")

    @app.get("/", response_class=HTMLResponse)
    def ui():
        """Tiny web chat UI (served same-origin — no CORS)."""
        try:
            with open(CHAT_UI, encoding="utf-8") as f:
                return f.read()
        except OSError:
            raise HTTPException(
                404, "chat_ui.html not found next to serve_real_endpoint.py"
            )

    def thought_meta():
        on = args.thought_enabled and is_wrapped(lm)
        return {
            "enabled": on,
            "default_seed": args.thought_seed,
            "gain": args.gain if args.thought_enabled else None,
            "wrapped_layers": n_wrapped(lm),
            "trajectory_enabled": bool(on and is_recorder_installed(lm)),
            "dim": target_dim(lm) if on else None,
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
            "weights": args.weights,
            "device": args.device,
            "gated_deltanet_layers": n_layers,
            "enable_thinking": args.enable_thinking,
            **thought_meta(),
        }

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    @app.get("/thought")
    def thought():
        return thought_meta()

    def _generate(
        input_ids: List[int],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        seed: Optional[int] = None,
    ):
        """Return ``(text, trajectory)``; trajectory is the p_t thought wave."""
        buf = (
            TrajectoryBuffer(target_dim(lm))
            if (args.thought_enabled and is_wrapped(lm))
            else None
        )
        if buf is not None:
            set_active_buffer(buf)
        if seed is not None:
            set_request_seed(seed)
        try:
            ids = torch.tensor([input_ids], dtype=torch.long, device=args.device)
            do_sample = temperature > 0
            kwargs = dict(
                input_ids=ids,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=safe_pad_id(tok),
            )
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
            with torch.inference_mode():
                out = lm.generate(**kwargs)
        finally:
            if seed is not None:
                clear_request_seed()
            if buf is not None:
                clear_active_buffer()
        new = out[0][ids.shape[1] :]
        text = tok.decode(new, skip_special_tokens=True)
        text = text if args.enable_thinking else strip_thinking(text)
        trajectory = buf.snapshot() if buf is not None else None
        return text, trajectory

    def _sse(payload: dict) -> str:
        return "data: " + json.dumps(payload) + "\n\n"

    def _chat_stream(
        input_ids: List[int],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        seed: Optional[int] = None,
    ):
        """Yield OpenAI-format SSE chunks (streaming chat completion).

        ``seed`` is applied inside the *generation* thread (where
        ``lm.generate`` runs), so the per-request thought seed is isolated
        from any other concurrent request.
        """
        from transformers import TextIteratorStreamer

        buf = (
            TrajectoryBuffer(target_dim(lm))
            if (args.thought_enabled and is_wrapped(lm))
            else None
        )
        streamer = TextIteratorStreamer(tok, timeout=300, skip_special_tokens=True)
        ids = torch.tensor([input_ids], dtype=torch.long, device=args.device)
        do_sample = temperature > 0
        gk = dict(
            input_ids=ids,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=safe_pad_id(tok),
            streamer=streamer,
        )
        if top_p is not None:
            gk["top_p"] = top_p
        if top_k is not None:
            gk["top_k"] = top_k

        def _run():
            if seed is not None:
                set_request_seed(seed)
            if buf is not None:
                set_active_buffer(buf)
            try:
                with torch.inference_mode():
                    lm.generate(**gk)
            finally:
                if seed is not None:
                    clear_request_seed()
                if buf is not None:
                    clear_active_buffer()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        created = int(time.time())
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]

        def chunk(delta: dict, finish: Optional[str] = None):
            return _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )

        close_tag = "<" + "/think" + ">"
        passed = args.enable_thinking  # when thinking is on, pass everything through
        hold = ""
        MAX_HOLD = 1000

        def _thought():
            """Flush any p_t coords captured so far as an SSE 'thought' chunk."""
            if buf is None:
                return
            pts = buf.drain()
            if pts:
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "type": "thought",
                        "points": pts,
                        "dim": target_dim(lm),
                    }
                )

        try:
            yield chunk({"role": "assistant", "content": ""})
            for text in streamer:
                yield from _thought()
                if not text:
                    continue
                if not passed:
                    hold += text
                    ci = hold.find(close_tag)
                    if ci != -1:
                        passed = True
                        rest = hold[ci + len(close_tag) :].lstrip()
                        hold = ""
                        if rest:
                            yield chunk({"content": rest})
                    elif len(hold) >= MAX_HOLD:
                        passed = True
                        yield chunk({"content": hold})
                        hold = ""
                    # else: still inside a thinking block - keep holding
                else:
                    yield chunk({"content": text})
            yield from _thought()
            if hold:
                yield chunk({"content": hold})
        finally:
            thread.join()
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        active_seed = None
        if args.thought_enabled and is_wrapped(lm):
            active_seed = (
                req.thought_seed if req.thought_seed is not None else args.thought_seed
            )

        ids = make_chat_fn(lm, tok, args.enable_thinking)(req.messages)
        prompt_tokens = len(ids)

        if req.stream:

            def _gen():
                yield from _chat_stream(
                    ids,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    req.top_k,
                    seed=active_seed,
                )

            return StreamingResponse(_gen(), media_type="text/event-stream")

        text, trajectory = _generate(
            ids,
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.top_k,
            seed=active_seed,
        )
        comp_tokens = len(tok(text)["input_ids"]) if text else 0
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens": prompt_tokens + comp_tokens,
            },
            "thought": {
                "enabled": bool(active_seed is not None),
                "seed": active_seed,
                "trajectory": trajectory,
                "dim": target_dim(lm) if trajectory else None,
            },
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        active_seed = None
        if args.thought_enabled and is_wrapped(lm):
            active_seed = (
                req.thought_seed if req.thought_seed is not None else args.thought_seed
            )
        ids = make_chat_fn(lm, tok, args.enable_thinking)(
            [ChatMessage(role="user", content=req.prompt)]
        )
        prompt_tokens = len(ids)
        text, trajectory = _generate(
            ids,
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.top_k,
            seed=active_seed,
        )
        return {
            "id": "cmpl-" + uuid.uuid4().hex[:12],
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tok(text)["input_ids"]) if text else 0,
            },
            "thought": {
                "enabled": bool(active_seed is not None),
                "seed": active_seed,
                "trajectory": trajectory,
                "dim": target_dim(lm) if trajectory else None,
            },
        }

    print(f"chat endpoint ready on http://{args.host}:{args.port}/v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
