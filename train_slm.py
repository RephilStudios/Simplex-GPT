"""
train_slm.py
============

Train a tiny language model (SLM) from scratch, then show it generating.

Two corpori are supported:
  * ``python``    - synthetic Python; the target is ``print("hello world")``.
  * ``assistant`` - the assistant's own prose; the target is fluent, on-style
                    text (a higher-entropy, natural-language test of the arch).

Run::

    python train_slm.py --corpus python                # coding SLM
    python train_slm.py --corpus assistant --steps 6000 --hidden 384 --layers 4

Artifacts (written to the project dir): named per corpus by default,
``slm_weights.pt`` + ``tokenizer.json`` for python, ``assistant_weights.pt`` +
``assistant_tokenizer.json`` for assistant.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from assistant_corpus import build_corpus as build_assistant_corpus
from llm_thought import LMConfig, TinyThoughtLM
from python_corpus import build_corpus as build_python_corpus
from tokenizer import CharTokenizer


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="python", choices=["python", "assistant"])
    ap.add_argument("--snippets", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--device", default=None, help="cpu|cuda (default: cuda if available)"
    )
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    # per-corpus artifact names (unless the caller overrides them)
    if args.corpus == "assistant":
        args.out = "assistant_weights.pt"
        args.tokenizer_out = "assistant_tokenizer.json"
        args.corpus_out = "assistant_corpus.txt"
    else:
        args.out = "slm_weights.pt"
        args.tokenizer_out = "tokenizer.json"
        args.corpus_out = "python_corpus.txt"
    return args


def make_config(args):
    cfg = LMConfig
    cfg.hidden_size = args.hidden
    cfg.num_layers = args.layers
    cfg.intermediate_size = args.hidden * 3
    return cfg


@torch.no_grad()
def hello_tests(model: TinyThoughtLM, tok: CharTokenizer) -> bool:
    """A handful of prompts that should complete to hello-world-ish code."""
    prompts = [
        "def main():\n    ",
        "def hello():\n    ",
        "# say hello to the world\n",
        "print(",
    ]
    print("\n--- hello-world generation tests (greedy) ---")
    hit = False
    for p in prompts:
        out = model.generate_text(p, tok, max_new_tokens=24)
        good = "hello world" in out.lower()
        hit = hit or good
        print(f"[{'ok' if good else '  '}] {p!r}\n       -> {out!r}")
    return hit


@torch.no_grad()
def prose_tests(model: TinyThoughtLM, tok: CharTokenizer) -> bool:
    """Complete a few assistant-style prompts and show the model's voice."""
    prompts = [
        "The core idea is that ",
        "Let me be honest about ",
        "One thing to flag. ",
        "The result is ",
    ]
    print("\n--- assistant-style generation (greedy) ---")
    ok = False
    for p in prompts:
        out = model.generate_text(p, tok, max_new_tokens=40)
        letters = sum(c.isalpha() for c in out)
        good = len(out) > 0 and letters >= max(8, int(0.4 * len(out)))
        ok = ok or good
        print(f"[{'ok' if good else '  '}] {p!r}\n       -> {out!r}")
    return ok


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # -- data ----------------------------------------------------------------
    t0 = time.time()
    if args.corpus == "assistant":
        corpus, n_snip = build_assistant_corpus()
    else:
        corpus, n_snip = build_python_corpus(n_snippets=args.snippets, seed=args.seed)
    tok = CharTokenizer.from_text(corpus)
    data = torch.tensor(tok.encode(corpus), dtype=torch.long)
    n = data.numel()
    print(
        f"corpus: {n_snip} snippets, {len(corpus)} chars -> {n} tokens | "
        f"vocab: {tok.vocab_size} chars | device: {device}  ({time.time() - t0:.1f}s)"
    )
    if not args.no_save:
        with open(args.corpus_out, "w", encoding="utf-8") as f:
            f.write(corpus)
        tok.save(args.tokenizer_out)

    # -- model ---------------------------------------------------------------
    cfg = make_config(args)
    model = TinyThoughtLM(cfg, thought=None, vocab_size=tok.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model: {cfg.num_layers} layers, hidden {cfg.hidden_size}, {n_params / 1e6:.2f}M params"
    )

    # -- training ------------------------------------------------------------
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    seq_len, batch = args.seq_len, args.batch

    def get_batch():
        ix = torch.randint(0, n - seq_len - 1, (batch,))
        x = torch.stack([data[i : i + seq_len] for i in ix.tolist()]).to(device)
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix.tolist()]).to(device)
        return x, y

    t0 = time.time()
    best = float("inf")
    for step in range(1, args.steps + 1):
        x, y = get_batch()
        logits = model(x)  # (B, T, V)
        loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            ppl = loss.item() ** (1.0)  # cross-entropy already in nats/token
            print(
                f"step {step:5d}  loss {loss.item():.4f} nats/token  ({time.time() - t0:.0f}s)"
            )
            best = min(best, loss.item())

    model.eval()
    print(
        f"\ntraining done in {time.time() - t0:.0f}s  (best loss {best:.4f} nats/token)"
    )

    # -- save + test ---------------------------------------------------------
    if not args.no_save:
        torch.save(model.state_dict(), args.out)
        print(f"saved weights -> {args.out}\nsaved tokenizer -> {args.tokenizer_out}")

    if args.corpus == "python":
        hit = hello_tests(model, tok)
        print(
            f"\nhello-world: {'PASS' if hit else 'not yet (try more steps / capacity)'}"
        )
        return 0 if hit else 1

    ok = prose_tests(model, tok)
    print(
        f"\nassistant-style: {'looks fluent' if ok else 'still rough (try more steps / data)'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
