"""
train_slm.py
============

Train a tiny Python coding model (SLM) from scratch on the synthetic Python
corpus, then show it producing ``hello world`` and (optionally) that the
thought field can steer the *real* code output.

Run::

    python train_slm.py                     # sensible defaults
    python train_slm.py --steps 8000 --hidden 320 --layers 4
    python train_slm.py --snippets 60000 --seq-len 64 --lr 2e-3

Artifacts (written to the project dir):
    slm_weights.pt     trained vanilla weights
    tokenizer.json     char vocabulary
    python_corpus.txt  the corpus that was trained on (for reference)
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from llm_thought import LMConfig, TinyThoughtLM
from python_corpus import build_corpus
from tokenizer import CharTokenizer


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument("--out", default="slm_weights.pt")
    ap.add_argument("--tokenizer-out", default="tokenizer.json")
    ap.add_argument("--corpus-out", default="python_corpus.txt")
    ap.add_argument("--no-save", action="store_true")
    return ap.parse_args()


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


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # -- data ----------------------------------------------------------------
    t0 = time.time()
    corpus, n_snip = build_corpus(n_snippets=args.snippets, seed=args.seed)
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

    hit = hello_tests(model, tok)
    print(f"\nhello-world: {'PASS' if hit else 'not yet (try more steps / capacity)'}")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
