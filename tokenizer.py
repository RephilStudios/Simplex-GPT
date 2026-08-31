"""
tokenizer.py
============

A tiny char-level tokenizer tuned for Python source.

Character-level is the right call for a from-scratch tiny model: Python uses a
small, fixed alphabet, so the model learns syntax at the character scale with
no subword complexity. The vocabulary is built from the corpus, so it only
contains characters that actually appear (plus an explicit set of code
characters), keeping the LM head small and the per-position entropy low.
"""

from __future__ import annotations

import json
import os
from typing import List

# Characters we always want available for Python, even if a given corpus
# happens not to use them (so the model can still produce them).
CODE_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \n\t\"'`()[]{}<>,.;:=+-*/%&|^!~@#$?_\\"
)


class CharTokenizer:
    def __init__(self, vocab: List[str]):
        self.vocab = list(vocab)
        self.itos = {i: s for i, s in enumerate(self.vocab)}
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.unk = self.stoi.get("\u0000", 0)  # never emitted; safety only

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from the characters present in ``text``."""
        chars = sorted(set(text) | set(CODE_CHARS))
        return cls(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] if c in self.stoi else self.unk for c in text]

    def decode(self, ids) -> str:
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(self.itos[i] for i in ids if i in self.itos)

    # -- (de)serialization ---------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"])


def ensure_dir(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    return d
