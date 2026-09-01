"""
openai_client_demo.py
=====================

Proves the endpoint is a drop-in target for the official ``openai`` Python
SDK — point the client at ``http://127.0.0.1:8100/v1`` and use it exactly
like the real API.

Demonstrates:
  1. ``client.models.list()``
  2. a normal chat completion,
  3. a **streaming** chat completion (tokens print as they arrive),
  4. A/B steering via ``extra_body={"thought_seed": ...}``.

The server must be running (``serve_real_endpoint.py --port 8100``).

Run::

    python openai_client_demo.py --port 8100
"""

from __future__ import annotations

import argparse

from openai import OpenAI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--seed-a", type=int, default=42)
    ap.add_argument("--seed-b", type=int, default=7)
    args = ap.parse_args()

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="not-needed")

    print("=== 1. models.list() ===")
    for m in client.models.list().data:
        print("  ", m.id)
    print()

    print("=== 2. chat.completions.create() ===")
    r = client.chat.completions.create(
        model="qwen3.5-4b-simplex",
        messages=[{"role": "user", "content": "What sound does a bee make? One word."}],
        max_tokens=16,
        temperature=0.0,
    )
    print("  reply:", repr(r.choices[0].message.content.strip()))
    print()

    print("=== 3. streaming (stream=True) ===")
    stream = client.chat.completions.create(
        model="qwen3.5-4b-simplex",
        messages=[
            {"role": "user", "content": "Count from one to five. Just the words."}
        ],
        max_tokens=64,
        temperature=0.0,
        stream=True,
    )
    print("  ", end="", flush=True)
    for ev in stream:
        d = ev.choices[0].delta
        if getattr(d, "content", None):
            print(d.content, end="", flush=True)
    print()
    print()

    print("=== 4. A/B steering via extra_body.thought_seed ===")
    msgs = [{"role": "user", "content": "Choose a four-letter dog breed. One word."}]
    for s in (args.seed_a, args.seed_b):
        r = client.chat.completions.create(
            model="qwen3.5-4b-simplex",
            messages=msgs,
            max_tokens=16,
            temperature=0.0,
            extra_body={"thought_seed": s},
        )
        print(f"  seed {s:>3}: {r.choices[0].message.content.strip()!r}")
    print()
    print("done.")


if __name__ == "__main__":
    main()
