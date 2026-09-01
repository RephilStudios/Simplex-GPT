"""
chat_demo.py
============

Quick interactive demo against a *running* ``serve_real_endpoint.py``.
Waits for /health, then runs a short multi-turn conversation (showing context
is retained) and an A/B steering pair.

    python chat_demo.py --port 8100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def http(base: str, path: str, body: dict | None = None, timeout: int = 180) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(base: str, timeout: int = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            http(base, "/health", timeout=5)
            return
        except Exception:
            time.sleep(3)
    raise SystemExit("server did not become ready in time")


def port_ready(base: str) -> bool:
    try:
        http(base, "/health", timeout=3)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--seed-a", type=int, default=42)
    ap.add_argument("--seed-b", type=int, default=7)
    ap.add_argument(
        "--spawn",
        action="store_true",
        help="start serve_real_endpoint.py ourselves if the port is busy-free",
    )
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--gain", type=float, default=4.0)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    proc = None
    if not port_ready(base):
        if not args.spawn:
            raise SystemExit(
                f"no server at {base} — start serve_real_endpoint.py or pass --spawn"
            )
        log = open("server.log", "wb")
        proc = subprocess.Popen(
            [
                sys.executable,
                "serve_real_endpoint.py",
                "--port",
                str(args.port),
                "--weights",
                args.weights,
                "--thought-enabled",
                "--gain",
                str(args.gain),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        print(f"spawned server (pid {proc.pid}), waiting for ready ...")
    wait_ready(base)
    h = http(base, "/health")
    print(f"=== /health ===")
    print(json.dumps(h, indent=2))
    print()

    # multi-turn conversation (context should carry over)
    print("=== multi-turn chat ===")
    history = []
    turns = [
        "I'm thinking of a European capital famous for a river running through it. Which city?",
        "Nice. Name one famous bridge on that river. One or two words.",
        "And what river is it? One word.",
    ]
    for i, user in enumerate(turns, 1):
        history.append({"role": "user", "content": user})
        r = http(
            base,
            "/v1/chat/completions",
            {
                "messages": history,
                "max_tokens": 32,
                "temperature": 0.0,
            },
        )
        reply = r["choices"][0]["message"]["content"].strip()
        print(f"  [{i}] user : {user}")
        print(f"  [{i}] model: {reply}")
        history.append({"role": "assistant", "content": reply})
    print()

    # A/B steering on a genuine-choice prompt
    print("=== A/B steering (same prompt, two seeds) ===")
    for s in (args.seed_a, args.seed_b):
        r = http(
            base,
            "/v1/chat/completions",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Choose a four-letter dog breed. One word.",
                    }
                ],
                "max_tokens": 16,
                "temperature": 0.0,
                "thought_seed": s,
            },
        )
        reply = r["choices"][0]["message"]["content"].strip()
        print(f"  seed {s:>3}: {reply!r}   (thought={r['thought']})")
    print()
    print("done.")
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("(spawned server stopped)")


if __name__ == "__main__":
    main()
