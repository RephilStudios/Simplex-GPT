"""
test_concurrency.py
===================

End-to-end proof that the real endpoint (Qwen3.5-4B, thought field ON)
handles **concurrent** requests correctly:

  1. parallel non-stream chats with *different* seeds each get their own
     seed's reply and the right seed echoed back (no cross-contamination),
  2. parallel chats with the *same* seed are retraceable (identical text),
  3. parallel *streamed* chats are retraceable too, and agree exactly with
     the non-stream result for the same seed,
  4. parallel requests genuinely overlap in wall time (with the old global
     generation lock, wall(parallel) == sum(sequential)).

Run::

    python test_concurrency.py --weights models/Qwen3.5-4B --gain 4.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

PORT = 8138
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = "Choose a four-letter dog breed. One word."


def http(path: str, body: dict | None = None, timeout: int = 240) -> dict:
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(timeout: int = 300) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            http("/health")
            return True
        except Exception:
            time.sleep(3)
    return False


def chat(seed: int, max_tokens: int = 16) -> tuple[str, dict]:
    r = http(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "thought_seed": seed,
        },
    )
    return r["choices"][0]["message"]["content"].strip(), r


def chat_stream(seed: int, max_tokens: int = 16) -> str:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(
            {
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
                "thought_seed": seed,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text = ""
    with urllib.request.urlopen(req, timeout=240) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
            except Exception:
                continue
            ch = (j.get("choices") or [{}])[0]
            d = ch.get("delta") or {}
            if d.get("content"):
                text += d["content"]
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/Qwen3.5-4B")
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    proc = subprocess.Popen(
        [
            args.python,
            "serve_real_endpoint.py",
            "--port",
            str(PORT),
            "--weights",
            args.weights,
            "--thought-enabled",
            "--gain",
            str(args.gain),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ok = False
    try:
        if not wait_ready():
            print("FAIL: server did not become ready in time")
            tail = proc.stdout.read() if proc.stdout else ""
            print(tail[-3000:] if tail else "(no output)")
            return 1

        checks: list[tuple[str, bool, str]] = []

        def run_threads(fns: list) -> None:
            ts = [threading.Thread(target=f) for f in fns]
            [t.start() for t in ts]
            [t.join() for t in ts]

        # 0. warmup (allocator / CUDA init) — not timed
        warm, _ = chat(42)
        checks.append(("warmup: non-empty reply", len(warm) > 0, repr(warm)))

        # 1. parallel chats, DIFFERENT seeds -> isolated, seed echoed
        res: dict = {}

        def p_chat(s: int) -> None:
            res[s] = chat(s)

        run_threads([lambda s=s: p_chat(s) for s in (42, 7, 123)])
        all_nonempty = all(len(res[s][0]) > 0 for s in res)
        seeds_ok = all(res[s][1]["thought"]["seed"] == s for s in res)
        checks.append(
            (
                "parallel(42,7,123): all non-empty",
                all_nonempty,
                {s: repr(res[s][0]) for s in res},
            )
        )
        checks.append(
            (
                "parallel: each seed echoed back",
                seeds_ok,
                f"seeds={[res[s][1]['thought']['seed'] for s in res]}",
            )
        )

        # 2. parallel chats, SAME seed -> retraceable
        pair: dict = {}

        def p_chat_pair(i: int) -> None:
            pair[i] = chat(42)[0]

        run_threads([lambda: p_chat_pair(0), lambda: p_chat_pair(1)])
        checks.append(
            (
                "parallel(42,42): identical (retraceable)",
                pair[0] == pair[1] and len(pair[0]) > 0,
                f"a={pair[0]!r} b={pair[1]!r}",
            )
        )

        # 3. parallel STREAMED chats, same seed -> identical, and equal to
        #    the non-stream result for that seed
        spair: dict = {}

        def p_stream(i: int) -> None:
            spair[i] = chat_stream(42)

        run_threads([lambda: p_stream(0), lambda: p_stream(1)])
        ns42 = res[42][0]
        checks.append(
            (
                "parallel stream(42,42): identical",
                spair[0] == spair[1] and len(spair[0]) > 0,
                f"a={spair[0]!r} b={spair[1]!r}",
            )
        )
        checks.append(
            (
                "stream == non-stream (same seed)",
                spair[0] == ns42,
                f"stream={spair[0]!r} non-stream={ns42!r}",
            )
        )

        # 4. genuine overlap: parallel wall time must beat the sequential sum
        #    (with the old global lock, parallel == exactly sequential sum)
        t0 = time.time()
        chat(42, max_tokens=32)
        t1 = time.time()
        chat(7, max_tokens=32)
        seq = time.time() - t0  # t1-t0 + (t2-t1)
        t0 = time.time()
        run_threads(
            [
                lambda: chat(42, max_tokens=32),
                lambda: chat(7, max_tokens=32),
            ]
        )
        par = time.time() - t0
        ratio = par / seq if seq > 0 else float("inf")
        checks.append(
            (
                "overlap: parallel faster than sequential",
                par < 0.98 * seq,
                f"parallel={par:.1f}s sequential={seq:.1f}s ratio={ratio:.2f}",
            )
        )

        print(f"--- concurrency tests (gain {args.gain}) ---")
        passed = 0
        for name, good, detail in checks:
            passed += int(good)
            print(f"[{'ok' if good else 'FAIL'}] {name}\n       {detail}")
        print(f"\n{passed}/{len(checks)} checks passed")
        ok = passed == len(checks)
        return 0 if ok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
