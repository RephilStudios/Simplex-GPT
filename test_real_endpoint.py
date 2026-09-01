"""
test_real_endpoint.py
=====================

Boots ``serve_real_endpoint.py`` (thought enabled) as a subprocess, then
checks over HTTP:

  1. ``/health`` shows the thought field enabled + 24 wrapped layers,
  2. ``/v1/models`` lists the model,
  3. a chat completion returns non-empty text,
  4. **retrace** — same messages + same ``thought_seed`` => identical reply,
  5. **A/B** — different ``thought_seed`` on a genuine-choice prompt =>
     (may differ; reported, not asserted, since steering is confidence-gated),
  6. ``/thought`` reflects the active config.

Tears the server down when done.

Run::

    python test_real_endpoint.py --weights models/Qwen3.5-4B --gain 4.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8137
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = "Choose a four-letter dog breed. One word."


def http(path: str, body: dict | None = None) -> dict:
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
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

        h = http("/health")
        wrapped = h.get("wrapped_layers", 0)
        checks.append(
            (
                "health: thought enabled",
                h.get("enabled") is True,
                f"enabled={h.get('enabled')}",
            )
        )
        checks.append(
            ("health: 24 layers wrapped", wrapped == 24, f"wrapped={wrapped}")
        )

        m = http("/v1/models")
        checks.append(
            (
                "models: lists model",
                any(d["id"] == "qwen3.5-4b-simplex" for d in m["data"]),
                f"models={[d['id'] for d in m['data']]}",
            )
        )

        def chat(seed):
            r = http(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "thought_seed": seed,
                },
            )
            return r["choices"][0]["message"]["content"], r

        a, ra = chat(42)
        checks.append(("chat: non-empty reply", len(a.strip()) > 0, f"reply={a!r}"))
        checks.append(
            (
                "chat: thought seed echoed",
                ra["thought"]["seed"] == 42,
                f"seed={ra['thought']['seed']}",
            )
        )

        b, _ = chat(42)
        checks.append(("retrace: same seed identical", a == b, f"a={a!r} b={b!r}"))

        c, rc = chat(7)
        checks.append(
            (
                "A/B: seed 7 requested",
                rc["thought"]["seed"] == 7,
                f"a(42)={a!r} c(7)={c!r} differ={a != c}",
            )
        )

        t = http("/thought")
        checks.append(
            (
                "thought: state endpoint",
                t.get("enabled") is True and t.get("gain") == args.gain,
                json.dumps(t),
            )
        )

        print(f"--- real-endpoint tests (gain {args.gain}) ---")
        passed = 0
        for name, good, detail in checks:
            passed += int(good)
            print(f"[{'ok' if good else 'FAIL'}] {name}   ({detail})")
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
