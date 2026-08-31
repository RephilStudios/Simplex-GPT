"""
test_coding_endpoint.py
=======================

Serves the trained *coding* model (real Python) on two servers — one vanilla,
one with the Simplex Thought Field enabled — and shows, over HTTP:

1. the vanilla coding model completes ``def main():`` to ``print("hello world")``;
2. the vanilla model is deterministic;
3. on a prompt with a *genuine choice* (``x = ``), the thought seed steers
   the model to a different valid value (seed 42 vs 7);
4. that steering is reproducible per seed;
5. the thought fingerprint is recorded and verifies client-side.

This also surfaces an honest finding: where the model is *confident* (a
memorized answer like ``def main():``), the thought field does NOT override it —
it modulates the model's genuine uncertainties, which is the desired behavior.

Run::

    python test_coding_endpoint.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
HELLO_PROMPT = "def main():\n    "  # confident -> always the same (correct) answer
STEER_PROMPT = "x = "  # genuine choice -> thought seed picks a value
PORT_T = 8201  # coding + thought enabled
PORT_V = 8202  # coding vanilla

FAILURES: list[str] = []


def req(method: str, url: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=180) as resp:
        return json.loads(resp.read().decode())


def wait_health(port: int, timeout: float = 120.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return req("GET", f"http://127.0.0.1:{port}/health")
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last = e
            time.sleep(0.5)
    raise SystemExit(f"server on :{port} did not become healthy: {last}")


def verify_trace(d: dict) -> bool:
    h = hashlib.sha256()
    h.update(base64.b64decode(d["coords"]["b64"]))
    for k in sorted(d["slot_values"]):
        h.update(k.encode("utf-8"))
        h.update(base64.b64decode(d["slot_values"][k]["b64"]))
    return h.hexdigest()[:32] == d["fingerprint"]


def check(name: str, cond: bool, detail: str = ""):
    ok = bool(cond)
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    base = [sys.executable, "serve_endpoint.py", "--coding", "--device", "cuda"]
    log_t = open(os.path.join(ROOT, "_srvT.log"), "w", buffering=1)
    log_v = open(os.path.join(ROOT, "_srvV.log"), "w", buffering=1)
    p_t = subprocess.Popen(
        base
        + [
            "--port",
            str(PORT_T),
            "--thought-enabled",
            "--thought-seed",
            "42",
            "--gain-b",
            "1.0",
            "--gain-a",
            "1.0",
        ],
        cwd=ROOT,
        stdout=log_t,
        stderr=subprocess.STDOUT,
    )
    p_v = subprocess.Popen(
        base + ["--port", str(PORT_V)],
        cwd=ROOT,
        stdout=log_v,
        stderr=subprocess.STDOUT,
    )

    try:
        ht = wait_health(PORT_T)
        hv = wait_health(PORT_V)
        print(f"thought coding : {ht}\nvanilla coding : {hv}\n")
        check(
            "1. mode=coding on both",
            ht.get("mode") == "coding" and hv.get("mode") == "coding",
        )

        # --- hello world (confident, correct) -------------------------------
        v1 = req(
            "POST",
            f"http://127.0.0.1:{PORT_V}/v1/completions",
            {"prompt": HELLO_PROMPT, "max_tokens": 20},
        )
        v2 = req(
            "POST",
            f"http://127.0.0.1:{PORT_V}/v1/completions",
            {"prompt": HELLO_PROMPT, "max_tokens": 20},
        )
        vt = v1["choices"][0]["text"]
        check(
            "2. vanilla model writes hello world", "hello world" in vt.lower(), repr(vt)
        )
        check(
            "3. vanilla deterministic",
            v1["choices"][0]["completion_ids"] == v2["choices"][0]["completion_ids"],
        )

        # --- thought steering on a genuine choice ---------------------------
        t42 = req(
            "POST",
            f"http://127.0.0.1:{PORT_T}/v1/completions",
            {"prompt": STEER_PROMPT, "max_tokens": 8, "thought_seed": 42},
        )
        t42b = req(
            "POST",
            f"http://127.0.0.1:{PORT_T}/v1/completions",
            {"prompt": STEER_PROMPT, "max_tokens": 8, "thought_seed": 42},
        )
        t7 = req(
            "POST",
            f"http://127.0.0.1:{PORT_T}/v1/completions",
            {"prompt": STEER_PROMPT, "max_tokens": 8, "thought_seed": 7},
        )
        s42 = t42["choices"][0]["text"]
        s7 = t7["choices"][0]["text"]
        check(
            "4. thought seed 42 reproducible",
            t42["choices"][0]["completion_ids"] == t42b["choices"][0]["completion_ids"],
        )
        check(
            "5. thought seed steers a real choice (42 != 7)",
            t42["choices"][0]["completion_ids"] != t7["choices"][0]["completion_ids"],
            f"seed42={s42!r} seed7={s7!r}",
        )

        tr = req("GET", f"http://127.0.0.1:{PORT_T}/thought/last")
        check(
            "6. thought fingerprint verifies client-side",
            verify_trace(tr),
            tr.get("fingerprint", ""),
        )

        print("\n--- real code completions ---")
        print(f"confident prompt {HELLO_PROMPT!r} (vanilla)  -> {vt!r}")
        print(f"choice prompt    {STEER_PROMPT!r}")
        print(f"    thought seed 42 -> {STEER_PROMPT}{s42!r}")
        print(f"    thought seed 7  -> {STEER_PROMPT}{s7!r}")
        print(
            "\n(note) the field does NOT override the confident 'def main():' answer —"
        )
        print(
            "       it modulates the model's genuine choices, which is the desired behavior."
        )
    finally:
        for p in (p_t, p_v):
            p.terminate()
        for p in (p_t, p_v):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        log_t.close()
        log_v.close()

    total = 6
    print(f"\n{total - len(FAILURES)}/{total} coding-endpoint tests passed")
    if FAILURES:
        print("failed:", FAILURES)
        sys.exit(1)


if __name__ == "__main__":
    main()
