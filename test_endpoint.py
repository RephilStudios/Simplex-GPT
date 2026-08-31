"""
test_endpoint.py
================

LLM-endpoint test suite for the Simplex Thought Field.

Starts two servers (thought enabled / thought disabled, same model seed),
runs behavioral tests against them over HTTP, and verifies the recorded
ThoughtTrace fingerprint *client-side* with pure stdlib (no torch) — an
independent check of the server's record.

Tests
-----
1.  /health reports the correct thought config on both servers
2.  retraceable: same prompt + same thought seed  -> identical tokens
3.  seed steering: thought_seed 42 vs 7            -> different tokens
4.  vanilla server is deterministic (two calls identical)
5.  thought field changes generation (A != B, same model seed)
6.  /thought/last: trace recorded, fingerprint verifies client-side
7.  seeded sampling (temperature + top_k + rng_seed) is deterministic

Run::

    python test_endpoint.py
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
PROMPT = "The quick brown fox jumps over the lazy dog. 0123456789"
PORT_A = 8101  # thought enabled
PORT_B = 8102  # thought disabled (vanilla)

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
    last_err = None
    while time.time() < deadline:
        try:
            return req("GET", f"http://127.0.0.1:{port}/health")
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
            time.sleep(0.5)
    raise SystemExit(f"server on :{port} did not become healthy: {last_err}")


def verify_trace_client(d: dict) -> bool:
    """Recompute the ThoughtTrace fingerprint from the raw JSON payload."""
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
    log_a = open(os.path.join(ROOT, "_srvA.log"), "w", buffering=1)
    log_b = open(os.path.join(ROOT, "_srvB.log"), "w", buffering=1)
    common = [sys.executable, "serve_endpoint.py", "--model-seed", "1234"]
    p_a = subprocess.Popen(
        common
        + [
            "--port",
            str(PORT_A),
            "--thought-enabled",
            "--thought-seed",
            "42",
            "--gain-b",
            "0.6",
            "--gain-a",
            "0.6",
        ],
        cwd=ROOT,
        stdout=log_a,
        stderr=subprocess.STDOUT,
    )
    p_b = subprocess.Popen(
        common + ["--port", str(PORT_B)],
        cwd=ROOT,
        stdout=log_b,
        stderr=subprocess.STDOUT,
    )

    base = {"prompt": PROMPT, "max_tokens": 32, "temperature": 0.0}
    try:
        h_a = wait_health(PORT_A)
        h_b = wait_health(PORT_B)
        print(f"server A (thought on) : {h_a}")
        print(f"server B (vanilla)    : {h_b}\n")

        check(
            "1. health A: thought enabled", h_a.get("thought_enabled") is True, str(h_a)
        )
        check(
            "1. health B: thought disabled",
            h_b.get("thought_enabled") is False,
            str(h_b),
        )

        a1 = req(
            "POST",
            f"http://127.0.0.1:{PORT_A}/v1/completions",
            dict(base, thought_seed=42),
        )
        a2 = req(
            "POST",
            f"http://127.0.0.1:{PORT_A}/v1/completions",
            dict(base, thought_seed=42),
        )
        ids_a1 = a1["choices"][0]["completion_ids"]
        check(
            "2. retraceable: same seed -> identical tokens",
            ids_a1 == a2["choices"][0]["completion_ids"],
            f"fp={a1['thought']['fingerprint']}",
        )

        a7 = req(
            "POST",
            f"http://127.0.0.1:{PORT_A}/v1/completions",
            dict(base, thought_seed=7),
        )
        check(
            "3. seed steering: seed 42 vs 7 -> different tokens",
            ids_a1 != a7["choices"][0]["completion_ids"],
            f"n_diff={sum(x != y for x, y in zip(ids_a1, a7['choices'][0]['completion_ids']))}",
        )

        b1 = req("POST", f"http://127.0.0.1:{PORT_B}/v1/completions", dict(base))
        b2 = req("POST", f"http://127.0.0.1:{PORT_B}/v1/completions", dict(base))
        check(
            "4. vanilla server deterministic",
            b1["choices"][0]["completion_ids"] == b2["choices"][0]["completion_ids"],
        )

        check(
            "5. thought field changes generation (A != B)",
            ids_a1 != b1["choices"][0]["completion_ids"],
            f"n_diff={sum(x != y for x, y in zip(ids_a1, b1['choices'][0]['completion_ids']))}",
        )

        tr = req("GET", f"http://127.0.0.1:{PORT_A}/thought/last")
        check(
            "6. trace recorded + client-side fingerprint verifies",
            verify_trace_client(tr) and tr.get("seq_len", 0) > 0,
            f"fp={tr.get('fingerprint')}",
        )

        s1 = req(
            "POST",
            f"http://127.0.0.1:{PORT_A}/v1/completions",
            dict(base, temperature=1.0, top_k=5, rng_seed=99, thought_seed=42),
        )
        s2 = req(
            "POST",
            f"http://127.0.0.1:{PORT_A}/v1/completions",
            dict(base, temperature=1.0, top_k=5, rng_seed=99, thought_seed=42),
        )
        check(
            "7. seeded sampling deterministic",
            s1["choices"][0]["completion_ids"] == s2["choices"][0]["completion_ids"],
        )

        # sample output for the log
        print("\nsample completion (server A, seed 42, first 8 tokens):")
        print("  ", a1["choices"][0]["text"][:120])
        print("\nsample completion (server B, vanilla, first 8 tokens):")
        print("  ", b1["choices"][0]["text"][:120])
    finally:
        for p in (p_a, p_b):
            p.terminate()
        for p in (p_a, p_b):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        log_a.close()
        log_b.close()

    print(f"\n{7 - len(FAILURES)}/7 endpoint tests passed")
    if FAILURES:
        print("failed:", FAILURES)
        sys.exit(1)


if __name__ == "__main__":
    main()
