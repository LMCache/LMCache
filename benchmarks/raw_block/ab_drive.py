# SPDX-License-Identifier: Apache-2.0
"""E8 A/B driver: force raw-block (L2) hits via WORKING-SET >> L1. Warms K distinct
long prefixes (stored to L2), then measures TTFT by cycling them round-robin so
each is evicted from the small L1 before revisit -> loads come from L2 (raw_block).
Measure TTFT at concurrency 1 and a saturation point."""

# Standard
import argparse
import json
import os
import random
import threading
import time

# Third Party
import requests

URL = os.environ.get("KVIO_URL", "http://localhost:8000/v1/completions")
MODEL = os.environ.get("KVIO_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango"
).split()


def make_prefix(seed, nwords):
    r = random.Random(seed)
    return (
        "Document "
        + str(seed)
        + ": "
        + " ".join(r.choice(VOCAB) + str(r.randint(0, 999)) for _ in range(nwords))
    )


def ttft_once(prompt):
    t0 = time.perf_counter()
    try:
        r = requests.post(
            URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 4,
                "temperature": 0,
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
    except Exception:
        return None
    ttft = None
    for line in r.iter_lines():
        if line and line.startswith(b"data:"):
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            if j.get("choices") and j["choices"][0].get("text") and ttft is None:
                ttft = time.perf_counter() - t0
    r.close()
    return ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--words", type=int, default=11000)  # ~14K tokens ~ 1.75GB KV
    ap.add_argument("--nprefix", type=int, default=24)  # working set ~ 42GB >> L1
    ap.add_argument("--duration", type=float, default=25.0)
    a = ap.parse_args()

    prefixes = [make_prefix(1000 + i, a.words) for i in range(a.nprefix)]
    # WARM: store all prefixes to L2 (once each). Only done for concurrency==1 pass;
    # the 2nd pass reuses what's already on the device.
    if a.concurrency == 1:
        for p in prefixes:
            ttft_once(p)
        print(f"[{a.arm}] warmed {a.nprefix} prefixes to L2", flush=True)
        time.sleep(6)

    # MEASURE: round-robin cycle -> each prefix maximally evicted from L1 -> L2 load
    ttfts = []
    idx = [0]
    stop = time.time() + a.duration
    lock = threading.Lock()

    def worker():
        while time.time() < stop:
            with lock:
                p = prefixes[idx[0] % a.nprefix]
                idx[0] += 1
            t = ttft_once(p)
            if t is not None:
                with lock:
                    ttfts.append(t)

    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)
    ]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=a.duration + 40)
    elapsed = time.time() - t0

    ttfts.sort()
    n = len(ttfts)
    if n == 0:
        print(f"RESULT arm={a.arm} C={a.concurrency} n=0 NO_COMPLETED", flush=True)
        return
    p50 = ttfts[n // 2]
    p99 = ttfts[min(n - 1, int(0.99 * n))]
    print(
        f"RESULT arm={a.arm} C={a.concurrency} n={n} "
        f"TTFT_p50={p50 * 1000:.0f}ms TTFT_p99={p99 * 1000:.0f}ms "
        f"tput={n / elapsed:.2f}req/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
