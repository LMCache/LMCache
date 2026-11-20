#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-node P2P stress test.

Phases, each with ``--concurrency`` requests in flight:
  warm   : every synthetic document is sent once to its *home* node (documents alternate
           between A and B), so each document's KV lives in exactly one node's L1.
  peer   : ``--queries-per-doc`` questions per document go to the *other* node: every
           prefill must be served over P2P from the home node.
  mixedN : the same again, each query on a random node (local L1 hits and P2P hits
           interleaved), ``--mixed-rounds`` times.
Requests are non-streaming (the only mode that returns cache stats); with the default
``--max-tokens 1`` the request latency is the TTFT. Each phase prints requests, errors,
wall time, req/s, prefill tokens/s, TTFT p50/p95/p99/max, the LMCache hit ratio (cached
tokens / prompt tokens; the chunk-aligned maximum is about 0.98 for 7k-token prompts)
and writes ``stress_<tag>.csv`` (per request) and ``stress_<tag>.json`` (per phase, with
``start_ts``/``end_ts`` for Prometheus range exports) under ``--out-dir``.

usage: stress_p2p.py [--a URL] [--b URL] [--docs 16] [--doc-words 6000]
                     [--queries-per-doc 3] [--concurrency 8] [--mixed-rounds 1]
                     [--tag NAME] [--out-dir DIR]
"""

# Standard
import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time

# Third Party
import aiohttp
import p2penv

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()
QUESTIONS = [
    "What is the first word?",
    "What is the last word?",
    "How many words are there?",
    "Name a word that appears twice in a row.",
    "Summarize the stream in one sentence.",
]


def make_doc(seed: int, n_words: int) -> str:
    """Return a deterministic synthetic document of ``n_words`` Greek-letter names."""
    rng = random.Random(seed)
    return f"Document {seed}. Memorize this token stream.\n\n" + " ".join(
        rng.choice(WORDS) for _ in range(n_words)
    )


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile of ``xs`` (``nan`` when empty)."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize_phase(
    name: str, rows: list[dict], wall: float, start_ts: float, end_ts: float
) -> dict:
    """Aggregate one phase's request rows into the printed/JSON summary."""
    ok = [r for r in rows if r["error"] is None and r["ttft_s"] is not None]
    errs = [r for r in rows if r["error"] is not None]
    ttfts = [r["ttft_s"] for r in ok]
    tot = [r["total_s"] for r in ok]
    cached = [r["cached_tokens"] or 0 for r in ok]
    ptoks = [r["prompt_tokens"] or 0 for r in ok]
    return dict(
        phase=name,
        start_ts=round(start_ts, 1),
        end_ts=round(end_ts, 1),
        requests=len(rows),
        errors=len(errs),
        wall_s=round(wall, 2),
        req_per_s=round(len(ok) / wall, 2) if wall else 0,
        prompt_tok_per_s=round(sum(ptoks) / wall) if wall else 0,
        ttft_p50=round(percentile(ttfts, 50), 3),
        ttft_p95=round(percentile(ttfts, 95), 3),
        ttft_p99=round(percentile(ttfts, 99), 3),
        ttft_max=round(max(ttfts), 3) if ttfts else None,
        total_p50=round(percentile(tot, 50), 3),
        lmcache_hit_ratio=round(sum(cached) / max(1, sum(ptoks)), 3),
        mean_prompt_tokens=round(statistics.fmean(ptoks)) if ptoks else 0,
    )


async def one(session, base, model, prompt, max_tokens, rec):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "kv_transfer_params": {"cached_token_stats": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    cached = None
    ptoks = None
    err = None
    ntext = 0
    try:
        async with session.post(
            f"{base}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as r:
            if r.status != 200:
                err = f"http {r.status}: {(await r.text())[:120]}"
            else:
                d = await r.json()
                cached = (
                    (d.get("kv_transfer_params") or {}).get("cached_token_stats") or {}
                ).get("num_lmcache_cached_tokens")
                ptoks = d["usage"]["prompt_tokens"]
                ntext = d["usage"]["completion_tokens"]
    except Exception as e:  # noqa: BLE001 - any client-side failure is an error row
        err = f"{type(e).__name__}: {e}"[:160]
    lat = time.perf_counter() - t0
    rec.update(
        ttft_s=lat if err is None else None,
        total_s=lat,
        cached_tokens=cached,
        prompt_tokens=ptoks,
        error=err,
        out_tokens=ntext,
    )
    return rec


async def run_phase(name, jobs, concurrency, model, max_tokens, rows):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:

        async def guarded(base, prompt, meta):
            async with sem:
                return await one(
                    session,
                    base,
                    model,
                    prompt,
                    max_tokens,
                    dict(phase=name, node=base, **meta),
                )

        t0 = time.perf_counter()
        start_ts = time.time()
        res = await asyncio.gather(*(guarded(b, p, m) for b, p, m in jobs))
        wall = time.perf_counter() - t0
    rows.extend(res)
    s = summarize_phase(name, res, wall, start_ts, time.time())
    print(
        f"[{name:6}] n={s['requests']} err={s['errors']} wall={s['wall_s']}s "
        f"rps={s['req_per_s']} prefill_tok/s={s['prompt_tok_per_s']} "
        f"ttft p50/p95/p99/max={s['ttft_p50']}/{s['ttft_p95']}/"
        f"{s['ttft_p99']}/{s['ttft_max']}s hit={s['lmcache_hit_ratio']} "
        f"prompt_tok={s['mean_prompt_tokens']}",
        flush=True,
    )
    for e in [r for r in res if r["error"]][:3]:
        print("   error:", e["node"], e["error"])
    return s


async def main(a):
    nodes = [a.a, a.b]
    home = {i: nodes[i % 2] for i in range(a.docs)}
    other = {n: nodes[1 - nodes.index(n)] for n in nodes}
    base_seed = a.seed if a.seed is not None else time.time_ns() % 10**9
    docs = {i: make_doc(base_seed + i, a.doc_words) for i in range(a.docs)}
    rows, summary = [], []
    jobs = [
        (home[i], f"{docs[i]}\n\n{QUESTIONS[0]}", dict(doc=i, q=0, home=home[i]))
        for i in range(a.docs)
    ]
    summary.append(
        await run_phase("warm", jobs, a.concurrency, a.model, a.max_tokens, rows)
    )
    time.sleep(a.gap)
    jobs = [
        (
            other[home[i]],
            f"{docs[i]}\n\n{QUESTIONS[q % len(QUESTIONS)]}",
            dict(doc=i, q=q, home=home[i]),
        )
        for i in range(a.docs)
        for q in range(1, a.queries_per_doc + 1)
    ]
    random.Random(base_seed).shuffle(jobs)
    summary.append(
        await run_phase("peer", jobs, a.concurrency, a.model, a.max_tokens, rows)
    )
    rng = random.Random(base_seed + 7)
    for r in range(a.mixed_rounds):
        jobs = [
            (
                rng.choice(nodes),
                f"{docs[i]}\n\n{QUESTIONS[q % len(QUESTIONS)]}",
                dict(doc=i, q=q, home=home[i]),
            )
            for i in range(a.docs)
            for q in range(1, a.queries_per_doc + 1)
        ]
        rng.shuffle(jobs)
        summary.append(
            await run_phase(
                f"mixed{r}", jobs, a.concurrency, a.model, a.max_tokens, rows
            )
        )
    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or time.strftime("%H%M%S")
    with open(f"{a.out_dir}/stress_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(
        dict(args=vars(a), seed=base_seed, phases=summary),
        open(f"{a.out_dir}/stress_{tag}.json", "w"),
        indent=2,
    )
    total_err = sum(s["errors"] for s in summary)
    print(
        f"wrote {a.out_dir}/stress_{tag}.{{csv,json}}; "
        f"STRESS: {'PASS' if total_err == 0 else 'FAIL'} ({total_err} errors)"
    )
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    env = p2penv.load()
    va, vb, _, _ = p2penv.url_defaults(env)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", default=va)
    ap.add_argument("--b", default=vb)
    ap.add_argument("--model", default=env.get("MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--doc-words", type=int, default=6000)
    ap.add_argument("--queries-per-doc", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--mixed-rounds", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument(
        "--out-dir", default=os.path.join(env.get("P2P_HOME", "."), "results")
    )
    ap.add_argument("--tag")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
