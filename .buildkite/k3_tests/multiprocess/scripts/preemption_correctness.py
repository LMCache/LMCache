# SPDX-License-Identifier: Apache-2.0
"""End-to-end preemption correctness ladder (T0-T3 in the preemption TDD).

Drives two OpenAI-compatible vLLM servers -- a baseline without LMCache and
one with the LMCache MP connector -- through the same token-id workload and
compares generated token ids exactly.

Passes:
  0. baseline, high concurrency          -> reference token ids
  1. lmcache, cold cache, high concurrency -> T0 (no crash / no wedge) and
                                              T2 (preempted requests match)
  2. lmcache, warm cache, low concurrency  -> T3 (everything stored during the
                                              preemption storm is intact)

Preemption is asserted from vLLM's own ``/metrics`` counter, never from the
connector's log.  Prompts are lists of token ids with shared prefixes so a
corrupted chunk is re-hit by other requests, and generation uses
``ignore_eos`` with a fixed ``max_tokens`` so KV demand is deterministic.

The workload must overflow the KV pool by arithmetic: run the server with
``--num-gpu-blocks-override N`` and choose
``concurrency * (prompt + max_tokens) >> N * block_size``.
"""

# Standard
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Third Party
import aiohttp

PREEMPTION_METRIC = "vllm:num_preemptions_total"


@dataclass
class Prompt:
    request_id: str
    token_ids: list[int]
    max_tokens: int


@dataclass
class Completion:
    request_id: str
    token_ids: list[int]
    text: str
    finish_reason: str
    latency_s: float
    lmcache_cached_tokens: int | None = None
    error: str | None = None
    # Per output position: (best token id, best logprob, runner-up token id,
    # runner-up logprob).  Only collected for the reference pass.
    top2: list[tuple[int, float, int, float]] = field(default_factory=list)


@dataclass
class PassResult:
    name: str
    completions: dict[str, Completion] = field(default_factory=dict)
    preemptions_before: float = 0.0
    preemptions_after: float = 0.0
    wall_s: float = 0.0

    @property
    def preemptions(self) -> float:
        return self.preemptions_after - self.preemptions_before

    @property
    def errors(self) -> list[Completion]:
        return [c for c in self.completions.values() if c.error]


_TEXT = (
    "The history of computing is a history of people trying to make machines "
    "do arithmetic faster and more reliably than they could by hand. Early "
    "mechanical calculators used gears and levers; later designs replaced "
    "them with relays, then vacuum tubes, then transistors. Each generation "
    "was smaller, faster, and cheaper than the last, and each opened up new "
    "kinds of problems that could be attacked with computation. Programming "
    "languages evolved alongside the hardware, moving from raw machine codes "
    "to assembly mnemonics to high level notations that let a person describe "
    "an algorithm in something closer to ordinary prose. Operating systems "
    "appeared to share expensive machines among many users, and networks "
    "connected those machines into systems that spanned buildings, cities, "
    "and eventually the whole planet. Storage grew from punched cards and "
    "paper tape to magnetic drums, disks, and solid state memory, with "
    "capacities that increased by orders of magnitude every decade. Along "
    "the way, ideas from mathematics and logic gave the field its theoretical "
    "foundation: models of computation, notions of complexity, and proofs "
    "about what can and cannot be decided by any machine at all. "
)


def _text_token_pool(model: str, need: int) -> list[int] | None:
    """Tokenize real prose with the model's tokenizer; None if unavailable."""
    try:
        # Third Party
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
    except Exception:  # noqa: BLE001 - fall back to random ids
        return None
    ids: list[int] = []
    paragraph = 0
    while len(ids) < need:
        # Vary the paragraph slightly so repeated text does not create
        # spurious shared prefixes between unrelated requests.
        ids.extend(
            tok.encode(f"Section {paragraph}. " + _TEXT, add_special_tokens=False)
        )
        paragraph += 1
    return ids


def build_workload(
    *,
    model: str,
    num_requests: int,
    prompt_min: int,
    prompt_max: int,
    max_tokens: int,
    num_shared_prefixes: int,
    shared_prefix_len: int,
    vocab: int,
    seed: int,
) -> list[Prompt]:
    """Token-id prompts: a shared prefix (one of a few) plus a unique tail.

    Prompts are slices of real prose when the model's tokenizer can be
    loaded; natural text has peaked next-token distributions, which keeps
    numerical near-ties (and therefore benign divergences) rare.  Random
    token ids are the fallback.
    """
    rng = random.Random(seed)
    pool = _text_token_pool(
        model, need=num_requests * prompt_max + shared_prefix_len * 4
    )

    def draw(n: int) -> list[int]:
        if pool is None:
            return [rng.randrange(10, vocab) for _ in range(n)]
        start = rng.randrange(0, max(1, len(pool) - n))
        return pool[start : start + n]

    prefixes = [draw(shared_prefix_len) for _ in range(num_shared_prefixes)]
    prompts = []
    for i in range(num_requests):
        total = rng.randint(prompt_min, prompt_max)
        prefix = prefixes[i % num_shared_prefixes]
        tail = draw(max(1, total - len(prefix)))
        prompts.append(Prompt(f"preempt-{seed}-{i}", prefix + tail, max_tokens))
    print("prompt source:", "random token ids" if pool is None else "tokenized prose")
    return prompts


def scrape_metric(base_url: str, name: str) -> float:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
            body = resp.read().decode()
    except Exception:
        return float("nan")
    total = 0.0
    found = False
    for line in body.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            total += float(line.rsplit(" ", 1)[1])
            found = True
    return total if found else float("nan")


async def _one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: Prompt,
    sem: asyncio.Semaphore,
    want_cache_stats: bool,
    want_logprobs: bool,
) -> Completion:
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt.token_ids,
        "max_tokens": prompt.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "return_token_ids": True,
        "request_id": prompt.request_id,
    }
    if want_cache_stats:
        payload["kv_transfer_params"] = {"cached_token_stats": {}}
    if want_logprobs:
        payload["logprobs"] = 2
        payload["return_tokens_as_token_ids"] = True
    async with sem:
        start = time.monotonic()
        try:
            async with session.post(f"{url}/v1/completions", json=payload) as resp:
                body = await resp.json()
                if resp.status != 200:
                    return Completion(
                        prompt.request_id, [], "", "", 0.0, error=json.dumps(body)[:300]
                    )
        except Exception as exc:  # noqa: BLE001 - report, do not crash the run
            return Completion(prompt.request_id, [], "", "", 0.0, error=repr(exc))
        latency = time.monotonic() - start
    choice = body["choices"][0]
    stats = (body.get("kv_transfer_params") or {}).get("cached_token_stats") or {}
    top2: list[tuple[int, float, int, float]] = []
    logprobs = choice.get("logprobs") or {}
    for entry in logprobs.get("top_logprobs") or []:
        ranked = sorted(
            ((_token_id(k), float(v)) for k, v in entry.items()), key=lambda t: -t[1]
        )
        if len(ranked) >= 2:
            top2.append((ranked[0][0], ranked[0][1], ranked[1][0], ranked[1][1]))
        elif ranked:
            top2.append((ranked[0][0], ranked[0][1], -1, float("-inf")))
    return Completion(
        request_id=prompt.request_id,
        token_ids=list(choice.get("token_ids") or []),
        text=choice.get("text", ""),
        finish_reason=choice.get("finish_reason", ""),
        latency_s=latency,
        lmcache_cached_tokens=stats.get("num_lmcache_cached_tokens"),
        top2=top2,
    )


def _token_id(key: str) -> int:
    """``return_tokens_as_token_ids`` renders keys as ``token_id:123``."""
    if key.startswith("token_id:"):
        return int(key.split(":", 1)[1])
    return -1


async def run_pass(
    name: str,
    url: str,
    model: str,
    prompts: list[Prompt],
    concurrency: int,
    want_cache_stats: bool,
    want_logprobs: bool = False,
) -> PassResult:
    result = PassResult(name=name)
    result.preemptions_before = scrape_metric(url, PREEMPTION_METRIC)
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    start = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _one(session, url, model, p, sem, want_cache_stats, want_logprobs)
            for p in prompts
        ]
        for coro in asyncio.as_completed(tasks):
            c = await coro
            result.completions[c.request_id] = c
    result.wall_s = time.monotonic() - start
    result.preemptions_after = scrape_metric(url, PREEMPTION_METRIC)
    return result


@dataclass
class Divergence:
    request_id: str
    position: int
    ref_token: int
    got_token: int
    gap: float  # reference top-1 minus top-2 logprob at ``position``
    got_is_runner_up: bool
    cached_tokens: int | None

    def benign(self, near_tie_gap: float) -> bool:
        """Numerical noise flips a near-tie to the runner-up; nothing else."""
        return self.got_is_runner_up and self.gap <= near_tie_gap

    def describe(self, total: int) -> str:
        kind = "near-tie" if self.got_is_runner_up else "not in top-2"
        return (
            f"{self.request_id}: diverges at output token {self.position}/{total} "
            f"ref={self.ref_token} got={self.got_token} top1-top2 gap={self.gap:.4f} "
            f"({kind}, lmcache_cached_tokens={self.cached_tokens})"
        )


def compare(
    reference: PassResult, other: PassResult
) -> tuple[list[Divergence], list[str]]:
    """Classify every request whose token ids differ from the reference.

    Returns the divergences plus lines for requests that are missing or
    errored (always hard failures).
    """
    divergences: list[Divergence] = []
    hard: list[str] = []
    for rid, ref in reference.completions.items():
        got = other.completions.get(rid)
        if got is None or got.error:
            hard.append(f"{rid}: missing or errored ({got.error if got else 'absent'})")
            continue
        if ref.token_ids == got.token_ids:
            continue
        first = next(
            (
                i
                for i, (a, b) in enumerate(
                    zip(ref.token_ids, got.token_ids, strict=False)
                )
                if a != b
            ),
            min(len(ref.token_ids), len(got.token_ids)),
        )
        ref_tok = ref.token_ids[first] if first < len(ref.token_ids) else -1
        got_tok = got.token_ids[first] if first < len(got.token_ids) else -1
        gap = float("inf")
        runner_up = False
        if first < len(ref.top2):
            _t1, lp1, t2, lp2 = ref.top2[first]
            gap = lp1 - lp2
            runner_up = got_tok == t2
        divergences.append(
            Divergence(
                rid, first, ref_tok, got_tok, gap, runner_up, got.lmcache_cached_tokens
            )
        )
    return divergences, hard


def summarize(result: PassResult) -> str:
    n = len(result.completions)
    errors = len(result.errors)
    lengths = [c.finish_reason == "length" for c in result.completions.values()]
    cached = [
        c.lmcache_cached_tokens
        for c in result.completions.values()
        if c.lmcache_cached_tokens is not None
    ]
    lat = sorted(c.latency_s for c in result.completions.values() if not c.error)
    p99 = lat[int(0.99 * (len(lat) - 1))] if lat else float("nan")
    cache_line = ""
    if cached:
        cache_line = f" lmcache_cached_tokens(mean)={sum(cached) / len(cached):.1f}"
    return (
        f"[{result.name}] requests={n} errors={errors} finish=length:{sum(lengths)} "
        f"preemptions={result.preemptions:.0f} wall={result.wall_s:.1f}s "
        f"p99_latency={p99:.2f}s{cache_line}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-url", required=True)
    ap.add_argument("--lmcache-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--hot-concurrency", type=int, default=64)
    ap.add_argument("--replay-concurrency", type=int, default=4)
    ap.add_argument("--prompt-min", type=int, default=256)
    ap.add_argument("--prompt-max", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--shared-prefixes", type=int, default=4)
    ap.add_argument("--shared-prefix-len", type=int, default=192)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="hot+replay rounds")
    ap.add_argument(
        "--min-replay-hit-fraction",
        type=float,
        default=0.8,
        help="replay pass must report at least this fraction of prompt tokens "
        "as served from LMCache, else the pass proves nothing",
    )
    ap.add_argument("--require-preemption", action="store_true", default=True)
    ap.add_argument(
        "--no-require-preemption", dest="require_preemption", action="store_false"
    )
    ap.add_argument(
        "--near-tie-gap",
        type=float,
        default=0.1,
        help="a divergence is benign only if the reference's top-1/top-2 logprob "
        "gap at that position is at most this many nats and the other run "
        "produced the runner-up token",
    )
    ap.add_argument(
        "--noise-floor",
        action="store_true",
        default=True,
        help="also run the baseline at replay concurrency and report its own "
        "divergence count against the reference (informational)",
    )
    ap.add_argument("--no-noise-floor", dest="noise_floor", action="store_false")
    ap.add_argument("--output-dir", type=Path, default=Path("preemption_results"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_workload(
        model=args.model,
        num_requests=args.num_requests,
        prompt_min=args.prompt_min,
        prompt_max=args.prompt_max,
        max_tokens=args.max_tokens,
        num_shared_prefixes=args.shared_prefixes,
        shared_prefix_len=args.shared_prefix_len,
        vocab=args.vocab,
        seed=args.seed,
    )
    demand = sum(len(p.token_ids) + p.max_tokens for p in prompts)
    print(f"workload: {len(prompts)} requests, total KV demand {demand} tokens")

    failures: list[str] = []

    baseline = asyncio.run(
        run_pass(
            "baseline",
            args.baseline_url,
            args.model,
            prompts,
            args.hot_concurrency,
            False,
            want_logprobs=True,
        )
    )
    print(summarize(baseline))
    if baseline.errors:
        failures.append(f"baseline had {len(baseline.errors)} errors")
    if not all(c.token_ids for c in baseline.completions.values() if not c.error):
        failures.append(
            "baseline returned no token ids; does this vLLM support return_token_ids?"
        )
    if not all(c.top2 for c in baseline.completions.values() if not c.error):
        print(
            "WARNING: baseline returned no top-2 logprobs; "
            "every divergence counts as hard"
        )

    def judge(label: str, other: PassResult) -> None:
        divergences, hard = compare(baseline, other)
        benign = [d for d in divergences if d.benign(args.near_tie_gap)]
        real = [d for d in divergences if not d.benign(args.near_tie_gap)]
        print(
            f"    {label}: {len(divergences)} divergent requests "
            f"({len(benign)} near-tie, {len(real)} hard), {len(hard)} missing/errored"
        )
        for d in divergences[:10]:
            print("      ", d.describe(args.max_tokens))
        for line in hard:
            failures.append(f"{label}: {line}")
        if real:
            failures.append(f"{label}: {len(real)} hard divergences from baseline")

    if args.noise_floor:
        floor = asyncio.run(
            run_pass(
                "baseline-noise-floor",
                args.baseline_url,
                args.model,
                prompts,
                args.replay_concurrency,
                False,
            )
        )
        print(summarize(floor))
        divergences, _hard = compare(baseline, floor)
        benign = sum(d.benign(args.near_tie_gap) for d in divergences)
        print(
            f"    A/A noise floor (baseline hot vs baseline replay-concurrency): "
            f"{len(divergences)} divergent, {benign} near-tie, "
            f"{len(divergences) - benign} would count as hard"
        )
        for d in divergences[:5]:
            print("      ", d.describe(args.max_tokens))

    for rnd in range(args.repeats):
        hot = asyncio.run(
            run_pass(
                f"lmcache-hot-{rnd}",
                args.lmcache_url,
                args.model,
                prompts,
                args.hot_concurrency,
                True,
            )
        )
        print(summarize(hot))
        if hot.errors:
            failures.append(f"T0 {hot.name}: {len(hot.errors)} errored requests")
        if args.require_preemption and not hot.preemptions > 0:
            failures.append(
                f"T0 {hot.name}: no preemptions observed; workload too small"
            )
        judge(f"T2 {hot.name}", hot)

        replay = asyncio.run(
            run_pass(
                f"lmcache-replay-{rnd}",
                args.lmcache_url,
                args.model,
                prompts,
                args.replay_concurrency,
                True,
            )
        )
        print(summarize(replay))
        if replay.errors:
            failures.append(f"T3 {replay.name}: {len(replay.errors)} errored requests")
        cached = [
            (c.lmcache_cached_tokens or 0) / len(p.token_ids)
            for p in prompts
            if (c := replay.completions.get(p.request_id)) is not None
        ]
        hit_fraction = sum(cached) / len(cached) if cached else 0.0
        print(f"    replay LMCache hit fraction of prompt tokens: {hit_fraction:.2f}")
        if hit_fraction < args.min_replay_hit_fraction:
            failures.append(
                f"T3 {replay.name}: replay hit fraction {hit_fraction:.2f} < "
                f"{args.min_replay_hit_fraction}; the replay did not exercise the cache"
            )
        judge(f"T3 {replay.name}", replay)

        for res in (hot, replay):
            with open(args.output_dir / f"{res.name}.json", "w") as f:
                json.dump(
                    {rid: c.__dict__ for rid, c in res.completions.items()}, f, indent=1
                )
    with open(args.output_dir / "baseline.json", "w") as f:
        json.dump(
            {rid: c.__dict__ for rid, c in baseline.completions.items()}, f, indent=1
        )

    if failures:
        print("FAILED:")
        for line in failures:
            print("  -", line)
        return 1
    print("PASSED: T0 no crash, T2 hot outputs match, T3 replay outputs match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
