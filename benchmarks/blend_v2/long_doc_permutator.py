# SPDX-License-Identifier: Apache-2.0
# Adapted from long_doc_qa.py for the blend_v2 permutation stress test.
#
# Workload design (see README.md):
#   Each request is: [System Prompt] + [Doc_i1] + [Doc_i2] + ... + [Doc_iN]
#   where (i1, i2, ..., iN) is one permutation of the N contexts.
#
#   Stress test axes (controlled by CLI args):
#     1. Blended Context Boundaries  -> --num-contexts
#     2. Eviction                    -> --num-permutations
#     3. Chunk Homogeneity           -> --vocab-size
#     4. Prefix Domination           -> --system-prompt-length

# Standard
from dataclasses import dataclass
import argparse
import asyncio
import itertools
import json
import math
import os
import random
import sys
import time

# Third Party
from openai import AsyncOpenAI
import pandas as pd

# ---------------------------------------------------------------------------
# Globals (set in __main__)
# ---------------------------------------------------------------------------
OUTPUT_FILE = None

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RequestStats:
    prompt_id: int  # linear index in the request list
    permutation_id: int  # which permutation (index into the enumerated set)
    request_start: float
    ttft: float
    request_end: float
    successful: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def generate_vocab_pool(size: int, seed: int = 42) -> list[str]:
    """Generate a vocabulary pool of `size` unique pseudo-words.

    For small sizes we use hand-picked short words; for larger sizes we
    deterministically generate synthetic words so every token is unique.
    """
    rng = random.Random(seed)
    # Build synthetic words: base + suffix to guarantee uniqueness
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    pool: set[str] = set()
    while len(pool) < size:
        length = rng.randint(3, 7)
        word = ""
        for j in range(length):
            if j % 2 == 0:
                word += rng.choice(consonants)
            else:
                word += rng.choice(vowels)
        # append a number suffix to guarantee uniqueness
        word = f"{word}{len(pool)}"
        pool.add(word)
    return sorted(pool)


def generate_system_prompt(length: int, seed: int = 42) -> str:
    """Generate a deterministic shared system prompt of ~`length` tokens."""
    rng = random.Random(seed)
    # Use a small fixed pool so the system prompt is repetitive (realistic)
    words = [
        "the",
        "system",
        "will",
        "process",
        "your",
        "request",
        "and",
        "provide",
        "an",
        "answer",
        "based",
        "on",
        "context",
    ]
    return " ".join(rng.choices(words, k=length))


def generate_contexts(
    num_contexts: int,
    length: int,
    vocab_pool: list[str],
    seed: int = 123,
) -> list[str]:
    """Generate `num_contexts` unique context blocks of ~`length` tokens each.

    Each context draws from `vocab_pool` with a per-context seed so the
    token sequences genuinely diverge.
    """
    contexts = []
    for i in range(num_contexts):
        rng = random.Random(seed + i)
        body = " ".join(rng.choices(vocab_pool, k=length))
        contexts.append(body)
    return contexts


def enumerate_permutations(
    num_contexts: int,
    num_permutations: int,
    seed: int = 0,
) -> list[tuple[int, ...]]:
    """Enumerate up to `num_permutations` distinct permutations of range(num_contexts).

    Uses itertools.permutations (lazy) and stops after collecting enough.
    If num_permutations >= N!, returns all permutations.
    """
    total_possible = math.factorial(num_contexts)
    if num_permutations >= total_possible:
        return list(itertools.permutations(range(num_contexts)))

    # For large N where N! >> num_permutations, sample randomly
    # to avoid iterating through too many permutations
    if total_possible > num_permutations * 10:
        rng = random.Random(seed)
        seen: set[tuple[int, ...]] = set()
        indices = list(range(num_contexts))
        while len(seen) < num_permutations:
            perm = tuple(rng.sample(indices, len(indices)))
            seen.add(perm)
        return sorted(seen)

    # For smaller N!, enumerate and pick first num_permutations
    result = []
    for perm in itertools.permutations(range(num_contexts)):
        result.append(perm)
        if len(result) >= num_permutations:
            break
    return result


def build_request_list(
    system_prompt: str,
    contexts: list[str],
    permutations: list[tuple[int, ...]],
) -> list[tuple[list[dict], int]]:
    """Build the full request list from permutations of contexts.

    Each request concatenates all contexts in the permutation order into
    a single user message, preceded by the system prompt.

    Returns list of (messages, permutation_index).
    """
    requests = []
    for perm_idx, perm in enumerate(permutations):
        concatenated = "\n\n".join(contexts[i] for i in perm)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": concatenated},
        ]
        requests.append((messages, perm_idx))
    return requests


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------


def write_resp(text: str):
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "a") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


async def process_single_request(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    request_index: int,
    total_requests: int,
    permutation_id: int,
    output_len: int,
    semaphore: asyncio.Semaphore,
) -> RequestStats:
    """Send one chat completion request and record timing stats."""
    async with semaphore:
        write_resp(
            f"\n--- Request {request_index + 1}/{total_requests} "
            f"(perm={permutation_id}) ---\n"
        )
        start_time = time.time()
        first_token_time = None

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=output_len,
            temperature=0.0,
            stream_options={"include_usage": True},
        )

        chunks = []
        prompt_tokens = 0
        completion_tokens = 0
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                chunks.append(delta.content)

        end_time = time.time()
        write_resp(f"Response: {''.join(chunks)}\n")

        ttft = (first_token_time - start_time) if first_token_time is not None else -1
        return RequestStats(
            prompt_id=request_index,
            permutation_id=permutation_id,
            request_start=start_time,
            ttft=ttft,
            request_end=end_time,
            successful=ttft > 0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


async def run_benchmark(
    client: AsyncOpenAI,
    model: str,
    request_list: list[tuple[list[dict], int]],
    output_len: int,
    max_inflight: int,
) -> list[RequestStats]:
    """Execute all requests with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_inflight)
    tasks = [
        process_single_request(
            client=client,
            model=model,
            messages=messages,
            request_index=i,
            total_requests=len(request_list),
            permutation_id=perm_id,
            output_len=output_len,
            semaphore=semaphore,
        )
        for i, (messages, perm_id) in enumerate(request_list)
    ]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def relative_time(df: pd.DataFrame, start_time: float):
    df["request_start"] = df["request_start"] - start_time
    df["request_end"] = df["request_end"] - start_time
    df["ttft_time"] = df["request_start"] + df["ttft"]


def ttft_stats(series: pd.Series) -> dict:
    """Compute summary statistics for a TTFT series."""
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "std": float(s.std()),
    }


def print_results(df: pd.DataFrame, wall_time: float, label: str):
    ok = df.query("successful == True")
    stats = ttft_stats(ok["ttft"])
    CSI = "\x1b["
    RESET = CSI + "0m"
    print(f"\n{CSI}36;1m=== {label} ==={RESET}")
    print(f"  Total requests : {len(df)}")
    print(f"  Successful     : {len(ok)}")
    print(f"  Wall time      : {wall_time:.3f}s")
    if stats:
        print(f"{CSI}32m  TTFT mean      : {stats['mean']:.3f}s{RESET}")
        print(f"{CSI}32m  TTFT median    : {stats['median']:.3f}s{RESET}")
        print(f"  TTFT min       : {stats['min']:.3f}s")
        print(f"  TTFT max       : {stats['max']:.3f}s")
        print(f"  TTFT p95       : {stats['p95']:.3f}s")
        print(f"  TTFT p99       : {stats['p99']:.3f}s")
        print(f"  TTFT std       : {stats['std']:.3f}s")
    if len(ok) > 0:
        total_tokens = ok["prompt_tokens"].sum() + ok["completion_tokens"].sum()
        print(f"  Throughput     : {len(ok) / wall_time:.2f} req/s")
        print(f"  Throughput     : {total_tokens / wall_time:.2f} tok/s")


def plot_ttft_distribution(df: pd.DataFrame, filename: str = "ttft_distribution.png"):
    """Save a histogram + box plot of TTFT values."""
    # Third Party
    import matplotlib

    matplotlib.use("Agg")
    # Third Party
    import matplotlib.pyplot as plt

    ok = df.query("successful == True")
    if ok.empty:
        return

    ttft = ok["ttft"]
    stats = ttft_stats(ttft)

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Histogram
    ax_hist.hist(
        ttft, bins=min(50, len(ttft)), color="steelblue", edgecolor="white", alpha=0.85
    )
    ax_hist.axvline(
        stats["mean"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={stats['mean']:.3f}s",
    )
    ax_hist.axvline(
        stats["median"],
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"median={stats['median']:.3f}s",
    )
    ax_hist.axvline(
        stats["p95"],
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"p95={stats['p95']:.3f}s",
    )
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=9)
    ax_hist.set_title("TTFT Distribution (Permutation Benchmark)")

    # Box plot
    ax_box.boxplot(
        ttft,
        vert=False,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.5),
    )
    ax_box.set_xlabel("TTFT (s)")
    ax_box.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Distribution plot saved to {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(args):
    random.seed(args.seed)

    base_url = (
        args.base_url or f"http://{args.host or 'localhost'}:{args.port or 8000}/v1"
    )
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    print(f"Using base URL: {base_url}")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=None)

    model = args.model
    if model == "auto":
        models = await client.models.list()
        model = models.data[0].id
        print(f"Auto-selected model: {model}")

    # ---- Generate vocab pool & contexts ----
    vocab_pool = generate_vocab_pool(args.vocab_size, seed=args.seed)
    print(f"Vocab pool: {args.vocab_size} words")

    system_prompt = generate_system_prompt(args.system_prompt_length, seed=args.seed)
    contexts = generate_contexts(
        args.num_contexts, args.context_length, vocab_pool, seed=args.seed + 1
    )

    # ---- Enumerate permutations ----
    permutations = enumerate_permutations(
        args.num_contexts, args.num_permutations, seed=args.seed
    )
    print(
        f"Enumerated {len(permutations)} permutations of {args.num_contexts} contexts "
        f"(max {math.factorial(args.num_contexts)} possible)"
    )

    # ---- Build request list ----
    request_list = build_request_list(system_prompt, contexts, permutations)
    total = len(request_list)
    print(f"Built {total} requests")
    print(f"  System prompt: ~{args.system_prompt_length} tokens")
    print(f"  Each context:  ~{args.context_length} tokens")
    print(
        f"  Total per req: ~"
        f"{args.system_prompt_length + args.num_contexts * args.context_length}"
        f" tokens"
    )

    # max_inflight == 0 means unlimited (flood)
    max_inflight = (
        args.max_inflight_requests if args.max_inflight_requests > 0 else total
    )

    if args.output_dir and args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Dummy warmup (1 request to get the engine started) ----
    print("\n--- Dummy warmup (1 request) ---")
    dummy_prompt = " ".join(["warmup"] * 500)
    dummy_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": dummy_prompt},
    ]
    dummy_list = [(dummy_messages, -1)]
    await run_benchmark(client, model, dummy_list, args.output_len, 1)
    print("Warmup complete.")

    if args.sleep_after_warmup > 0:
        print(f"Sleeping {args.sleep_after_warmup}s after warmup...")
        await asyncio.sleep(args.sleep_after_warmup)

    # ---- Clear LMCache log before benchmark ----
    if args.lmcache_log and os.path.exists(args.lmcache_log):
        with open(args.lmcache_log, "w") as f:
            pass
        print(f"Cleared {args.lmcache_log} before benchmark")

    # ---- Run benchmark ----
    print(f"\n--- Benchmark: {total} requests (max_inflight={max_inflight}) ---")
    bench_start = time.time()
    bench_stats = await run_benchmark(
        client, model, request_list, args.output_len, max_inflight
    )
    bench_wall = time.time() - bench_start

    bench_df = pd.DataFrame([s.__dict__ for s in bench_stats])
    relative_time(bench_df, bench_start)
    bench_df.to_csv(os.path.join(args.output_dir, "benchmark_round.csv"), index=False)
    print_results(bench_df, bench_wall, "BENCHMARK RESULTS")
    plot_ttft_distribution(
        bench_df, os.path.join(args.output_dir, "ttft_distribution.png")
    )

    # ---- Build summary ----
    ok = bench_df.query("successful == True")
    stats = ttft_stats(ok["ttft"])
    total_tokens = (
        float(ok["prompt_tokens"].sum() + ok["completion_tokens"].sum())
        if not ok.empty
        else 0.0
    )
    summary = {
        "num_contexts": args.num_contexts,
        "context_length": args.context_length,
        "system_prompt_length": args.system_prompt_length,
        "vocab_size": args.vocab_size,
        "num_permutations": len(permutations),
        "total_requests": len(bench_df),
        "successful_requests": len(ok),
        "wall_time": round(bench_wall, 3),
        "throughput_rps": round(len(ok) / bench_wall, 2) if bench_wall > 0 else 0,
        "throughput_tps": round(total_tokens / bench_wall, 2) if bench_wall > 0 else 0,
        **{f"ttft_{k}": round(v, 4) for k, v in stats.items()},
    }

    if args.json_output:
        print(json.dumps(summary))

    # ---- LMCache Log Parsing ----
    cache_summary = {}
    if args.lmcache_log:
        if os.path.exists(args.lmcache_log):
            print(f"\n--- Parsing LMCache Log ({args.lmcache_log}) ---")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            # Third Party
            from parse_lmcache_log import parse_log
            from parse_lmcache_log import report as cache_report

            raw = parse_log(args.lmcache_log, mode=args.lmcache_mode)
            cache_summary = cache_report(
                raw, args.lmcache_workers, args.lmcache_mode, args.output_dir
            )
        else:
            print(f"\n  WARNING: LMCache log not found at {args.lmcache_log}")

    # ---- Write combined summary.txt ----
    summary.update(cache_summary)
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  Full summary saved to {summary_path}")


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Blend V2 permutation stress test: send permutations of "
        "context documents to stress-test blended KV cache reuse."
    )

    # Prompt shape
    parser.add_argument(
        "--num-contexts",
        type=int,
        default=5,
        help="Number of unique context documents (default: 5)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=5000,
        help="Length of each context in tokens (default: 5000)",
    )
    parser.add_argument(
        "--system-prompt-length",
        type=int,
        default=1000,
        help="Length of the shared system prompt in tokens (default: 1000). "
        "Use 0 for no system prompt.",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=10,
        help="Number of distinct permutations to send (default: 10). "
        "Capped at N! where N = --num-contexts.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Size of the word vocabulary used to build contexts (default: 8000). "
        "Smaller values increase chunk hash collision risk.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1,
        help="Decode tokens per request (default: 1)",
    )

    # Execution mode
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=1,
        help="Max concurrent requests. 0 = flood all at once (throughput mode), "
        "1 = sequential (TTFT mode). Default: 1",
    )

    # Server
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default="auto")

    # Warmup
    parser.add_argument(
        "--sleep-after-warmup",
        type=float,
        default=0.0,
        help="Seconds to sleep after the dummy warmup request",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save artifacts (csv, png). Created if needed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File to write response text to (default: stdout)",
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Print JSON summary line at end"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    # LMCache log parsing
    parser.add_argument(
        "--lmcache-log",
        type=str,
        default=None,
        help="Path to lmcache.log to parse after benchmark",
    )
    parser.add_argument(
        "--lmcache-workers",
        type=int,
        default=4,
        help="Number of parallel TP workers for log parsing",
    )
    parser.add_argument(
        "--lmcache-mode",
        type=str,
        choices=["blend", "prefix"],
        default="blend",
        help="Mode for log parsing",
    )

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.output:
        if args.output_dir and args.output_dir != ".":
            OUTPUT_FILE = os.path.join(args.output_dir, args.output)
        else:
            OUTPUT_FILE = args.output
    asyncio.run(main(args))
