# SPDX-License-Identifier: Apache-2.0
"""Measure isolated LRU victim selection, excluding key creation and deletion.

Run the same script on the base and candidate revisions, with the same CPU
affinity, interpreter, and arguments. For example, from the repository root::

    PYTHONPATH=. python benchmarks/microbenchmark/isolated_lru_benchmark.py \
        --sizes 1000 10000 100000 --output /tmp/isolated-lru.json

Timings include the public policy call and disposal of its result. Peak Python
allocation is measured separately with tracemalloc, outside the timed samples.
This is a CPU policy microbenchmark, not an inference throughput benchmark.
"""

# Standard
from collections.abc import Callable
from functools import partial
from pathlib import Path
import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
import tracemalloc

# First Party
from lmcache.v1.common.object_key import ObjectKey
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)


def _measure(
    select: Callable[[], object], rounds: int, loops: int
) -> tuple[list[float], int]:
    """Warm up selection, then measure latency and separate peak allocation.

    Args:
        select: Non-mutating policy call to benchmark.
        rounds: Number of timed batches.
        loops: Calls per batch; results are released between calls.

    Returns:
        Per-call batch latencies in microseconds and peak allocated bytes.
    """
    for _ in range(5):
        select()
    gc.collect()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(loops):
            select()
        samples.append((time.perf_counter_ns() - start) / loops / 1000)

    tracemalloc.start()
    select()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return samples, peak


def main() -> None:
    """Benchmark bucket sizes, eviction ratios, and eligibility filters.

    Command-line arguments select bucket sizes, sample counts, and an optional
    JSON output file. Prints median and batch p95 per-call latency plus peak
    allocation. JSON retains every sample and the environment for comparison.

    Raises:
        AssertionError: If selection returns incorrect victims or mutates keys.
        OSError: If the output file cannot be written or git cannot be invoked.
        subprocess.CalledProcessError: If the working directory is not in git.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.sizes, args.rounds, args.loops) <= 0:
        parser.error("sizes, rounds, and loops must be positive")

    results: list[dict[str, object]] = []
    for size in args.sizes:
        keys = [
            ObjectKey(
                chunk_hash=ObjectKey.IntHash2Bytes(i),
                model_name="benchmark",
                kv_rank=0,
                cache_salt="tenant",
            )
            for i in range(size)
        ]
        policy = IsolatedLRUEvictionPolicy()
        policy.on_keys_created(keys)
        oldest_first = list(reversed(keys))
        pinned = set(oldest_first[: size // 10])

        def is_unpinned(key: ObjectKey, pinned: set[ObjectKey] = pinned) -> bool:
            """Return whether a key is outside this bucket's pinned prefix."""
            return key not in pinned

        cases: list[tuple[str, float, Callable[[ObjectKey], bool] | None]] = [
            ("zero", 0.0, None),
            ("one", 1 / size, None),
            ("up_to_64", min(64 / size, 1.0), None),
            ("one_percent", 0.01, None),
            ("twenty_percent", 0.2, None),
            ("full", 1.0, None),
            ("pinned_head", 0.01, is_unpinned),
            ("reject_all", 0.2, lambda key: False),
        ]
        for label, ratio, eligible in cases:
            target = max(1, int(size * ratio)) if ratio > 0 else 0
            expected = [
                key for key in oldest_first if eligible is None or eligible(key)
            ][:target]

            select = partial(
                policy.get_eviction_actions,
                ratio,
                key_eligible_filter=eligible,
                cache_salt="tenant",
            )
            actions = select()
            assert [key for action in actions for key in action.keys] == expected
            samples, peak = _measure(select, args.rounds, args.loops)
            assert policy.get_num_tracked_keys("tenant") == size
            median = statistics.median(samples)
            p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
            results.append(
                {
                    "size": size,
                    "case": label,
                    "ratio": ratio,
                    "selected": len(expected),
                    "median_us": median,
                    "batch_p95_us": p95,
                    "peak_bytes": peak,
                    "samples_us": samples,
                }
            )
            print(
                f"{size:>9} {label:<16} median={median:>12.3f}us "
                f"batch_p95={p95:>12.3f}us peak={peak:>10}B",
                flush=True,
            )

    if args.output is not None:
        args.output.write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "affinity": sorted(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity")
                    else None,
                    "head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "working_tree_status": subprocess.check_output(
                        ["git", "status", "--short"], text=True
                    ),
                    "rounds": args.rounds,
                    "loops": args.loops,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
