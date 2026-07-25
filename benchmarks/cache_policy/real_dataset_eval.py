# SPDX-License-Identifier: Apache-2.0
"""
Statistically robust real-data (ShareGPT) evaluation of cache policies.

Reuses the existing simulator (``lmcache.tools.cache_policy_bench.runner``)
and the ShareGPT loader
(``lmcache.tools.cache_policy_bench.sharegpt_workload``) unchanged; this
script only adds repeated bootstrap-resampled runs + confidence intervals
and a corpus-scale sweep on top.

Prerequisite -- prepare the corpus once (see
``lmcache/tools/cache_policy_bench/sharegpt_workload.py`` module docstring
or ``benchmarks/cache_policy/README.md`` for the exact commands):
``benchmarks/multi_round_qa/ShareGPT.json`` must exist.

Usage::

    python benchmarks/cache_policy/real_dataset_eval.py \\
        --sharegpt-path benchmarks/multi_round_qa/ShareGPT.json \\
        -o benchmarks/cache_policy/results/real_data
"""

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import csv
import json

# First Party
from benchmarks.cache_policy.stats import bootstrap_ci
from lmcache.tools.cache_policy_bench.cost_model import CostModel, CostModelConfig
from lmcache.tools.cache_policy_bench.runner import (
    DEFAULT_KV_BYTES_PER_CHUNK,
    DEFAULT_POLICIES,
    run_workload,
)
from lmcache.tools.cache_policy_bench.sharegpt_workload import (
    load_sharegpt_conversations,
    requests_from_conversations,
)

_MIB = 2**20
DEFAULT_SCALES: list[Optional[int]] = [500, 2000, 5000]
DEFAULT_CACHE_SIZES_MIB: list[float] = [50.0, 100.0, 200.0]
DEFAULT_N_REPEATS = 6


@dataclass
class AggregatedResult:
    """Bootstrap-CI-aggregated metrics for one (policy, scale, cache-size) cell."""

    policy_name: str
    max_conversations: str
    cache_capacity_bytes: int
    n_repeats: int
    hit_rate_mean: float
    hit_rate_ci_lo: float
    hit_rate_ci_hi: float
    latency_p95_mean: float
    latency_p95_ci_lo: float
    latency_p95_ci_hi: float
    eviction_count_mean: float
    eviction_count_ci_lo: float
    eviction_count_ci_hi: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_repeated(
    sharegpt_path: Path,
    scales: list[Optional[int]],
    cache_sizes_bytes: list[int],
    policies: list[str],
    n_repeats: int,
    chunk_size: int = 256,
) -> tuple[list[dict[str, Any]], list[AggregatedResult]]:
    """
    Run the full (policy x scale x cache-size) grid, ``n_repeats`` times
    each with a fresh bootstrap resample of the corpus, and aggregate.

    Returns:
        ``(raw_rows, aggregated)`` -- the raw per-repeat rows (as plain
        dicts, for full transparency) and the bootstrap-CI-aggregated
        table used for reporting.
    """
    cost_model = CostModel(CostModelConfig())
    raw_rows: list[dict[str, Any]] = []
    aggregated: list[AggregatedResult] = []

    # Parse the corpus once; each repeat below only resamples/rebuilds
    # Request objects from the in-memory list, not from disk.
    conversations = load_sharegpt_conversations(sharegpt_path)

    for scale in scales:
        scale_label = "full" if scale is None else str(scale)
        for policy_name in policies:
            for cache_bytes in cache_sizes_bytes:
                hit_rates: list[float] = []
                p95s: list[float] = []
                evictions: list[float] = []
                for repeat in range(n_repeats):
                    requests = requests_from_conversations(
                        conversations,
                        chunk_size=chunk_size,
                        max_conversations=scale,
                        seed=repeat,
                    )
                    result = run_workload(
                        policy_name,
                        requests,
                        cache_bytes,
                        DEFAULT_KV_BYTES_PER_CHUNK,
                        cost_model,
                        workload_name=f"sharegpt[{scale_label}]",
                    )
                    hit_rates.append(result.token_hit_rate)
                    p95s.append(result.latency_p95_seconds)
                    evictions.append(float(result.eviction_count))
                    row = result.to_dict()
                    row["max_conversations"] = scale_label
                    row["repeat"] = repeat
                    raw_rows.append(row)

                hr_mean, hr_lo, hr_hi = bootstrap_ci(hit_rates)
                p95_mean, p95_lo, p95_hi = bootstrap_ci(p95s)
                ev_mean, ev_lo, ev_hi = bootstrap_ci(evictions)
                aggregated.append(
                    AggregatedResult(
                        policy_name=policy_name,
                        max_conversations=scale_label,
                        cache_capacity_bytes=cache_bytes,
                        n_repeats=n_repeats,
                        hit_rate_mean=hr_mean,
                        hit_rate_ci_lo=hr_lo,
                        hit_rate_ci_hi=hr_hi,
                        latency_p95_mean=p95_mean,
                        latency_p95_ci_lo=p95_lo,
                        latency_p95_ci_hi=p95_hi,
                        eviction_count_mean=ev_mean,
                        eviction_count_ci_lo=ev_lo,
                        eviction_count_ci_hi=ev_hi,
                    )
                )
                print(
                    f"scale={scale_label:>6s} policy={policy_name:10s} "
                    f"cache={cache_bytes / _MIB:6.0f}MiB "
                    f"hit_rate={hr_mean:.3f} [{hr_lo:.3f},{hr_hi:.3f}] "
                    f"p95={p95_mean * 1000:.2f}ms"
                )

    return raw_rows, aggregated


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt-path", required=True, type=Path)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=DEFAULT_SCALES,
        help="Conversation-count scales to sweep; pass 'full' for the whole corpus",
    )
    parser.add_argument(
        "--cache-sizes-mib", nargs="+", type=float, default=DEFAULT_CACHE_SIZES_MIB
    )
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "-o", "--output-dir", default="benchmarks/cache_policy/results/real_data"
    )
    args = parser.parse_args()

    scales: list[Optional[int]] = [None if s == "full" else int(s) for s in args.scales]
    cache_sizes_bytes = [int(mib * _MIB) for mib in args.cache_sizes_mib]

    raw_rows, aggregated = run_repeated(
        args.sharegpt_path,
        scales,
        cache_sizes_bytes,
        args.policies,
        args.repeats,
        chunk_size=args.chunk_size,
    )

    out_dir = Path(args.output_dir)
    _write_json(raw_rows, out_dir / "real_dataset_raw.json")
    _write_csv(raw_rows, out_dir / "real_dataset_raw.csv")
    agg_dicts = [a.to_dict() for a in aggregated]
    _write_json(agg_dicts, out_dir / "real_dataset_ci.json")
    _write_csv(agg_dicts, out_dir / "real_dataset_ci.csv")
    print(
        f"\nWrote {len(raw_rows)} raw rows and "
        f"{len(aggregated)} aggregated rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
