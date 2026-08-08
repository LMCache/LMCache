# SPDX-License-Identifier: Apache-2.0
"""
Direction-finding comparison: baseline policies vs. the three candidate
improvements (score rebalancing, admission control, hierarchical caching),
across both the synthetic suite and a time-boxed real-data (ShareGPT) leg.

See ``docs/design/v1/storage_backend/cache_policy/cost-aware-policy-eval.md``,
"Direction-finding experiment" section, for the motivating diagnosis and
the resulting recommendation.

Usage::

    python benchmarks/cache_policy/experiments/compare_directions.py \\
        --synthetic -o benchmarks/cache_policy/results/experiments

    python benchmarks/cache_policy/experiments/compare_directions.py \\
        --real-data --sharegpt-path benchmarks/multi_round_qa/ShareGPT.json \\
        -o benchmarks/cache_policy/results/experiments
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import csv
import json

# First Party
from benchmarks.cache_policy.experiments._common import run_policy_instance_workload
from benchmarks.cache_policy.experiments.admission_control import (
    run_admission_controlled_workload,
)
from benchmarks.cache_policy.experiments.hierarchical_cache import (
    run_hierarchical_workload,
)
from benchmarks.cache_policy.experiments.variant_policies import (
    BlendedPolicy,
    FrequencyFirstPolicy,
)
from benchmarks.cache_policy.stats import bootstrap_ci
from lmcache.tools.cache_policy_bench.cost_model import CostModel, CostModelConfig
from lmcache.tools.cache_policy_bench.runner import BenchResult, run_workload
from lmcache.tools.cache_policy_bench.sharegpt_workload import (
    load_sharegpt_conversations,
    requests_from_conversations,
)
from lmcache.tools.cache_policy_bench.workloads import (
    Request,
    mixed_zipfian,
    multi_round_chat,
)

_MIB = 2**20
DEFAULT_KV_BYTES_PER_CHUNK = 256 * 1024
_TIER1_FRACTION = 0.2  # hierarchical: fast tier is 20% of the total budget


def _run_all_directions(
    requests: list[Request],
    cache_bytes: int,
    cost_model: CostModel,
    workload_name: str,
) -> list[BenchResult]:
    """Run baseline policies + all three candidate directions at one cell."""
    results: list[BenchResult] = []

    for policy_name in ("LRU", "LFU", "COST_AWARE"):
        results.append(
            run_workload(
                policy_name,
                requests,
                cache_bytes,
                DEFAULT_KV_BYTES_PER_CHUNK,
                cost_model,
                workload_name=workload_name,
            )
        )

    for label, policy in (
        ("freq_first", FrequencyFirstPolicy()),
        ("blended", BlendedPolicy()),
    ):
        results.append(
            run_policy_instance_workload(
                policy,
                label,
                requests,
                cache_bytes,
                DEFAULT_KV_BYTES_PER_CHUNK,
                cost_model,
                workload_name=workload_name,
            )
        )

    for base_policy in ("LRU", "COST_AWARE"):
        results.append(
            run_admission_controlled_workload(
                base_policy,
                requests,
                cache_bytes,
                DEFAULT_KV_BYTES_PER_CHUNK,
                cost_model,
                workload_name=workload_name,
            )
        )

    tier1_bytes = max(DEFAULT_KV_BYTES_PER_CHUNK, int(cache_bytes * _TIER1_FRACTION))
    tier2_bytes = max(DEFAULT_KV_BYTES_PER_CHUNK, cache_bytes - tier1_bytes)
    for tier1_policy in ("LRU", "COST_AWARE"):
        results.append(
            run_hierarchical_workload(
                tier1_policy,
                "LRU",
                requests,
                tier1_bytes,
                tier2_bytes,
                DEFAULT_KV_BYTES_PER_CHUNK,
                cost_model,
                workload_name=workload_name,
            )
        )

    return results


def run_synthetic_comparison() -> list[BenchResult]:
    cost_model = CostModel(CostModelConfig())
    workloads = {
        "mixed_zipfian": mixed_zipfian(3000, unique_prefixes=300, seed=1),
        "multi_round_chat": multi_round_chat(40, rounds_per_session=12),
    }
    cache_sizes_bytes = [int(mib * _MIB) for mib in (50, 100, 200)]

    results: list[BenchResult] = []
    for workload_name, requests in workloads.items():
        for cache_bytes in cache_sizes_bytes:
            cell = _run_all_directions(requests, cache_bytes, cost_model, workload_name)
            results.extend(cell)
            for r in cell:
                print(
                    f"[synthetic] {workload_name:18s} {cache_bytes / _MIB:5.0f}MiB "
                    f"{r.policy_name:16s} hit={r.token_hit_rate:.3f} "
                    f"p95={r.latency_p95_seconds * 1000:.2f}ms"
                )
    return results


def run_real_data_comparison(
    sharegpt_path: Path,
    scales: list[int],
    cache_sizes_mib: list[float],
    n_repeats: int,
) -> list[dict[str, Any]]:
    """Time-boxed real-data comparison with bootstrap CI across repeats.

    Returns aggregated rows (mean/CI per direction/scale/cache-size), not
    raw per-repeat rows -- see ``real_dataset_eval.py`` for the raw-row
    pattern this mirrors.
    """
    cost_model = CostModel(CostModelConfig())
    conversations = load_sharegpt_conversations(sharegpt_path)
    cache_sizes_bytes = [int(mib * _MIB) for mib in cache_sizes_mib]

    aggregated: list[dict[str, Any]] = []
    for scale in scales:
        for cache_bytes in cache_sizes_bytes:
            per_direction: dict[str, list[BenchResult]] = {}
            for repeat in range(n_repeats):
                requests = requests_from_conversations(
                    conversations, chunk_size=256, max_conversations=scale, seed=repeat
                )
                cell = _run_all_directions(
                    requests, cache_bytes, cost_model, f"sharegpt[{scale}]"
                )
                for r in cell:
                    per_direction.setdefault(r.policy_name, []).append(r)

            for direction, runs in per_direction.items():
                hit_mean, hit_lo, hit_hi = bootstrap_ci(
                    [r.token_hit_rate for r in runs]
                )
                p95_mean, p95_lo, p95_hi = bootstrap_ci(
                    [r.latency_p95_seconds for r in runs]
                )
                row = {
                    "direction": direction,
                    "max_conversations": scale,
                    "cache_capacity_bytes": cache_bytes,
                    "n_repeats": n_repeats,
                    "hit_rate_mean": hit_mean,
                    "hit_rate_ci_lo": hit_lo,
                    "hit_rate_ci_hi": hit_hi,
                    "latency_p95_mean": p95_mean,
                    "latency_p95_ci_lo": p95_lo,
                    "latency_p95_ci_hi": p95_hi,
                }
                aggregated.append(row)
                print(
                    f"[real-data] scale={scale:>6d} {cache_bytes / _MIB:5.0f}MiB "
                    f"{direction:16s} hit={hit_mean:.3f} [{hit_lo:.3f},{hit_hi:.3f}] "
                    f"p95={p95_mean * 1000:.2f}ms"
                )
    return aggregated


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write ``rows`` to CSV. Different directions carry different
    ``extra_params`` keys (e.g. admission control's ``rejected_admissions``
    vs. hierarchical's ``tier1_capacity_bytes``), so the column set is the
    union across all rows, not just the first row's keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--sharegpt-path", type=Path)
    parser.add_argument("--scales", nargs="+", type=int, default=[500, 2000])
    parser.add_argument("--cache-sizes-mib", nargs="+", type=float, default=[100.0])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "-o", "--output-dir", default="benchmarks/cache_policy/results/experiments"
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if not args.synthetic and not args.real_data:
        parser.error("pass --synthetic and/or --real-data")

    if args.synthetic:
        synthetic_results = run_synthetic_comparison()
        rows = [r.to_dict() for r in synthetic_results]
        _write_json(rows, out_dir / "synthetic_comparison.json")
        _write_csv(rows, out_dir / "synthetic_comparison.csv")
        print(f"Wrote {len(rows)} synthetic comparison rows to {out_dir}")

    if args.real_data:
        if args.sharegpt_path is None:
            parser.error("--real-data requires --sharegpt-path")
        real_rows = run_real_data_comparison(
            args.sharegpt_path, args.scales, args.cache_sizes_mib, args.repeats
        )
        _write_json(real_rows, out_dir / "real_data_comparison.json")
        _write_csv(real_rows, out_dir / "real_data_comparison.csv")
        print(f"Wrote {len(real_rows)} real-data comparison rows to {out_dir}")


if __name__ == "__main__":
    main()
