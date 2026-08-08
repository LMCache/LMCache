# SPDX-License-Identifier: Apache-2.0
"""
Ablation study isolating the two ideas combined in
``CostAwareEvictionPolicy``'s score:

    score = (estimated_recompute_tokens / memory_size_bytes)
            / (1 + age_seconds / half_life_seconds)

* EWMA-smoothed recompute-cost density (the numerator)
* Recency decay (the denominator)

Variants:
    full            -- default half_life_seconds=60, cost_ewma_alpha=0.2
    no_recency      -- half_life_seconds effectively infinite (huge value),
                        so the decay denominator stays ~1: pure cost-density
                        ranking, recency plays no role.
    no_ewma         -- cost_ewma_alpha=1.0: no smoothing, each put()
                        overwrites the estimate with the latest observation.
    cost_agnostic   -- LRU, included as the non-cost-aware reference point.

Usage::

    python benchmarks/cache_policy/run_ablation.py \\
        -o benchmarks/cache_policy/results
"""

# Standard
from pathlib import Path
import argparse

# First Party
from lmcache.tools.cache_policy_bench.cost_model import CostModel, CostModelConfig
from lmcache.tools.cache_policy_bench.runner import (
    DEFAULT_KV_BYTES_PER_CHUNK,
    run_workload,
    to_csv,
    to_json,
)
from lmcache.tools.cache_policy_bench.workloads import mixed_zipfian, multi_round_chat

_MIB = 2**20

VARIANTS: dict[str, dict] = {
    "full": {"half_life_seconds": 60.0, "cost_ewma_alpha": 0.2},
    "no_recency": {"half_life_seconds": 1.0e9, "cost_ewma_alpha": 0.2},
    "no_ewma": {"half_life_seconds": 60.0, "cost_ewma_alpha": 1.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default="benchmarks/cache_policy/results")
    parser.add_argument("--cache-mib", type=float, default=100.0)
    args = parser.parse_args()

    cache_bytes = int(args.cache_mib * _MIB)
    cost_model = CostModel(CostModelConfig())

    workloads = {
        "mixed_zipfian": mixed_zipfian(3000, unique_prefixes=300, seed=1),
        "multi_round_chat": multi_round_chat(40, rounds_per_session=12),
    }

    results = []
    for workload_name, requests in workloads.items():
        for variant_name, kwargs in VARIANTS.items():
            results.append(
                run_workload(
                    "COST_AWARE",
                    requests,
                    cache_bytes,
                    DEFAULT_KV_BYTES_PER_CHUNK,
                    cost_model,
                    workload_name=f"{workload_name}[{variant_name}]",
                    **kwargs,
                )
            )
        # Reference point: LRU, no cost-awareness at all.
        results.append(
            run_workload(
                "LRU",
                requests,
                cache_bytes,
                DEFAULT_KV_BYTES_PER_CHUNK,
                cost_model,
                workload_name=f"{workload_name}[cost_agnostic]",
            )
        )

    out_dir = Path(args.output_dir)
    to_csv(results, out_dir / "ablation_results.csv")
    to_json(results, out_dir / "ablation_results.json")

    print(
        f"{'workload[variant]':45s} {'hit_rate':>10s} "
        f"{'p95_ms':>10s} {'evictions':>10s}"
    )
    for r in results:
        print(
            f"{r.workload_name:45s} {r.token_hit_rate:10.3f} "
            f"{r.latency_p95_seconds * 1000:10.3f} {r.eviction_count:10d}"
        )


if __name__ == "__main__":
    main()
