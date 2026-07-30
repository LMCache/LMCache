# SPDX-License-Identifier: Apache-2.0
"""
Ablation study isolating ``AdmissionControlledPolicy``'s one tunable
parameter: ``halve_every``, the frequency sketch's decay window.

Variants:
    fast_decay    -- halve_every=2,000 (short memory, popularity estimates
                      forget quickly)
    default       -- halve_every=20,000 (shipped default)
    slow_decay    -- halve_every=200,000 (long memory, popularity estimates
                      persist)
    no_admission  -- plain LRU, included as the non-admission-controlled
                      reference point

Usage::

    python benchmarks/cache_policy/run_admission_control_ablation.py \\
        -o benchmarks/cache_policy/results/admission_control
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
    "fast_decay": {"halve_every": 2_000},
    "default": {"halve_every": 20_000},
    "slow_decay": {"halve_every": 200_000},
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
                    "ADMISSION_LRU",
                    requests,
                    cache_bytes,
                    DEFAULT_KV_BYTES_PER_CHUNK,
                    cost_model,
                    workload_name=f"{workload_name}[{variant_name}]",
                    **kwargs,
                )
            )
        # Reference point: LRU, no admission control at all.
        results.append(
            run_workload(
                "LRU",
                requests,
                cache_bytes,
                DEFAULT_KV_BYTES_PER_CHUNK,
                cost_model,
                workload_name=f"{workload_name}[no_admission]",
            )
        )

    out_dir = Path(args.output_dir)
    to_csv(results, out_dir / "admission_control_ablation.csv")
    to_json(results, out_dir / "admission_control_ablation.json")

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
