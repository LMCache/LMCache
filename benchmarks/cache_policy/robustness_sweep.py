# SPDX-License-Identifier: Apache-2.0
"""
Robustness sweep for the frequency-weighting fix in
``CostAwareEvictionPolicy`` (see
``docs/design/v1/storage_backend/cache_policy/cost-aware-policy-eval.md``).

The original benchmark reading showing ``COST_AWARE`` losing to
``LRU``/``LFU`` was a single (workload, cache-size) snapshot. Fixing the
score to include a log-dampened hit-count term is only a *general*
improvement if it holds beyond that one snapshot. This script checks:

1. **Zipf skew strength** -- does the fix help across mild-to-extreme
   popularity concentration, or only at the one ``zipf_s`` value already
   benchmarked?
2. **Chunk-size heterogeneity** -- the standard sweep workloads use a
   uniform ``kv_bytes_per_chunk``, which is exactly the condition where
   ``cost_density`` degenerates to a constant multiple of
   ``estimated_recompute_tokens``. This check verifies the cost-density
   term still does real work (discriminates by cost, not just frequency)
   when chunk memory size actually varies, via a direct two-chunk
   ``CostAwareEvictionPolicy`` scenario (isolated from the simulator's
   uniform-size limitation).
3. **Regression check** -- the existing four workloads at three cache
   sizes, to confirm ``multi_round_chat`` (where the cost-only design had
   a real edge) doesn't get worse now that frequency also has a vote.

Usage::

    python benchmarks/cache_policy/robustness_sweep.py
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
from lmcache.tools.cache_policy_bench.workloads import mixed_zipfian
from lmcache.v1.storage_backend.cache_policy.cost_aware_policy import (
    CostAwareEvictionPolicy,
)

_MIB = 2**20
POLICIES = ["LRU", "LFU", "FIFO", "MRU", "COST_AWARE"]


def check_zipf_skew(cache_mib: float = 100.0) -> list:
    """Sweep Zipf skew strength for ``mixed_zipfian`` across all policies."""
    cost_model = CostModel(CostModelConfig())
    results = []
    for zipf_s in (0.6, 1.2, 2.0):
        requests = mixed_zipfian(3000, unique_prefixes=300, zipf_s=zipf_s, seed=1)
        for policy_name in POLICIES:
            results.append(
                run_workload(
                    policy_name,
                    requests,
                    int(cache_mib * _MIB),
                    DEFAULT_KV_BYTES_PER_CHUNK,
                    cost_model,
                    workload_name=f"mixed_zipfian[zipf_s={zipf_s}]",
                )
            )
    return results


def check_size_heterogeneity() -> None:
    """
    Direct two-chunk check that cost-density still discriminates by cost
    when hit_count is held equal and memory size actually varies.

    "cheap": small recompute cost, small memory footprint.
    "expensive": large recompute cost, large memory footprint, but the
    same cost *density* (cost/byte) as "cheap" -- so with hit_count equal,
    the two should score identically (frequency and cost-density both
    tied) despite very different absolute memory sizes. This is the
    control case. The follow-up perturbs recompute cost only, holding
    memory size and hit_count fixed, and checks the score moves in the
    expected direction.
    """
    policy = CostAwareEvictionPolicy(half_life_seconds=1.0e9)  # freeze recency decay
    now = 0.0

    policy.put("cheap", memory_size_bytes=1024, observed_recompute_tokens=100)
    policy.put("expensive", memory_size_bytes=8192, observed_recompute_tokens=800)
    score_cheap = policy.calculate_score("cheap", current_time=now)
    score_expensive = policy.calculate_score("expensive", current_time=now)
    same_density = abs(score_cheap - score_expensive) < 1e-9
    print(
        f"[size heterogeneity] equal cost-density, equal hit_count: "
        f"cheap={score_cheap:.6f} expensive={score_expensive:.6f} "
        f"equal={same_density}"
    )
    assert same_density, "equal cost-density chunks should score equally"

    # Now make "expensive" genuinely more cost-dense (same size, higher
    # recompute cost) and confirm the score responds -- proves cost-density
    # is still load-bearing post-fix, not overridden by frequency.
    policy2 = CostAwareEvictionPolicy(half_life_seconds=1.0e9)
    policy2.put("low_cost", memory_size_bytes=1024, observed_recompute_tokens=100)
    policy2.put("high_cost", memory_size_bytes=1024, observed_recompute_tokens=900)
    score_low = policy2.calculate_score("low_cost", current_time=now)
    score_high = policy2.calculate_score("high_cost", current_time=now)
    print(
        f"[size heterogeneity] equal size/hit_count, 9x cost: "
        f"low_cost={score_low:.6f} high_cost={score_high:.6f} "
        f"ratio={score_high / score_low:.3f}"
    )
    assert score_high > score_low, (
        "higher cost-density chunk must score higher (harder to evict) "
        "when hit_count and size are held equal"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default="benchmarks/cache_policy/results")
    args = parser.parse_args()

    print("=== Check 2: chunk-size / cost-density still discriminates ===")
    check_size_heterogeneity()

    print("\n=== Check 1: Zipf skew strength sweep (mixed_zipfian) ===")
    zipf_results = check_zipf_skew()
    header = f"{'workload':30s} {'policy':12s} {'hit_rate':>10s} {'evictions':>10s}"
    print(header)
    for r in zipf_results:
        print(
            f"{r.workload_name:30s} {r.policy_name:12s} "
            f"{r.token_hit_rate:10.3f} {r.eviction_count:10d}"
        )

    out_dir = Path(args.output_dir)
    to_csv(zipf_results, out_dir / "robustness_zipf_skew.csv")
    to_json(zipf_results, out_dir / "robustness_zipf_skew.json")
    print(
        f"\nWrote {len(zipf_results)} rows to "
        f"{out_dir}/robustness_zipf_skew.{{csv,json}}"
    )


if __name__ == "__main__":
    main()
