# SPDX-License-Identifier: Apache-2.0
"""
Statistically robust real-data (ShareGPT) evaluation of cache policies.

Reuses the existing simulator (``lmcache.tools.cache_policy_bench.runner``)
and the ShareGPT loader
(``lmcache.tools.cache_policy_bench.sharegpt_workload``) unchanged; this
script adds repeated-subsample runs, confidence intervals, and a
corpus-scale sweep on top.

Sampling method -- read this before interpreting any interval here: each
repeat draws ``max_conversations`` conversations from the full corpus via
``random.sample`` (see
:func:`lmcache.tools.cache_policy_bench.sharegpt_workload.requests_from_conversations`),
which samples *without* replacement. This is repeated subsampling, not a
corpus bootstrap (which would sample the full corpus size *with*
replacement, allowing duplicates) -- the two are not the same procedure
and the difference matters for how these intervals should be interpreted
(a subsampling interval reflects sensitivity to which conversations were
drawn, not a bootstrap approximation of the full-corpus sampling
distribution).

Every policy in a given (scale, cache-size) cell is replayed against the
exact same ``n_repeats`` subsamples (repeat ``i`` uses ``seed=i`` for
every policy), so per-repeat readings are *paired* across policies, not
independent. This script reports both:

1. Each policy's own descriptive mean +/- CI (independent bootstrap of
   its own repeat values) -- useful for reading off a single policy's
   typical performance, but comparing two policies by checking whether
   their independent CIs overlap discards the pairing and understates
   the evidence for a real difference (or overstates it, in principle,
   though in practice it is the conservative direction).
2. A paired comparison against a baseline policy (default: ``LRU``) for
   every other policy in the cell: the per-repeat difference's own
   bootstrap CI (:func:`benchmarks.cache_policy.stats.paired_bootstrap_ci_diff`)
   plus an exact paired sign test
   (:func:`benchmarks.cache_policy.stats.paired_sign_test`) as a second,
   distribution-free check. This is the statistically appropriate
   comparison given the paired sampling design, and is what any
   "policy X beats policy Y" claim in the report should cite.

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
from benchmarks.cache_policy.stats import (
    bootstrap_ci,
    paired_bootstrap_ci_diff,
    paired_sign_test,
)
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
from lmcache.tools.cache_policy_bench.workloads import Request

_MIB = 2**20
DEFAULT_SCALES: list[Optional[int]] = [500, 2000, 5000]
DEFAULT_CACHE_SIZES_MIB: list[float] = [50.0, 100.0, 200.0]
DEFAULT_N_REPEATS = 6
DEFAULT_BASELINE_POLICY = "LRU"


@dataclass
class AggregatedResult:
    """
    Descriptive bootstrap-CI-aggregated metrics for one (policy, scale,
    cache-size) cell, from that policy's own repeat values independently
    -- see the module docstring for why this alone should not be used to
    compare two policies.
    """

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


@dataclass
class PairedComparison:
    """
    Paired comparison of one policy against ``baseline_policy_name`` in
    one (scale, cache-size) cell, using the same ``n_repeats`` subsample
    seeds for both -- see the module docstring for why this, not two
    independent :class:`AggregatedResult` CIs, is the valid comparison.
    """

    policy_name: str
    baseline_policy_name: str
    max_conversations: str
    cache_capacity_bytes: int
    n_repeats: int
    hit_rate_diff_mean: float
    hit_rate_diff_ci_lo: float
    hit_rate_diff_ci_hi: float
    sign_test_p_value: float
    significant_at_p05: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_repeated(
    sharegpt_path: Path,
    scales: list[Optional[int]],
    cache_sizes_bytes: list[int],
    policies: list[str],
    n_repeats: int,
    chunk_size: int = 256,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
) -> tuple[list[dict[str, Any]], list[AggregatedResult], list[PairedComparison]]:
    """
    Run the full (policy x scale x cache-size) grid, ``n_repeats`` times
    each with a fresh subsample of the corpus, and aggregate.

    Every policy at a given (scale, cache-size, repeat) is replayed
    against the identical subsample (same ``seed=repeat``) -- requests
    are built once per (scale, repeat) and reused across cache sizes and
    policies, both for efficiency and so the paired structure is
    explicit: ``requests_by_repeat[repeat]`` is literally the same list
    object passed to every policy.

    Returns:
        ``(raw_rows, aggregated, paired_comparisons)`` -- the raw
        per-repeat rows (full transparency), the descriptive per-policy
        CI table, and the paired-vs-baseline comparison table (empty if
        ``baseline_policy`` is not in ``policies``).
    """
    cost_model = CostModel(CostModelConfig())
    raw_rows: list[dict[str, Any]] = []
    aggregated: list[AggregatedResult] = []
    paired_comparisons: list[PairedComparison] = []

    conversations = load_sharegpt_conversations(sharegpt_path)
    have_baseline = baseline_policy in policies
    if not have_baseline:
        print(
            f"Note: baseline policy {baseline_policy!r} not in --policies; "
            "skipping paired comparisons (descriptive per-policy CIs only)."
        )

    for scale in scales:
        scale_label = "full" if scale is None else str(scale)
        requests_by_repeat: list[list[Request]] = [
            requests_from_conversations(
                conversations, chunk_size=chunk_size, max_conversations=scale, seed=r
            )
            for r in range(n_repeats)
        ]

        for cache_bytes in cache_sizes_bytes:
            per_policy_hit_rates: dict[str, list[float]] = {p: [] for p in policies}
            per_policy_p95s: dict[str, list[float]] = {p: [] for p in policies}
            per_policy_evictions: dict[str, list[float]] = {p: [] for p in policies}

            for repeat, requests in enumerate(requests_by_repeat):
                for policy_name in policies:
                    result = run_workload(
                        policy_name,
                        requests,
                        cache_bytes,
                        DEFAULT_KV_BYTES_PER_CHUNK,
                        cost_model,
                        workload_name=f"sharegpt[{scale_label}]",
                    )
                    per_policy_hit_rates[policy_name].append(result.token_hit_rate)
                    per_policy_p95s[policy_name].append(result.latency_p95_seconds)
                    per_policy_evictions[policy_name].append(
                        float(result.eviction_count)
                    )
                    row = result.to_dict()
                    row["max_conversations"] = scale_label
                    row["repeat"] = repeat
                    raw_rows.append(row)

            for policy_name in policies:
                hr_mean, hr_lo, hr_hi = bootstrap_ci(per_policy_hit_rates[policy_name])
                p95_mean, p95_lo, p95_hi = bootstrap_ci(per_policy_p95s[policy_name])
                ev_mean, ev_lo, ev_hi = bootstrap_ci(per_policy_evictions[policy_name])
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
                    f"scale={scale_label:>6s} policy={policy_name:26s} "
                    f"cache={cache_bytes / _MIB:6.0f}MiB "
                    f"hit_rate={hr_mean:.3f} [{hr_lo:.3f},{hr_hi:.3f}] "
                    f"p95={p95_mean * 1000:.2f}ms"
                )

            if have_baseline:
                baseline_hit_rates = per_policy_hit_rates[baseline_policy]
                for policy_name in policies:
                    if policy_name == baseline_policy:
                        continue
                    diff_mean, diff_lo, diff_hi = paired_bootstrap_ci_diff(
                        per_policy_hit_rates[policy_name], baseline_hit_rates
                    )
                    sign_p = paired_sign_test(
                        per_policy_hit_rates[policy_name], baseline_hit_rates
                    )
                    significant = diff_lo > 0 or diff_hi < 0
                    paired_comparisons.append(
                        PairedComparison(
                            policy_name=policy_name,
                            baseline_policy_name=baseline_policy,
                            max_conversations=scale_label,
                            cache_capacity_bytes=cache_bytes,
                            n_repeats=n_repeats,
                            hit_rate_diff_mean=diff_mean,
                            hit_rate_diff_ci_lo=diff_lo,
                            hit_rate_diff_ci_hi=diff_hi,
                            sign_test_p_value=sign_p,
                            significant_at_p05=significant,
                        )
                    )
                    print(
                        f"  paired vs {baseline_policy}: {policy_name:24s} "
                        f"diff={diff_mean:+.4f} [{diff_lo:+.4f},{diff_hi:+.4f}] "
                        f"sign_test_p={sign_p:.4f} "
                        f"{'SIGNIFICANT' if significant else 'not significant'}"
                    )

    return raw_rows, aggregated, paired_comparisons


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to CSV, tolerating rows with different key sets.

    Different policies can populate different keys (e.g. only
    admission-control policies report ``param_sketch_halvings_triggered``
    -- see ``BenchResult.to_dict``), so the column set is the union
    across all rows, in first-seen order, not just the first row's keys.
    """
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
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
    parser.add_argument("--baseline-policy", default=DEFAULT_BASELINE_POLICY)
    parser.add_argument(
        "-o", "--output-dir", default="benchmarks/cache_policy/results/real_data"
    )
    args = parser.parse_args()

    scales: list[Optional[int]] = [None if s == "full" else int(s) for s in args.scales]
    cache_sizes_bytes = [int(mib * _MIB) for mib in args.cache_sizes_mib]

    raw_rows, aggregated, paired_comparisons = run_repeated(
        args.sharegpt_path,
        scales,
        cache_sizes_bytes,
        args.policies,
        args.repeats,
        chunk_size=args.chunk_size,
        baseline_policy=args.baseline_policy,
    )

    out_dir = Path(args.output_dir)
    _write_json(raw_rows, out_dir / "real_dataset_raw.json")
    _write_csv(raw_rows, out_dir / "real_dataset_raw.csv")
    agg_dicts = [a.to_dict() for a in aggregated]
    _write_json(agg_dicts, out_dir / "real_dataset_ci.json")
    _write_csv(agg_dicts, out_dir / "real_dataset_ci.csv")
    paired_dicts = [p.to_dict() for p in paired_comparisons]
    _write_json(paired_dicts, out_dir / "real_dataset_paired_diff.json")
    _write_csv(paired_dicts, out_dir / "real_dataset_paired_diff.csv")
    print(
        f"\nWrote {len(raw_rows)} raw rows, {len(aggregated)} aggregated rows, "
        f"and {len(paired_comparisons)} paired-comparison rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
