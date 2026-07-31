# SPDX-License-Identifier: Apache-2.0
"""
Multi-seed synthetic cache-size sweep with bootstrap confidence intervals.

Supersedes the single-seed ``runner.py --sweep`` reading as the basis for
any headline claim in the report: a single workload instance cannot
support a statistical claim (the finding "58.0% -> 83.3%" in an earlier
report draft came from exactly one ``multi_round_chat`` run and one
``mixed_zipfian`` run each). This script runs each (policy, workload,
cache-size) cell across ``N_SEEDS`` independently generated workload
instances for every seed-capable generator (``repetitive_short``,
``novel_long``, ``mixed_zipfian``) and reports mean +/- 95% bootstrap CI.

``multi_round_chat`` ignores its ``seed`` argument by design (see its
docstring) -- it is *not* included in the multi-seed statistics here.
Its result is reported separately, once, explicitly labeled as a
deterministic case study rather than statistical evidence, alongside a
small supplementary sweep over its own structural parameters
(``n_sessions``, ``rounds_per_session``) to check the qualitative finding
holds across configurations even though no individual reading has an
associated uncertainty interval.

Usage::

    python benchmarks/cache_policy/main_sweep_multiseed.py \\
        -o benchmarks/cache_policy/results/admission_control
"""

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
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
    run_workload,
)
from lmcache.tools.cache_policy_bench.workloads import (
    mixed_zipfian,
    multi_round_chat,
    novel_long,
    repetitive_short,
)

_MIB = 2**20
N_SEEDS = 10
POLICIES = ["LRU", "LFU", "COST_AWARE", "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU"]
CACHE_SIZES_MIB = [50.0, 100.0, 200.0]

_SEEDED_WORKLOAD_BUILDERS = {
    "repetitive_short": lambda seed: repetitive_short(3000, vocab_size=100, seed=seed),
    "novel_long": lambda seed: novel_long(750, seed=seed),
    "mixed_zipfian": lambda seed: mixed_zipfian(3000, unique_prefixes=300, seed=seed),
}

DEFAULT_BASELINE_POLICY = "LRU"

_MULTI_ROUND_CHAT_VARIANTS = {
    "default": {"n_sessions": 40, "rounds_per_session": 12},
    "fewer_longer_sessions": {"n_sessions": 20, "rounds_per_session": 20},
    "more_shorter_sessions": {"n_sessions": 80, "rounds_per_session": 6},
}


@dataclass
class AggregatedCell:
    """Bootstrap-CI-aggregated metrics for one (policy, workload, cache-size) cell."""

    policy_name: str
    workload_name: str
    cache_capacity_bytes: int
    n_seeds: int
    hit_rate_mean: float
    hit_rate_ci_lo: float
    hit_rate_ci_hi: float
    eviction_count_mean: float
    latency_p95_mean_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairedComparison:
    """
    Paired comparison of one policy against ``baseline_policy_name`` in
    one (workload, cache-size) cell, using the same ``N_SEEDS`` workload
    instances for both. Valid because ``_SEEDED_WORKLOAD_BUILDERS`` are
    deterministic functions of ``seed`` -- every policy at a given seed
    replays the *identical* generated workload instance, so per-seed
    readings are paired across policies, not independent (the same
    reasoning ``real_dataset_eval.py`` uses for the real-data grid).
    """

    policy_name: str
    baseline_policy_name: str
    workload_name: str
    cache_capacity_bytes: int
    n_seeds: int
    hit_rate_diff_mean: float
    hit_rate_diff_ci_lo: float
    hit_rate_diff_ci_hi: float
    sign_test_p_value: float
    significant_at_p05: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_paired_comparisons(
    raw_rows: list[dict[str, Any]],
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
) -> list[PairedComparison]:
    """
    Derive paired policy-vs-baseline comparisons from already-collected
    ``raw_rows`` (no new runs) -- see :class:`PairedComparison`.
    """
    by_cell: dict[tuple[str, int, str], dict[int, float]] = {}
    for row in raw_rows:
        key = (row["workload_name"], row["cache_capacity_bytes"], row["policy_name"])
        by_cell.setdefault(key, {})[row["seed"]] = row["token_hit_rate"]

    cells = {(wl, cb) for wl, cb, _p in by_cell}
    comparisons: list[PairedComparison] = []
    for workload_name, cache_bytes in sorted(cells):
        baseline_by_seed = by_cell.get((workload_name, cache_bytes, baseline_policy))
        if not baseline_by_seed:
            continue
        seeds = sorted(baseline_by_seed)
        baseline_vals = [baseline_by_seed[s] for s in seeds]
        for (wl, cb, policy_name), values_by_seed in by_cell.items():
            same_cell = wl == workload_name and cb == cache_bytes
            if not same_cell or policy_name == baseline_policy:
                continue
            policy_vals = [values_by_seed[s] for s in seeds]
            diff_mean, lo, hi = paired_bootstrap_ci_diff(policy_vals, baseline_vals)
            sign_p = paired_sign_test(policy_vals, baseline_vals)
            comparisons.append(
                PairedComparison(
                    policy_name=policy_name,
                    baseline_policy_name=baseline_policy,
                    workload_name=workload_name,
                    cache_capacity_bytes=cache_bytes,
                    n_seeds=len(seeds),
                    hit_rate_diff_mean=diff_mean,
                    hit_rate_diff_ci_lo=lo,
                    hit_rate_diff_ci_hi=hi,
                    sign_test_p_value=sign_p,
                    significant_at_p05=(lo > 0 or hi < 0),
                )
            )
    return comparisons


def run_seeded_sweep() -> tuple[list[dict[str, Any]], list[AggregatedCell]]:
    """
    Run every (policy, seed-capable workload, cache-size) cell across
    ``N_SEEDS`` independently generated workload instances.

    Returns:
        ``(raw_rows, aggregated)`` -- per-seed raw readings (full
        transparency) and the bootstrap-CI-aggregated table used for
        reporting.
    """
    cost_model = CostModel(CostModelConfig())
    raw_rows: list[dict[str, Any]] = []
    aggregated: list[AggregatedCell] = []

    for workload_name, build in _SEEDED_WORKLOAD_BUILDERS.items():
        for cache_mib in CACHE_SIZES_MIB:
            cache_bytes = int(cache_mib * _MIB)
            for policy_name in POLICIES:
                hit_rates: list[float] = []
                evictions: list[float] = []
                p95s: list[float] = []
                for seed in range(N_SEEDS):
                    requests = build(seed)
                    result = run_workload(
                        policy_name,
                        requests,
                        cache_bytes,
                        DEFAULT_KV_BYTES_PER_CHUNK,
                        cost_model,
                        workload_name=workload_name,
                    )
                    hit_rates.append(result.token_hit_rate)
                    evictions.append(float(result.eviction_count))
                    p95s.append(result.latency_p95_seconds)
                    row = result.to_dict()
                    row["seed"] = seed
                    raw_rows.append(row)

                hr_mean, hr_lo, hr_hi = bootstrap_ci(hit_rates)
                aggregated.append(
                    AggregatedCell(
                        policy_name=policy_name,
                        workload_name=workload_name,
                        cache_capacity_bytes=cache_bytes,
                        n_seeds=N_SEEDS,
                        hit_rate_mean=hr_mean,
                        hit_rate_ci_lo=hr_lo,
                        hit_rate_ci_hi=hr_hi,
                        eviction_count_mean=sum(evictions) / len(evictions),
                        latency_p95_mean_seconds=sum(p95s) / len(p95s),
                    )
                )
                print(
                    f"{workload_name:18s} {cache_mib:5.0f}MiB {policy_name:24s} "
                    f"hit_rate={hr_mean:.4f} [{hr_lo:.4f},{hr_hi:.4f}] "
                    f"(n={N_SEEDS})"
                )

    return raw_rows, aggregated


def run_multi_round_chat_case_study() -> list[dict[str, Any]]:
    """
    Deterministic-case-study reading for multi_round_chat: one run per
    (policy, variant, cache-size) cell, no seed variation (none is
    possible -- see module docstring), explicitly not pooled into any
    confidence interval. Varies structural parameters instead, to check
    the qualitative finding generalizes across configurations.
    """
    cost_model = CostModel(CostModelConfig())
    rows: list[dict[str, Any]] = []
    for variant_name, kwargs in _MULTI_ROUND_CHAT_VARIANTS.items():
        requests = multi_round_chat(**kwargs)
        for cache_mib in CACHE_SIZES_MIB:
            cache_bytes = int(cache_mib * _MIB)
            for policy_name in POLICIES:
                result = run_workload(
                    policy_name,
                    requests,
                    cache_bytes,
                    DEFAULT_KV_BYTES_PER_CHUNK,
                    cost_model,
                    workload_name=f"multi_round_chat[{variant_name}]",
                )
                row = result.to_dict()
                row["variant"] = variant_name
                rows.append(row)
                print(
                    f"multi_round_chat[{variant_name}] {cache_mib:5.0f}MiB "
                    f"{policy_name:24s} hit_rate={result.token_hit_rate:.4f} "
                    "(single deterministic run, not statistical evidence)"
                )
    return rows


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output-dir",
        default="benchmarks/cache_policy/results/admission_control",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print(f"=== Multi-seed sweep ({N_SEEDS} seeds/cell, seed-capable workloads) ===")
    raw_rows, aggregated = run_seeded_sweep()
    _write_json(raw_rows, out_dir / "multiseed_sweep_raw.json")
    _write_json([a.to_dict() for a in aggregated], out_dir / "multiseed_sweep_ci.json")

    paired = compute_paired_comparisons(raw_rows)
    _write_json(
        [p.to_dict() for p in paired], out_dir / "multiseed_sweep_paired_diff.json"
    )
    print(
        f"\n=== Paired comparisons vs {DEFAULT_BASELINE_POLICY} "
        "(derived, no new runs) ==="
    )
    for p in paired:
        print(
            f"{p.workload_name:18s} {p.cache_capacity_bytes / _MIB:5.0f}MiB "
            f"{p.policy_name:24s} diff={p.hit_rate_diff_mean * 100:+.2f}pp "
            f"[{p.hit_rate_diff_ci_lo * 100:+.2f},{p.hit_rate_diff_ci_hi * 100:+.2f}] "
            f"sign_p={p.sign_test_p_value:.4f} "
            f"{'SIGNIFICANT' if p.significant_at_p05 else 'not significant'}"
        )

    print(
        "\n=== multi_round_chat deterministic case study "
        "(not statistical evidence) ==="
    )
    chat_rows = run_multi_round_chat_case_study()
    _write_json(chat_rows, out_dir / "multi_round_chat_case_study.json")

    print(
        f"\nWrote {len(raw_rows)} raw rows + {len(aggregated)} aggregated cells to "
        f"{out_dir}/multiseed_sweep_{{raw,ci}}.json, and {len(chat_rows)} "
        f"multi_round_chat case-study rows to "
        f"{out_dir}/multi_round_chat_case_study.json"
    )


if __name__ == "__main__":
    main()
