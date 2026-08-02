# SPDX-License-Identifier: Apache-2.0
"""
Committed, scriptable ADMISSION_LRU-vs-LFU comparison for the mixed-Zipfian
multi-seed sweep -- the headline result of the cache-policy evaluation
report, and the reason this script exists at all: that comparison must
not live only as a one-off calculation made outside the repository.

Reads the raw per-seed rows already produced by
``main_sweep_multiseed.py`` (``multiseed_sweep_raw.json``), pairs
``--candidate-policy`` against ``--baseline-policy`` by
``(cache_capacity_bytes, seed)``, and reports, per cache capacity, the
paired mean hit-rate difference with a 95% bootstrap CI
(:func:`benchmarks.cache_policy.stats.paired_bootstrap_ci_diff`), an
exact paired sign test (:func:`benchmarks.cache_policy.stats.paired_sign_test`),
win/tie/loss counts, and a Holm-Bonferroni-corrected significance
decision across the three capacities (the family of comparisons this
script makes in one run).

Usage::

    python benchmarks/cache_policy/compare_admission_vs_lfu.py \\
        --input benchmarks/cache_policy/results/admission_control/multiseed_sweep_raw.json \\
        --output benchmarks/cache_policy/results/admission_control/admission_vs_lfu_paired.json
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json

# First Party
from benchmarks.cache_policy.stats import (
    paired_bootstrap_ci_diff,
    paired_sign_test,
)

_MIB = 2**20
_EXPECTED_CACHE_SIZES_MIB = (50.0, 100.0, 200.0)
_ALPHA = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw per-seed sweep JSON (main_sweep_multiseed.py output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the paired comparison JSON to.",
    )
    parser.add_argument(
        "--workload",
        default="mixed_zipfian",
        help="Workload name to compare on (default: mixed_zipfian).",
    )
    parser.add_argument(
        "--candidate-policy",
        default="ADMISSION_LRU",
        help="Treatment policy name (default: ADMISSION_LRU).",
    )
    parser.add_argument(
        "--baseline-policy",
        default="LFU",
        help="Baseline policy name (default: LFU).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="RNG seed for the paired bootstrap CI (default: 0).",
    )
    return parser.parse_args()


def _load_rows(
    input_path: Path,
    workload: str,
    candidate_policy: str,
    baseline_policy: str,
) -> dict[int, dict[str, dict[int, float]]]:
    """
    Load and validate the raw sweep rows, grouped as
    ``{cache_capacity_bytes: {policy_name: {seed: token_hit_rate}}}``.

    Raises:
        ValueError: If the input is missing rows, has duplicate rows, has
            mismatched seed sets between the two policies at some
            capacity, has an out-of-range hit rate, has fewer than 2
            seeds, or does not contain exactly the three expected cache
            capacities (50/100/200 MiB).
    """
    with open(input_path) as f:
        raw_rows: list[dict[str, Any]] = json.load(f)

    relevant = [
        row
        for row in raw_rows
        if row["workload_name"] == workload
        and row["policy_name"] in {candidate_policy, baseline_policy}
    ]
    if not relevant:
        raise ValueError(
            f"No rows found for workload={workload!r} with policies "
            f"{candidate_policy!r}/{baseline_policy!r} in {input_path}"
        )

    by_capacity: dict[int, dict[str, dict[int, float]]] = {}
    seen: set[tuple[int, int, str]] = set()
    for row in relevant:
        capacity = int(row["cache_capacity_bytes"])
        seed = int(row["seed"])
        policy = row["policy_name"]
        hit_rate = float(row["token_hit_rate"])

        key = (capacity, seed, policy)
        if key in seen:
            raise ValueError(
                f"Duplicate row for (cache_capacity_bytes={capacity}, "
                f"seed={seed}, policy_name={policy!r}) in {input_path}"
            )
        seen.add(key)

        if not (0.0 <= hit_rate <= 1.0):
            raise ValueError(
                f"token_hit_rate out of [0, 1] for {key}: {hit_rate!r}"
            )

        by_capacity.setdefault(capacity, {}).setdefault(policy, {})[seed] = hit_rate

    expected_capacities = {int(mib * _MIB) for mib in _EXPECTED_CACHE_SIZES_MIB}
    found_capacities = set(by_capacity.keys())
    if found_capacities != expected_capacities:
        raise ValueError(
            f"Expected cache capacities {sorted(expected_capacities)} bytes "
            f"(50/100/200 MiB), found {sorted(found_capacities)} in {input_path}"
        )

    for capacity, by_policy in by_capacity.items():
        for policy in (candidate_policy, baseline_policy):
            if policy not in by_policy:
                raise ValueError(
                    f"Missing policy {policy!r} at cache_capacity_bytes={capacity}"
                )
        candidate_seeds = set(by_policy[candidate_policy].keys())
        baseline_seeds = set(by_policy[baseline_policy].keys())
        if candidate_seeds != baseline_seeds:
            raise ValueError(
                f"Seed sets differ between {candidate_policy!r} and "
                f"{baseline_policy!r} at cache_capacity_bytes={capacity}: "
                f"{sorted(candidate_seeds)} vs {sorted(baseline_seeds)}"
            )
        if len(candidate_seeds) < 2:
            raise ValueError(
                f"At least 2 seeds are required at cache_capacity_bytes="
                f"{capacity}, found {len(candidate_seeds)}"
            )

    return by_capacity


def _holm_correct(p_values_by_capacity: dict[int, float]) -> dict[int, dict[str, Any]]:
    """
    Holm-Bonferroni step-down correction across the given p-values.

    Returns a per-capacity dict of ``holm_rank`` (1-indexed, ascending
    p-value order), ``holm_threshold`` (``alpha / (family_size - rank + 1)``),
    ``holm_adjusted_p_value``, and ``holm_reject_at_p05`` -- ``False`` for
    every capacity from the first non-rejection onward, per the Holm
    step-down procedure.
    """
    family_size = len(p_values_by_capacity)
    # Break ties on the p-value itself by capacity, so the ranking (and
    # therefore every downstream Holm field) doesn't depend on dict
    # insertion order, which in turn depends on the input JSON's row
    # order -- see test_compare_row_order_independent_of_input_order.
    ordered = sorted(p_values_by_capacity.items(), key=lambda kv: (kv[1], kv[0]))

    result: dict[int, dict[str, Any]] = {}
    still_rejecting = True
    running_max_adjusted = 0.0
    for rank, (capacity, p_value) in enumerate(ordered, start=1):
        threshold = _ALPHA / (family_size - rank + 1)
        adjusted = min(1.0, (family_size - rank + 1) * p_value)
        running_max_adjusted = max(running_max_adjusted, adjusted)
        if still_rejecting and p_value > threshold:
            still_rejecting = False
        result[capacity] = {
            "holm_rank": rank,
            "holm_threshold": threshold,
            "holm_adjusted_p_value": running_max_adjusted,
            "holm_reject_at_p05": still_rejecting,
        }
    return result


def compare(
    by_capacity: dict[int, dict[str, dict[int, float]]],
    candidate_policy: str,
    baseline_policy: str,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Compute the per-capacity paired comparison rows, Holm-corrected."""
    p_values: dict[int, float] = {}
    per_capacity: dict[int, dict[str, Any]] = {}

    for capacity, by_policy in by_capacity.items():
        shared_seeds = sorted(by_policy[candidate_policy].keys())
        candidate = [by_policy[candidate_policy][seed] for seed in shared_seeds]
        baseline = [by_policy[baseline_policy][seed] for seed in shared_seeds]
        diffs = [a - b for a, b in zip(candidate, baseline, strict=True)]

        diff_mean, ci_lo, ci_hi = paired_bootstrap_ci_diff(
            candidate, baseline, seed=bootstrap_seed
        )
        sign_p = paired_sign_test(candidate, baseline)
        p_values[capacity] = sign_p

        per_capacity[capacity] = {
            "cache_capacity_bytes": capacity,
            "cache_capacity_mib": capacity / _MIB,
            "n_seeds": len(shared_seeds),
            "seed_ids": shared_seeds,
            "candidate_hit_rate_mean": sum(candidate) / len(candidate),
            "baseline_hit_rate_mean": sum(baseline) / len(baseline),
            "hit_rate_diff_mean": diff_mean,
            "hit_rate_diff_percentage_points": diff_mean * 100.0,
            "hit_rate_diff_ci_lo": ci_lo,
            "hit_rate_diff_ci_hi": ci_hi,
            "wins": sum(1 for d in diffs if d > 0),
            "ties": sum(1 for d in diffs if d == 0),
            "losses": sum(1 for d in diffs if d < 0),
            "sign_test_p_value": sign_p,
        }

    holm = _holm_correct(p_values)
    for capacity, row in per_capacity.items():
        row.update(holm[capacity])

    return [per_capacity[capacity] for capacity in sorted(per_capacity.keys())]


def main() -> None:
    args = _parse_args()
    by_capacity = _load_rows(
        args.input, args.workload, args.candidate_policy, args.baseline_policy
    )
    comparisons = compare(
        by_capacity, args.candidate_policy, args.baseline_policy, args.bootstrap_seed
    )

    output = {
        "analysis": {
            "candidate_policy": args.candidate_policy,
            "baseline_policy": args.baseline_policy,
            "workload_name": args.workload,
            "pairing_key": "seed",
            "multiple_testing_method": "Holm",
            "family_definition": "three cache capacities",
            "alpha": _ALPHA,
        },
        "comparisons": comparisons,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(comparisons)} capacity comparisons to {args.output}")


if __name__ == "__main__":
    main()
