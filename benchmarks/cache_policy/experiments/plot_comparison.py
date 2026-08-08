# SPDX-License-Identifier: Apache-2.0
"""
Render a bar chart comparing all direction-finding candidates on hit rate
and p95 latency, synthetic and real-data side by side.

Usage::

    python benchmarks/cache_policy/experiments/plot_comparison.py \\
        --synthetic .../results/experiments/synthetic_comparison.json \\
        --real-data .../results/experiments/real_data_comparison.json \\
        -o .../results/charts/direction_comparison.png
"""

# Standard
from pathlib import Path
import argparse
import json

_MIB = 2**20


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_comparison(
    synthetic_rows: list[dict],
    real_rows: list[dict],
    output: Path,
    cache_mib: float = 100.0,
) -> None:
    # Third Party
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cache_bytes = int(cache_mib * _MIB)
    synth_cell = [
        r
        for r in synthetic_rows
        if r["workload_name"] == "mixed_zipfian"
        and r["cache_capacity_bytes"] == cache_bytes
    ]
    directions = [r["policy_name"] for r in synth_cell]

    real_scale = max(r["max_conversations"] for r in real_rows)
    real_cell = {
        r["direction"]: r
        for r in real_rows
        if r["max_conversations"] == real_scale
        and r["cache_capacity_bytes"] == cache_bytes
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    synth_hits = [r["token_hit_rate"] * 100 for r in synth_cell]
    real_hits = [
        real_cell[d]["hit_rate_mean"] * 100 if d in real_cell else 0 for d in directions
    ]
    real_err = [
        (
            (real_cell[d]["hit_rate_mean"] - real_cell[d]["hit_rate_ci_lo"]) * 100,
            (real_cell[d]["hit_rate_ci_hi"] - real_cell[d]["hit_rate_mean"]) * 100,
        )
        if d in real_cell
        else (0, 0)
        for d in directions
    ]
    x = range(len(directions))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x], synth_hits, width, label="synthetic (mixed_zipfian)"
    )
    ax.bar(
        [i + width / 2 for i in x],
        real_hits,
        width,
        yerr=[[e[0] for e in real_err], [e[1] for e in real_err]],
        capsize=3,
        label=f"real data ({real_scale} conversations)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(directions, rotation=45, ha="right")
    ax.set_ylabel("Token hit rate (%)")
    ax.set_title(f"Hit rate by direction @ {cache_mib:.0f} MiB")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    synth_p95 = [r["latency_p95_seconds"] * 1000 for r in synth_cell]
    real_p95 = [
        real_cell[d]["latency_p95_mean"] * 1000 if d in real_cell else 0
        for d in directions
    ]
    ax.bar(
        [i - width / 2 for i in x], synth_p95, width, label="synthetic (mixed_zipfian)"
    )
    ax.bar(
        [i + width / 2 for i in x],
        real_p95,
        width,
        label=f"real data ({real_scale} conversations)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(directions, rotation=45, ha="right")
    ax.set_ylabel("p95 modeled latency (ms)")
    ax.set_title(f"p95 latency by direction @ {cache_mib:.0f} MiB")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Direction-finding comparison: baselines vs. candidate improvements")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic", required=True, help="synthetic_comparison.json path"
    )
    parser.add_argument(
        "--real-data", required=True, help="real_data_comparison.json path"
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--cache-mib", type=float, default=100.0)
    args = parser.parse_args()

    synthetic_rows = _load(Path(args.synthetic))
    real_rows = _load(Path(args.real_data))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(synthetic_rows, real_rows, out_path, cache_mib=args.cache_mib)


if __name__ == "__main__":
    main()
