# SPDX-License-Identifier: Apache-2.0
"""
Render a hit-rate-vs-corpus-scale chart (with bootstrap CI error bars) from
``real_dataset_eval.py``'s aggregated output.

Usage::

    python benchmarks/cache_policy/plot_real_dataset.py \\
        -i benchmarks/cache_policy/results/real_data/real_dataset_ci.json \\
        -o benchmarks/cache_policy/results/charts/real_data_hit_rate.png
"""

# Standard
from pathlib import Path
import argparse
import json

_MIB = 2**20


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_hit_rate_vs_scale(rows: list[dict], output: Path) -> None:
    # Third Party
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cache_sizes = sorted({r["cache_capacity_bytes"] for r in rows})
    policies = list(dict.fromkeys(r["policy_name"] for r in rows))

    fig, axes = plt.subplots(
        1, len(cache_sizes), figsize=(5 * len(cache_sizes), 4), sharey=True
    )
    if len(cache_sizes) == 1:
        axes = [axes]

    for ax, cache_bytes in zip(axes, cache_sizes, strict=False):
        for policy in policies:
            cell_rows = [
                r
                for r in rows
                if r["cache_capacity_bytes"] == cache_bytes
                and r["policy_name"] == policy
            ]
            cell_rows.sort(
                key=lambda r: (
                    float("inf")
                    if r["max_conversations"] == "full"
                    else int(r["max_conversations"])
                )
            )
            xs = list(range(len(cell_rows)))
            means = [r["hit_rate_mean"] * 100 for r in cell_rows]
            lo = [
                r["hit_rate_mean"] * 100 - r["hit_rate_ci_lo"] * 100 for r in cell_rows
            ]
            hi = [
                r["hit_rate_ci_hi"] * 100 - r["hit_rate_mean"] * 100 for r in cell_rows
            ]
            ax.errorbar(xs, means, yerr=[lo, hi], marker="o", capsize=3, label=policy)
        ax.set_xticks(range(len(cell_rows)))
        ax.set_xticklabels([r["max_conversations"] for r in cell_rows])
        ax.set_title(f"{cache_bytes / _MIB:.0f} MiB cache")
        ax.set_xlabel("Corpus scale (# conversations)")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Token hit rate (%), mean +/- 95% bootstrap CI")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Real ShareGPT data: hit rate vs. corpus scale")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", required=True, help="real_dataset_ci.json path"
    )
    parser.add_argument("-o", "--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_hit_rate_vs_scale(rows, out_path)


if __name__ == "__main__":
    main()
