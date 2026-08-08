# SPDX-License-Identifier: Apache-2.0
"""
Render charts from a cache-policy benchmark sweep CSV.

Usage::

    python benchmarks/cache_policy/plot_results.py \\
        -i benchmarks/cache_policy/results/sweep_results.csv \\
        -o benchmarks/cache_policy/results/charts
"""

# Standard
from pathlib import Path
import argparse
import csv

_MIB = 2**20


def _load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _policies(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["policy_name"] not in seen:
            seen.append(r["policy_name"])
    return seen


def _workloads(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["workload_name"] not in seen:
            seen.append(r["workload_name"])
    return seen


def plot_hit_rate_vs_cache_size(rows: list[dict], output: Path) -> None:
    # Third Party
    import matplotlib.pyplot as plt  # noqa: PLC0415

    workloads = _workloads(rows)
    policies = _policies(rows)
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(5 * len(workloads), 4), sharey=True
    )
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads, strict=False):
        for policy in policies:
            pts = sorted(
                (
                    (
                        float(r["cache_capacity_bytes"]) / _MIB,
                        float(r["token_hit_rate"]),
                    )
                    for r in rows
                    if r["workload_name"] == workload and r["policy_name"] == policy
                ),
                key=lambda p: p[0],
            )
            xs = [p[0] for p in pts]
            ys = [p[1] * 100 for p in pts]
            ax.plot(xs, ys, marker="o", label=policy)
        ax.set_title(workload)
        ax.set_xlabel("Cache capacity (MiB)")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Token hit rate (%)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Token hit rate vs. cache capacity")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def plot_latency_p95(rows: list[dict], output: Path, cache_mib: float) -> None:
    # Third Party
    import matplotlib.pyplot as plt  # noqa: PLC0415

    workloads = _workloads(rows)
    policies = _policies(rows)
    target_bytes = cache_mib * _MIB

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(policies)
    x_base = range(len(workloads))

    for i, policy in enumerate(policies):
        vals = []
        for workload in workloads:
            match = [
                r
                for r in rows
                if r["workload_name"] == workload
                and r["policy_name"] == policy
                and abs(float(r["cache_capacity_bytes"]) - target_bytes) < 1
            ]
            vals.append(float(match[0]["latency_p95_seconds"]) * 1000 if match else 0.0)
        xs = [x + i * width for x in x_base]
        ax.bar(xs, vals, width=width, label=policy)

    ax.set_xticks([x + 0.4 - width / 2 for x in x_base])
    ax.set_xticklabels(workloads)
    ax.set_ylabel("p95 modeled latency (ms)")
    ax.set_title(f"p95 modeled latency by policy/workload @ {cache_mib:.0f} MiB cache")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def plot_throughput(rows: list[dict], output: Path, cache_mib: float) -> None:
    # Third Party
    import matplotlib.pyplot as plt  # noqa: PLC0415

    workloads = _workloads(rows)
    policies = _policies(rows)
    target_bytes = cache_mib * _MIB

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(policies)
    x_base = range(len(workloads))

    for i, policy in enumerate(policies):
        vals = []
        for workload in workloads:
            match = [
                r
                for r in rows
                if r["workload_name"] == workload
                and r["policy_name"] == policy
                and abs(float(r["cache_capacity_bytes"]) - target_bytes) < 1
            ]
            vals.append(float(match[0]["requests_per_second"]) if match else 0.0)
        xs = [x + i * width for x in x_base]
        ax.bar(xs, vals, width=width, label=policy)

    ax.set_xticks([x + 0.4 - width / 2 for x in x_base])
    ax.set_xticklabels(workloads)
    ax.set_ylabel("Requests / second (simulator throughput)")
    ax.set_title(f"Simulator throughput by policy/workload @ {cache_mib:.0f} MiB cache")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="sweep_results.csv path")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory for PNGs")
    parser.add_argument(
        "--cache-mib",
        type=float,
        default=100.0,
        help="Cache size (MiB) to use for the latency/throughput bar charts",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_hit_rate_vs_cache_size(rows, out_dir / "hit_rate_vs_cache_size.png")
    plot_latency_p95(rows, out_dir / "latency_p95.png", args.cache_mib)
    plot_throughput(rows, out_dir / "throughput.png", args.cache_mib)


if __name__ == "__main__":
    main()
