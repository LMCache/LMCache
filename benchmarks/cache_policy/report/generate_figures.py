# SPDX-License-Identifier: Apache-2.0
"""
Render every figure used in the cache-policy evaluation report from the
already-committed benchmark result artifacts (no new sweeps run here,
except a small one-off replay for the freeze-illustration figure --
cheap enough to redo on every report build rather than caching another
result file for one plot).

Usage::

    python benchmarks/cache_policy/report/generate_figures.py \\
        -o benchmarks/cache_policy/report/figures
"""

# Standard
from pathlib import Path
import argparse
import json

# Third Party
import matplotlib

matplotlib.use("Agg")
# Third Party
import matplotlib.pyplot as plt  # noqa: E402

# First Party
from lmcache.tools.cache_policy_bench.cost_model import (  # noqa: E402
    CostModel,
    CostModelConfig,
)
from lmcache.tools.cache_policy_bench.runner import (  # noqa: E402
    DEFAULT_KV_BYTES_PER_CHUNK,
    run_workload,
)
from lmcache.tools.cache_policy_bench.workloads import novel_long  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

_MIB = 2**20
_RESULTS = Path("benchmarks/cache_policy/results")
_AC_RESULTS = _RESULTS / "admission_control"
_REAL_DATA = _RESULTS / "real_data"

_LINE_POLICIES = ["LRU", "LFU", "COST_AWARE", "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU"]
_COLORS = {
    "LRU": "#888888",
    "LFU": "#1f77b4",
    "FIFO": "#9467bd",
    "MRU": "#7f7f7f",
    "COST_AWARE": "#ff7f0e",
    "ADMISSION_LRU": "#2ca02c",
    "WINDOWED_ADMISSION_LRU": "#d62728",
    "ADMISSION_COST_AWARE": "#8c564b",
    "WINDOWED_ADMISSION_COST_AWARE": "#e377c2",
}
_MARKERS = {
    "LRU": "o",
    "LFU": "s",
    "FIFO": "^",
    "MRU": "v",
    "COST_AWARE": "D",
    "ADMISSION_LRU": "P",
    "WINDOWED_ADMISSION_LRU": "X",
    "ADMISSION_COST_AWARE": "*",
    "WINDOWED_ADMISSION_COST_AWARE": "h",
}


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cache_mib(row: dict) -> float:
    return int(row["cache_capacity_bytes"]) / _MIB


def fig_hit_rate_vs_cache_size(rows: list[dict], out: Path) -> None:
    workloads = ["repetitive_short", "novel_long", "mixed_zipfian", "multi_round_chat"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for ax, workload in zip(axes.flat, workloads, strict=False):
        for policy in _LINE_POLICIES:
            pts = sorted(
                (
                    (_cache_mib(r), float(r["token_hit_rate"]))
                    for r in rows
                    if r["workload_name"] == workload and r["policy_name"] == policy
                ),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts, strict=False)
            ax.plot(
                xs,
                [y * 100 for y in ys],
                marker=_MARKERS[policy],
                color=_COLORS[policy],
                label=policy,
                linewidth=1.8,
            )
        ax.set_title(workload)
        ax.set_xlabel("Cache size (MiB)")
        ax.set_xticks([50, 100, 200])
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Token hit rate (%)")
    axes[1, 0].set_ylabel("Token hit rate (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Hit rate vs. cache size, by workload (synthetic sweep)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "fig1_hit_rate_vs_cache_size.png")
    plt.close(fig)


def fig_latency_vs_cache_size(rows: list[dict], out: Path) -> None:
    workloads = ["mixed_zipfian", "multi_round_chat"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, workload in zip(axes, workloads, strict=False):
        for policy in _LINE_POLICIES:
            pts = sorted(
                (
                    (_cache_mib(r), float(r["latency_p95_seconds"]) * 1000)
                    for r in rows
                    if r["workload_name"] == workload and r["policy_name"] == policy
                ),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts, strict=False)
            ax.plot(
                xs, ys, marker=_MARKERS[policy], color=_COLORS[policy],
                label=policy, linewidth=1.8,
            )
        ax.set_title(workload)
        ax.set_xlabel("Cache size (MiB)")
        ax.set_ylabel("Modeled p95 latency (ms)")
        ax.set_xticks([50, 100, 200])
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Modeled p95 latency vs. cache size")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out / "fig2_latency_p95_vs_cache_size.png")
    plt.close(fig)


def fig_eviction_and_rejections(rows: list[dict], out: Path) -> None:
    policies = [
        "LRU", "LFU", "FIFO", "MRU", "COST_AWARE",
        "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU",
    ]
    target = 100 * _MIB
    evictions, rejections = [], []
    for p in policies:
        match = [
            r for r in rows
            if r["workload_name"] == "mixed_zipfian"
            and r["policy_name"] == p
            and int(r["cache_capacity_bytes"]) == target
        ]
        r = match[0]
        evictions.append(int(r["eviction_count"]))
        rejections.append(int(r.get("param_rejected_admissions", 0) or 0))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(policies))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x], evictions, width, label="Evictions",
        color="#1f77b4",
    )
    ax.bar(
        [i + width / 2 for i in x], rejections, width, label="Rejected admissions",
        color="#d62728",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylabel("Count over 3,000 requests")
    ax.set_title("Evictions vs. rejected admissions -- mixed_zipfian, 100 MiB")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig3_evictions_vs_rejections.png")
    plt.close(fig)


def fig_real_data_ci(ci_rows: list[dict], out: Path) -> None:
    scales = ["500", "2000", "5000"]
    policies_lru = ["LRU", "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU"]
    policies_cost = [
        "COST_AWARE", "ADMISSION_COST_AWARE", "WINDOWED_ADMISSION_COST_AWARE",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    titles = ["LRU family", "COST_AWARE family"]
    for ax, policies, title in zip(
        axes, [policies_lru, policies_cost], titles, strict=False
    ):
        width = 0.8 / len(policies)
        for i, p in enumerate(policies):
            means, los, his, xs = [], [], [], []
            for si, scale in enumerate(scales):
                match = [
                    r for r in ci_rows
                    if r["policy_name"] == p
                    and r["max_conversations"] == scale
                    and int(r["cache_capacity_bytes"]) == 200 * _MIB
                ]
                if not match:
                    continue
                m = match[0]
                means.append(float(m["hit_rate_mean"]) * 100)
                los.append(
                    (float(m["hit_rate_mean"]) - float(m["hit_rate_ci_lo"])) * 100
                )
                his.append(
                    (float(m["hit_rate_ci_hi"]) - float(m["hit_rate_mean"])) * 100
                )
                xs.append(si + i * width)
            ax.bar(
                xs, means, width, yerr=[los, his], capsize=3,
                label=p, color=_COLORS[p],
            )
        ax.set_xticks([i + width for i in range(len(scales))])
        ax.set_xticklabels([f"{s} conv." for s in scales])
        ax.set_title(f"{title} -- 200 MiB cache")
        ax.set_ylabel("Token hit rate (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "Real ShareGPT data: hit rate with 95% bootstrap CI (6 repeats), 200 MiB"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig4_real_data_ci_200mib.png")
    plt.close(fig)


def fig_zipf_robustness(rows: list[dict], out: Path) -> None:
    policies = ["LRU", "LFU", "COST_AWARE", "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in policies:
        pts = []
        for r in rows:
            if r["policy_name"] != p:
                continue
            # workload_name looks like "mixed_zipfian[zipf_s=1.2]"
            tag = r["workload_name"]
            if "zipf_s=" not in tag:
                continue
            zipf_s = float(tag.split("zipf_s=")[1].rstrip("]"))
            pts.append((zipf_s, float(r["token_hit_rate"]) * 100))
        pts.sort()
        if not pts:
            continue
        xs, ys = zip(*pts, strict=False)
        ax.plot(xs, ys, marker=_MARKERS[p], color=_COLORS[p], label=p, linewidth=1.8)
    ax.set_xlabel("Zipf skew parameter (zipf_s)")
    ax.set_ylabel("Token hit rate (%)")
    ax.set_title("Hit rate vs. Zipf skew strength (mixed_zipfian, 100 MiB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig5_zipf_robustness.png")
    plt.close(fig)


def fig_ablation(
    admission_rows: list[dict], windowed_rows: list[dict], out: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    variants = ["fast_decay", "default", "slow_decay"]
    workloads = ["mixed_zipfian", "multi_round_chat"]
    width = 0.35
    for wi, workload in enumerate(workloads):
        ys = []
        for v in variants:
            match = [
                r for r in admission_rows
                if r["workload_name"] == f"{workload}[{v}]"
            ]
            ys.append(float(match[0]["token_hit_rate"]) * 100 if match else 0.0)
        xs = [i + wi * width for i in range(len(variants))]
        ax.bar(xs, ys, width, label=workload)
    ax.set_xticks([i + width / 2 for i in range(len(variants))])
    ax.set_xticklabels(["fast\n(2k)", "default\n(20k)", "slow\n(200k)"])
    ax.set_xlabel("halve_every")
    ax.set_ylabel("Token hit rate (%)")
    ax.set_title("ADMISSION_LRU: halve_every ablation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    variants = [
        "tiny_window", "default", "large_window",
        "lenient_promotion", "strict_promotion",
    ]
    labels = [
        "tiny\nwin=5", "default\nwin=20,t=2", "large\nwin=80",
        "lenient\nt=1", "strict\nt=4",
    ]
    for wi, workload in enumerate(workloads):
        ys = []
        for v in variants:
            match = [
                r for r in windowed_rows
                if r["workload_name"] == f"{workload}[windowed_{v}]"
            ]
            ys.append(float(match[0]["token_hit_rate"]) * 100 if match else 0.0)
        xs = [i + wi * width for i in range(len(variants))]
        ax.bar(xs, ys, width, label=workload)
    ax.set_xticks([i + width / 2 for i in range(len(variants))])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Token hit rate (%)")
    ax.set_title(
        "WINDOWED_ADMISSION_LRU: window_capacity / promotion_threshold ablation"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out / "fig6_ablation.png")
    plt.close(fig)


def fig_freeze_illustration(out: Path) -> None:
    requests = novel_long(500, min_tokens=2048, max_tokens=4096, chunk_size=256, seed=0)
    cost_model = CostModel(CostModelConfig())
    small_cache_bytes = 2 * 1024 * 1024
    policies = ["LRU", "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU"]
    evictions = []
    for p in policies:
        result = run_workload(
            p, requests, small_cache_bytes, DEFAULT_KV_BYTES_PER_CHUNK, cost_model,
            workload_name="freeze_check",
        )
        evictions.append(result.eviction_count)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bars = ax.bar(policies, evictions, color=[_COLORS[p] for p in policies])
    for bar, v in zip(bars, evictions, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), str(v),
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Eviction count over the run")
    ax.set_title("Purely one-shot traffic (novel_long): freeze illustration")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig7_freeze_illustration.png")
    plt.close(fig)


def fig_latency_distribution(raw_rows: list[dict], out: Path) -> None:
    policies = ["LRU", "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU"]
    target_bytes = 100 * _MIB
    data = []
    for p in policies:
        vals = [
            float(r["latency_p95_seconds"]) * 1000
            for r in raw_rows
            if r["policy_name"] == p
            and r["max_conversations"] == "500"
            and int(r["cache_capacity_bytes"]) == target_bytes
        ]
        data.append(vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, tick_labels=policies, patch_artist=True, widths=0.5)
    for patch, p in zip(bp["boxes"], policies, strict=False):
        patch.set_facecolor(_COLORS[p])
        patch.set_alpha(0.6)
    ax.set_ylabel("Modeled p95 latency (ms)")
    ax.set_title(
        "Distribution of p95 latency across 6 bootstrap repeats\n"
        "(real ShareGPT, 500 conversations, 100 MiB)"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig8_latency_distribution.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output-dir", default="benchmarks/cache_policy/report/figures"
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sweep_rows = _load(_AC_RESULTS / "sweep_results.json")
    zipf_rows = _load(_AC_RESULTS / "robustness_zipf_skew.json")
    admission_ablation = _load(_AC_RESULTS / "admission_control_ablation.json")
    windowed_ablation = _load(_AC_RESULTS / "windowed_admission_control_ablation.json")
    real_ci = _load(_REAL_DATA / "real_dataset_ci.json")
    real_raw = _load(_REAL_DATA / "real_dataset_raw.json")

    fig_hit_rate_vs_cache_size(sweep_rows, out)
    fig_latency_vs_cache_size(sweep_rows, out)
    fig_eviction_and_rejections(sweep_rows, out)
    fig_real_data_ci(real_ci, out)
    fig_zipf_robustness(zipf_rows, out)
    fig_ablation(admission_ablation, windowed_ablation, out)
    fig_freeze_illustration(out)
    fig_latency_distribution(real_raw, out)

    print(f"Wrote 8 figures to {out}")


if __name__ == "__main__":
    main()
