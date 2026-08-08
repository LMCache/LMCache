# SPDX-License-Identifier: Apache-2.0
"""
Render every figure used in the cache-policy evaluation report from the
already-committed benchmark result artifacts (no new sweeps run here,
except a small one-off replay for the freeze-illustration figure --
cheap enough to redo on every report build rather than caching another
result file for one plot).

Uses the multi-seed/paired-comparison result files as the source for any
figure that supports a statistical claim (see
``benchmarks/cache_policy/main_sweep_multiseed.py`` and
``benchmarks/cache_policy/real_dataset_eval.py``) -- single-run sweep
data is used only for figures that are explicitly illustrative
(mechanism figures, not "policy X beats policy Y" claims), and is
labeled as such in its own caption text here rather than left for the
report prose to clarify.

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


def fig_hit_rate_vs_cache_size_multiseed(ci_rows: list[dict], out: Path) -> None:
    """Mean hit rate +/- 95% CI across 10 seeds, seed-capable workloads only."""
    workloads = ["repetitive_short", "novel_long", "mixed_zipfian"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, workload in zip(axes, workloads, strict=False):
        for policy in _LINE_POLICIES:
            pts = sorted(
                (
                    (
                        _cache_mib(r),
                        float(r["hit_rate_mean"]),
                        float(r["hit_rate_mean"]) - float(r["hit_rate_ci_lo"]),
                        float(r["hit_rate_ci_hi"]) - float(r["hit_rate_mean"]),
                    )
                    for r in ci_rows
                    if r["workload_name"] == workload and r["policy_name"] == policy
                ),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] * 100 for p in pts]
            lo = [p[2] * 100 for p in pts]
            hi = [p[3] * 100 for p in pts]
            ax.errorbar(
                xs,
                ys,
                yerr=[lo, hi],
                marker=_MARKERS[policy],
                color=_COLORS[policy],
                label=policy,
                linewidth=1.6,
                capsize=3,
                elinewidth=1,
            )
        ax.set_title(workload)
        ax.set_xlabel("Cache size (MiB)")
        ax.set_xticks([50, 100, 200])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Token hit rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Mean token hit rate with 95% bootstrap CI across 10 independent seeds")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out / "fig1_hit_rate_vs_cache_size_multiseed.png")
    plt.close(fig)


def fig_latency_vs_cache_size_multiseed(ci_rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in _LINE_POLICIES:
        pts = sorted(
            (
                (_cache_mib(r), float(r["latency_p95_mean_seconds"]) * 1000)
                for r in ci_rows
                if r["workload_name"] == "mixed_zipfian" and r["policy_name"] == policy
            ),
            key=lambda t: t[0],
        )
        if not pts:
            continue
        xs, ys = zip(*pts, strict=False)
        ax.plot(
            xs,
            ys,
            marker=_MARKERS[policy],
            color=_COLORS[policy],
            label=policy,
            linewidth=1.8,
        )
    ax.set_xlabel("Cache size (MiB)")
    ax.set_ylabel("Mean analytical p95 estimate (ms), across 10 seeds")
    ax.set_title("Analytical recomputation-latency estimate vs. cache size (mixed_zipfian)")
    ax.set_xticks([50, 100, 200])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_latency_p95_vs_cache_size_multiseed.png")
    plt.close(fig)


def fig_eviction_and_rejections(rows: list[dict], out: Path) -> None:
    """Single-run illustration of the eviction/rejection mechanism."""
    policies = [
        "LRU",
        "LFU",
        "FIFO",
        "MRU",
        "COST_AWARE",
        "ADMISSION_LRU",
        "WINDOWED_ADMISSION_LRU",
    ]
    target = 100 * _MIB
    evictions, rejections = [], []
    for p in policies:
        match = [
            r
            for r in rows
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
        [i - width / 2 for i in x],
        evictions,
        width,
        label="Evictions",
        color="#1f77b4",
    )
    ax.bar(
        [i + width / 2 for i in x],
        rejections,
        width,
        label="Rejected admissions",
        color="#d62728",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylabel("Chunk-level events over 3,000 requests")
    ax.set_title(
        "Evictions vs. rejected admissions -- mixed_zipfian, 100 MiB\n"
        "(single run; see Figure 1 for the CI'd hit-rate effect)"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig3_evictions_vs_rejections.png")
    plt.close(fig)


def fig_real_data_paired_diff(paired_rows: list[dict], out: Path) -> None:
    """Paired hit-rate difference vs. LRU, with 95% CI, at 200 MiB across scales."""
    scales = ["500", "2000", "5000"]
    policies = [
        "LFU",
        "COST_AWARE",
        "ADMISSION_LRU",
        "WINDOWED_ADMISSION_LRU",
    ]
    present = sorted(
        {r["policy_name"] for r in paired_rows if r["policy_name"] in policies},
        key=lambda p: policies.index(p),
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / max(1, len(present))
    for i, p in enumerate(present):
        means, los, his, xs = [], [], [], []
        for si, scale in enumerate(scales):
            match = [
                r
                for r in paired_rows
                if r["policy_name"] == p
                and r["max_conversations"] == scale
                and int(r["cache_capacity_bytes"]) == 200 * _MIB
            ]
            if not match:
                continue
            m = match[0]
            diff_mean = float(m["hit_rate_diff_mean"]) * 100
            means.append(diff_mean)
            los.append(diff_mean - float(m["hit_rate_diff_ci_lo"]) * 100)
            his.append(float(m["hit_rate_diff_ci_hi"]) * 100 - diff_mean)
            xs.append(si + i * width)
        color = _COLORS.get(p, None)
        ax.bar(xs, means, width, yerr=[los, his], capsize=2, label=p, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([i + width * (len(present) - 1) / 2 for i in range(len(scales))])
    ax.set_xticklabels([f"{s} conv." for s in scales])
    ax.set_ylabel("Paired hit-rate difference vs. LRU (percentage points)")
    ax.set_title(
        "ShareGPT-derived round-robin replay: paired hit-rate difference vs. LRU\n"
        "95% paired-bootstrap CI across 6 matched subsamples"
    )
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig4_real_data_paired_diff_200mib.png")
    plt.close(fig)


def fig_zipf_robustness(rows: list[dict], out: Path) -> None:
    """Single-run-per-point sweep across Zipf skew (illustrative, not a CI'd claim)."""
    policies = ["LRU", "LFU", "COST_AWARE", "ADMISSION_LRU", "WINDOWED_ADMISSION_LRU"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in policies:
        pts = []
        for r in rows:
            if r["policy_name"] != p:
                continue
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
    ax.set_ylabel("Token hit rate (%), single run per point")
    ax.set_title("Single-run sensitivity to Zipf skew (mixed_zipfian, 100 MiB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig5_zipf_robustness.png")
    plt.close(fig)


def fig_ablation(
    admission_rows: list[dict], windowed_rows: list[dict], out: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    variants = ["fast_decay", "default", "slow_decay"]
    workloads = ["mixed_zipfian"]
    width = 0.55
    for wi, workload in enumerate(workloads):
        ys, halvings = [], []
        for v in variants:
            match = [
                r for r in admission_rows if r["workload_name"] == f"{workload}[{v}]"
            ]
            ys.append(float(match[0]["token_hit_rate"]) * 100 if match else 0.0)
            halvings.append(
                match[0].get("param_sketch_halvings_triggered", "?") if match else "?"
            )
        xs = [i + wi * width for i in range(len(variants))]
        bars = ax.bar(xs, ys, width, label=workload)
        for bar, h in zip(bars, halvings, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"h={h}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks([i + width / 2 for i in range(len(variants))])
    ax.set_xticklabels(["fast\n(5k)", "default\n(20k)", "slow\n(80k)"])
    ax.set_xlabel("halve_every  (h=actual halving passes triggered)")
    ax.set_ylabel("Token hit rate (%)")
    ax.set_title("ADMISSION_LRU: halve_every sensitivity (single run)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    variants = [
        "tiny_window",
        "default",
        "large_window",
        "lenient_promotion",
        "strict_promotion",
    ]
    labels = [
        "tiny\nwin=5",
        "default\nwin=20,t=2",
        "large\nwin=80",
        "lenient\nt=1",
        "strict\nt=4",
    ]
    for wi, workload in enumerate(workloads):
        ys = []
        for v in variants:
            match = [
                r
                for r in windowed_rows
                if r["workload_name"] == f"{workload}[windowed_{v}]"
            ]
            ys.append(float(match[0]["token_hit_rate"]) * 100 if match else 0.0)
        xs = [i + wi * width for i in range(len(variants))]
        ax.bar(xs, ys, width, label=workload)
    ax.set_xticks([i + width / 2 for i in range(len(variants))])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Token hit rate (%)")
    ax.set_title(
        "WINDOWED_ADMISSION_LRU: parameter sensitivity (single run)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out / "fig6_ablation.png")
    plt.close(fig)


def fig_multi_round_chat_case_study(rows: list[dict], out: Path) -> None:
    """Deterministic case study across structural parameters (not stats evidence)."""
    variants = ["default", "fewer_longer_sessions", "more_shorter_sessions"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, variant in zip(axes, variants, strict=False):
        for policy in _LINE_POLICIES:
            pts = sorted(
                (
                    (_cache_mib(r), float(r["token_hit_rate"]) * 100)
                    for r in rows
                    if r["variant"] == variant and r["policy_name"] == policy
                ),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts, strict=False)
            ax.plot(
                xs,
                ys,
                marker=_MARKERS[policy],
                color=_COLORS[policy],
                label=policy,
                linewidth=1.6,
            )
        ax.set_title(variant, fontsize=10)
        ax.set_xlabel("Cache size (MiB)")
        ax.set_xticks([50, 100, 200])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Token hit rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "multi_round_chat deterministic case study "
        "(single run per point -- not statistical evidence)"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(out / "fig9_multi_round_chat_case_study.png")
    plt.close(fig)


def fig_freeze_illustration(out: Path) -> None:
    """Deterministic one-shot stress test showing turnover and rejection."""
    requests = novel_long(
        500,
        min_tokens=2048,
        max_tokens=4096,
        chunk_size=256,
        seed=0,
    )
    cost_model = CostModel(CostModelConfig())
    small_cache_bytes = 2 * 1024 * 1024

    policies = [
        "LRU",
        "ADMISSION_LRU",
        "WINDOWED_ADMISSION_LRU",
    ]

    evictions: list[int] = []
    rejections: list[int] = []

    for policy_name in policies:
        result = run_workload(
            policy_name,
            requests,
            small_cache_bytes,
            DEFAULT_KV_BYTES_PER_CHUNK,
            cost_model,
            workload_name="freeze_check",
        )

        evictions.append(int(result.eviction_count))
        rejections.append(
            int(result.extra_params.get("rejected_admissions", 0))
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    x = list(range(len(policies)))
    width = 0.36

    eviction_bars = ax.bar(
        [i - width / 2 for i in x],
        evictions,
        width,
        label="Evictions",
        color="#1f77b4",
    )

    rejection_bars = ax.bar(
        [i + width / 2 for i in x],
        rejections,
        width,
        label="Rejected admissions",
        color="#d62728",
    )

    ax.bar_label(eviction_bars, padding=3, fontsize=9)
    ax.bar_label(rejection_bars, padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Chunk-level events over 500 requests")
    ax.set_title(
        "One-shot traffic: cache turnover vs. admission rejection\n"
        "(novel_long, deterministic stress test)"
    )
    ax.legend()
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
    ax.set_ylabel("Analytical p95 estimate (ms)")
    ax.set_title(
        "Per-subsample analytical p95 estimates across 6 matched replays\n"
        "(ShareGPT-derived round-robin replay, 500 conversations, 100 MiB)"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig8_latency_distribution.png")
    plt.close(fig)


def fig_admission_vs_lfu(paired_comparison: dict, out: Path) -> None:
    """
    ADMISSION_LRU vs. LFU paired hit-rate difference across cache sizes,
    from ``compare_admission_vs_lfu.py``'s output -- no hardcoded diff
    numbers, everything here is read from that committed JSON.
    """
    analysis = paired_comparison["analysis"]
    comparisons = sorted(
        paired_comparison["comparisons"], key=lambda c: c["cache_capacity_mib"]
    )

    xs = list(range(len(comparisons)))
    means = [c["hit_rate_diff_percentage_points"] for c in comparisons]
    los = [
        c["hit_rate_diff_percentage_points"] - c["hit_rate_diff_ci_lo"] * 100
        for c in comparisons
    ]
    his = [
        c["hit_rate_diff_ci_hi"] * 100 - c["hit_rate_diff_percentage_points"]
        for c in comparisons
    ]
    colors = ["#2ca02c" if c["holm_reject_at_p05"] else "#888888" for c in comparisons]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xs, means, yerr=[los, his], capsize=4, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c['cache_capacity_mib']:.0f} MiB" for c in comparisons])
    ax.set_ylabel(
        f"{analysis['candidate_policy']} - {analysis['baseline_policy']} "
        "hit rate (percentage points)"
    )
    n_seeds = comparisons[0]["n_seeds"] if comparisons else 0
    ax.set_title(
        f"{analysis['candidate_policy']} vs. {analysis['baseline_policy']} "
        f"({analysis['workload_name']}), paired 95% CI, n={n_seeds} paired seeds\n"
        "Green = Holm-significant at alpha=0.05 across the 3-capacity family"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig10_admission_vs_lfu.png")
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
    multiseed_ci = _load(_AC_RESULTS / "multiseed_sweep_ci.json")
    chat_case_study = _load(_AC_RESULTS / "multi_round_chat_case_study.json")
    zipf_rows = _load(_AC_RESULTS / "robustness_zipf_skew.json")
    admission_ablation = _load(_AC_RESULTS / "admission_control_ablation.json")
    windowed_ablation = _load(_AC_RESULTS / "windowed_admission_control_ablation.json")
    real_paired = _load(_REAL_DATA / "real_dataset_paired_diff.json")
    real_raw = _load(_REAL_DATA / "real_dataset_raw.json")
    with open(_AC_RESULTS / "admission_vs_lfu_paired.json", encoding="utf-8") as f:
        admission_vs_lfu = json.load(f)

    fig_hit_rate_vs_cache_size_multiseed(multiseed_ci, out)
    fig_latency_vs_cache_size_multiseed(multiseed_ci, out)
    fig_eviction_and_rejections(sweep_rows, out)
    fig_real_data_paired_diff(real_paired, out)
    fig_zipf_robustness(zipf_rows, out)
    fig_ablation(admission_ablation, windowed_ablation, out)
    fig_multi_round_chat_case_study(chat_case_study, out)
    fig_freeze_illustration(out)
    fig_latency_distribution(real_raw, out)
    fig_admission_vs_lfu(admission_vs_lfu, out)

    print(f"Wrote 10 figures to {out}")


if __name__ == "__main__":
    main()
