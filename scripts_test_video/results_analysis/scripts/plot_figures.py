from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_ratio_num(series: pd.Series) -> Tuple[pd.Series, bool]:
    try:
        return pd.to_numeric(series), True
    except Exception:
        return series.astype(str), False


def _ratio_display_label(r: str) -> str:
    r_str = str(r)
    try:
        r_num = float(r_str)
        if np.isclose(r_num, 0.0):
            return "CacheBlend"
        if np.isclose(r_num, -1.0):
            return "No Reuse"    
        return f"{r_num:g}%"
    except Exception:
        return "CacheBlend" if r_str in {"0", "0.0"} else f"{r_str}%"


def _make_ratio_palette(ratios_sorted: list[str], cmap_name: str = "tab20b"):
    cmap = plt.get_cmap(cmap_name)
    n = max(len(ratios_sorted), 1)
    colors = [cmap(i % cmap.N) for i in range(n)]
    return {r: colors[i] for i, r in enumerate(ratios_sorted)}


def _annotate_bars(ax, bars, fmt="{:.2f}", ypad=2):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, ypad),
            ha="center",
            va="bottom",
            rotation=80,
            fontsize=6,
        )

def plot_grouped_metrics_pdf(
    csv_path: str | Path,
    pdf_path: str | Path,
    metrics: Iterable[str] = ("precision", "recall", "f1-score"),
    dpi: int = 150,
    show: bool = False,
) -> None:
    csv_path, pdf_path = Path(csv_path), Path(pdf_path)
    df = pd.read_csv(csv_path)

    required = {"category", "recompute_ratio", *set(metrics)}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["category"] = df["category"].astype(str)
    df["_ratio_num"], ratio_is_num = _to_ratio_num(df["recompute_ratio"])

    categories = sorted(df["category"].unique().tolist())
    if ratio_is_num:
        ratios_sorted = (
            df[["_ratio_num", "recompute_ratio"]]
            .drop_duplicates()
            .sort_values("_ratio_num")["recompute_ratio"]
            .astype(str)
            .tolist()
        )
    else:
        ratios_sorted = df["recompute_ratio"].astype(str).drop_duplicates().sort_values().tolist()

    plot_df = (
        df[["category", "recompute_ratio", *list(metrics)]]
        .groupby(["category", "recompute_ratio"], as_index=False)
        .mean(numeric_only=True)
    )
    overall_df = (
        df[["recompute_ratio", *list(metrics)]]
        .groupby(["recompute_ratio"], as_index=False)
        .mean(numeric_only=True)
        .assign(category="Avg")
    )

    plot_df = pd.concat([plot_df, overall_df], ignore_index=True)
    categories_plot = categories + ["Avg"]

    mlist = list(metrics)
    fig, axes = plt.subplots(
        1, len(mlist),
        figsize=(4 * len(mlist), 2.6),
        constrained_layout=True,
        dpi=dpi,
    )
    if len(mlist) == 1:
        axes = [axes]

    x = np.arange(len(categories_plot))
    group_w = 0.82
    bar_w = group_w / max(len(ratios_sorted), 1)

    for ax, metric in zip(axes, mlist):
        for j, r in enumerate(ratios_sorted):
            sub = plot_df[plot_df["recompute_ratio"].astype(str) == str(r)].set_index("category")
            y = [float(sub.loc[c, metric]) if c in sub.index else np.nan for c in categories_plot]
            offsets = x - group_w / 2 + j * bar_w + bar_w / 2
            bars = ax.bar(
                offsets,
                y,
                width=bar_w,
                label=_ratio_display_label(r),
                color=plt.cm.tab20b(j / 6),
                linewidth=0.3,
            )
            _annotate_bars(ax, bars, fmt="{:.2f}", ypad=2)

        ax.set_title(str(metric).replace("-", " ").title())
        ax.set_ylabel(str(metric).replace("-", " ").title())
        ax.set_ylim(0, 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(categories_plot, rotation=25, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)

    handles, labels = axes[-1].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h); l2.append(l); seen.add(l)

    fig.legend(
        h2, l2,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.subplots_adjust(right=0.84)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")


def plot_latency_pdf(
    csv_path: str | Path,
    pdf_path: str | Path,
    latency_category: str = "abuse",
    dpi: int = 150,
    show: bool = False,
) -> None:
    csv_path, pdf_path = Path(csv_path), Path(pdf_path)
    df = pd.read_csv(csv_path)

    required = {"category", "recompute_ratio", "latency"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["category"] = df["category"].astype(str)
    df["_ratio_num"], ratio_is_num = _to_ratio_num(df["recompute_ratio"])

    g = (
        df[["category", "recompute_ratio", "latency", "_ratio_num"]]
        .groupby(["category", "recompute_ratio"], as_index=False)
        .mean(numeric_only=True)
    )
    d = g[g["category"] == str(latency_category)].copy()

    fig = plt.figure(figsize=(3.8, 2.4), dpi=dpi)
    if d.empty:
        plt.title(f"No data: category='{latency_category}'")
        plt.axis("off")
    else:
        d = d.sort_values("_ratio_num" if (ratio_is_num and "_ratio_num" in d.columns) else "recompute_ratio")
        ratios = d["recompute_ratio"].astype(str).tolist()
        labels = [_ratio_display_label(r) for r in ratios]
        palette = _make_ratio_palette(ratios, cmap_name="tab20b")
        vals = d["latency"].to_numpy()

        bars = plt.bar(labels, vals, width=0.5, color=[palette[r] for r in ratios], linewidth=0.3)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            plt.annotate(
                f"{v:.2f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                va="bottom",
            )
        plt.ylim(0, max(vals)*1.2)
        plt.xlabel("Recompute Ratio")
        plt.ylabel("Latency (s)")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        plt.tight_layout()

    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    plot_grouped_metrics_pdf("vlcache_dataset.csv", "vlcache_metrics.pdf", dpi=150, show=False)
    # plot_latency_pdf("vlcache_dataset.csv", "vlcache_latency_abuse.pdf", latency_category="abuse", dpi=150, show=False)