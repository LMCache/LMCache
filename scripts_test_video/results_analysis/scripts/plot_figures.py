from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
            fontsize=10,
        )


def plot_grouped_metrics_pdf(
    csv_path: str | Path,
    pdf_path: str | Path,
    metrics: Iterable[str] = ("precision", "recall", "f1_score"),
    system_col: str = "system",
    category_col: str = "category",
    dpi: int = 120,
) -> None:
    csv_path, pdf_path = Path(csv_path), Path(pdf_path)
    df = pd.read_csv(csv_path)

    required = {system_col, category_col, *set(metrics)}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[system_col] = df[system_col].astype(str)
    df[category_col] = df[category_col].astype(str)

    categories = sorted(df[category_col].unique().tolist())

    def system_sort_key(s: str):
        s = str(s)
        m = re.fullmatch(r"gop(\d+)", s)
        if m:
            return (0, int(m.group(1)))
        return (1, s)

    systems_sorted = sorted(df[system_col].unique().tolist(), key=system_sort_key)

    plot_df = (
        df[[category_col, system_col, *list(metrics)]]
        .groupby([category_col, system_col], as_index=False)
        .mean(numeric_only=True)
    )
    overall_df = (
        df[[system_col, *list(metrics)]]
        .groupby([system_col], as_index=False)
        .mean(numeric_only=True)
        .assign(**{category_col: "Avg"})
    )

    plot_df = pd.concat([plot_df, overall_df], ignore_index=True)
    categories_plot = categories + ["Avg"]

    x = np.arange(len(categories_plot))
    group_w = 0.9
    bar_w = group_w / max(len(systems_sorted), 1)
    cmap = plt.get_cmap("tab20")

    out_prefix = pdf_path.with_suffix("")  # remove .pdf

    for metric in list(metrics):
        fig, ax = plt.subplots(figsize=(7.0, 2.6), constrained_layout=True, dpi=dpi)

        for j, sys in enumerate(systems_sorted):
            sub = plot_df[plot_df[system_col] == sys].set_index(category_col)
            y = [float(sub.loc[c, metric]) if c in sub.index else np.nan for c in categories_plot]
            offsets = x - group_w / 2 + j * bar_w + bar_w / 2
            bars = ax.bar(
                offsets,
                y,
                width=bar_w,
                label=sys,
                color=cmap(j % cmap.N),
                linewidth=0.3,
            )
            _annotate_bars(ax, bars, fmt="{:.2f}", ypad=2)

        title = str(metric).replace("_", " ").replace("-", " ").title()
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(categories_plot, rotation=25, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)

        # legend 放右侧
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            frameon=False,
        )
        fig.subplots_adjust(right=0.84)

        out_path = Path(f"{out_prefix}_{metric}.pdf")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    plot_grouped_metrics_pdf("csv/with_codec_accuracy.csv", "metrics_grouped.pdf", dpi=150)
