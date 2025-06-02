import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(paths):
    ttft_vals, f1_vals, labels = [], [], []
    for path in paths:
        df = pd.read_csv(path)
        ttft_vals.append(df["ttft"].mean())
        f1_vals.append(df["ROUGEL"].mean())
        labels.append(path.split('/')[-1])
        print(f"Loaded {path}: ttft={ttft_vals[-1]:.2f}, ROUGEL={f1_vals[-1]:.4f}")
    return ttft_vals, f1_vals, labels

def plot_series(ax, baseline_paths, ours_paths, prefill_paths, title):
    # load each series
    ttft_b, f1_b, _ = load_metrics(baseline_paths)
    ttft_o, f1_o, _ = load_metrics(ours_paths)
    ttft_p, f1_p, _ = load_metrics(prefill_paths)

    # plot
    ax.plot(ttft_b, f1_b,
            color='tab:blue', marker='o', markersize=10, linewidth=5, label='LRU')
    ax.plot(ttft_o, f1_o,
            color='tab:orange', marker='^', markersize=10, linewidth=5, label='Ours')
    ax.plot(ttft_p, f1_p,
            color='tab:green', marker='D', markersize=10, linewidth=5, label='Prefill')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Average Delay (s)", fontsize=16)
    ax.set_ylabel("Average ROUGE-L Score", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True)
    ax.set_xlim(left=0)
    # **no** ax.legend() here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ROUGE-L vs TTFT for two compression setups side by side"
    )
    parser.add_argument(
        "--plot-filename",
        type=str,
        default="Figure8.pdf",
        help="Output filename for the combined figure"
    )
    args = parser.parse_args()

    # first panel: processed.csv
    baseline1 = [
        '../results/May_12_1/baseline/02_processed.csv',
        '../results/May_12_1/baseline/03_processed.csv',
        '../results/May_12_1/baseline/06_processed.csv',
    ]
    ours1 = [
        '../results/May_12_1/ours/01_processed.csv',
        '../results/May_12_1/ours/1_processed.csv',
        '../results/May_12_1/ours/10_processed.csv',
    ]
    prefill = [
        '../results/May_10_3/prefill/0_processed.csv'
    ]

    # second panel: processed2.csv
    baseline2 = [
        '../results/May_12_1/baseline/02_processed2.csv',
        '../results/May_12_1/baseline/03_processed2.csv',
        '../results/May_12_1/baseline/06_processed2.csv',
    ]
    ours2 = [
        '../results/May_13_1_samsum_rr/ours/01_processed2.csv',
        '../results/May_13_1_samsum_rr/ours/04_processed2.csv',
        '../results/May_13_1_samsum_rr/ours/1_processed2.csv',
        '../results/May_13_1_samsum_rr/ours/10_processed2.csv',
    ]
    prefill2 = prefill  # same prefill file for both panels

    # create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    # plot each panel
    plot_series(axes[0], baseline1, ours1, prefill,    "Compression Method: KIVI")
    plot_series(axes[1], baseline2, ours2, prefill2,   "Compression Method: StreamingLLM")

    # remove the y-axis label on the second subplot
    axes[1].set_ylabel('')

    # unify y-axis limits across both subplots
    y0 = axes[0].get_ylim()
    y1 = axes[1].get_ylim()
    ymin = min(y0[0], y1[0])
    ymax = max(y0[1], y1[1])
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin, ymax)

    # --- add one shared legend at the top ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=3,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.15)
    )
    # increase top margin so legend doesn't overlap
    fig.subplots_adjust(top=0.80)

    # save figure
    fig.savefig(args.plot_filename, dpi=300, bbox_inches="tight")
    print(f"Saved combined figure as {args.plot_filename}")
