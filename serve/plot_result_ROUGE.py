import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --- Parse command-line arguments for plot filename (only used for naming first plot) ---
parser = argparse.ArgumentParser(
    description="Plot Average ROUGEL vs Average ttft from CSV results"
)
parser.add_argument(
    "--plot-filename",
    type=str,
    default="all_results.pdf",
    help="Filename for saving the 'all_results' plot"
)
args = parser.parse_args()

# File lists
file_paths_kivi = [
    'results/May_23_1_sum/baseline_kivi/02_processed.csv',
    'results/May_23_1_sum/baseline_kivi/03_processed.csv',
    'results/May_23_1_sum/baseline_kivi/06_processed.csv',
]
file_paths_ours = [
]
file_paths_prefill = [
    'results/May_23_1_sum/prefill/0_processed.csv'
]
file_paths_streaming = [
    'results/May_23_1_sum/baseline_streaming/02_processed.csv',
    'results/May_23_1_sum/baseline_streaming/03_processed.csv',
    'results/May_23_1_sum/baseline_streaming/06_processed.csv',
]
file_paths_offload = [
    'results/May_23_1_sum/prefill/1_processed.csv'
]

def load_metrics(file_list, filter_first=False):
    """
    Load CSVs from file_list and compute mean 'ttft' and 'ROUGEL'.
    If filter_first=True, drop all rows where occurrence_number == 1.
    """
    ttft_vals, f1_vals, labels = [], [], []
    for path in file_list:
        df = pd.read_csv(path)
        if filter_first:
            df = df[df["occurrence_number"] != 1]
        ttft_vals.append(df["ttft"].mean())
        f1_vals.append(df["ROUGEL"].mean())
        labels.append(path.split('/')[-1])
        print(f"File: {path}")
        if filter_first:
            print("  (filtered out occurrence_number == 1)")
        print(f"  Average ttft: {ttft_vals[-1]:.2f}")
        print(f"  Average ROUGEL: {f1_vals[-1]:.4f}")
    return ttft_vals, f1_vals, labels

# --- 1) Compute metrics for "all_results" (no filtering) ---
ttft_base_all, f1_base_all, labels_base_all       = load_metrics(file_paths_kivi,     filter_first=False)
ttft_ours_all, f1_ours_all, labels_ours_all       = load_metrics(file_paths_ours,     filter_first=False)
ttft_pre_all, f1_pre_all, labels_pre_all          = load_metrics(file_paths_prefill,  filter_first=False)
ttft_cat4_all, f1_cat4_all, labels_cat4_all       = load_metrics(file_paths_streaming,filter_first=False)
ttft_cat5_all, f1_cat5_all, labels_cat5_all       = load_metrics(file_paths_offload, filter_first=False)

# Plot "all_results"
plt.figure(figsize=(8, 6))

# Baseline
if file_paths_kivi:
    plt.plot(
        ttft_base_all, f1_base_all,
        color='tab:blue',
        marker='o',
        markersize=10,
        linewidth=5,
        label='KIVI LRU'
    )

# Ours
if file_paths_ours:
    plt.plot(
        ttft_ours_all, f1_ours_all,
        color='tab:orange',
        marker='^',
        markersize=10,
        linewidth=5,
        label='Ours'
    )

# Prefill
if file_paths_prefill:
    plt.plot(
        ttft_pre_all, f1_pre_all,
        color='tab:green',
        marker='D',
        markersize=10,
        linewidth=5,
        label='Prefill'
    )

# Category 4 (StreamingLLM LRU)
if file_paths_streaming:
    plt.plot(
        ttft_cat4_all, f1_cat4_all,
        color='tab:pink',
        marker='s',
        markersize=10,
        linewidth=5,
        label='StreamingLLM LRU'
    )

# Category 5 (Offload)
if file_paths_offload:
    plt.plot(
        ttft_cat5_all, f1_cat5_all,
        color='tab:red',
        marker='X',
        markersize=10,
        linewidth=5,
        label='Offload'
    )

plt.xlabel("Average Delay (s)", fontsize=16)
plt.ylabel("Average ROUGE-L Score", fontsize=16)
plt.title("Dataset: qmsum & samsum", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.xlim(xmin=0)

# Save the "all_results" figure
plt.savefig(args.plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {args.plot_filename}")

# --- 2) Compute metrics for "no_first" (filter out occurrence_number == 1) ---
ttft_base_nf, f1_base_nf, labels_base_nf       = load_metrics(file_paths_kivi,     filter_first=True)
ttft_ours_nf, f1_ours_nf, labels_ours_nf       = load_metrics(file_paths_ours,     filter_first=True)
ttft_pre_nf, f1_pre_nf, labels_pre_nf          = load_metrics(file_paths_prefill,  filter_first=True)
ttft_cat4_nf, f1_cat4_nf, labels_cat4_nf       = load_metrics(file_paths_streaming,filter_first=True)
ttft_cat5_nf, f1_cat5_nf, labels_cat5_nf       = load_metrics(file_paths_offload, filter_first=True)

# Plot "no_first"
plt.figure(figsize=(8, 6))

# Baseline
if file_paths_kivi:
    plt.plot(
        ttft_base_nf, f1_base_nf,
        color='tab:blue',
        marker='o',
        markersize=10,
        linewidth=5,
        label='KIVI LRU'
    )

# Ours
if file_paths_ours:
    plt.plot(
        ttft_ours_nf, f1_ours_nf,
        color='tab:orange',
        marker='^',
        markersize=10,
        linewidth=5,
        label='Ours'
    )

# Prefill
if file_paths_prefill:
    plt.plot(
        ttft_pre_nf, f1_pre_nf,
        color='tab:green',
        marker='D',
        markersize=10,
        linewidth=5,
        label='Prefill'
    )

# Category 4 (StreamingLLM LRU)
if file_paths_streaming:
    plt.plot(
        ttft_cat4_nf, f1_cat4_nf,
        color='tab:pink',
        marker='s',
        markersize=10,
        linewidth=5,
        label='StreamingLLM LRU'
    )

# Category 5 (Offload)
if file_paths_offload:
    plt.plot(
        ttft_cat5_nf, f1_cat5_nf,
        color='tab:red',
        marker='X',
        markersize=10,
        linewidth=5,
        label='Offload'
    )

plt.xlabel("Average Delay (s)", fontsize=16)
plt.ylabel("Average ROUGE-L Score", fontsize=16)
plt.title("Dataset: qmsum & samsum", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.xlim(xmin=0)

# Save the "no_first" figure
plt.savefig("no_first.pdf", dpi=300, bbox_inches="tight")
print("Plot saved as no_first.pdf")
