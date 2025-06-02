import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --- Parse command-line arguments for plot filename ---
parser = argparse.ArgumentParser(
    description="Plot Average ROUGEL vs Average ttft from CSV results"
)
parser.add_argument(
    "--plot-filename",
    type=str,
    default="ttft_vs_ROUGEL_similarity.pdf",
    help="Filename for saving the plot"
)
args = parser.parse_args()

# File lists
file_paths_baseline = [
    '/home/ubuntu/st-prodstack-v/press/qmsum/results_rate_0.271428571_processed.csv',
    '/home/ubuntu/st-prodstack-v/press/qmsum/results_rate_0.514285714_processed.csv',
    '/home/ubuntu/st-prodstack-v/press/qmsum/results_rate_0.628571429_processed.csv',
    # 'results/Apr_14/baseline_kivi/1_processed.csv',
]
file_paths_ours = [
    'results/May_13_1/ours/01_processed2.csv',
    'results/May_13_1/ours/04_processed2.csv',
    'results/May_13_1/ours/07_processed2.csv',
    'results/May_13_1/ours/1_processed2.csv',
    'results/May_13_1/ours/10_processed2.csv',
]
file_paths_prefill = [
    'results/Apr_14/baseline_kivi/0_processed.csv'
]

def load_metrics(file_list):
    ttft_vals, f1_vals, labels = [], [], []
    for path in file_list:
        df = pd.read_csv(path)
        ttft_vals.append(df["ttft"].mean())
        f1_vals.append(df["ROUGEL"].mean())
        labels.append(path.split('/')[-1])
        print(f"File: {path}")
        print(f"  Average ttft: {ttft_vals[-1]:.2f}")
        print(f"  Average ROUGEL: {f1_vals[-1]:.4f}")
    return ttft_vals, f1_vals, labels

# Load all three series without sorting
ttft_base, f1_base, labels_base = load_metrics(file_paths_baseline)
ttft_ours, f1_ours, labels_ours     = load_metrics(file_paths_ours)
ttft_pre, f1_pre, labels_pre       = load_metrics(file_paths_prefill)

# --- Plot all three series in original order ---
plt.figure(figsize=(8, 6))

# Baseline
plt.plot(
    ttft_base, f1_base,
    color='tab:blue',        # 论文常用的蓝色
    marker='o',              # 圆形
    markersize=10,            # 点大小
    linewidth=5,             # 线宽
    label='LRU'
)
plt.plot(
    ttft_ours, f1_ours,
    color='tab:orange',      # 论文常用的橙色
    marker='^',              # X 形
    markersize=10,
    linewidth=5,
    label='Ours'
)
plt.plot(
    ttft_pre, f1_pre,
    color='tab:red',       # 论文常用的绿色
    marker='X',              # 菱形
    markersize=10,
    linewidth=5,
    label='Prefill'
)

plt.xlabel("Average Delay (s)", fontsize=16)
plt.ylabel("Average ROUGE-L Score", fontsize=16)
plt.title("Compression Method: StreamingLLM", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.xlim(xmin=0)
# plt.ylim(ymax=1)

# Save the figure
plt.savefig(args.plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {args.plot_filename}")
