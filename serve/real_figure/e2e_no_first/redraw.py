import pandas as pd
import matplotlib.pyplot as plt

# ---- Subplot 1 file paths ----
file_paths1_kivi = [
    '../../results/Jun_4_2_sum/baseline_kivi/02_processed.csv',
    '../../results/Jun_4_2_sum/baseline_kivi/03_processed.csv',
    '../../results/Jun_4_2_sum/baseline_kivi/06_processed.csv',
]
file_paths1_ours = [
    '../../results/Jun_4_2_sum/ours/01_processed_updated.csv',
    '../../results/Jun_4_2_sum/ours/05_processed_updated.csv',
    '../../results/Jun_4_2_sum/ours/1_processed_updated.csv',
    '../../results/Jun_4_2_sum/ours/10_processed_updated.csv',
]
file_paths1_prefill = [
    '../../results/Jun_4_2_sum/prefill/0_processed.csv'
]
file_paths1_streaming = [
    '../../results/Jun_4_2_sum/baseline_streaming/02_processed.csv',
    '../../results/Jun_4_2_sum/baseline_streaming/03_processed.csv',
    '../../results/Jun_4_2_sum/baseline_streaming/06_processed.csv',
]
file_paths1_offload = [
    '../../results/Jun_4_2_sum/prefill/1_processed.csv'
]

# ---- Subplot 2 file paths ----
file_paths2_kivi = [
    '../../results/Jun_5_1_qa/baseline_kivi/02_processed.csv',
    '../../results/Jun_5_1_qa/baseline_kivi/03_processed.csv',
    '../../results/Jun_5_1_qa/baseline_kivi/06_processed.csv',
]
file_paths2_ours = [
    '../../results/Jun_5_1_qa/ours/01_processed_updated.csv',
    # '../../results/Jun_5_1_qa/ours/02_processed_updated.csv',
    '../../results/Jun_5_1_qa/ours/03_processed_updated.csv',
    # '../../results/Jun_5_1_qa/ours/04_processed_updated.csv',
    # '../../results/Jun_5_1_qa/ours/06_processed_updated.csv',
    # '../../results/Jun_5_1_qa/ours/07_processed_updated.csv',
    # '../../results/Jun_5_1_qa/ours/08_processed_updated.csv',
    '../../results/Jun_5_1_qa/ours/09_processed_updated.csv',
    # '../../results/Jun_5_1_qa/ours/1_processed_updated.csv',
    # '../../results/Jun_5_1_qa/ours/10_processed_updated.csv',
]
file_paths2_prefill = [
    '../../results/Jun_5_1_qa/prefill/0_processed.csv'
]
file_paths2_streaming = [
    '../../results/Jun_5_1_qa/baseline_streaming/02_processed.csv',
    '../../results/Jun_5_1_qa/baseline_streaming/03_processed.csv',
    '../../results/Jun_5_1_qa/baseline_streaming/06_processed.csv',
]
file_paths2_offload = [
    '../../results/Jun_5_1_qa/prefill/1_processed.csv'
]

# ---- Subplot 3 file paths ----
file_paths3_kivi = [
    '../../results/Jun_19_1_coding/baseline_kivi/02_processed.csv',
    '../../results/Jun_19_1_coding/baseline_kivi/03_processed.csv',
    '../../results/Jun_19_1_coding/baseline_kivi/06_processed.csv',
]
file_paths3_ours = [
    '../../results/Jun_19_1_coding/ours/01_processed_updated.csv',
    '../../results/Jun_19_1_coding/ours/05_processed_updated.csv',
    '../../results/Jun_19_1_coding/ours/1_processed_updated.csv',
    '../../results/Jun_19_1_coding/ours/10_processed_updated.csv',
]
file_paths3_prefill = [
    '../../results/Jun_19_1_coding/prefill/0_processed.csv'
]
file_paths3_streaming = [
    '../../results/Jun_19_1_coding/baseline_streaming/02_processed.csv',
    '../../results/Jun_19_1_coding/baseline_streaming/03_processed.csv',
    '../../results/Jun_19_1_coding/baseline_streaming/06_processed.csv',
]
file_paths3_offload = [
    '../../results/Jun_19_1_coding/prefill/1_processed.csv'
]

def load_metrics(file_list, filter_first=False):
    ttft_vals, f1_vals = [], []
    for path in file_list:
        df = pd.read_csv(path)
        if filter_first:
            df = df[df["occurrence_number"] != 1]
        ttft_vals.append(df["ttft"].mean())
        f1_vals.append(df["ROUGEL"].mean())
    return ttft_vals, f1_vals

subplot_filepaths = [
    {
        "kivi": file_paths1_kivi,
        "ours": file_paths1_ours,
        "prefill": file_paths1_prefill,
        "streaming": file_paths1_streaming,
        "offload": file_paths1_offload
    },
    {
        "kivi": file_paths2_kivi,
        "ours": file_paths2_ours,
        "prefill": file_paths2_prefill,
        "streaming": file_paths2_streaming,
        "offload": file_paths2_offload
    },
    {
        "kivi": file_paths3_kivi,
        "ours": file_paths3_ours,
        "prefill": file_paths3_prefill,
        "streaming": file_paths3_streaming,
        "offload": file_paths3_offload
    },
]

subplot_titles = [
    "Dataset: qmsum & samsum",
    "Dataset: triviaqa & hotpotqa",
    "Dataset: lcc_e & repobench_p_e"
]
y_labels = [
    "Average ROUGE-L Score",
    "Average F1 Score",
    "Average CodeBLEU Score"
]
methods = [
    ('Ours', 'ours', 'tab:orange', '^'),
    ('KIVI LRU', 'kivi', 'tab:blue', 'o'),
    ('StreamingLLM LRU', 'streaming', 'tab:pink', 's'),
    ('Prefill', 'prefill', 'tab:green', 'D'),
    ('Offload', 'offload', 'tab:red', 'X')
]

fig, axs = plt.subplots(1, 3, figsize=(24, 6), sharex=False, sharey=False)
for i, ax in enumerate(axs):
    filepaths = subplot_filepaths[i]
    for label, key, color, marker in methods:
        ttft, f1 = load_metrics(filepaths[key], filter_first=True)
        if ttft and f1:
            ax.plot(
                ttft, f1,
                color=color,
                marker=marker,
                markersize=10,
                linewidth=5,
                label=label
            )
    ax.set_xlabel("Average Delay (s)", fontsize=22)
    ax.set_ylabel(y_labels[i], fontsize=22)
    ax.set_title(subplot_titles[i], fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)
    ax.set_xlim(left=0)

# Shared legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=20, frameon=False, bbox_to_anchor=(0.5, 1.07))
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("all_metrics.pdf", dpi=300, bbox_inches="tight")
print("Plot saved as all_metrics.pdf")
