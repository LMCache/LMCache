import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ...（你的路径部分，完全照抄不变）...
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
file_paths2_kivi = [
    '../../results/Jun_5_1_qa/baseline_kivi/02_processed.csv',
    '../../results/Jun_5_1_qa/baseline_kivi/03_processed.csv',
    '../../results/Jun_5_1_qa/baseline_kivi/06_processed.csv',
]
file_paths2_ours = [
    '../../results/Jun_5_1_qa/ours/01_processed_updated.csv',
    '../../results/Jun_5_1_qa/ours/03_processed_updated.csv',
    '../../results/Jun_5_1_qa/ours/09_processed_updated.csv',
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

def load_metrics_weighted(file_list, w):
    """加权平均 occurrence_number=1 和 =2 的值"""
    ttft_1, f1_1, ttft_2, f1_2 = [], [], [], []
    for path in file_list:
        df = pd.read_csv(path)
        df1 = df[df["occurrence_number"] == 1]
        df2 = df[df["occurrence_number"] == 2]
        ttft_1.append(df1["ttft"].mean() if not df1.empty else np.nan)
        f1_1.append(df1["ROUGEL"].mean() if not df1.empty else np.nan)
        ttft_2.append(df2["ttft"].mean() if not df2.empty else np.nan)
        f1_2.append(df2["ROUGEL"].mean() if not df2.empty else np.nan)
    # 加权组合，忽略缺失项
    ttft, f1 = [], []
    for a1, a2, b1, b2 in zip(ttft_1, ttft_2, f1_1, f1_2):
        # nan处理：若有一项缺失就用另一项（或者跳过）
        if np.isnan(a1) and not np.isnan(a2):
            ttft_val = a2
        elif not np.isnan(a1) and np.isnan(a2):
            ttft_val = a1
        elif np.isnan(a1) and np.isnan(a2):
            ttft_val = np.nan
        else:
            ttft_val = (1-w) * a1 + w * a2
        if np.isnan(b1) and not np.isnan(b2):
            f1_val = b2
        elif not np.isnan(b1) and np.isnan(b2):
            f1_val = b1
        elif np.isnan(b1) and np.isnan(b2):
            f1_val = np.nan
        else:
            f1_val = (1-w) * b1 + w * b2
        ttft.append(ttft_val)
        f1.append(f1_val)
    # 把nan的点过滤掉
    ttft, f1 = zip(*[(x, y) for x, y in zip(ttft, f1) if not (np.isnan(x) or np.isnan(y))])
    return list(ttft), list(f1)

weights = np.arange(0.5, 0.96, 0.05)  # 0.5 ~ 0.95, 步长0.05

for w in weights:
    fig, axs = plt.subplots(1, 3, figsize=(24, 6), sharex=False, sharey=False)
    for i, ax in enumerate(axs):
        filepaths = subplot_filepaths[i]
        for label, key, color, marker in methods:
            try:
                ttft, f1 = load_metrics_weighted(filepaths[key], w)
                if ttft and f1:
                    ax.plot(
                        ttft, f1,
                        color=color,
                        marker=marker,
                        markersize=10,
                        linewidth=5,
                        label=label
                    )
            except Exception as e:
                print(f"Failed {label} subplot {i}: {e}")
        ax.set_xlabel("Average Delay (s)", fontsize=22)
        ax.set_ylabel(y_labels[i], fontsize=22)
        ax.set_title(subplot_titles[i], fontsize=22)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True)
        ax.set_xlim(left=0)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=20, frameon=False, bbox_to_anchor=(0.5, 1.07))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = f"all_metrics_weight_{w:.2f}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {save_path}")
    plt.close(fig)  # 防止内存溢出

print("All done.")
