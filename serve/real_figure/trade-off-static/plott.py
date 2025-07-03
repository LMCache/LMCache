import matplotlib.pyplot as plt

# 你的token/TTFT/ROUGE-L计算逻辑
rates = [1.0, 0.728571429, 0.485714286, 0.371428571]
rate_to_filename = {
    1.0: "1.csv",
    0.728571429: "06.csv",
    0.485714286: "03.csv",
    0.371428571: "02.csv"
}
filename_to_ab = {
    "06.csv": (9.199533115315706e-05, 0.05496935327559265),
    "03.csv": (5.591602459317067e-05, 0.07696992164728042),
    "02.csv": (2.3640076152514657e-05, 0.3193163771870893),
    "1.csv":  (0.00013438977394253005, -0.0006046225223680943)
}
base_rate = 0.728571429
base_size_gb = 321

qmsum_sum = 11506816
samsum_sum = 6906491
total_sum = qmsum_sum + samsum_sum
qmsum_ratio = qmsum_sum / total_sum
samsum_ratio = samsum_sum / total_sum

quality_dict = {
    'qmsum': {
        1.0: 0.8780,
        0.728571429: 0.8599,
        0.485714286: 0.7473,
        0.371428571: 0.5056
    },
    'samsum': {
        1.0: 0.9567,
        0.728571429: 0.9565,
        0.485714286: 0.9095,
        0.371428571: 0.7313
    }
}

# ====== token/GB 换算 ======
gb_per_token = 0.0313 / 256
token_per_gb = 256 / 0.0313
base_num_token = int(base_size_gb * token_per_gb)
rate_to_num_token = {r: int(base_num_token / base_rate * r) for r in rates}
tokens_qmsum = [rate_to_num_token[r] * qmsum_ratio for r in rates]
tokens_samsum = [rate_to_num_token[r] * samsum_ratio for r in rates]

def calc_ttft(rate, num_token):
    filename = rate_to_filename[rate]
    a, b = filename_to_ab[filename]
    return a * num_token + b

ttft_qmsum = [calc_ttft(r, s) for r, s in zip(rates, tokens_qmsum)]
ttft_samsum = [calc_ttft(r, s) for r, s in zip(rates, tokens_samsum)]
quality_qmsum = [quality_dict['qmsum'][r] for r in rates]
quality_samsum = [quality_dict['samsum'][r] for r in rates]

# ----------- 画图 -----------
fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

# QMSum
axes[0].plot(ttft_qmsum, quality_qmsum,
             color='tab:blue', marker='o', markersize=10, linewidth=5)
axes[0].set_title('QMSum', fontsize=16)
axes[0].set_xlabel("Sum of TTFT (s)", fontsize=16)
axes[0].set_ylabel("Average ROUGE-L Score", fontsize=16)
axes[0].tick_params(axis='both', labelsize=14)
axes[0].grid(True)
axes[0].set_xlim(left=0)

# SAMSum
axes[1].plot(ttft_samsum, quality_samsum,
             color='tab:orange', marker='s', markersize=10, linewidth=5)
axes[1].set_title('SAMSum', fontsize=16)
axes[1].set_xlabel("Sum of TTFT (s)", fontsize=16)
axes[1].set_ylabel("")
axes[1].tick_params(axis='both', labelsize=14)
axes[1].grid(True)
axes[1].set_xlim(left=0)

# 统一y轴范围
y0 = axes[0].get_ylim()
y1 = axes[1].get_ylim()
ymin = min(y0[0], y1[0])
ymax = max(y0[1], y1[1])
axes[0].set_ylim(ymin, ymax)
axes[1].set_ylim(ymin, ymax)

# 没有legend
# plt.legend() 这一行完全不写

fig.savefig("quality_vs_ttft_token_single.pdf", dpi=300, bbox_inches="tight")
print("Saved as quality_vs_ttft_token_single.pdf")
