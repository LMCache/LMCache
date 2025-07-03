import matplotlib.pyplot as plt
import numpy as np

compression = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
xvals = 1 - compression

# 数据
streaming_a = [0.218, 0.193, 0.163, 0.072, 0.121, 0.122]
expected_a  = [0.219, 0.219, 0.219, 0.183, 0.199, 0.138]
snap_a      = [0.218, 0.218, 0.172, 0.136, 0.183, 0.072]

streaming_b = [0.227, 0.226, 0.208, 0.213, 0.131, 0.216]
expected_b  = [0.227, 0.202, 0.138, 0.180, 0.168, 0.142]
snap_b      = [0.227, 0.163, 0.117, 0.153, 0.032, 0.209]

title_fontsize = 24
label_fontsize = 22
tick_fontsize = 20
legend_fontsize = 22

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 只要这些x轴tick
xtick_vals = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
xtick_vals = np.sort(xtick_vals)[::-1]  # 保持递减

axes[0].plot(xvals, streaming_a, marker='o', color='tab:blue', markersize=12, linewidth=5, label='StreamingLLMPress')
axes[0].plot(xvals, expected_a, marker='^', color='tab:orange', markersize=12, linewidth=5, label='ExpectedAttentionPress')
axes[0].plot(xvals, snap_a, marker='D', color='tab:green', markersize=12, linewidth=5, label='SnapKVPress')
axes[0].set_title('Sample Index 23', fontsize=title_fontsize)
axes[0].set_xlabel('Compression Ratio', fontsize=label_fontsize)
axes[0].set_ylabel('Evaluation Score', fontsize=label_fontsize)
axes[0].set_xticks(xtick_vals)
axes[0].set_xticklabels([f"{x:.1f}" for x in xtick_vals], fontsize=tick_fontsize)
axes[0].tick_params(axis='y', labelsize=tick_fontsize)
axes[0].grid(True)

axes[1].plot(xvals, streaming_b, marker='o', color='tab:blue', markersize=12, linewidth=5, label='StreamingLLMPress')
axes[1].plot(xvals, expected_b, marker='^', color='tab:orange', markersize=12, linewidth=5, label='ExpectedAttentionPress')
axes[1].plot(xvals, snap_b, marker='D', color='tab:green', markersize=12, linewidth=5, label='SnapKVPress')
axes[1].set_title('Sample Index 25', fontsize=title_fontsize)
axes[1].set_xlabel('Compression Ratio', fontsize=label_fontsize)
axes[1].set_xticks(xtick_vals)
axes[1].set_xticklabels([f"{x:.1f}" for x in xtick_vals], fontsize=tick_fontsize)
axes[1].tick_params(axis='y', labelsize=tick_fontsize)
axes[1].grid(True)

# 统一y轴范围
y0 = axes[0].get_ylim()
y1 = axes[1].get_ylim()
ymin = min(y0[0], y1[0])
ymax = max(y0[1], y1[1])
axes[0].set_ylim(ymin, ymax)
axes[1].set_ylim(ymin, ymax)

plt.subplots_adjust(top=0.78)

handles, labels = axes[0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color='none'))
labels.append('')
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=2,
    fontsize=legend_fontsize,
    bbox_to_anchor=(0.5, 1.1),
    columnspacing=2,
    handletextpad=1,
    frameon=True
)

fig.savefig("compression_eval_examples_final.pdf", dpi=300, bbox_inches='tight')
plt.close()
