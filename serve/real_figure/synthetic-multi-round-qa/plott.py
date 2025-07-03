import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

systems = ['Prefill', 'GPU Prefix Caching', 'CPU offloading', 'GPU+CPU+SSD (0.5GB/s)']
model1_files = [
    ['prefill-3b_output_0.5.csv'],
    ['gpu-3b_output_0.5.csv'],
    ['cpu-3b_output_0.5.csv'],
    ['disk-3b_output_0.5.csv']
]
model2_files = [
    ['prefill_output_0.5.csv'],
    ['gpu_output_0.5.csv'],
    ['cpu_output_0.5.csv'],
    ['disk_output_0.5.csv']
]

def create_dummy_csv(filename, ttft_value):
    if not os.path.exists(filename):
        df = pd.DataFrame({'ttft': np.random.normal(ttft_value, ttft_value * 0.05, size=30)})
        df.to_csv(filename, index=False)

dummy_ttft_values1 = [60, 80, 120, 150]
dummy_ttft_values2 = [55, 90, 110, 170]

for files, vals in zip(model1_files, dummy_ttft_values1):
    for f in files:
        create_dummy_csv(f, vals)
for files, vals in zip(model2_files, dummy_ttft_values2):
    for f in files:
        create_dummy_csv(f, vals)

def get_ttft_and_throughput(file_lists):
    ttft_avgs = []
    throughput_avgs = []
    for files in file_lists:
        ttfts = []
        for f in files:
            df = pd.read_csv(f)
            if 'ttft' not in df.columns:
                raise ValueError(f"'ttft' column not found in {f}")
            ttfts.extend(df['ttft'].dropna().values)
        ttft_mean = np.mean(ttfts)
        throughput_mean = np.mean(1 / np.array(ttfts)) * 1000
        ttft_avgs.append(ttft_mean)
        throughput_avgs.append(throughput_mean)
    return ttft_avgs, throughput_avgs

ttft1, throughput1 = get_ttft_and_throughput(model1_files)
ttft2, throughput2 = get_ttft_and_throughput(model2_files)

x = np.arange(len(systems))
width = 0.5

fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=False)
plt.subplots_adjust(wspace=0.18)

handles = []
labels = []

# 字体参数全部再+4
title_fontsize = 30
axislabel_fontsize = 29
tick_fontsize = 26
xtick_fontsize = 26
annotate_fontsize = 26
legend_fontsize = 29

# 1. 设置主y轴和副y轴的最大值
ymax1 = max(ttft1) * 1.10
ymax2 = max(ttft2) * 1.10
ymax1b = max(throughput1) * 1.10
ymax2b = max(throughput2) * 1.10

# Model 1 subplot
ax1 = axes[0]
bars1 = ax1.bar(x, ttft1, width=width, color='#7fa2ff', edgecolor='black', alpha=0.85, label='TTFT (s)')
ax1.set_ylabel('TTFT (s)', fontsize=axislabel_fontsize, color='#34495e')
ax1.set_xticks(x)
ax1.set_title('Llama-3.2-3B', fontsize=title_fontsize, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#34495e', labelsize=tick_fontsize)
ax1.tick_params(axis='x', pad=55, labelsize=tick_fontsize)
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax1.set_facecolor('#f7faff')
ax1.set_ylim(0, ymax1)
ax1.set_xticklabels(systems, fontsize=xtick_fontsize, rotation=25, ha='center', rotation_mode='anchor')

for rect in bars1:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 10),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=annotate_fontsize, color='#34495e')

ax1b = ax1.twinx()
line1 = ax1b.plot(x, throughput1, 'o-', color='#ff5e62', lw=3, markersize=15, label='Throughput (QPS)')
ax1b.tick_params(axis='y', labelcolor='#ff5e62', labelsize=tick_fontsize)
ax1b.set_ylim(0, ymax1b)
for i, v in enumerate(throughput1):
    ax1b.annotate(f"{v:.1f}", (x[i], throughput1[i]), textcoords="offset points", xytext=(0,14), ha='center', fontsize=annotate_fontsize, color='#ff5e62')

h1, l1 = ax1.get_legend_handles_labels()
h1b, l1b = ax1b.get_legend_handles_labels()
handles += h1 + h1b
labels += l1 + l1b

# Model 2 subplot
ax2 = axes[1]
bars2 = ax2.bar(x, ttft2, width=width, color='#7fa2ff', edgecolor='black', alpha=0.85, label='TTFT (s)')
ax2.set_ylabel('', fontsize=axislabel_fontsize, color='#34495e')
ax2.set_xticks(x)
ax2.set_title('Llama-3.1-8B', fontsize=title_fontsize, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#34495e', labelsize=tick_fontsize)
ax2.tick_params(axis='x', pad=55, labelsize=tick_fontsize)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.set_facecolor('#f7faff')
ax2.set_ylim(0, ymax2)
ax2.set_xticklabels(systems, fontsize=xtick_fontsize, rotation=25, ha='center', rotation_mode='anchor')

for rect in bars2:
    height = rect.get_height()
    ax2.annotate(f'{height:.1f}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 10),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=annotate_fontsize, color='#34495e')

ax2b = ax2.twinx()
line2 = ax2b.plot(x, throughput2, 'o-', color='#ff5e62', lw=3, markersize=15, label='Prefill Throughput (t/s)')
ax2b.set_ylabel('', fontsize=axislabel_fontsize, color='#ff5e62')
ax2b.tick_params(axis='y', labelcolor='#ff5e62', labelsize=tick_fontsize)
ax2b.set_ylim(0, ymax2b)
for i, v in enumerate(throughput2):
    ax2b.annotate(f"{v:.1f}", (x[i], throughput2[i]), textcoords="offset points", xytext=(0,14), ha='center', fontsize=annotate_fontsize, color='#ff5e62')

h2, l2 = ax2.get_legend_handles_labels()
h2b, l2b = ax2b.get_legend_handles_labels()
handles += h2 + h2b
labels += l2 + l2b

unique = dict()
for h, l in zip(handles, labels):
    if l not in unique:
        unique[l] = h

fig.legend(list(unique.values()), list(unique.keys()),
           loc='upper center', bbox_to_anchor=(0.5, 1.07),
           ncol=2, fontsize=legend_fontsize, labelcolor='#34495e', frameon=True, fancybox=True)

fig.text(0.97, 0.5, 'Prefill Throughput (t/s)', va='center', rotation=90, fontsize=axislabel_fontsize, color='#ff5e62')

plt.tight_layout(rect=[0, 0, 0.97, 0.97])
plt.savefig("model1_model2_barline.pdf", dpi=200, bbox_inches='tight')
plt.close(fig)

print("Plot saved as 'model1_model2_barline.pdf'")
