#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_selected_weights(csv_path, output_path=None):
    """
    读取 CSV 并绘制权重为 0.6, 0.7, 0.8, 0.9 附近的数据点的折线图，
    并根据指定样式调整图表元素，同时增加 Prefill 平线。
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 目标权重和容差
    target_weights = [0.6, 0.7, 0.8, 0.9]
    tol = 0.0001

    # 筛选出最接近目标权重的行，并按 weight 排序
    mask = df['weight'].apply(lambda x: any(abs(x - w) < tol for w in target_weights))
    subset = df[mask].sort_values('weight')

    if len(subset) < len(target_weights):
        missing = set(target_weights) - set(round(w, 4) for w in subset['weight'])
        print(f"警告：未找到以下权重的数据行：{missing}")

    # 提取 x 和 y
    x = subset['weight']
    y_ours = subset['min_ttft_for_rougel_>=95']

    # Prefill 平线值
    prefill_value = 1.3807121301349252
    y_prefill = [prefill_value] * len(x)

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(8, 4))

    # 绘制 “Ours” 曲线
    ax.plot(
        x,
        y_ours,
        marker='o',
        markersize=10,
        linewidth=5,
        label='Ours'
    )

    # 绘制 “Prefill” 平线
    ax.plot(
        x,
        y_prefill,
        linestyle='--',
        linewidth=3,
        label='Prefill'
    )

    # 设置标题与坐标轴标签，增大字体
    ax.set_title('Min TTFT for F1 Score Drop < 5%', fontsize=16)
    ax.set_xlabel('Hit Rate', fontsize=16)
    ax.set_ylabel('Average Delay (s)', fontsize=16)

    # 坐标轴刻度字体大小
    ax.tick_params(axis='both', labelsize=14)

    # 网格
    ax.grid(True, linestyle='--', alpha=0.6)

    # 图例
    ax.legend(fontsize=14, loc='best')

    # 布局紧凑
    fig.tight_layout()

    # 保存或展示
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到：{output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='从 CSV 中读取指定权重行并绘制 TTFT 曲线（0.6,0.7,0.8,0.9），并添加 Prefill 平线'
    )
    parser.add_argument(
        'csv_path',
        help='输入 CSV 文件路径，例如 /mnt/data/result.csv'
    )
    parser.add_argument(
        '-o', '--output',
        help='输出图片路径，例如 ./ttft_plot.png，如果不指定则直接显示',
        default=None
    )
    args = parser.parse_args()

    plot_selected_weights(args.csv_path, args.output)
