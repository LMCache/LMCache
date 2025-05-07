#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np

def simulate_polya_urn(n_items: int, sequence_length: int, alpha: float) -> list[int]:
    """
    用对称 Dirichlet–Multinomial (Pólya Urn) 模型生成索引序列。
    n_items: 数据集中类别数（行数）。
    sequence_length: 模拟请求序列的长度。
    alpha: 每个类别的初始权重。
    返回: 长度为 sequence_length 的索引列表，每个元素是 [0, n_items) 之间的整数。
    """
    counts = np.zeros(n_items, dtype=float)
    selected = []
    for t in range(sequence_length):
        probs = (alpha + counts) / (n_items * alpha + t)
        idx = np.random.choice(n_items, p=probs)
        selected.append(idx)
        counts[idx] += 1
    return selected

def main():
    parser = argparse.ArgumentParser(
        description="使用对称 Dirichlet–Multinomial (Pólya Urn) 模型，"
                    "根据输入 CSV 模拟请求序列并输出新的 CSV，同时可复现并报告未被选中的请求数。"
    )
    parser.add_argument(
        "-i", "--input-csv", required=True,
        help="输入 CSV 文件路径 (包含 title 和约200行数据)。"
    )
    parser.add_argument(
        "-o", "--output-csv", required=True,
        help="输出 CSV 文件路径。"
    )
    parser.add_argument(
        "-n", "--sequence-length", type=int, required=True,
        help="要模拟的请求数（序列长度）。"
    )
    parser.add_argument(
        "--alpha", type=float, default=20,
        help="Dirichlet 浓度参数 α，默认 1.0。"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="随机数种子，用于结果可复现（可选）。"
    )
    args = parser.parse_args()

    # 可复现设置
    if args.seed is not None:
        np.random.seed(args.seed)

    # 读取原始数据
    df = pd.read_csv(args.input_csv)
    n_items = len(df)

    # 模拟索引序列
    indices = simulate_polya_urn(n_items, args.sequence_length, args.alpha)

    # 根据模拟的索引，选取对应的行并重置索引
    result_df = df.iloc[indices].reset_index(drop=True)
    # 添加原始行索引列
    result_df["index_in_dataset"] = indices

    # 添加出现次数列：第 k 次出现该条记录
    counts_so_far = {}
    occurrences = []
    for idx in indices:
        counts_so_far[idx] = counts_so_far.get(idx, 0) + 1
        occurrences.append(counts_so_far[idx])
    result_df["occurrence_number"] = occurrences

    # 输出到新的 CSV
    result_df.to_csv(args.output_csv, index=False)
    print(f"Simulation complete: wrote {args.sequence_length} rows to {args.output_csv}")

    # 统计未被选中的请求数量
    never_selected = set(range(n_items)) - set(indices)
    print(f"{len(never_selected)} of the {n_items} requests were never selected in the simulation.")

if __name__ == "__main__":
    main()
