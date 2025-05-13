#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="""
        将输入 CSV 中的所有记录循环两遍输出，
        第一遍保持原始顺序，occurrence_number=1；
        第二遍按随机抽样（可重复），使用固定随机种子42，occurrence_number根据实际出现次数动态计算。
        最终输出长度为输入长度的两倍。
        """
    )
    parser.add_argument(
        "-i", "--input-csv", required=True,
        help="输入 CSV 文件路径（包含标题和数据行）。"
    )
    parser.add_argument(
        "-o", "--output-csv", required=True,
        help="输出 CSV 文件路径。"
    )
    args = parser.parse_args()

    # 读取原始数据并记录原始索引
    df = pd.read_csv(args.input_csv)
    df['index_in_dataset'] = df.index
    n = len(df)

    # 第一遍：按原始顺序
    first = df.copy()

    # 第二遍：随机抽样（可重复），保留原索引列
    second = df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

    # 合并两遍
    combined = pd.concat([first, second], ignore_index=True)

    # 动态计算 occurrence_number：统计每条记录的出现次数
    combined['occurrence_number'] = combined.groupby('index_in_dataset').cumcount() + 1

    # 输出
    combined.to_csv(args.output_csv, index=False)
    print(f"Doubling complete: wrote {len(combined)} rows to {args.output_csv}")


if __name__ == '__main__':
    main()
