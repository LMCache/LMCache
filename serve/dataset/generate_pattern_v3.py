#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="""
        将输入 CSV 中的所有记录循环两遍输出，第一遍 occurrence_number=1（原始顺序），
        第二遍 occurrence_number=2（随机顺序）。
        数据集长度变为原来的两倍。
        接口参数与原 Polya Urn 脚本保持一致，仅替换采样逻辑。
        """
    )
    parser.add_argument(
        "-i", "--input-csv", required=True,
        help="输入 CSV 文件路径 (包含 title 和若干行数据)。"
    )
    parser.add_argument(
        "-o", "--output-csv", required=True,
        help="输出 CSV 文件路径。"
    )
    parser.add_argument(
        "-n", "--sequence-length", type=int, required=True,
        help="（可选）序列长度参数，将被忽略，实际长度为输入行数的两倍。"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="（可选）浓度参数 α，将被忽略，仅保留原接口一致性。"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="随机数种子（可选），用于第二遍顺序的可复现性。"
    )
    args = parser.parse_args()

    # 可复现设置（仅影响第二遍打乱顺序）
    if args.seed is not None:
        np.random.seed(args.seed)

    # 读取原始数据
    df = pd.read_csv(args.input_csv)
    n_items = len(df)

    # --- 第一遍：原始顺序，occurrence_number = 1 ---
    df_first = df.copy().reset_index(drop=True)
    df_first["index_in_dataset"] = df_first.index
    df_first["occurrence_number"] = 1

    # --- 第二遍：随机顺序，occurrence_number = 2 ---
    df_second = df.copy().reset_index(drop=True)
    df_second["index_in_dataset"] = df_second.index
    df_second["occurrence_number"] = 2
    # 随机打乱行顺序
    df_second = df_second.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 合并两遍数据
    result_df = pd.concat([df_first, df_second], ignore_index=True)

    # 输出到新的 CSV
    result_df.to_csv(args.output_csv, index=False)
    total = len(result_df)
    print(f"Doubling complete: wrote {total} rows to {args.output_csv}")

    # 统计未被选中的请求数量（固定为 0）
    never_selected = 0
    print(f"{never_selected} of the {n_items} requests were never selected in the simulation.")


if __name__ == "__main__":
    main()
