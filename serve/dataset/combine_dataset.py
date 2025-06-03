#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Combine two CSVs, replicate each row three times with Poisson‐distributed time intervals, and output a single processed CSV.")
    parser.add_argument(
        "--input1", "-i1",
        required=True,
        help="Path to the first input CSV (e.g., samsum.csv)."
    )
    parser.add_argument(
        "--input2", "-i2",
        required=True,
        help="Path to the second input CSV (e.g., qmsum.csv)."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for the output CSV (e.g., sum_processed.csv)."
    )
    args = parser.parse_args()

    # ——— 步骤 1：读取两个 CSV，并添加 index_in_dataset（0–N） ———
    df1 = pd.read_csv(args.input1)
    df2 = pd.read_csv(args.input2)
    df1['index_in_dataset'] = np.arange(len(df1))
    df2['index_in_dataset'] = np.arange(len(df2))

    # ——— 步骤 2：合并并随机打乱 ———
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ——— 步骤 3：为每一行生成三次重复（occurrence_number），
    #         并为“原始行”之间的 start_time 间隔 ~ Poisson(λ=1) ———
    dt_global = np.random.poisson(lam=1, size=len(df))
    base_times = dt_global.cumsum()

    # ——— 步骤 4：在每组（三次重复）内部，
    #         相邻 occurrence 的时间间隔 ~ Poisson(λ=100) ———
    records = []
    for i, row in df.iterrows():
        base = base_times[i]
        dt_within = np.random.poisson(lam=100, size=2)
        times = [
            base,
            base + dt_within[0],
            base + dt_within[0] + dt_within[1]
        ]
        for occ_num, start in enumerate(times, start=1):
            rec = row.to_dict()
            rec['occurrence_number'] = occ_num
            rec['start_time'] = start
            records.append(rec)

    df_out = pd.DataFrame(records)

    # ——— 步骤 5：按 start_time 升序排列并输出 ———
    df_out = df_out.sort_values('start_time').reset_index(drop=True)
    df_out.to_csv(args.output, index=False)
    print(f"生成完成：{args.output} 共 {len(df_out)} 行")

if __name__ == "__main__":
    main()
