#!/usr/bin/env python3
import pandas as pd
import numpy as np

def main():
    # ——— 步骤 1：读取两个 CSV，并添加 index_in_dataset（0–199） ———
    df1 = pd.read_csv('samsum.csv')        # 把 file1.csv 换成你的第一个文件名
    df2 = pd.read_csv('qmsum.csv')        # 把 file2.csv 换成你的第二个文件名
    df1['index_in_dataset'] = np.arange(len(df1))
    df2['index_in_dataset'] = np.arange(len(df2))

    # ——— 步骤 2：合并并随机打乱 ———
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ——— 步骤 3：为每一行生成三次重复（occurrence_number），
    #         并为“原始行”之间的 start_time 间隔 ~ Poisson(λ=1) ———
    # 先为每条原始记录生成一个基础时间
    dt_global = np.random.poisson(lam=1, size=len(df))
    base_times = dt_global.cumsum()

    # ——— 步骤 4：在每组（三次重复）内部，
    #         相邻 occurrence 的时间间隔 ~ Poisson(λ=100) ———
    records = []
    for i, row in df.iterrows():
        base = base_times[i]
        # 生成两段间隔，分别对应 1→2、2→3
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
    df_out.to_csv('sum_processed.csv', index=False)
    print("生成完成：combined_output.csv 共 %d 行" % len(df_out))

if __name__ == '__main__':
    main()
