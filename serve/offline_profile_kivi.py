import pandas as pd

df = pd.read_csv('results/Jun_19_1_coding/baseline_kivi/02-fixed_processed.csv')
mask = (df['dataset'] == 'repobench-p_e') & (df['occurrence_number'] == 2)
sub_df = df.loc[mask]
average_rougel = sub_df['ROUGEL'].mean()
print(f"ROUGEL 平均值（dataset='repobench-p_e' 且 occurrence_number=2）：{average_rougel}")
