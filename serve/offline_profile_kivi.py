import pandas as pd

df = pd.read_csv('results/Jun_5_1_qa/baseline_kivi/06_processed.csv')
mask = (df['dataset'] == 'triviaqa') & (df['occurrence_number'] == 2)
sub_df = df.loc[mask]
average_rougel = sub_df['ROUGEL'].mean()
print(f"ROUGEL 平均值（dataset='samsum' 且 occurrence_number=2）：{average_rougel}")
