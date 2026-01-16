import pandas as pd
import matplotlib.pyplot as plt

# org_df = pd.read_csv('win40_stride10-100_all_metrics_results.csv')

# org_df['diff'] = org_df['first_token_time'] - org_df['first_scheduled_time']
# filtered_df = org_df[org_df['start_frame_idx'] != 0].copy()
# # filtered the num_frames < 80
# filtered_df = filtered_df[filtered_df['num_frames'] == 80]

# filtered_df = filtered_df[['stride_ratio', 'diff', 'start_frame_idx', 'num_frames']]
# filtered_df.to_csv('stride_ratio_diff_filtered.csv', index=False)


df = pd.read_csv('stride_ratio_diff_filtered.csv')
df_filtered = df[df['start_frame_idx'] != 0]
plot_data = df_filtered.groupby('stride_ratio')['diff'].mean().reset_index().sort_values('stride_ratio')
# normalize the diff values for better visualization
plot_data['diff'] = plot_data['diff'] / plot_data['diff'].min()
plt.figure(figsize=(3.5, 2), dpi=150)
x = plot_data['stride_ratio'].astype(str)
y = plot_data['diff']
bars = plt.bar(x, y, color='skyblue', alpha=0.8)
# stride-ratio=1.0 should be highlighted
for i, bar in enumerate(bars):
    if x.iloc[i] == '1.0':
        bar.set_color('orange')
# stride-ratio=1.0, xtics label should be "no reuse"
xtick_labels = [label if label != '1.0' else 'no reuse' for label in x]
plt.xticks(range(len(x)), xtick_labels, rotation=10, fontsize=8)        
plt.title('Latency vs Stride Ratio', fontsize=8)
plt.xlabel('Stride Ratio', fontsize=8)
plt.ylabel('Norm. Latency', fontsize=8)
plt.yticks(fontsize=8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=40)

plt.ylim(0, max(y) * 1.3) 
plt.tight_layout(pad=0)
plt.savefig('csv_processed_bar.pdf', bbox_inches='tight')
plt.close()