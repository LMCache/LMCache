import pandas as pd
import matplotlib.pyplot as plt

# List of file paths for baseline and ours
file_paths = ['results/Apr_1/baseline_kivi/1_processed.csv', 
              'results/Apr_1/baseline_kivi/02_processed.csv',
              'results/Apr_1/baseline_kivi/03_processed.csv',
              'results/Apr_1/baseline_kivi/06_processed.csv',
              'results/Apr_1/baseline_kivi/0_processed.csv']

file_paths_ours = ['results/Apr_1/ours/1_processed.csv']

# Lists to store computed values for baseline
ttft_values = []
f1_scores = []
file_labels = []

# Process baseline files
for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    # Compute averages
    average_ttft = df["ttft"].mean()
    average_f1_score = df["f1_score"].mean()
    
    # Store values
    ttft_values.append(average_ttft)
    f1_scores.append(average_f1_score)
    file_labels.append(file_path.split('/')[-1])  # Use file name as label
    
    # Print the results
    print(f"File: {file_path}")
    print(f"  Average ttft: {average_ttft}")
    print(f"  Average f1_score: {average_f1_score}")

# 对baseline数据按照ttft从小到大排序
sorted_indices = sorted(range(len(ttft_values)), key=lambda i: ttft_values[i])
ttft_values_sorted = [ttft_values[i] for i in sorted_indices]
f1_scores_sorted = [f1_scores[i] for i in sorted_indices]
file_labels_sorted = [file_labels[i] for i in sorted_indices]

# Lists to store computed values for ours
ttft_values_ours = []
f1_scores_ours = []
file_labels_ours = []

# Process ours files
for file_path in file_paths_ours:
    df = pd.read_csv(file_path)
    
    # Compute averages
    average_ttft = df["ttft"].mean()
    average_f1_score = df["f1_score"].mean()
    
    # Store values
    ttft_values_ours.append(average_ttft)
    f1_scores_ours.append(average_f1_score)
    file_labels_ours.append(file_path.split('/')[-1])  # Use file name as label
    
    # Print the results
    print(f"File: {file_path}")
    print(f"  Average ttft: {average_ttft}")
    print(f"  Average f1_score: {average_f1_score}")

# Plot the results
plt.figure(figsize=(8, 6))

# Plot baseline points and connect them with a line (按ttft从小到大连接)
plt.plot(ttft_values_sorted, f1_scores_sorted, color='b', marker='o', linestyle='-', label='Baseline')
# Annotate baseline points with file names
for i, label in enumerate(file_labels_sorted):
    plt.annotate(label, (ttft_values_sorted[i], f1_scores_sorted[i]), fontsize=10, xytext=(5,5), textcoords='offset points')

# Plot ours points in green (单独散点，不连接)
plt.scatter(ttft_values_ours, f1_scores_ours, color='g', marker='o', label='Ours')
# Annotate ours points with file names
for i, label in enumerate(file_labels_ours):
    plt.annotate(label, (ttft_values_ours[i], f1_scores_ours[i]), fontsize=10, xytext=(5,5), textcoords='offset points')

# Labels and title
plt.xlabel("Average ttft")
plt.ylabel("Average f1_similarity")
plt.title("Average f1_similarity vs. Average ttft")
plt.grid(True)
plt.legend()

# Save the plot instead of showing it
plot_filename = "ttft_vs_f1_similarity.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {plot_filename}")
