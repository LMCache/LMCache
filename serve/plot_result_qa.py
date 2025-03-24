import pandas as pd
import matplotlib.pyplot as plt

# List of file paths
file_paths = ['result_1.csv', 
              'result_02.csv',
              'result_03.csv',
              'result_06.csv']

# Lists to store computed values
ttft_values = []
f1_scores = []
file_labels = []

# Process each file
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

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(ttft_values, f1_scores, color='b', marker='o')

# Annotate points with file names
for i, label in enumerate(file_labels):
    plt.annotate(label, (ttft_values[i], f1_scores[i]), fontsize=10, xytext=(5,5), textcoords='offset points')

# Labels and title
plt.xlabel("Average ttft")
plt.ylabel("Average f1_similarity")
plt.title("Average f1_similarity vs. Average ttft")
plt.grid(True)

# Save the plot instead of showing it
plot_filename = "ttft_vs_f1_similarity.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {plot_filename}")
