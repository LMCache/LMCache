import pandas as pd
import matplotlib.pyplot as plt

# List of file paths
file_paths = ['results/baseline_kivi_2.csv', 'results/baseline_kivi_4.csv', 'results/baseline_kivi_8.csv', 'results/baseline_no_compression.csv', 'results/baseline_prefill.csv']

# Lists to store computed values
ttft_values = []
rougeL_fmeasure_values = []
file_labels = []

# Process each file
for file_path in file_paths:
    df = pd.read_csv(file_path)

    # Compute averages
    average_ttft = df["ttft"].mean()
    average_rougeL_fmeasure = df["rougeL_fmeasure"].mean()

    # Store values
    ttft_values.append(average_ttft)
    rougeL_fmeasure_values.append(average_rougeL_fmeasure)
    file_labels.append(file_path.split('/')[-1])  # Use file name as label

    # Print the results
    print(f"File: {file_path}")
    print(f"  Average ttft: {average_ttft}")
    print(f"  Average rougeL_fmeasure: {average_rougeL_fmeasure}")

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(ttft_values, rougeL_fmeasure_values, color='b', marker='o')

# Annotate points with file names
for i, label in enumerate(file_labels):
    plt.annotate(label, (ttft_values[i], rougeL_fmeasure_values[i]), fontsize=10, xytext=(5,5), textcoords='offset points')

# Labels and title
plt.xlabel("Average ttft")
plt.ylabel("Average rougeL_fmeasure")
plt.title("Average rougeL_fmeasure vs. Average ttft")
plt.grid(True)

# Save the plot instead of showing it
plot_filename = "ttft_vs_rougeL_fmeasure.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {plot_filename}")
