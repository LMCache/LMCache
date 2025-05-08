import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --- Parse command-line arguments for plot filename ---
parser = argparse.ArgumentParser(
    description="Plot Average ROUGEL vs Average ttft from CSV results"
)
parser.add_argument(
    "--plot-filename",
    type=str,
    default="ttft_vs_ROUGEL_similarity.png",
    help="Filename for saving the plot"
)
args = parser.parse_args()

# List of file paths for baseline, ours, and prefill
file_paths = [
    'results/May_7_5/baseline/02_processed.csv',
    'results/May_7_5/baseline/03_processed.csv',
]

file_paths_ours = [
    'results/May_7_5/ours/01_processed.csv',
    'results/May_7_5/ours/001_processed.csv',
    'results/May_7_5/ours/1_processed.csv',
    'results/May_7_5/ours/10_processed.csv',
]

file_path_prefill = [
    'results/May_7_3/prefill/0_processed.csv'
]

# --- Process baseline files ---
ttft_values = []
f1_scores = []
file_labels = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    average_ttft = df["ttft"].mean()
    average_f1_score = df["ROUGEL"].mean()

    ttft_values.append(average_ttft)
    f1_scores.append(average_f1_score)
    file_labels.append(file_path.split('/')[-1])

    print(f"File: {file_path}")
    print(f"  Average ttft: {average_ttft}")
    print(f"  Average ROUGEL: {average_f1_score}")

# Sort baseline by ttft
sorted_indices = sorted(range(len(ttft_values)), key=lambda i: ttft_values[i])
ttft_values_sorted = [ttft_values[i] for i in sorted_indices]
f1_scores_sorted   = [f1_scores[i]   for i in sorted_indices]
file_labels_sorted = [file_labels[i] for i in sorted_indices]

# --- Process "ours" files ---
ttft_values_ours = []
f1_scores_ours   = []
file_labels_ours = []

for file_path in file_paths_ours:
    df = pd.read_csv(file_path)
    average_ttft = df["ttft"].mean()
    average_f1_score = df["ROUGEL"].mean()

    ttft_values_ours.append(average_ttft)
    f1_scores_ours.append(average_f1_score)
    file_labels_ours.append(file_path.split('/')[-1])

    print(f"File: {file_path}")
    print(f"  Average ttft: {average_ttft}")
    print(f"  Average ROUGEL: {average_f1_score}")

# Sort ours by ttft
sorted_indices_ours = sorted(range(len(ttft_values_ours)), key=lambda i: ttft_values_ours[i])
ttft_values_ours_sorted   = [ttft_values_ours[i]   for i in sorted_indices_ours]
f1_scores_ours_sorted     = [f1_scores_ours[i]     for i in sorted_indices_ours]
file_labels_ours_sorted   = [file_labels_ours[i]   for i in sorted_indices_ours]

# --- Process prefill files (same as "ours") ---
ttft_values_prefill = []
f1_scores_prefill   = []
file_labels_prefill = []

for file_path in file_path_prefill:
    df = pd.read_csv(file_path)
    average_ttft = df["ttft"].mean()
    average_f1_score = df["ROUGEL"].mean()

    ttft_values_prefill.append(average_ttft)
    f1_scores_prefill.append(average_f1_score)
    file_labels_prefill.append(file_path.split('/')[-1])

    print(f"File: {file_path}")
    print(f"  Average ttft: {average_ttft}")
    print(f"  Average ROUGEL: {average_f1_score}")

# Sort prefill by ttft
sorted_indices_prefill = sorted(range(len(ttft_values_prefill)), key=lambda i: ttft_values_prefill[i])
ttft_values_prefill_sorted = [ttft_values_prefill[i] for i in sorted_indices_prefill]
f1_scores_prefill_sorted   = [f1_scores_prefill[i]   for i in sorted_indices_prefill]
file_labels_prefill_sorted = [file_labels_prefill[i] for i in sorted_indices_prefill]

# --- Plot all three series ---
plt.figure(figsize=(8, 6))

# Baseline
plt.plot(ttft_values_sorted, f1_scores_sorted, marker='o', linestyle='-', label='Baseline')
for i, label in enumerate(file_labels_sorted):
    plt.annotate(label, (ttft_values_sorted[i], f1_scores_sorted[i]),
                 fontsize=10, xytext=(5,5), textcoords='offset points')

# Ours
plt.plot(ttft_values_ours_sorted, f1_scores_ours_sorted, marker='o', linestyle='-', label='Ours')
for i, label in enumerate(file_labels_ours_sorted):
    plt.annotate(label, (ttft_values_ours_sorted[i], f1_scores_ours_sorted[i]),
                 fontsize=10, xytext=(5,5), textcoords='offset points')

# Prefill
plt.plot(ttft_values_prefill_sorted, f1_scores_prefill_sorted, marker='o', linestyle='-', label='Prefill')
for i, label in enumerate(file_labels_prefill_sorted):
    plt.annotate(label, (ttft_values_prefill_sorted[i], f1_scores_prefill_sorted[i]),
                 fontsize=10, xytext=(5,5), textcoords='offset points')

plt.xlabel("Average ttft")
plt.ylabel("Average ROUGEL")
plt.title("Average ROUGEL vs. Average ttft")
plt.grid(True)
plt.legend()
plt.xlim(xmin=0)
plt.ylim(ymax=1)

# Save the figure using the provided filename argument
plt.savefig(args.plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {args.plot_filename}")
