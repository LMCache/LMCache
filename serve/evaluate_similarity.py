# This script evaluates the similarity of answers in a CSV file using the ROUGE metric.
import pandas as pd
from our_metrics import evaluate_answer, f1_score
import os

INPUT02       = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/May_23_1_sum/baseline_kivi/02.csv'
INPUT03       = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/May_23_1_sum/baseline_kivi/03.csv'
INPUT06       = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/May_23_1_sum/baseline_kivi/06.csv'
INPUT_1       = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/May_23_1_sum/prefill/1.csv'
INPUT0        = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/May_23_1_sum/prefill/0.csv'

def main():
    # Load & filter the reference answers, then reset its index
    df0 = pd.read_csv(INPUT0)
    df0 = df0.reset_index(drop=True)
    reference_answers = df0['answer'].tolist()

    # Collect all INPUT* paths except the reference INPUT0
    input_paths = [
        path for name, path in globals().items()
        if name.startswith("INPUT")
    ]

    # Generate processed filenames
    filenames = [
        os.path.splitext(path)[0] + "_processed.csv"
        for path in input_paths
    ]

    # Process each CSV in turn
    for path, fname in zip(input_paths, filenames):
        # Load & filter this CSV, then reset its index
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)

        # Compute ROUGE‑L by looking up the reference answer via row number
        df['ROUGEL'] = df.apply(
            lambda row: evaluate_answer(
                row['answer'],
                reference_answers[row.name]
            ),
            axis=1
        )

        # Save
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == "__main__":
    main()
