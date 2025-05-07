# This script evaluates the similarity of answers in a CSV file using the ROUGE metric.
import pandas as pd
from our_metrics import evaluate_answer, f1_score
import os

INPUT02       = 'results/May_7/baseline/02.csv'
INPUT03       = 'results/May_7/baseline/03.csv'
INPUT06       = 'results/May_7/baseline/06.csv'
INPUT0        = 'results/May_7_2/prefill/0.csv'
INPUT_OURS01 = 'results/May_7/ours/01.csv'
INPUT_OURS04 = 'results/May_7/ours/04.csv'
INPUT_OURS07 = 'results/May_7/ours/07.csv'
INPUT_OURS1   = 'results/May_7/ours/1.csv'


def main():
    # Load & filter the reference answers, then reset its index
    df0 = pd.read_csv(INPUT0)
    df0 = df0[df0['occurrence_number'] != 1].reset_index(drop=True)

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
        df = df[df['occurrence_number'] != 1].reset_index(drop=True)

        # Compute ROUGE‑L by zipping answers row‑wise
        df['ROUGEL'] = [
            evaluate_answer(hyp, ref)
            for hyp, ref in zip(df['answer'], df0['answer'])
        ]

        # Save
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == "__main__":
    main()
