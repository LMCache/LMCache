# This script evaluates the similarity of answers in a CSV file using the ROUGE metric.
import pandas as pd
from our_metrics import evaluate_answer, f1_score
import os

INPUT01       = '../May_13_streaming/results_rate_0.1.csv'
INPUT025      = '../May_13_streaming/results_rate_0.25.csv'
INPUT05       = '../May_13_streaming/results_rate_0.5.csv'
INPUT06       = '../May_13_streaming/results_rate_06.csv'
INPUT03       = '../May_13_streaming/results_rate_03.csv'
INPUT02       = '../May_13_streaming/results_rate_02.csv'
INPUT0        = '../May_10_3/prefill/0.csv'

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

    # before your loop, build a dict from df0
    ref_map = df0.set_index('index_in_dataset')['answer'].to_dict()

    # Process each CSV in turn
    for path, fname in zip(input_paths, filenames):
        # Load & filter this CSV, then reset its index
        df = pd.read_csv(path)

        # Compute ROUGE‑L by looking up the reference answer via index_in_dataset
        df['ROUGEL'] = df.apply(
            lambda row: evaluate_answer(
                row['generated_text'],
                ref_map[row['index_in_dataset']]
            ),
            axis=1
        )

        # Save
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == "__main__":
    main()
