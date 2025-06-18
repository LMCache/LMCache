# This script evaluates the similarity of answers in multiple CSV files using the ROUGE metric (or F1 score).
import pandas as pd
from our_metrics import evaluate_answer, f1_score
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ROUGE-L (or F1) similarity for a set of input CSVs against a reference CSV"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        metavar="INPUT",
        required=True,
        help="One or more input CSV paths (formerly INPUT02, INPUT03, INPUT06, etc.)"
    )
    parser.add_argument(
        "--input0",
        required=True,
        help="Path to the reference CSV (formerly INPUT0)"
    )
    parser.add_argument(
        "--metric",
        choices=["rouge", "f1"],
        default="rouge",
        help="Which metric to use: 'rouge' for ROUGE-L, 'f1' for F1 score"
    )
    args = parser.parse_args()

    # Choose metric function based on --metric argument
    metric_func = f1_score if args.metric == "f1" else evaluate_answer

    # Gather all input CSVs (e.g. INPUT02, INPUT03, INPUT06, …) and the reference
    input_paths = args.inputs + [args.input0]
    INPUT0 = args.input0

    # Load & filter the reference answers, then reset its index
    df0 = pd.read_csv(INPUT0)
    df0 = df0.reset_index(drop=True)
    reference_answers = df0["answer"].tolist()

    # Generate processed filenames for each CSV (including reference)
    filenames = [
        os.path.splitext(path)[0] + "_processed.csv"
        for path in input_paths
    ]

    # Process each CSV in turn
    for path, fname in zip(input_paths, filenames):
        # Load & reset index
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)

        # Compute ROUGE-L (or F1) by looking up the reference answer via row number
        df["ROUGEL"] = df.apply(
            lambda row: metric_func(
                row["answer"],
                reference_answers[row.name]
            ),
            axis=1
        )

        # Save
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == "__main__":
    main()
