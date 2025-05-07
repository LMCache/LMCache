# This script evaluates the similarity of answers in a CSV file using the ROUGE metric.
import pandas as pd
from our_metrics import evaluate_answer, f1_score
import os

INPUT1        = 'results/Apr_28_samsum/baseline_kivi/1.csv'
INPUT02       = 'results/Apr_28_samsum/baseline_kivi/02.csv'
INPUT03       = 'results/Apr_28_samsum/baseline_kivi/03.csv'
INPUT06       = 'results/Apr_28_samsum/baseline_kivi/06.csv'
INPUT0        = 'results/Apr_28_samsum/0.csv'
INPUT_OURS0001 = 'results/Apr_28_samsum/ours_token_based_decision/0001.csv'
INPUT_OURS001 = 'results/Apr_28_samsum/ours_token_based_decision/001.csv'
INPUT_OURS01 = 'results/Apr_28_samsum/ours_token_based_decision/01.csv'
INPUT_OURS04 = 'results/Apr_28_samsum/ours_token_based_decision/04.csv'
INPUT_OURS07 = 'results/Apr_28_samsum/ours_token_based_decision/07.csv'
INPUT_OURS1   = 'results/Apr_28_samsum/ours_token_based_decision/1.csv'
INPUT_OURS100 = 'results/Apr_28_samsum/ours_token_based_decision/100.csv'
INPUT_OURS1000 = 'results/Apr_28_samsum/ours_token_based_decision/1000.csv'

def main():
    # Load the reference answers
    df0 = pd.read_csv(INPUT0)

    # Collect all INPUT* paths except the reference INPUT0
    input_paths = [
        path for name, path in globals().items()
        if name.startswith("INPUT")
    ]

    # Read all target CSVs
    dataframes = [pd.read_csv(path) for path in input_paths]

    # Generate processed-filenames
    filenames = [os.path.splitext(path)[0] + "_processed.csv" for path in input_paths]

    # Evaluate ROUGE-L against the reference and save
    for df, fname in zip(dataframes, filenames):
        df['ROUGEL'] = df.apply(
            lambda row, ref=df0: evaluate_answer(
                row['answer'],
                ref.loc[
                    ref['index_in_dataset'] == row['index_in_dataset'],
                    'answer'
                ].iloc[0]
            ),
            axis=1
        )
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == "__main__":
    main()
