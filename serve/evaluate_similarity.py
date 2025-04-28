# This script evaluates the similarity of answers in a CSV file using the ROUGE metric.
import pandas as pd
from our_metrics import evaluate_answer, f1_score

INPUT1 = 'results/Apr_14/baseline_kivi/1_processed.csv'
INPUT02 = 'results/Apr_14/baseline_kivi/02.csv'
INPUT03 = 'results/Apr_14/baseline_kivi/03.csv'
INPUT06 = 'results/Apr_14/baseline_kivi/06.csv'
INPUT0 = 'results/Apr_14/baseline_kivi/0.csv'

INPUT_OURS01 = 'results/Apr_14/context_based/01_processed.csv'
INPUT_OURS10 = 'results/Apr_14/context_based/10_processed.csv'
INPUT_OURS100 = 'results/Apr_14/context_based/100_processed.csv'

def main():
    # Read the CSV file into a DataFrame.
    df1 = pd.read_csv(INPUT1)
    df02 = pd.read_csv(INPUT02)
    df03 = pd.read_csv(INPUT03)
    df06 = pd.read_csv(INPUT06)
    df0 = pd.read_csv(INPUT0)
    df_ours01 = pd.read_csv(INPUT_OURS01)
    df_ours10 = pd.read_csv(INPUT_OURS10)
    df_ours100 = pd.read_csv(INPUT_OURS100)
    dataframes = [df1, df02, df03, df06, df0, df_ours01, df_ours10, df_ours100]
    filenames = ['results/Apr_14/baseline_kivi/1_processed.csv', 'results/Apr_14/baseline_kivi/02_processed.csv', 'results/Apr_14/baseline_kivi/03_processed.csv', 'results/Apr_14/baseline_kivi/06_processed.csv', 'results/Apr_14/baseline_kivi/0_processed.csv', 'results/Apr_14/context_based/01_processed.csv', 'results/Apr_14/context_based/10_processed.csv', 'results/Apr_14/context_based/100_processed.csv']
        
    # Apply the evaluate_answer function to each row and store the result in a new column.
    for df, fname in zip(dataframes, filenames):
        # df['ROUGEL'] = df.apply(
        #     lambda row: evaluate_answer(
        #         row['answer'],
        #         df0.loc[df0['index_in_dataset'] == row['index_in_dataset'], 'answer'].iloc[0]
        #     ),
        #     axis=1
        # )
        df['ROUGEL'] = df.apply(
            lambda row: evaluate_answer(
                row['answer'],
                df0.loc[
                    df0.iloc[:, 0] == row.iloc[0],  # compare the first column by position
                    'answer'
                ].iat[0]
            ),
            axis=1
        )
            
        # Write the updated DataFrame back to a new CSV file.
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == '__main__':
    main()
