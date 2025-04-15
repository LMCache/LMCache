# This script evaluates the similarity of answers in a CSV file using the ROUGE metric.
import pandas as pd
from our_metrics import evaluate_answer, evaluate_answer

INPUT1 = 'results/Apr_14/baseline_kivi/1.csv'
INPUT02 = 'results/Apr_14/baseline_kivi/02.csv'
INPUT03 = 'results/Apr_14/baseline_kivi/03.csv'
INPUT06 = 'results/Apr_14/baseline_kivi/06.csv'

INPUT_OURS001 = 'results/Apr_14/ours/001.csv'

INPUT_TRUTH = 'results/Apr_1/1.csv'

def main():
    # Read the CSV file into a DataFrame.
    df1 = pd.read_csv(INPUT1)
    df02 = pd.read_csv(INPUT02)
    df03 = pd.read_csv(INPUT03)
    df06 = pd.read_csv(INPUT06)
    df_ours001 = pd.read_csv(INPUT_OURS001)
    df_truth1 = pd.read_csv(INPUT_TRUTH)
    dataframes = [df1, df02, df03, df06, df_ours001]
    filenames = ['results/Apr_14/baseline_kivi/1_processed.csv', 'results/Apr_14/baseline_kivi/02_processed.csv', 'results/Apr_14/baseline_kivi/03_processed.csv', 'results/Apr_14/baseline_kivi/06_processed.csv', 'results/Apr_14/ours/001_processed.csv']
        
    # Apply the evaluate_answer function to each row and store the result in a new column.
    for df, fname in zip(dataframes, filenames):
        df['f1_score'] = df.apply(
            lambda row: evaluate_answer(
                row['answer'],
                df_truth1.loc[df_truth1['index_in_dataset'] == row['index_in_dataset'], 'answer'].iloc[0]
            ),
            axis=1
        )
            
        # Write the updated DataFrame back to a new CSV file.
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == '__main__':
    main()
