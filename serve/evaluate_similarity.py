import pandas as pd
from our_metrics import evaluate_answer, f1_score

INPUT1 = 'results/Apr_1/baseline_kivi/1.csv'
INPUT02 = 'results/Apr_1/baseline_kivi/02.csv'
INPUT03 = 'results/Apr_1/baseline_kivi/03.csv'
INPUT06 = 'results/Apr_1/baseline_kivi/06.csv'
INPUT0 = 'results/Apr_1/baseline_kivi/0.csv'

INPUT_OURS1 = 'results/Apr_1/ours/1.csv'

def main():
    # Read the CSV file into a DataFrame.
    df1 = pd.read_csv(INPUT1)
    df02 = pd.read_csv(INPUT02)
    df03 = pd.read_csv(INPUT03)
    df06 = pd.read_csv(INPUT06)
    df0 = pd.read_csv(INPUT0)
    df_ours1 = pd.read_csv(INPUT_OURS1)
    dataframes = [df1, df02, df03, df06, df0, df_ours1]
    filenames = ['results/Apr_1/baseline_kivi/1_processed.csv', 'results/Apr_1/baseline_kivi/02_processed.csv', 'results/Apr_1/baseline_kivi/03_processed.csv', 'results/Apr_1/baseline_kivi/06_processed.csv', 'results/Apr_1/baseline_kivi/0_processed.csv', 'results/Apr_1/ours/1_processed.csv']
        
    # Apply the evaluate_answer function to each row and store the result in a new column.
    for df, fname in zip(dataframes, filenames):
        df['f1_score'] = df.apply(
            lambda row: f1_score(row['answer'], df0['answer'].iloc[row.name]), 
            axis=1
        )
    
        # Write the updated DataFrame back to a new CSV file.
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == '__main__':
    main()
