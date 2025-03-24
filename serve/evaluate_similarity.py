import pandas as pd
from our_metrics import evaluate_answer, f1_score

INPUT1 = 'results/qmsum_1_original.csv'
INPUT02 = 'results/qmsum_02_original.csv'
INPUT03 = 'results/qmsum_03_original.csv'
INPUT06 = 'results/qmsum_06_original.csv'

def main():
    # Read the CSV file into a DataFrame.
    df1 = pd.read_csv(INPUT1)
    df02 = pd.read_csv(INPUT02)
    df03 = pd.read_csv(INPUT03)
    df06 = pd.read_csv(INPUT06)
    dataframes = [df1, df02, df03, df06]
    filenames = ['result_1.csv', 'result_02.csv', 'result_03.csv', 'result_06.csv']
        
    # Apply the evaluate_answer function to each row and store the result in a new column.
    for df, fname in zip(dataframes, filenames):
        df['f1_score'] = df.apply(
            lambda row: f1_score(row['answer'], df1['answer'].iloc[row.name]), 
            axis=1
        )
    
        # Write the updated DataFrame back to a new CSV file.
        df.to_csv(fname, index=False)
        print(f"Processed CSV saved to {fname}")

if __name__ == '__main__':
    main()
