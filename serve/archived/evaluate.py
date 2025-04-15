import pandas as pd
from our_metrics import evaluate_answer, f1_score

INPUT = 'results/nqa_02_original200.csv'

def main():
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(INPUT)
    
    # Ensure the CSV contains the required columns.
    if 'reference_answer' not in df.columns or 'answer' not in df.columns:
        print("Error: CSV file must contain 'reference_answer' and 'answer' columns.")
        return
    
    # Apply the evaluate_answer function to each row and store the result in a new column.
    df['rougeL_fmeasure'] = df.apply(lambda row: evaluate_answer(row['answer'], row['reference_answer']), axis=1)
    df['f1_score'] = df.apply(lambda row: f1_score(row['answer'], row['reference_answer']), axis=1)
    
    # Write the updated DataFrame back to a new CSV file.
    df.to_csv(INPUT, index=False)
    print(f"Processed CSV saved to {INPUT}")

if __name__ == '__main__':
    main()
