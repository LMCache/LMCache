#!/usr/bin/env python3
import argparse
import pandas as pd

def main(input_csv):
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    # Check the column exists
    if 'ROUGEL' not in df.columns:
        raise KeyError(f"'ROUGEL' column not found in {input_csv}")
    
    # Compute the mean (ignoring NaNs)
    avg = df['ROUGEL'].mean()
    print(f'Average ROUGEL: {avg:.4f}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compute average ROUGEL from CSV')
    p.add_argument('input_csv', help='Path to the input CSV file')
    args = p.parse_args()
    main(args.input_csv)
