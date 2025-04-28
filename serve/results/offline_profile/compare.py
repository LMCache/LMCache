#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def main(x1_path: str, x2_path: str):
    # Load the data
    df1 = pd.read_csv(x1_path)
    df2 = pd.read_csv(x2_path)

    # Extract the vectors
    A = df1['ROUGEL'].to_numpy()
    B = df2['avg_ROUGEL'].to_numpy()

    # Only use the first 180 rows
    A = A[:180]
    B = B[:180]

    # Compute metrics for model 1: \hat A_i = B_i
    mse_1 = np.mean((A - B) ** 2)
    mae_1 = np.mean(np.abs(A - B))

    # Compute metrics for model 2: \hat A_i = mean(B)
    B_mean = B.mean()
    mse_2 = np.mean((A - B_mean) ** 2)
    mae_2 = np.mean(np.abs(A - B_mean))

    # Compute Pearson correlation
    corr = np.corrcoef(A, B)[0, 1]

    # Output results
    print(f"MSE (A vs. B):           {mse_1:.4f}")
    print(f"MSE (A vs. mean(B)):     {mse_2:.4f}")
    print(f"MAE (A vs. B):           {mae_1:.4f}")
    print(f"MAE (A vs. mean(B)):     {mae_2:.4f}")
    print(f"Pearson corr(A, B):      {corr:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare how well B predicts A vs. how well a constant predicts A (using only first 180 rows)"
    )
    parser.add_argument('x1', help="Path to x1.csv (must contain 'ROUGEL' column)")
    parser.add_argument('x2', help="Path to x2.csv (must contain 'avg_ROUGEL' column)")
    args = parser.parse_args()
    main(args.x1, args.x2)
