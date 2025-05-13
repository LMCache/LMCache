#!/usr/bin/env python3
import argparse
import os
import csv
import pandas as pd
import numpy as np

def compute_ab(x_vals, y_vals):
    """
    Fit y = a*x + b for given x and y arrays.
    Returns (a, b).
    """
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    a, b = np.polyfit(x, y, 1)
    return a, b


def process_csv(filepath, fname):
    """
    Read a CSV, extract appropriate 'ttft' entries based on filename,
    and return the fitted (a, b).

    - For 'tmp2_disk_1.csv', use last 3 rows with x = [12115, 14115, 16115].
    - Otherwise, use last 4 rows with x = [10115, 12115, 14115, 16115].
    """
    df = pd.read_csv(filepath)
    if 'ttft' not in df.columns:
        raise ValueError(f"'ttft' column not found in {fname}")

    if fname == 'tmp2_disk_1.csv':
        if len(df) < 3:
            raise ValueError(f"Not enough rows (<3) in {fname}")
        y_vals = df['ttft'].iloc[-3:].tolist()
        x_vals = [12115, 14115, 16115]
    else:
        if len(df) < 4:
            raise ValueError(f"Not enough rows (<4) in {fname}")
        y_vals = df['ttft'].iloc[-5:].tolist()
        x_vals = [8115, 10115, 12115, 14115, 16115]

    return compute_ab(x_vals, y_vals)


def main(input_dir, output_csv):
    results = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.csv'):
            continue
        path = os.path.join(input_dir, fname)
        try:
            a, b = process_csv(path, fname)
            results.append((fname, a, b))
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['filename', 'a', 'b'])
        writer.writerows(results)

    print(f"Wrote {len(results)} fits to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="For each CSV in a directory, fit 'ttft' values to y = a*x + b"
    )
    parser.add_argument('input_dir', help="Directory containing your CSV files")
    parser.add_argument(
        '-o', '--output',
        default='results.csv',
        help="Name of the output CSV (default: results.csv)"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output)
