#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Sort a CSV file by its 'id' column (ascending)."
    )
    parser.add_argument(
        'input_csv',
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        help=(
            "Path to write the sorted CSV. "
            "If omitted, will append '_sorted' before the .csv extension."
        )
    )
    args = parser.parse_args()

    # Read
    df = pd.read_csv(args.input_csv)

    # Sort by numeric id
    df_sorted = df.sort_values(by='id', kind='mergesort')

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input_csv)
        out_path = f"{base}_sorted{ext}"

    # Write without the pandas index
    df_sorted.to_csv(out_path, index=False)
    print(f"Sorted CSV written to: {out_path}")

if __name__ == '__main__':
    main()
