#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

def find_min_ttft(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    results = []
    # Iterate weights in ascending order
    for w, group in sorted(df.groupby('weight'), key=lambda x: x[0]):
        valid = group[group['weighted_rougel'] >= 0.95]
        if valid.empty:
            print(f"Warning: no entries with weighted_rougel ≥ 95% at weight {w}", file=sys.stderr)
            # Record None if no valid entry
            results.append({'weight': w, 'min_ttft_for_rougel_>=95': None})
        else:
            min_ttft = valid['weighted_ttft'].min()
            results.append({'weight': w, 'min_ttft_for_rougel_>=95': min_ttft})

    out_df = pd.DataFrame(results, columns=['weight', 'min_ttft_for_rougel_>=95'])

    if output_csv:
        out_df.to_csv(output_csv, index=False)
        print(f"Wrote results to {output_csv}")
    else:
        print(out_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(
        description="For each weight in weighted_averages.csv, find the minimum ttft where weighted_rougel >= 95%."
    )
    parser.add_argument('input_csv', help="Path to weighted_averages.csv")
    parser.add_argument('-o', '--output', help="Optional path to write results (CSV). If omitted, prints to stdout.")
    args = parser.parse_args()

    find_min_ttft(args.input_csv, args.output)

if __name__ == '__main__':
    main()
