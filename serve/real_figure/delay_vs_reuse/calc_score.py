#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

def compute_avg(filepath):
    """Read CSV, filter occurrence_number==2, return (t_avg, r_avg)."""
    df = pd.read_csv(filepath)
    df2 = df[df['occurrence_number'] == 2]
    if df2.empty:
        print(f"Warning: no rows with occurrence_number==2 in {filepath}", file=sys.stderr)
    t_avg = df2['ttft'].mean()
    r_avg = df2['ROUGEL'].mean()
    name = os.path.splitext(os.path.basename(filepath))[0]
    return name, t_avg, r_avg

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-file averages and weighted combos against a baseline CSV.")
    parser.add_argument(
        'csv_files', nargs='+',
        help="CSV files to process; first one is baseline, the rest are 'others'")
    parser.add_argument(
        '-o', '--output', default='weighted_averages.csv',
        help="Output CSV file for weighted-average results")
    args = parser.parse_args()

    if len(args.csv_files) < 2:
        parser.error("Need at least two CSV files (baseline + ≥1 others)")

    # 1) compute per-file averages
    avgs = {}
    for fp in args.csv_files:
        name, t_avg, r_avg = compute_avg(fp)
        avgs[name] = {'ttft': t_avg, 'rougel': r_avg}

    baseline = list(avgs.keys())[0]
    b_t, b_r = avgs[baseline]['ttft'], avgs[baseline]['rougel']

    # 2) compute weighted averages vs. baseline
    weights = [i * 0.1 for i in range(11)]  # 0.0 .. 1.0
    records = []
    for name in list(avgs.keys())[1:]:
        o_t, o_r = avgs[name]['ttft'], avgs[name]['rougel']
        for w in weights:
            records.append({
                'baseline': baseline,
                'other':    name,
                'weight':   w,
                'weighted_ttft':   (1 - w) * b_t + w * o_t,
                'weighted_rougel': (1 - w) * b_r + w * o_r
            })

    # 3) assemble and sort results
    out_df = pd.DataFrame(records)
    # Sort by weight (ascending)
    out_df.sort_values(by='weight', inplace=True)

    # 4) write out
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(records)} rows to: {args.output}")

if __name__ == '__main__':
    main()
