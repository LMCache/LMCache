#!/usr/bin/env python3
"""
Process a CSV file with columns: context, question, reference answer.

1. Add an `id` column: first three rows → 0, next three → 1, etc.
2. Prepend a header row before each group of three, with:
   - same context
   - question = "0"
   - reference answer = "0"
   - id = "<group>_x"
3. Sort so that rows are grouped by `context`, and within each context, each header (`id_x`) appears just before its three rows.
4. Save to output CSV.

Usage:
    python process_csv.py input.csv output.csv
"""
import pandas as pd
import argparse


def process_csv(input_csv: str, output_csv: str) -> None:
    # Read input
    df = pd.read_csv(input_csv)

    # 1. Add `id` column: 3 rows per group
    df['id'] = (df.index // 3).astype(str)

    # 2. Build header rows
    header_rows = []
    for group_id, group in df.groupby('id', sort=False):
        header_rows.append({
            'context': group['context'].iloc[0],
            'question': '0',
            'reference answer': '0',
            'id': f"{group_id}_x"
        })
    header_df = pd.DataFrame(header_rows, columns=df.columns)

    # 3. Combine header rows and original data
    combined = pd.concat([header_df, df], ignore_index=True)

    # 4. Add helper fields for sorting: numeric group, header flag
    combined['group_num'] = combined['id'].apply(lambda v: int(v.split('_')[0]))
    combined['header_flag'] = combined['id'].apply(lambda v: 0 if v.endswith('_x') else 1)

    # 5. Sort by context, then group number, then header_flag
    processed_df = combined.sort_values(
        by=['context', 'group_num', 'header_flag'],
        ignore_index=True
    )

    # Drop helper columns
    processed_df = processed_df.drop(columns=['group_num', 'header_flag'])

    # Write output
    processed_df.to_csv(output_csv, index=False)
    print(f"Processed CSV written to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process CSV: group rows by three, add headers, sort by context and group."
    )
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('output_csv', help='Path where the processed CSV will be saved')
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv)
