#!/usr/bin/env python3
import pandas as pd
import argparse

def compute_rougel_stats(input_csv: str,
                         per_id_output: str,
                         summary_output: str = None):
    # 1. Load data
    df = pd.read_csv(input_csv)

    # 2. Filter out ids ending with "_x"
    df = df.loc[~df['id'].str.endswith('_x', na=False)]

    # 3. Group by id → mean & variance
    stats = (
        df
        .groupby('id', as_index=False)['ROUGEL']
        .agg(avg_ROUGEL = 'mean',
             var_ROUGEL = 'var')   # sample variance (ddof=1)
    )

    # 4. Save per‐id stats
    stats.to_csv(per_id_output, index=False)
    print(f"Saved per-id ROUGEL stats to '{per_id_output}' ({len(stats)} rows)")

    # 5. Compute overall
    overall_avg = df['ROUGEL'].mean()
    overall_var = df['ROUGEL'].var()

    print("\nOverall (non-*_x) ROUGEL metrics:")
    print(f"  • Average ROUGEL:    {overall_avg:.6f}")
    print(f"  • Variance ROUGEL:   {overall_var:.6f}")

    # 6. Optionally save summary
    if summary_output:
        pd.DataFrame([{
            'overall_avg_ROUGEL': overall_avg,
            'overall_var_ROUGEL': overall_var
        }]).to_csv(summary_output, index=False)
        print(f"Saved summary to '{summary_output}'")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Compute per-id mean & variance of ROUGEL (excluding ids ending with '_x'), "
                    "plus overall mean & variance."
    )
    p.add_argument('input_csv',     help="Input CSV path")
    p.add_argument('per_id_stats',  help="Where to write per-id stats (id, avg_ROUGEL, var_ROUGEL)")
    p.add_argument('summary_csv', nargs='?',
                   help="(Optional) Where to write overall summary")
    args = p.parse_args()

    compute_rougel_stats(args.input_csv,
                         args.per_id_stats,
                         args.summary_csv)
