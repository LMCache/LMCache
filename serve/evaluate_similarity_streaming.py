import argparse
import pandas as pd
import os
import sys

# constants
A_CPU = 8.54669335240032e-06
B_CPU = 0.017131611630247772
A_DISK = 0.00013438977394253005
B_DISK = -0.0006046225223680943

# mapping from rate to file code
RATE_CODE = {
    0.371428571: "02",
    0.485714286: "03",
    0.728571429: "06",
}

def find_row_value(df, idx_col, val_col, idx):
    """Helper to lookup val_col in df where idx_col == idx."""
    matches = df.loc[df[idx_col] == idx, val_col]
    if matches.empty:
        raise KeyError(f"No row with {idx_col}={idx}")
    return matches.iloc[0]

def main(args):
    # read input and filter
    df = pd.read_csv(args.input_csv)
    df = df[df["occurrence_number"] == 2].copy()
    if df.empty:
        print("No rows with occurence_number == 2", file=sys.stderr)
        sys.exit(1)

    # preload auxiliary CSVs
    prefill_csv = os.path.join(args.prefill_dir, "0.csv")
    df_prefill = pd.read_csv(prefill_csv)

    # preload rate-dependent ROUGEL CSVs
    df_rates = {}
    for rate, code in RATE_CODE.items():
        path = os.path.join(args.streaming_dir, f"results_rate_{code}_processed.csv")
        df_rates[code] = pd.read_csv(path)
    df_rates["1"] = pd.read_csv("../results/Apr_28_samsum/baseline_kivi/1_processed.csv")

    ttfts = []
    rouges = []

    for _, row in df.iterrows():
        token_num = row["Token Number"]
        device    = row["device"]
        idx       = row["index_in_dataset"]

        if token_num == 0:
            ttft  = find_row_value(df_prefill, "index_in_dataset", "ttft", idx)
            rouge = 1.0
        else:
            rate = row["Rates"]
            num = rate.strip("[]")  
            if num.endswith("."):
                num = num[:-1]
            rate = float(num)
            code = next((c for r, c in RATE_CODE.items() if abs(rate - r) < 1e-6), None)
            if rate == 1:
                code = "1"

            if device == "cpu":
                ttft = token_num * rate * A_CPU + B_CPU
            elif device == "disk":
                ttft = token_num * rate * A_DISK + B_DISK
            else:
                raise ValueError(f"Unknown device '{device}'")

            rouge = find_row_value(df_rates[code], "index_in_dataset", "ROUGEL", idx)

        ttfts.append(ttft)
        rouges.append(rouge)

    df["ttft"]   = ttfts
    df["ROUGEL"] = rouges

    # derive output path by appending _processed
    dirpath, filename = os.path.split(args.input_csv)
    # go one level up from that directory
    parent_dir = os.path.dirname(dirpath)
    # split filename into name and extension
    base_name, ext = os.path.splitext(filename)
    # build your output path in the parent directory
    output_csv = os.path.join(parent_dir, f"{base_name}_processed2{ext}")

    df.to_csv(output_csv, index=False)
    print(f"Wrote results to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute ttft & ROUGEL for occurrence_number=2 rows")
    p.add_argument("input_csv",   help="Path to your input CSV")
    p.add_argument("--prefill-dir",  default="results/May_13_2_triviaqa_rr/prefill",
                   help="Directory containing 0.csv")
    p.add_argument("--streaming-dir", default="results/May_14_1_triviaqa_press",
                   help="Base dir for results_rate_XX_processed.csv files")
    args = p.parse_args()
    main(args)