import argparse
import pandas as pd
import os

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
    df = pd.read_csv(args.input_tokens_csv)
    df_aux = pd.read_csv(args.input_csv)
    df = df.copy()

    # preload auxiliary CSVs
    prefill_csv = os.path.join(args.prefill_dir, "0.csv")
    df_prefill = pd.read_csv(prefill_csv)
    if args.rate_1_file:
        df_rate1 = pd.read_csv(args.rate_1_file)

    # preload rate-dependent ROUGEL CSVs
    df_rates1 = {}
    for rate, code in RATE_CODE.items():
        path = os.path.join(args.streaming_dir1, f"results_rate_{code}_processed.csv")
        df_rates1[code] = pd.read_csv(path)

    # preload rate-dependent ROUGEL CSVs
    df_rates2 = {}
    for rate, code in RATE_CODE.items():
        path = os.path.join(args.streaming_dir2, f"results_rate_{code}_processed.csv")
        df_rates2[code] = pd.read_csv(path)

    ttfts = []
    rouges = []

    for _idx_row, (_, row) in enumerate(df.iterrows()):
        token_num = row["Token Number"]
        device    = row["device"]
        idx       = row["index_in_dataset"]
        dataset   = df_aux.iloc[_idx_row]["dataset"]

        if token_num == 0:
            ttft  = df_prefill.iloc[_idx_row]["ttft"]
            rouge = 1.0
        else:
            rate = row["Rates"]
            num = rate.strip("[]")  
            if num.endswith("."):
                num = num[:-1]
            rate = float(num)
            code = next((c for r, c in RATE_CODE.items() if abs(rate - r) < 1e-6), None)

            if rate != 1:
                if device == "cpu":
                    ttft = token_num * rate * A_CPU + B_CPU
                elif device == "disk":
                    ttft = token_num * rate * A_DISK + B_DISK
                else:
                    raise ValueError(f"Unknown device '{device}'")

                if dataset == args.dataset1:
                    rouge = find_row_value(df_rates1[code], "index_in_dataset", "ROUGEL", idx)
                elif dataset == args.dataset2:
                    rouge = find_row_value(df_rates2[code], "index_in_dataset", "ROUGEL", idx)
            else:
                ttft = df_rate1.iloc[_idx_row]["ttft"]
                rouge = df_rate1.iloc[_idx_row]["ROUGEL"]

        ttfts.append(ttft)
        rouges.append(rouge)

    df["ttft"]   = ttfts
    df["ROUGEL"] = rouges

    # derive output path by appending _processed
    dirpath, filename = os.path.split(args.input_tokens_csv)
    # go one level up from that directory
    parent_dir = os.path.dirname(dirpath)
    # split filename into name and extension
    base_name, ext = os.path.splitext(filename)
    # build your output path in the parent directory
    if args.output_csv == None:
        output_csv = os.path.join(f"{parent_dir}/../baseline_streaming", f"{base_name}_processed{ext}")
    else:
        output_csv = args.output_csv

    df.to_csv(output_csv, index=False)
    print(f"Wrote results to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute ttft & ROUGEL")
    p.add_argument("input_tokens_csv",   help="Path to your input tokens CSV")
    p.add_argument("--input-csv",   help="Path to your input CSV")
    p.add_argument("--prefill-dir",  default="results/May_13_2_triviaqa_rr/prefill",
                   help="Directory containing 0.csv")
    p.add_argument("--streaming-dir1", default="results/May_14_1_triviaqa_press",
                   help="Base dir for results_rate_XX_processed.csv files")
    p.add_argument("--dataset1")
    p.add_argument("--streaming-dir2", default="results/May_14_1_triviaqa_press",
                   help="Base dir for results_rate_XX_processed.csv files")
    p.add_argument("--dataset2")
    p.add_argument("--rate-1-file", default="results/May_23_1_sum/prefill/1_processed.csv",)
    p.add_argument("--output-csv", default=None)
    args = p.parse_args()
    main(args)

'''
Usage for sum:
python3 evaluate_similarity_streaming.py \
    results/Jun_4_2_sum/baseline_kivi/tokens/02.csv \
    --input-csv results/Jun_4_2_sum/baseline_kivi/02.csv \
    --prefill-dir results/Jun_4_2_sum/prefill \
    --streaming-dir1 results/May_13_streaming \
    --dataset1 samsum \
    --streaming-dir2 ../../press/qmsum \
    --dataset2 qmsum

Usage for qa:
python3 evaluate_similarity_streaming.py \
    results/Jun_5_1_qa/baseline_kivi/tokens/02.csv \
    --input-csv results/Jun_5_1_qa/baseline_kivi/02.csv \
    --prefill-dir results/Jun_5_1_qa/prefill \
    --streaming-dir1 ../../press/triviaqa \
    --dataset1 triviaqa \
    --streaming-dir2 ../../press/hotpotqa \
    --dataset2 hotpotqa \
    --rate-1-file results/Jun_5_1_qa/prefill/1_processed.csv

Usage for coding:
python3 evaluate_similarity_streaming.py \
    results/Jun_19_1_coding/baseline_kivi/tokens/02.csv \
    --input-csv results/Jun_19_1_coding/baseline_kivi/02.csv \
    --prefill-dir results/Jun_19_1_coding/prefill \
    --streaming-dir1 ../../press/repobench-p_e \
    --dataset1 repobench-p_e \
    --streaming-dir2 ../../press/lcc_e \
    --dataset2 lcc_e \
    --rate-1-file results/Jun_19_1_coding/prefill/1_processed.csv
'''
