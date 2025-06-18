import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Filter rows with occurrence_number == 1 from a CSV"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for the filtered output CSV file"
    )
    args = parser.parse_args()

    # 读取 CSV
    df = pd.read_csv(args.input)

    # 筛选出 occurrence_number == 1 的行
    df_filtered = df[df["occurrence_number"] == 1]

    # 输出到新的 CSV，不保留原索引
    df_filtered.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
