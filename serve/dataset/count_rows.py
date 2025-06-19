#!/usr/bin/env python3
# count_rows.py

import argparse
import pandas as pd
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="统计 CSV 文件的行数")
    parser.add_argument("csv_path", help="待统计的 CSV 文件路径")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"读取 CSV 出错: {e}", file=sys.stderr)
        sys.exit(1)

    row_count = len(df)
    print(f"文件 {args.csv_path} 共包含 {row_count} 行（不含表头）")

if __name__ == "__main__":
    main()
