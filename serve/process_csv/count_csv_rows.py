#!/usr/bin/env python3
import csv
import argparse
import sys

csv.field_size_limit(sys.maxsize)

def count_rows(csv_path: str, has_header: bool = True) -> int:
    """
    Count the number of rows in a CSV file.

    :param csv_path: Path to the CSV file.
    :param has_header: Whether the first row is a header (default: True).
    :return: Number of data rows (excluding header if has_header=True).
    """
    try:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            total = sum(1 for _ in reader)
    except FileNotFoundError:
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return total - 1 if has_header and total > 0 else total

def main():
    parser = argparse.ArgumentParser(
        description="Count the number of rows in a CSV file."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file to be counted."
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the file as having no header row."
    )
    args = parser.parse_args()

    count = count_rows(args.csv_file, has_header=not args.no_header)
    print(count)

if __name__ == "__main__":
    main()
