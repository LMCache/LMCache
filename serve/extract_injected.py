import re
import csv
import sys

def extract_token_numbers(input_file, output_csv, drop_first=200):
    records = []
    token_pattern = r"Injected token number:\s*(\d+)"
    rate_pattern  = r"rate:\s*([\d\.]+)"
    
    current_rates = []

    with open(input_file, "r") as infile:
        for line in infile:
            # 1) Capture any LMCache rate
            rate_match = re.search(rate_pattern, line)
            if rate_match:
                current_rates.append(rate_match.group(1))
                continue

            # 2) On injection, record token + rates, then reset
            token_match = re.search(token_pattern, line)
            if token_match:
                token = token_match.group(1)
                records.append((token, current_rates.copy()))
                current_rates.clear()
                continue

    # Drop the first `drop_first` entries
    records = records[drop_first:] if len(records) > drop_first else []

    # Write out CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Token Number", "Rates"])
        for token, rates in records:
            # keep only distinct rates, preserving order
            seen = set()
            unique_rates = []
            for r in rates:
                if r not in seen:
                    seen.add(r)
                    unique_rates.append(r)
            # format as e.g. "[0.485714286, 0.485714286]" but now without duplicates
            rates_str = "[" + ", ".join(unique_rates) + "]"
            writer.writerow([token, rates_str])

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python extract_token_numbers.py input_file output_csv [drop_first]")
    else:
        input_file = sys.argv[1]
        output_csv = sys.argv[2]
        drop_first = int(sys.argv[3]) if len(sys.argv) == 4 else 200
        extract_token_numbers(input_file, output_csv, drop_first)
        print(f"Done: dropped first {drop_first}, kept distinct rates, wrote to {output_csv}")
