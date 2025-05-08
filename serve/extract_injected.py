import re
import csv
import sys

# Default reference CSV path
default_reference_csv = 'dataset/samsum_processed_v3.csv'

def extract_token_numbers(input_file, output_csv, reference_csv=default_reference_csv):
    # Parse log input for tokens and rates
    records = []
    token_pattern = r"Injected token number:\s*(\d+)"
    rate_pattern  = r"rate:\s*([\d\.]+)"
    current_rates = []

    with open(input_file, "r") as infile:
        for line in infile:
            rate_match = re.search(rate_pattern, line)
            if rate_match:
                current_rates.append(rate_match.group(1))
                continue
            token_match = re.search(token_pattern, line)
            if token_match:
                token = token_match.group(1)
                records.append((token, current_rates.copy()))
                current_rates.clear()

    # Read reference CSV for additional columns
    reference_rows = []
    with open(reference_csv, "r", newline="") as ref_file:
        reader = csv.DictReader(ref_file)
        for row in reader:
            reference_rows.append((row.get('length'), row.get('index_in_dataset'), row.get('occurrence_number')))

    # Warn if lengths mismatch
    if len(reference_rows) != len(records):
        print(f"Warning: reference CSV rows ({len(reference_rows)}) != extracted records ({len(records)})")

    # Write merged output CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Token Number", "Rates", "length", "index_in_dataset", "occurrence_number"])
        for i, (token, rates) in enumerate(records):
            # Deduplicate rates preserving order
            seen = set()
            unique_rates = []
            for r in rates:
                if r not in seen:
                    seen.add(r)
                    unique_rates.append(r)
            # If token non-zero and no rates, default to [1]
            try:
                if int(token) != 0 and len(unique_rates) == 0:
                    unique_rates = ['1']
            except ValueError:
                pass
            # Build rates string from unique rates
            rates_str = "[" + ", ".join(unique_rates) + "]"
            # Append reference fields
            length, idx, occ = reference_rows[i] if i < len(reference_rows) else (None, None, None)
            writer.writerow([token, rates_str, length, idx, occ])

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 3:
        input_file = sys.argv[1]
        output_csv = sys.argv[2]
        reference_csv = default_reference_csv
    elif argc == 4:
        input_file = sys.argv[1]
        reference_csv = sys.argv[2]
        output_csv = sys.argv[3]
    else:
        print("Usage: python extract_token_numbers.py input_file [reference_csv] output_csv")
        sys.exit(1)

    extract_token_numbers(input_file, output_csv, reference_csv)
    print(f"Done: extracted tokens and rates using reference '{reference_csv}', wrote to {output_csv}")
