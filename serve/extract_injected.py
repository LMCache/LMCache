import re
import csv
import sys

# Default reference CSV path
default_reference_csv = 'dataset/samsum_processed_v2.csv'

def extract_token_numbers(input_file, output_csv, reference_csv=default_reference_csv):
    # Patterns
    token_pattern  = r"Injected token number:\s*(\d+)"
    rate_pattern   = r"rate:\s*([\d\.]+)"
    device_pattern = r"Decompressed memory object from (disk|hot cache)"

    records = []
    current_rates   = []
    current_devices = []

    # 1) Parse log input for rates + device context
    with open(input_file, "r") as infile:
        for line in infile:
            # rate line?
            rm = re.search(rate_pattern, line)
            if rm:
                rate = rm.group(1)
                current_rates.append(rate)

                # detect device
                dm = re.search(device_pattern, line)
                if dm:
                    dev = dm.group(1)
                    # map “hot cache” → “cpu”
                    current_devices.append('disk' if dev == 'disk' else 'cpu')
                else:
                    # if no explicit device keyword, leave blank
                    current_devices.append('')

                continue

            # token line?
            tm = re.search(token_pattern, line)
            if tm:
                token = tm.group(1)
                # store token, plus the parallel lists of rates/devices
                records.append((token, current_rates.copy(), current_devices.copy()))
                current_rates.clear()
                current_devices.clear()

    # 2) Load reference CSV for extra columns
    reference_rows = []
    with open(reference_csv, newline="") as ref_file:
        reader = csv.DictReader(ref_file)
        for row in reader:
            reference_rows.append((
                row.get('length'),
                row.get('index_in_dataset'),
                row.get('occurrence_number')
            ))

    if len(reference_rows) != len(records):
        print(f"Warning: reference CSV rows ({len(reference_rows)}) != "
              f"extracted records ({len(records)})")

    # 3) Write merged output CSV, now including device
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Token Number",
            "Rates",
            "length",
            "index_in_dataset",
            "occurrence_number",
            "device"
        ])

        for i, (token, rates, devices) in enumerate(records):
            # dedupe rates, preserving order
            seen_rates = set()
            unique_rates = []
            for r in rates:
                if r not in seen_rates:
                    seen_rates.add(r)
                    unique_rates.append(r)

            # fallback rate = 1 if non-zero token but no rates found
            try:
                if int(token) != 0 and not unique_rates:
                    unique_rates = ['1']
            except ValueError:
                pass

            rates_str = "[" + ", ".join(unique_rates) + "]"

            # dedupe devices in same order
            seen_devs = set()
            unique_devs = []
            for d in devices:
                if d and d not in seen_devs:
                    seen_devs.add(d)
                    unique_devs.append(d)

            # pick first device (or blank)
            device_str = unique_devs[0] if unique_devs else ''

            # reference fields
            length, idx, occ = reference_rows[i] if i < len(reference_rows) else (None, None, None)

            writer.writerow([
                token,
                rates_str,
                length,
                idx,
                occ,
                device_str
            ])

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 3:
        input_file, output_csv = sys.argv[1], sys.argv[2]
        reference_csv = default_reference_csv
    elif argc == 4:
        input_file, reference_csv, output_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        print("Usage: python extract_token_numbers.py input_file [reference_csv] output_csv")
        sys.exit(1)

    extract_token_numbers(input_file, output_csv, reference_csv)
    print(f"Done: extracted tokens and rates using reference '{reference_csv}', wrote to {output_csv}")
