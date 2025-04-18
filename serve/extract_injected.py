import re
import csv
import sys

def extract_token_numbers(input_file, output_csv):
    token_numbers = []
    # This regex matches "Injected token number:" followed by optional whitespace and a series of digits.
    pattern = r"Injected token number:\s*(\d+)"
    
    with open(input_file, "r") as infile:
        for line in infile:
            match = re.search(pattern, line)
            if match:
                token = match.group(1)
                token_numbers.append(token)
            else:
                # Optionally, you can print a warning if the line does not match.
                print(f"Warning: Could not find token number in line: {line.strip()}")
    
    # Write the extracted token numbers to a CSV file.
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row; you can customize the header if needed.
        writer.writerow(["Token Number"])
        for token in token_numbers:
            writer.writerow([token])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_token_numbers.py input_file output_csv")
    else:
        input_file = sys.argv[1]
        output_csv = sys.argv[2]
        extract_token_numbers(input_file, output_csv)
        print(f"Extraction complete. Token numbers saved to {output_csv}")
