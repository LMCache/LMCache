import os
import re
import glob

RESULTS_DIR = "mmlu-results"
OUTFILE = "compare-results/comparison.txt"

def parse_file(path):
    acc, lat = None, None
    try:
        with open(path) as f:
            for line in f:
                if match := re.match(r"Average accuracy:?\s*([0-9.]+)", line):
                    acc = float(match.group(1))
                elif match := re.match(r"Total latency:?\s*([0-9.]+)", line):
                    lat = float(match.group(1))
    except Exception as e:
        print(f"⚠️ Failed to parse {path}: {e}")
    return acc, lat

def main():
    os.makedirs("compare-results", exist_ok=True)
    report = ["🔍 MMLU Benchmark Results\n"]

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "v0_*.txt")))
    if not files:
        print("❌ No result files found.")
        return

    for f in files:
        name = os.path.basename(f).replace(".txt", "")
        acc, lat = parse_file(f)
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        lat_str = f"{lat:.2f}" if lat is not None else "N/A"
        report.append(f"- **{name}** → accuracy: {acc_str}, latency: {lat_str}")

    text = "\n".join(report)
    print(text)

    with open(OUTFILE, "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()
