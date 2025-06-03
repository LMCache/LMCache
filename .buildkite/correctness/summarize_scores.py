# Standard
import glob
import os
import re

# Use environment variable for results directory if set, otherwise default
RESULTS_DIR = os.environ.get("RESULTS_DIR", "mmlu-results")
OUTFILE = "compare-results/comparison.txt"


def parse_file(path):
    acc, lat = None, None
    try:
        with open(path) as f:
            content = f.read()
            print(f"📄 Parsing {path} ({len(content)} chars)")
            for line in content.splitlines():
                if match := re.match(r"Average accuracy:?\s*([0-9.]+)", line):
                    acc = float(match.group(1))
                elif match := re.match(r"Total latency:?\s*([0-9.]+)", line):
                    lat = float(match.group(1))
    except Exception as e:
        print(f"⚠️ Failed to parse {path}: {e}")
    return acc, lat


def main():
    print(f"🔍 Looking for results in: {os.path.abspath(RESULTS_DIR)}")

    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Results directory '{RESULTS_DIR}' does not exist")
        print("📁 Current directory contents:")
        for item in os.listdir("."):
            print(f"   - {item}")
        return

    # List all files in results directory
    all_files = os.listdir(RESULTS_DIR)
    print(f"📁 Files in {RESULTS_DIR}: {all_files}")

    os.makedirs("compare-results", exist_ok=True)
    report = ["🔍 MMLU Benchmark Results\n"]

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.txt")))
    print(f"🎯 Found {len(files)} .txt files: {[os.path.basename(f) for f in files]}")

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

    print(f"✅ Summary written to {OUTFILE}")


if __name__ == "__main__":
    main()
