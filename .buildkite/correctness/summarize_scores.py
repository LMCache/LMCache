# Standard
import glob
import json
import os
import re

# Use environment variable for results directory if set, otherwise default
RESULTS_DIR = os.environ.get("RESULTS_DIR", "mmlu-results")
OUTFILE = "compare-results/comparison.txt"

# Also check nested buildkite artifact path
NESTED_RESULTS_DIR = ".buildkite/correctness/mmlu-results"


def parse_old_format_file(path):
    """Parse the old text format from mmlu_bench.py"""
    acc, lat = None, None
    try:
        with open(path) as f:
            content = f.read()
            print(f"📄 Parsing old format {path} ({len(content)} chars)")
            for line in content.splitlines():
                if match := re.match(r"Average accuracy:?\s*([0-9.]+)", line):
                    acc = float(match.group(1))
                elif match := re.match(r"Total latency:?\s*([0-9.]+)", line):
                    lat = float(match.group(1))
    except Exception as e:
        print(f"⚠️ Failed to parse {path}: {e}")
    return acc, lat


def parse_new_format_file(path):
    """Parse the new JSONL format from 1-mmlu.py and 2-mmlu.py"""
    acc, num_questions = None, None
    try:
        with open(path) as f:
            print(f"📄 Parsing new format {path}")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Look for the "total" key which contains overall results
                if "total" in data:
                    acc = data["total"]["accuracy"]
                    num_questions = data["total"]["num_questions"]
                    break
    except Exception as e:
        print(f"⚠️ Failed to parse {path}: {e}")
    return acc, num_questions


def get_detailed_results_from_jsonl(path):
    """Get detailed per-subject results from JSONL file"""
    results = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Each line should have one key-value pair
                for subject, metrics in data.items():
                    results[subject] = metrics
    except Exception as e:
        print(f"⚠️ Failed to parse detailed results from {path}: {e}")
    return results


def main():
    print(f"🔍 Looking for results in: {os.path.abspath(RESULTS_DIR)}")

    # Check both possible locations for results
    results_dirs = [RESULTS_DIR]
    if os.path.exists(NESTED_RESULTS_DIR):
        results_dirs.append(NESTED_RESULTS_DIR)
        print(f"🔍 Also checking nested path: {os.path.abspath(NESTED_RESULTS_DIR)}")

    # Check if any results directory exists
    existing_dirs = [d for d in results_dirs if os.path.exists(d)]
    if not existing_dirs:
        print(f"❌ No results directories found. Checked: {results_dirs}")
        print("📁 Current directory contents:")
        for item in os.listdir("."):
            print(f"   - {item}")
        return

    os.makedirs("compare-results", exist_ok=True)
    report = ["🔍 MMLU Benchmark Results\n"]

    # Look for both old format (.txt) and new format (.jsonl) files in all directories
    txt_files = []
    jsonl_files = []

    for results_dir in existing_dirs:
        print(f"📁 Checking directory: {results_dir}")
        all_files = os.listdir(results_dir)
        print(f"📁 Files in {results_dir}: {all_files}")

        txt_files.extend(sorted(glob.glob(os.path.join(results_dir, "*.txt"))))
        jsonl_files.extend(sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))))

    print(
        f"🎯 Found {len(txt_files)} .txt files: "
        f"{[os.path.basename(f) for f in txt_files]}"
    )
    print(
        f"🎯 Found {len(jsonl_files)} .jsonl files: "
        f"{[os.path.basename(f) for f in jsonl_files]}"
    )

    if not txt_files and not jsonl_files:
        print("❌ No result files found.")
        return

    # Process old format files
    for f in txt_files:
        name = os.path.basename(f).replace(".txt", "")
        acc, lat = parse_old_format_file(f)
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        lat_str = f"{lat:.2f}" if lat is not None else "N/A"
        report.append(f"- **{name}** → accuracy: {acc_str}, latency: {lat_str}")

    # Process new format files
    for f in jsonl_files:
        name = os.path.basename(f).replace(".jsonl", "")
        acc, num_questions = parse_new_format_file(f)
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        questions_str = f"{num_questions}" if num_questions is not None else "N/A"
        report.append(f"- **{name}** → accuracy: {acc_str}, questions: {questions_str}")

        # Optionally add detailed per-subject results
        detailed_results = get_detailed_results_from_jsonl(f)
        if detailed_results and len(detailed_results) > 1:  # More than just "total"
            report.append("  📊 Subject breakdown:")
            for subject, metrics in sorted(detailed_results.items()):
                if subject != "total":  # Skip total as we already showed it
                    subject_acc = metrics.get("accuracy", "N/A")
                    subject_questions = metrics.get("num_questions", "N/A")
                    if isinstance(subject_acc, float):
                        subject_acc = f"{subject_acc:.4f}"
                    report.append(
                        f"    - {subject}: {subject_acc} "
                        f"({subject_questions} questions)"
                    )

    text = "\n".join(report)
    print(text)

    with open(OUTFILE, "w") as f:
        f.write(text)

    print(f"✅ Summary written to {OUTFILE}")


if __name__ == "__main__":
    main()
