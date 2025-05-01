import sys
import re

def parse_file(path):
    acc, lat = None, None
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("Average accuracy:"):
                    acc = float(line.split(":")[1].strip())
                elif line.startswith("Total latency:"):
                    lat = float(line.split(":")[1].strip())
    except Exception as e:
        print(f"⚠️ Failed to parse {path}: {e}")
    return acc, lat

def main():
    if len(sys.argv) != 3:
        print("Usage: compare_mmlu_scores.py <file1> <file2>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    acc1, lat1 = parse_file(file1)
    acc2, lat2 = parse_file(file2)

    report = []
    report.append("🔍 MMLU Comparison: LMCache Disabled vs No LMCache\n")

    if acc1 is not None and acc2 is not None:
        report.append(f"- Average Accuracy (LMCache): {acc1}")
        report.append(f"- Average Accuracy (No Cache): {acc2}")
        report.append(f"- Δ Accuracy: {round(acc1 - acc2, 4)}")
    else:
        report.append("⚠️ Could not parse average accuracy for one or both files.")

    if lat1 is not None and lat2 is not None:
        report.append(f"\n- Total Latency (LMCache): {lat1}")
        report.append(f"- Total Latency (No Cache): {lat2}")
        report.append(f"- Δ Latency: {round(lat1 - lat2, 2)}")
    else:
        report.append("\n⚠️ Could not parse total latency for one or both files.")

    print("\n".join(report))

    # Save for annotation
    with open("compare-results/comparison.txt", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    main()
