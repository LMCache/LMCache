import re

def average_disk_speed(log_csv_path: str) -> float:
    """
    Reads a CSV (or any text file) containing LMCache INFO lines,
    extracts all reported disk speeds (in GiB/s) and returns their average.
    """
    speeds = []
    # matches “(≈ 3.67 GiB/s)” and captures “3.67”
    speed_re = re.compile(r'≈\s*([\d.]+)\s*GiB/s')

    with open(log_csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = speed_re.search(line)
            if m:
                speeds.append(float(m.group(1)))

    if not speeds:
        raise ValueError("No disk‑speed entries found in file.")
    return sum(speeds) / len(speeds)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "logs.csv"
    avg = average_disk_speed(path)
    print(f"Average disk speed: {avg:.3f} GiB/s")
