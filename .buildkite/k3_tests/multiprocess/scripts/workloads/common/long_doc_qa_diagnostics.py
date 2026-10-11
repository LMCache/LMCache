# SPDX-License-Identifier: Apache-2.0
"""Summarize a long-document QA phase without changing benchmark aggregates."""

# Standard
from pathlib import Path
import argparse
import csv
import math
import shlex
import statistics


def summarize_round(csv_path: Path) -> str:
    """Report sample counts and successful-request timing distributions."""
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    successful = [row for row in rows if row["successful"] == "True"]
    round_name = csv_path.stem.removesuffix("_round")
    lines = [
        f"{round_name}: samples={len(rows)} successful={len(successful)} "
        f"failed={len(rows) - len(successful)}"
    ]
    if not successful:
        lines.append("  timings unavailable: no successful requests")
        return "\n".join(lines)

    ttfts = [float(row["ttft"]) for row in successful]
    latencies = [
        float(row["request_end"]) - float(row["request_start"]) for row in successful
    ]
    for metric, values in (
        ("ttft", ttfts),
        ("request_latency", latencies),
    ):
        ordered = sorted(values)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        lines.append(
            f"  {metric}: min={ordered[0]:.6f} "
            f"median={statistics.median(ordered):.6f} "
            f"p95={p95:.6f} max={ordered[-1]:.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Print one phase's invocation and warmup/query CSV diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "lmcache"), required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()

    print(f"=== long_doc_qa diagnostics: {args.phase} ===")
    print(f"engine: {args.engine}")
    print(f"command: {shlex.join(args.command)}")
    print("Timings in seconds; successful requests only; untrimmed; p95=nearest-rank")
    for round_name in ("warmup", "query"):
        print(summarize_round(args.results_dir / f"{round_name}_round.csv"))


if __name__ == "__main__":
    main()
