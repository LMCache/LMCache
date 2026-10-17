# SPDX-License-Identifier: Apache-2.0
"""Append a suite's runs to the results file a static page renders.

One JSON line per run: its summary and its check against its own history,
plus where it ran. The results repo's GitHub Pages site reads this file, so
CI only ever appends to it. A line, summary shortened::

    {"host": "yotta-jap-h200-22", "suite": "nightly", "date": "20261015-011210",
     "build_url": "https://buildkite.com/...", "workload": "agentic",
     "deployment": "vllm_lmcache", "summary": {"ttft_p50": 1.84, ...},
     "vs_history": {"metrics": {...}, "min_runs": 10, ...},
     "images": {"vllm_source": "pinned", "pinned_since": 1791...}}

Usage::

    python -m perf_regression_testing.publish <report.json> <runs.jsonl> \\
        --host <name> [--build-url <url>]
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json

# Image facts the page shows next to a run.
IMAGE_KEYS = ("vllm_source", "pinned_since", "mooncake_commit", "mooncake_built")


def append(report: dict[str, Any], path: Path, host: str, build_url: str | None) -> int:
    """Append one line per run of a suite report to the results file.

    Args:
        report: A suite's ``report.json``.
        path: The results file; created if missing.
        host: Name of the host the suite ran on; the page groups runs by it.
        build_url: The CI build that ran the suite, or None outside CI.

    Returns:
        The number of lines appended: one per run, invalid ones included.
    """
    images = report.get("images", {})
    lines = [
        {
            "host": host,
            "suite": report["suite"],
            "date": report["date"],
            "build_url": build_url,
            "workload": r["workload"],
            "deployment": r["deployment"],
            "summary": r["summary"],
            "vs_history": r.get("vs_history"),
            "images": {k: images.get(k) for k in IMAGE_KEYS},
        }
        for r in report["runs"]
    ]
    with open(path, "a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return len(lines)


def main() -> None:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", type=Path, help="the suite's report.json")
    ap.add_argument("runs", type=Path, help="results file to append to")
    ap.add_argument("--host", required=True)
    ap.add_argument("--build-url", default="", help="empty outside CI")
    a = ap.parse_args()
    report = json.loads(a.report.read_text())
    n = append(report, a.runs, a.host, a.build_url or None)
    print(f"publish: {n} runs appended to {a.runs}")


if __name__ == "__main__":
    main()
