# SPDX-License-Identifier: Apache-2.0
"""Require executed tests from every selected GPU test file."""

# Standard
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def verify_results(report: Path, test_files: list[str]) -> None:
    cases = list(ET.parse(report).getroot().iter("testcase"))
    if not cases or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("skipped", "failure", "error")
    ):
        raise SystemExit("GPU tests missing, skipped, or failed")

    modules = {case.get("classname", "").split(".")[-1] for case in cases}
    missing = {Path(file).stem for file in test_files} - modules
    if missing:
        raise SystemExit(f"No executed tests from: {sorted(missing)}")
    print(f"{len(cases)} GPU test cases passed, 0 skipped")


if __name__ == "__main__":
    verify_results(Path(sys.argv[1]), sys.argv[2:])
