# SPDX-License-Identifier: Apache-2.0
"""Run the full prefix-cap module and require all CUDA payload cases."""

# Standard
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def main() -> int:
    """Retain fresh JUnit evidence; missing, skipped or partial results fail.

    Returns:
        The pytest failure status, or one if qualification evidence is invalid.
    """
    fd, filename = tempfile.mkstemp(prefix="prefix-cap-", suffix=".xml", dir=".")
    os.close(fd)
    report = Path(filename)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "tests/v1/test_mp_offload_cap.py",
            f"--junitxml={report}",
        ],
        check=False,
    )
    if completed.returncode != 0:
        return completed.returncode
    try:
        root = ET.parse(report).getroot()
    except (OSError, ET.ParseError) as error:
        print(f"Invalid prefix-cap report {report}: {error}", file=sys.stderr)
        return 1
    cases = list(root.iter("testcase"))
    required = {
        f"test_cap_controls_actual_cuda_and_storage_manager_payload[{cap}-{mode}]"
        for cap in ("0", "8", "19", "None")
        for mode in ("eager", "FIFO", "EVICTION_AWARE")
    }
    names = {case.get("name") for case in cases}
    if not required.issubset(names) or any(
        node.tag in {"failure", "error", "skipped"} for node in root.iter()
    ):
        print(
            f"Incomplete or unsuccessful prefix-cap report: {report}", file=sys.stderr
        )
        return 1
    print(f"Prefix-cap qualification: {len(cases)} passed, no skips; report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
