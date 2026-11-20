# SPDX-License-Identifier: Apache-2.0
"""The dedicated GPU qualification must reject missing or partial evidence."""

# Standard
from pathlib import Path
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

# Third Party
import pytest

RUNNER = (
    Path(__file__).resolve().parents[2]
    / ".buildkite/k3_tests/multiprocess/run-prefix-cap-tests.py"
)


@pytest.mark.parametrize(
    "outcome",
    ["pass", "missing", "partial", "skip", "failure", "error", "crash", "malformed"],
)
def test_qualification_entrypoint_requires_complete_success(
    tmp_path: Path, outcome: str
) -> None:
    """Exercise the real runner with a disposable child, not its parser alone."""
    suite = ET.Element("testsuite")
    for cap in ("0", "8", "19", "None"):
        for mode in ("eager", "FIFO", "EVICTION_AWARE"):
            ET.SubElement(
                suite,
                "testcase",
                name=f"test_cap_controls_actual_cuda_and_storage_manager_payload[{cap}-{mode}]",
            )
    if outcome == "partial":
        suite.remove(suite[-1])
    if outcome in ("skip", "failure", "error"):
        ET.SubElement(suite[-1], "skipped" if outcome == "skip" else outcome)
    payload = ET.tostring(suite, encoding="unicode")
    if outcome == "malformed":
        payload = "<testsuite>"
    (tmp_path / "pytest.py").write_text(
        "import os, pathlib, sys\n"
        "report = next(a.split('=', 1)[1] for a in sys.argv\n"
        "              if a.startswith('--junitxml='))\n"
        "if os.environ['OUTCOME'] != 'missing':\n"
        "    pathlib.Path(report).write_text(os.environ['REPORT'])\n"
        "print('12 passed')\n"
        "sys.exit(1 if os.environ['OUTCOME'] == 'crash' else 0)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "REPORT": payload,
            "OUTCOME": outcome,
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if outcome == "pass":
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode != 0
