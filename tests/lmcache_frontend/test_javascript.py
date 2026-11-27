# SPDX-License-Identifier: Apache-2.0
"""Run browser-script regressions with Node's built-in test runner."""

# Standard
from pathlib import Path
import shutil
import subprocess

# Third Party
import pytest


@pytest.mark.parametrize("script", sorted(Path(__file__).parent.glob("test_*.cjs")))
def test_javascript(script: Path) -> None:
    """Run a JavaScript regression file, or skip when Node is unavailable.

    Args:
        script: JavaScript test file next to this module.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for browser-script regression tests")
        return
    result = subprocess.run(
        [node, "--test", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
