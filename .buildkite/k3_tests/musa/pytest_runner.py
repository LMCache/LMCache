# SPDX-License-Identifier: Apache-2.0
"""Spawn-safe pytest entry point for the MUSA CI unit and smoke suites."""

# Standard
import sys

# Third Party
import pytest
import torch_musa  # noqa: F401 - registers torch.musa before test collection


def _main() -> int:
    """Run pytest without making spawned children re-enter the test runner."""
    return int(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(_main())
