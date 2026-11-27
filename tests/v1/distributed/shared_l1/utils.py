# SPDX-License-Identifier: Apache-2.0
"""Helpers for shared Device-DAX L1 tests."""

# Standard
from pathlib import Path

CAPACITY = 128 * 4096


def region_file(path: Path, size_bytes: int) -> Path:
    """Create a zero-filled file standing in for a Device-DAX device.

    The ``DEVDAX`` option maps a regular file through the same code path, so
    CPU tests share a region by mapping one file.
    """
    with open(path, "wb") as region:
        region.truncate(size_bytes)
    return path
