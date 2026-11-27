# SPDX-License-Identifier: Apache-2.0
"""Shared system topology types."""

# Standard
from dataclasses import dataclass


@dataclass
class NUMAMapping:
    """Map a GPU index to its NUMA node."""

    gpu_to_numa_mapping: dict[int, int]
