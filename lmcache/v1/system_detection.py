# SPDX-License-Identifier: Apache-2.0
"""Shared system topology types."""

# Standard
from dataclasses import dataclass
import os

_NUMA_NODE_SYSFS = "/sys/devices/system/node"
_PROC_SELF_STATUS = "/proc/self/status"


def _parse_id_list(text: str) -> set[int]:
    """Parse a kernel id list such as ``"0-3,8,10-11"`` into a set of ids."""
    ids: set[int] = set()
    for part in filter(None, text.strip().split(",")):
        first, _, last = part.partition("-")
        ids.update(range(int(first), int(last or first) + 1))
    return ids


def get_numa_node_cpus(node: int) -> set[int]:
    """Return the CPUs of a NUMA node that this process is allowed to run on.

    Args:
        node: NUMA node id.

    Returns:
        The intersection of the node's ``cpulist`` and the process affinity.

    Raises:
        ValueError: If the node does not exist, if none of its CPUs are allowed,
            or if the process may not allocate memory on it (e.g. a container
            cpuset that excludes the node's CPUs or memory).
    """
    try:
        with open(f"{_NUMA_NODE_SYSFS}/node{node}/cpulist") as f:
            cpulist = f.read().strip()
    except FileNotFoundError:
        raise ValueError(f"NUMA node {node} does not exist") from None
    allowed = _parse_id_list(cpulist) & os.sched_getaffinity(0)
    if not allowed:
        raise ValueError(
            f"None of NUMA node {node}'s CPUs ({cpulist}) are in this "
            "process's CPU affinity; check the container cpuset."
        )
    with open(_PROC_SELF_STATUS) as f:
        mems = next(line for line in f if line.startswith("Mems_allowed_list:"))
    if node not in _parse_id_list(mems.split(":", 1)[1]):
        raise ValueError(
            f"This process may not allocate memory on NUMA node {node} "
            f"({mems.strip()}); check the container cpuset."
        )
    return allowed


@dataclass
class NUMAMapping:
    """Map a GPU index to its NUMA node."""

    gpu_to_numa_mapping: dict[int, int]
