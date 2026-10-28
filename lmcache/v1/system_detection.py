# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Callable, Optional
import os
import platform

# Third Party
import psutil

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig


def _get_unavailable_gpu_pci_bus_id(_device_id: int) -> None:
    return None


get_gpu_pci_bus_id: Callable[[int], str | None]
try:
    # First Party
    from lmcache import device_ops

    get_gpu_pci_bus_id = device_ops.get_gpu_pci_bus_id
except ImportError:
    # Fallback if device ops are not available
    get_gpu_pci_bus_id = _get_unavailable_gpu_pci_bus_id

logger = init_logger(__name__)

_NUMA_NODE_SYSFS = "/sys/devices/system/node"
_PROC_SELF_STATUS = "/proc/self/status"


@dataclass
class NUMAMapping:
    gpu_to_numa_mapping: dict[int, int]


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


class SystemMemoryDetector:
    @staticmethod
    def get_available_memory_gb() -> float:
        """
        Get system available memory in GB using psutil.
        This method is cross-platform and doesn't require subprocess calls.

        Returns:
            Available memory in GB, or 0.0 if detection fails.
        """
        try:
            # Use psutil to get virtual memory information
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            system = platform.system()
            logger.info("%s system available memory: %.2f GB", system, available_gb)
            return available_gb

        except Exception as e:
            logger.warning("Failed to get system available memory using psutil: %s", e)
            return 0.0


class NUMADetector:
    @staticmethod
    def get_numa_mapping(config: LMCacheEngineConfig) -> Optional[NUMAMapping]:
        """
        Get NUMA mapping.
        """
        assert config.numa_mode in ["manual", "auto", None], (
            "NUMA mode must be either 'auto',  'manual', or None."
            f" Current mode: {config.numa_mode}"
        )

        numa_mapping: Optional[NUMAMapping] = None
        if config.numa_mode == "manual":
            numa_mapping = NUMADetector._read_from_config(config)
        elif config.numa_mode == "auto":
            numa_mapping = NUMADetector._read_from_sys()

        return numa_mapping

    @staticmethod
    def _read_from_config(config) -> NUMAMapping:
        """
        Read NUMA mapping from the LMCache configuration.
        """

        assert config.extra_config is not None, (
            "NUMA mode is set but extra_config is None. "
            "Please ensure the configuration is properly set."
        )

        assert "gpu_to_numa_mapping" in config.extra_config, (
            "NUMA mode is set to `manual` but gpu_to_numa_mapping is None. "
            "Please ensure the configuration is properly set."
        )

        gpu_to_numa_mapping = config.extra_config.get("gpu_to_numa_mapping")

        return NUMAMapping(gpu_to_numa_mapping)

    @staticmethod
    def _read_from_sys() -> Optional[NUMAMapping]:
        """
        Read NUMA mapping from system configuration.
        """

        try:
            device_index = torch_dev.current_device()
            pci_bus_id = get_gpu_pci_bus_id(device_index)
            if pci_bus_id is None:
                return None
            pci_bus_id = pci_bus_id.lower()

            numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            with open(numa_node_file) as f:
                numa_node = int(f.read())

            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_node})
        except Exception as e:
            logger.warning("Failed to auto read NUMA mapping from system: %s", e)
            return None
