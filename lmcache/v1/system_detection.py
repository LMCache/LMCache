# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import platform

# Third Party
import psutil

try:
    # First Party
    from lmcache.c_ops import get_gpu_pci_bus_id
except ImportError:
    # Fallback if c_ops is not available
    get_gpu_pci_bus_id = None

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


@dataclass
class NUMAMapping:
    gpu_to_numa_mapping: dict[int, int]


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
    def detect(
        numa_mode: str | None,
        gpu_to_numa_mapping: dict[int, int] | None = None,
    ) -> Optional[NUMAMapping]:
        """Resolve the GPU-to-NUMA mapping for a bare mode string.

        Entry point for callers without an ``LMCacheEngineConfig``.

        Args:
            numa_mode: ``"auto"`` to detect from sysfs, ``"manual"`` to use
                *gpu_to_numa_mapping*, ``None`` to disable.
            gpu_to_numa_mapping: GPU index to NUMA node mapping; required
                when *numa_mode* is ``"manual"``, ignored otherwise.

        Returns:
            The resolved mapping, or ``None`` when *numa_mode* is ``None``.

        Raises:
            ValueError: If *numa_mode* is not ``"auto"``, ``"manual"``, or
                ``None``, or is ``"manual"`` without a mapping.
        """
        if numa_mode is None:
            return None
        if numa_mode == "auto":
            return NUMADetector._read_from_sys()
        if numa_mode == "manual":
            if not gpu_to_numa_mapping:
                raise ValueError("numa_mode 'manual' requires a GPU-to-NUMA mapping.")
            return NUMAMapping(gpu_to_numa_mapping=gpu_to_numa_mapping)
        raise ValueError(
            f"Unsupported numa_mode {numa_mode!r}; expected 'auto', 'manual', or None."
        )

    @staticmethod
    def get_numa_mapping(config: LMCacheEngineConfig) -> Optional[NUMAMapping]:
        """Resolve the NUMA mapping from an engine config.

        Adapter over :meth:`detect` for callers holding an
        ``LMCacheEngineConfig``; ``"manual"`` mode reads the mapping from
        ``config.extra_config["gpu_to_numa_mapping"]``.

        Args:
            config: Engine config carrying ``numa_mode`` and, for manual
                mode, the mapping in ``extra_config``.

        Returns:
            The resolved mapping, or ``None`` when NUMA is disabled.

        Raises:
            ValueError: If ``numa_mode`` is invalid or ``"manual"`` without
                a mapping in ``extra_config``.
        """
        mapping = (config.extra_config or {}).get("gpu_to_numa_mapping")
        return NUMADetector.detect(config.numa_mode, mapping)

    @staticmethod
    def _read_from_sys() -> Optional[NUMAMapping]:
        """
        Read NUMA mapping from system configuration.
        """

        try:
            device_index = torch_dev.current_device()
            pci_bus_id = get_gpu_pci_bus_id(device_index).lower()

            numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            with open(numa_node_file) as f:
                numa_node = int(f.read())

            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_node})
        except Exception as e:
            logger.warning("Failed to auto read NUMA mapping from system: %s", e)
            return None
