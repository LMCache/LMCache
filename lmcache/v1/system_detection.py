# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.c_ops import get_gpu_pci_bus_id
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


@dataclass
class NUMAMapping:
    gpu_to_numa_mapping: dict[int, int]


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
            device_index = torch.cuda.current_device()
            pci_bus_id = get_gpu_pci_bus_id(device_index).lower()

            numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            with open(numa_node_file) as f:
                numa_node = int(f.read())

            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_node})
        except Exception as e:
            logger.warning(f"Failed to auto read NUMA mapping from system: {e}")
            return None
