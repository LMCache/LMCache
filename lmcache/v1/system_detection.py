# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import platform
import subprocess

# Third Party
import torch

if torch.cuda.is_available():
    from lmcache.c_ops import get_gpu_pci_bus_id

# First Party
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
        Get system available memory in GB.
        Uses different methods based on the operating system:
        - Linux: 'free' command
        - macOS: 'vm_stat' command
        - Others: fallback to 0.0
        """
        system = platform.system().lower()

        if system == "linux":
            return SystemMemoryDetector._get_linux_available_memory()
        elif system == "darwin":  # macOS
            return SystemMemoryDetector._get_macos_available_memory()
        else:
            logger.warning(f"Unsupported operating system: {system}")
            return 0.0

    @staticmethod
    def _get_linux_available_memory() -> float:
        """
        Get available memory on Linux using 'free' command.
        """
        try:
            # Run 'free -b' to get memory info in bytes
            result = subprocess.run(
                ["free", "-b"], capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split("\n")

            # Parse the memory line (second line)
            # Format: Mem: total used free shared buff/cache available
            mem_line = lines[1].split()
            if len(mem_line) >= 7:
                available_bytes = int(mem_line[6])  # available column
                available_gb = available_bytes / (1024**3)
                logger.info(f"Linux system available memory: {available_gb:.2f} GB")
                return available_gb
            else:
                logger.warning("Unable to parse 'free' command output format")
                return 0.0
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            IndexError,
        ) as e:
            logger.warning(f"Failed to get Linux system available memory: {e}")
            return 0.0

    @staticmethod
    def _get_macos_available_memory() -> float:
        """
        Get available memory on macOS using 'vm_stat' command.
        """
        try:
            # Get page size
            page_size_result = subprocess.run(
                ["pagesize"], capture_output=True, text=True, check=True
            )
            page_size = int(page_size_result.stdout.strip())

            # Get vm_stat output
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split("\n")

            free_pages = 0
            inactive_pages = 0

            for line in lines:
                if "Pages free:" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive:" in line:
                    inactive_pages = int(line.split(":")[1].strip().rstrip("."))

            # Calculate available memory (free + inactive pages)
            available_pages = free_pages + inactive_pages
            available_bytes = available_pages * page_size
            available_gb = available_bytes / (1024**3)

            logger.info(f"macOS system available memory: {available_gb:.2f} GB")
            return available_gb

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            IndexError,
        ) as e:
            logger.warning(f"Failed to get macOS system available memory: {e}")
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
            device_index = torch.cuda.current_device()
            pci_bus_id = get_gpu_pci_bus_id(device_index).lower()

            numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            with open(numa_node_file) as f:
                numa_node = int(f.read())

            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_node})
        except Exception as e:
            logger.warning(f"Failed to auto read NUMA mapping from system: {e}")
            return None
