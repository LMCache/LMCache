# SPDX-License-Identifier: Apache-2.0
"""Pinned-DRAM L1 manager."""

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.memory_manager import L1MemoryManager


class DramL1Manager(BaseL1Manager):
    """L1 manager backed by a pinned host-DRAM pool.

    The default medium: a single registerable buffer, so every L2 adapter
    and transfer channel can register it.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        """Create the DRAM pool described by ``config.memory_config``.

        Args:
            config: L1 settings; ``gds_l1_config`` and ``devdax_path`` must be
                unset (the factory picks another medium for those).
        """
        super().__init__(config, L1MemoryManager(config.memory_config))

    @property
    def backend_type(self) -> L1BackendType:
        """Return ``L1BackendType.DRAM``."""
        return L1BackendType.DRAM

    def get_memory_tags(self) -> MemoryTags:
        """Return ``(DRAM, LOCAL)``."""
        return MemoryTags(segment=MemorySegment.DRAM, residency=MemoryResidency.LOCAL)
