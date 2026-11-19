# SPDX-License-Identifier: Apache-2.0
"""Pinned-DRAM L1 manager."""

# Standard
import threading

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.memory_management import MemoryObj


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
        self._streaming_reservation_lock = threading.Lock()
        self._streaming_reserved_bytes = 0
        super().__init__(config, L1MemoryManager(config.memory_config))

    @property
    def backend_type(self) -> L1BackendType:
        """Return ``L1BackendType.DRAM``."""
        return L1BackendType.DRAM

    def get_memory_tags(self) -> MemoryTags:
        """Return ``(DRAM, LOCAL)``."""
        return MemoryTags(segment=MemorySegment.DRAM, residency=MemoryResidency.LOCAL)

    def reserve_streaming_arena(self, capacity: int) -> MemoryObj:
        """Reserve streaming bytes from DRAM before hot-cache admission."""
        layout = MemoryLayoutDesc([torch.Size([capacity])], [torch.uint8])
        with self._streaming_reservation_lock:
            if (
                capacity
                >= super().get_capacity_bytes() - self._streaming_reserved_bytes
            ):
                raise ValueError("streaming reservation must leave space for hot L1")
            error, objects = self._memory_manager.allocate(layout, 1)
            if error != L1Error.SUCCESS:
                raise RuntimeError(f"cannot reserve streaming arena: {error}")
            self._streaming_reserved_bytes += objects[0].get_physical_size()
            return objects[0]

    def release_streaming_arena(self, arena: MemoryObj) -> None:
        """Return an idle streaming arena during shutdown."""
        with self._streaming_reservation_lock:
            size = arena.get_physical_size()
            if self._memory_manager.free([arena]) != L1Error.SUCCESS:
                raise RuntimeError("cannot release streaming arena")
            self._streaming_reserved_bytes -= size

    def get_memory_usage(self) -> tuple[int, int]:
        """Return the used and available hot-L1 bytes, excluding reserved windows."""
        with self._streaming_reservation_lock:
            used, total = super().get_memory_usage()
            return (
                used - self._streaming_reserved_bytes,
                total - self._streaming_reserved_bytes,
            )

    def get_capacity_bytes(self) -> int:
        """Return hot-L1 capacity after the permanent streaming reservation."""
        with self._streaming_reservation_lock:
            return super().get_capacity_bytes() - self._streaming_reserved_bytes
