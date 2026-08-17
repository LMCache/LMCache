# SPDX-License-Identifier: Apache-2.0
"""L1 memory managers for the distributed cache.

Interchangeable tiers behind :class:`L1ManagerProtocol`:

- :class:`L1MemoryManager` -- CPU pinned-DRAM slab.
- :class:`DevDaxL1MemoryManager` -- Device-DAX-backed L1 slab.
- :class:`GDSL1MemoryManager` -- GDS slab file (cuFile DMA).
- :class:`DeviceResidentL1MemoryManager` -- device-resident L1.
"""

# First Party
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.device_resident_l1_memory_manager import (
    DeviceMemoryPool,
    DeviceResidentL1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.gds_l1_memory_manager import (
    GDSL1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.l1_manager_protocol import L1ManagerProtocol
from lmcache.v1.distributed.memory_manager.l1_memory_manager import (
    L1MemoryManager,
    create_memory_allocator,
)

__all__ = [
    "DevDaxL1MemoryManager",
    "DeviceResidentL1MemoryManager",
    "DeviceMemoryPool",
    "GDSL1MemoryManager",
    "L1ManagerProtocol",
    "L1MemoryManager",
    "create_memory_allocator",
]
