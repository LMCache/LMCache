# SPDX-License-Identifier: Apache-2.0

"""
Configuration for distributed storage manager
"""

# Standard
from dataclasses import dataclass, field


@dataclass
class L1MemoryManagerConfig:
    """
    The configuration for L1 memory manager.
    """

    size_in_bytes: int
    """ The size of L1 memory in bytes. """

    use_lazy: bool
    """ Whether to use lazy loading for L1 memory. """

    init_size_in_bytes: int = field(default=20 << 30)
    """ The initial size when using lazy allocation. Default is 20GB. """

    align_bytes: int = field(default=0x1000)
    """ The alignment size in bytes. Default is 4KB. """


@dataclass
class L1ManagerConfig:
    """
    Special config for the L1 Object/Key manager
    """

    memory_config: L1MemoryManagerConfig
    """ The memory manager configuration for L1 cache. """

    write_ttl_seconds: int = field(default=600)
    """ Time to live for each object's write lock. Default is 600s (10 minutes). """

    read_ttl_seconds: int = field(default=300)
    """ Time to live for each object's read lock. Default is 300s (5 minutes). """
