# SPDX-License-Identifier: Apache-2.0
"""A test L1 plugin: pinned host memory under a plugin's own name.

Loaded by ``--l1-manager '{"type": "PLUGIN", "module_path":
"tests.v1.distributed.plugin_l1_fixture", "class_name": "HostL1Manager",
...}'``. It hands out ordinary host-tensor objects, so the existing data path
and L2 adapters work with it unchanged; what it exercises is the factory's
loading and the plugin's own ``params`` validation.
"""

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


class HostL1Manager(BaseL1Manager):
    """The plugin class the factory loads; accepts one param, ``note``."""

    def __init__(self, config: L1ManagerConfig) -> None:
        plugin = config.plugin_l1_config
        if plugin is None:
            raise ValueError("HostL1Manager needs plugin_l1_config")
        unknown = set(plugin.params) - {"note"}
        if unknown:
            raise ValueError(f"Unknown HostL1Manager params: {sorted(unknown)}")
        super().__init__(config, L1MemoryManager(config.memory_config))

    @property
    def backend_type(self) -> L1BackendType:
        return L1BackendType.PLUGIN

    def get_memory_tags(self) -> MemoryTags:
        return MemoryTags(segment=MemorySegment.DRAM, residency=MemoryResidency.LOCAL)


class NotAManager:
    """Something a plugin config may name by mistake."""
