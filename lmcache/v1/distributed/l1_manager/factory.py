# SPDX-License-Identifier: Apache-2.0
"""Build L1 managers from their configurations."""

# Standard
from collections.abc import Callable, Sequence
from typing import cast
import importlib

# First Party
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    PluginL1Config,
    ShadowL1ManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.l1_manager.devdax import DevDaxL1Manager
from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
from lmcache.v1.distributed.l1_manager.gds import GdsL1Manager
from lmcache.v1.distributed.l1_manager.shadow import ShadowL1Manager


def load_plugin_l1_manager_class(
    plugin: PluginL1Config,
) -> Callable[[L1ManagerConfig], BaseL1Manager]:
    """Import the manager class a plugin L1 names.

    Args:
        plugin: The plugin's module path and class name.

    Returns:
        The class, a ``BaseL1Manager`` subclass that is constructed with the
        :class:`L1ManagerConfig` alone and builds its own memory manager.

    Raises:
        ImportError: If the module cannot be imported.
        ValueError: If the module has no such class, or it is not a
            ``BaseL1Manager`` subclass.
    """
    try:
        module = importlib.import_module(plugin.module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import L1 plugin module {plugin.module_path!r}: {e}"
        ) from e
    cls = getattr(module, plugin.class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, BaseL1Manager):
        raise ValueError(
            f"{plugin.module_path}.{plugin.class_name} is not a BaseL1Manager subclass"
        )
    return cast(Callable[[L1ManagerConfig], BaseL1Manager], cls)


def create_l1_manager(config: L1ManagerConfig) -> L1ManagerInterface:
    """Create the L1 manager for the medium ``config`` describes.

    Args:
        config: L1 settings. A :class:`ShadowL1ManagerConfig` selects a
            shadow, ``gds_l1_config`` the GDS slab, ``plugin_l1_config`` an
            externally provided manager class, ``memory_config.devdax_path``
            Device-DAX, otherwise the pinned-DRAM pool is used.

    Returns:
        A new, open L1 manager.

    Raises:
        ValueError: If both the GDS slab and a Device-DAX path are configured,
            no L1 medium backs a shadow's segment, the plugin class is
            invalid, or the medium's memory configuration is invalid.
        ImportError: If a plugin module cannot be imported.
    """
    if isinstance(config, ShadowL1ManagerConfig):
        return ShadowL1Manager(config)
    if config.gds_l1_config is not None:
        if config.memory_config.devdax_path:
            raise ValueError("GDS and Device-DAX require separate L1 managers")
        return GdsL1Manager(config)
    if config.plugin_l1_config is not None:
        return load_plugin_l1_manager_class(config.plugin_l1_config)(config)
    if config.memory_config.devdax_path:
        return DevDaxL1Manager(config)
    return DramL1Manager(config)


def create_l1_managers(configs: Sequence[L1ManagerConfig]) -> list[L1ManagerInterface]:
    """Create one L1 manager per configuration, in placement order.

    Args:
        configs: L1 settings in write-placement order.

    Returns:
        The open managers, in the order of ``configs``.

    Raises:
        ValueError: If a Device-DAX configuration maps a device range an
            earlier manager already owns, or a single configuration is invalid.
            Managers created before the failure are closed again.

    Note:
        Device ownership is checked against the live mappings of the earlier
        managers, so device aliases resolve the way the allocator sees them.
    """
    created: list[L1ManagerInterface] = []
    try:
        for config in configs:
            memory = config.memory_config
            path = memory.devdax_path
            if path and any(
                manager.owns_range(
                    path, memory.devdax_offset_in_bytes, memory.size_in_bytes
                )
                for manager in created
                if isinstance(manager, DevDaxL1Manager)
            ):
                raise ValueError(f"Device-DAX path already owned by an L1: {path}")
            created.append(create_l1_manager(config))
    except Exception:
        for manager in reversed(created):
            manager.close()
        raise
    return created
