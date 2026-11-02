# SPDX-License-Identifier: Apache-2.0
"""Build L1 managers from their configurations."""

# Standard
from collections.abc import Sequence

# First Party
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l1_manager.devdax import DevDaxL1Manager
from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
from lmcache.v1.distributed.l1_manager.gds import GdsL1Manager
from lmcache.v1.distributed.l1_manager.shadow import ShadowL1Manager


def create_l1_manager(config: L1ManagerConfig) -> L1ManagerInterface:
    """Create the L1 manager for the medium ``config`` describes.

    Args:
        config: L1 settings. ``gds_l1_config`` selects the GDS slab,
            ``memory_config.devdax_path`` selects Device-DAX, otherwise the
            pinned-DRAM pool is used.

    Returns:
        A new, open L1 manager.

    Raises:
        ValueError: If both the GDS slab and a Device-DAX path are configured,
            or the medium's memory configuration is invalid.
    """
    if config.gds_l1_config is not None:
        if config.memory_config.devdax_path:
            raise ValueError("GDS and Device-DAX require separate L1 managers")
        return GdsL1Manager(config)
    if config.memory_config.devdax_path:
        return DevDaxL1Manager(config)
    return DramL1Manager(config)


def create_l1_managers(configs: Sequence[L1ManagerConfig]) -> list[L1ManagerInterface]:
    """Create one L1 manager per configuration, in placement order.

    Each Device-DAX L1 that owns a shared CXL slab also gets a
    :class:`ShadowL1Manager`, appended after the configured managers, which
    indexes the views it borrows from peers' slabs.

    Args:
        configs: L1 settings in write-placement order.

    Returns:
        The open managers, in the order of ``configs``, then the shadow L1s.

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
        created.extend(
            ShadowL1Manager.next_to(manager)
            for manager in list(created)
            if isinstance(manager, DevDaxL1Manager) and manager.cxl_arena is not None
        )
    except Exception:
        for manager in reversed(created):
            manager.close()
        raise
    return created
