# SPDX-License-Identifier: Apache-2.0
"""GDS slab-file L1 manager."""

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
from lmcache.v1.gpu_connector.gds_context import (
    close_l1_gds_context,
    initialize_l1_gds_context,
)

logger = init_logger(__name__)


class GdsL1Manager(BaseL1Manager):
    """L1 manager whose medium is an NVMe slab reached through GPUDirect Storage.

    Objects are slab offsets rather than host buffers, so this L1 exposes no
    registerable memory descriptor. The manager owns the process-wide GDS
    context for its slab: it is initialized here and closed in :meth:`close`.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        """Create the slab address space and the GDS context behind it.

        Args:
            config: L1 settings with ``gds_l1_config`` set.

        Raises:
            ValueError: If ``config.gds_l1_config`` is None, or another GDS
                slab is already active in this process.
            RuntimeError, OSError: If the GDS context cannot be initialized;
                the slab address space is released again first.
        """
        if config.gds_l1_config is None:
            raise ValueError("GdsL1Manager requires gds_l1_config")
        super().__init__(config, GDSL1MemoryManager(config.gds_l1_config))
        # The context is keyed by the owner id drawn in the base constructor,
        # so it can only be created after it; undo the base registration if
        # the context fails, since the factory never sees this instance.
        try:
            initialize_l1_gds_context(self.l1_manager_id, config.gds_l1_config)
        except BaseException:
            super().close()
            raise
        if config.memory_config.use_hugepages:
            logger.warning(
                "l1-use-hugepages is a no-op for the GDS L1 tier; "
                "hugepages are ignored."
            )
        logger.info("GdsL1Manager: GDS L1 tier enabled; CPU pinned-DRAM L1 disabled")

    @property
    def backend_type(self) -> L1BackendType:
        """Return ``L1BackendType.GDS``."""
        return L1BackendType.GDS

    def get_memory_tags(self) -> MemoryTags:
        """Return ``(GDS, LOCAL)``."""
        return MemoryTags(segment=MemorySegment.GDS, residency=MemoryResidency.LOCAL)

    def close(self) -> None:
        """Close the GDS context, then free objects and the slab address space."""
        close_l1_gds_context(self.l1_manager_id)
        super().close()
