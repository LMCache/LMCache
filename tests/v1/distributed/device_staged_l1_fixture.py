# SPDX-License-Identifier: Apache-2.0
"""A test L1 plugin whose device-staged objects keep a host copy of their bytes.

Loaded by ``--l1-manager '{"type": "PLUGIN", "module_path":
"tests.v1.distributed.device_staged_l1_fixture", "class_name":
"HostCopyL1Manager", ...}'``. It exercises the two plugin hooks without any
vendor medium: the factory's class loading, and the transfer path driving
``DeviceStagedMemoryObj.transfer_through_device``.
"""

# Standard
from typing import Callable, Optional

# Third Party
import torch

# First Party
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.memory_management import (
    AddressManager,
    DeviceStagedMemoryObj,
    MemoryObj,
    MemoryObjMetadata,
)


class HostCopyObject(DeviceStagedMemoryObj):
    """Keeps what a store gathered as a host tensor; a retrieve uploads it."""

    def __init__(self, metadata: MemoryObjMetadata) -> None:
        super().__init__(metadata)
        self.saved: Optional[torch.Tensor] = None
        self.transfers: list[bool] = []

    def transfer_through_device(
        self,
        device: torch.device,
        store: bool,
        move: Callable[[torch.Tensor], None],
    ) -> None:
        if not store and self.saved is None:
            raise ValueError("HostCopyObject holds nothing to retrieve")
        area = torch.empty(self.get_physical_size(), dtype=torch.uint8, device=device)
        if self.saved is not None and not store:
            area.copy_(self.saved)
        move(area)
        if store:
            self.saved = area.cpu()
        self.transfers.append(store)


class BudgetMemoryManager:
    """Budget-only L1 memory manager handing out :class:`HostCopyObject`."""

    def __init__(self, size_in_bytes: int, align_bytes: int) -> None:
        self._address_manager = AddressManager(size_in_bytes, align_bytes)

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        chunk_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        objects: list[MemoryObj] = []
        for _ in range(count):
            try:
                address, allocated = self._address_manager.allocate(chunk_bytes)
            except RuntimeError:
                self.free(objects)
                return L1Error.OUT_OF_MEMORY, []
            objects.append(
                HostCopyObject(
                    MemoryObjMetadata(
                        shape=layout_desc.shapes[0],
                        dtype=layout_desc.dtypes[0],
                        address=address,
                        phy_size=allocated,
                        ref_count=0,
                        shapes=list(layout_desc.shapes),
                        dtypes=list(layout_desc.dtypes),
                    )
                )
            )
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        for mo in mem_objs:
            if isinstance(mo, HostCopyObject):
                mo.saved = None
            self._address_manager.free(mo.metadata.address, mo.get_physical_size())
            mo.invalidate()
        return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        total = self._address_manager.get_heap_size()
        return total - self._address_manager.get_free_size(), total

    def get_l1_memory_desc(self) -> Optional[L1MemoryDesc]:
        return None

    def close(self) -> None:
        return

    def memcheck(self) -> bool:
        return self._address_manager.check_consistency()


class HostCopyL1Manager(BaseL1Manager):
    """The plugin class the factory loads; accepts one param, ``note``."""

    def __init__(self, config: L1ManagerConfig) -> None:
        plugin = config.plugin_l1_config
        if plugin is None:
            raise ValueError("HostCopyL1Manager needs plugin_l1_config")
        unknown = set(plugin.params) - {"note"}
        if unknown:
            raise ValueError(f"Unknown HostCopyL1Manager params: {sorted(unknown)}")
        super().__init__(
            config,
            BudgetMemoryManager(plugin.size_in_bytes, config.memory_config.align_bytes),
        )

    @property
    def backend_type(self) -> L1BackendType:
        return L1BackendType.PLUGIN

    def get_memory_tags(self) -> MemoryTags:
        return MemoryTags(
            segment=MemorySegment.STORAGE, residency=MemoryResidency.LOCAL
        )


class NotAManager:
    """Something a plugin config may name by mistake."""
