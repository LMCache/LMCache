# SPDX-License-Identifier: Apache-2.0
"""NVMe DMA engine: objects held in a GDS slab."""

# Standard
from typing import Sequence, cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_context import SlabDirection, get_l1_gds_context
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
import lmcache.lmcache_native as lmcache_native


class NvmeDmaEngine:
    """Stages slab-backed objects with GDS reads and writes."""

    def stage(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        direction: "lmcache_native.TransferDirection",
    ) -> None:
        """Queue one slab read or write per object on the current stream.

        Args:
            memory_objs: Slab-backed objects (``GDSMemoryObject``).
            buffers: One registered staging region per object.
            direction: H2D reads the slab into the regions, D2H writes the
                regions to the slab.
        """
        slab_direction = (
            SlabDirection.READ
            if direction == lmcache_native.TransferDirection.H2D
            else SlabDirection.WRITE
        )
        for memory_obj, buffer in zip(memory_objs, buffers, strict=True):
            slab_obj = cast(GDSMemoryObject, memory_obj)
            get_l1_gds_context(slab_obj.get_l1_manager()).transfer_async(
                slab_obj, buffer, slab_direction
            )
