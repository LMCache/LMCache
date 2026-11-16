# SPDX-License-Identifier: Apache-2.0
"""NVMe DMA engine: objects held in a GDS slab."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_context import SlabDirection, get_l1_gds_context
from lmcache.v1.memory_management import GDSMemoryObject
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native


def stage(
    cache_context: BaseCacheContext,
    memory_objs: Sequence[GDSMemoryObject],
    buffers: Sequence[torch.Tensor],
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Queue one slab read or write per object on the current stream.

    Args:
        cache_context: Cache context owning the staging regions.
        memory_objs: Slab-backed objects.
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
        get_l1_gds_context(memory_obj.get_l1_manager()).transfer_async(
            memory_obj, buffer, slab_direction
        )
