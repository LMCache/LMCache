# SPDX-License-Identifier: Apache-2.0
"""NVMe DMA engine: objects held in a GDS slab."""

# Standard
from typing import Sequence, cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_context import (
    GDSContext,
    SlabDirection,
    get_l1_gds_context,
)
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.platform import stream as platform_stream
import lmcache.lmcache_native as lmcache_native


class NvmeDmaEngine:
    """Stages slab-backed objects with GDS reads and writes."""

    def stage(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        direction: "lmcache_native.TransferDirection",
    ) -> None:
        """Run one slab read or write per object and verify every transfer.

        Args:
            memory_objs: Slab-backed objects (``GDSMemoryObject``).
            buffers: One registered staging region per object.
            direction: H2D reads the slab into the regions, D2H writes the
                regions to the slab.

        Raises:
            RuntimeError: A GDS transfer reported an IO error or moved fewer
                bytes than requested.

        Notes:
            Blocks until the current stream drains. The GDS library reports
            transfer failures only after completion, so they must be checked
            before H2D bytes are scattered into the KV cache or a D2H store is
            committed. The callers turn the raise into a failed retrieve or an
            aborted store.
        """
        slab_direction = (
            SlabDirection.READ
            if direction == lmcache_native.TransferDirection.H2D
            else SlabDirection.WRITE
        )
        used: set[tuple[GDSContext, torch.device]] = set()
        for memory_obj, buffer in zip(memory_objs, buffers, strict=True):
            slab_obj = cast(GDSMemoryObject, memory_obj)
            context = get_l1_gds_context(slab_obj.get_l1_manager())
            context.transfer_async(slab_obj, buffer, slab_direction)
            used.add((context, buffer.device))
        for context, device in used:
            context.synchronize_and_check(
                device, platform_stream.current_stream(device)
            )
