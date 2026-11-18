# SPDX-License-Identifier: Apache-2.0
"""GPU DMA engine: objects held in host memory (DRAM, DevDAX)."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import MemoryObj
import lmcache.lmcache_native as lmcache_native


class GpuDmaEngine:
    """Stages host-memory objects with host/GPU copies."""

    def stage(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        direction: "lmcache_native.TransferDirection",
    ) -> None:
        """Queue one host/GPU copy per object on the current stream.

        Args:
            memory_objs: Host-memory objects.
            buffers: One staging region per object.
            direction: H2D fills the regions, D2H fills the objects.

        Raises:
            ValueError: If an object and its region differ in size.
        """
        is_h2d = direction == lmcache_native.TransferDirection.H2D
        for memory_obj, buffer in zip(memory_objs, buffers, strict=True):
            if is_h2d:
                lmcache_memcpy_async_h2d(memory_obj, buffer)
            else:
                lmcache_memcpy_async_d2h(buffer, memory_obj)
