# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Optional, Union

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.system_detection import NUMAMapping
import lmcache.v1.memory_management as memory_management


class PagedCpuGpuMemoryAllocator(MemoryAllocatorInterface):
    """
    Paged Memory Allocator for both CPU and GPU memory.
    This is a paged memory allocator for PD and P2P sharing
    when NIXL is enabled as NIXL relies on the paging abstraction.
    """

    def __init__(self):
        pass

    def init_gpu_memory_allocator(
        self,
        size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        device: str = torch_device_type,
    ):
        self.gpu_buffer = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
        )
        self.gpu_allocator = PagedTensorMemoryAllocator(
            self.gpu_buffer,
            shapes,
            dtypes,
            fmt,
        )

    def init_cpu_memory_allocator(
        self,
        size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        numa_mapping: Optional[NUMAMapping] = None,
    ):
        self.cpu_buffer = memory_management._allocate_cpu_memory(size, numa_mapping)
        self.cpu_allocator = PagedTensorMemoryAllocator(
            self.cpu_buffer,
            shapes,
            dtypes,
            fmt,
        )
        self.align_bytes = self.cpu_allocator.align_bytes

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "cpu",
    ) -> Optional[MemoryObj]:
        if allocator_type == "gpu":
            return self.gpu_allocator.allocate(shapes, dtypes, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.allocate(shapes, dtypes, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "gpu",
    ) -> Optional[List[MemoryObj]]:
        if allocator_type == "gpu":
            return self.gpu_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = "cpu"):
        if allocator_type == "gpu":
            self.gpu_allocator.free(memory_obj)
        elif allocator_type == "cpu":
            self.cpu_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        if allocator_type == "gpu":
            self.gpu_allocator.batched_free(memory_objs, update_stats=update_stats)
        elif allocator_type == "cpu":
            self.cpu_allocator.batched_free(memory_objs, update_stats=update_stats)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def __str__(self):
        return "PDMemoryAllocator"
