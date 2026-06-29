# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import nullcontext
from typing import List, Optional, Union
import threading

# Third Party
import torch

# First Party
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
import lmcache.v1.memory_management as memory_management


class PinMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        self.size = size
        self.buffer = memory_management._allocate_cpu_memory(size)
        self._unregistered = False

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shapes, dtypes, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()

    def close(self):
        if not self._unregistered:
            if self.buffer.numel() == 0:
                return
            memory_management._free_cpu_memory(self.buffer, self.size)
            self._unregistered = True

    def __str__(self):
        return "PinMemoryAllocator"
