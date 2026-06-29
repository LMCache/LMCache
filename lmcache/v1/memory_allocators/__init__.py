# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_allocators.buffer_allocator import BufferAllocator
from lmcache.v1.memory_allocators.cu_file_memory_allocator import CuFileMemoryAllocator
from lmcache.v1.memory_allocators.devdax_memory_allocator import DevDaxMemoryAllocator
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.hip_file_memory_allocator import (
    HipFileMemoryAllocator,
)
from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.paged_cpu_gpu_memory_allocator import (
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator

__all__ = [
    "AdHocMemoryAllocator",
    "BufferAllocator",
    "CuFileMemoryAllocator",
    "DevDaxMemoryAllocator",
    "GPUMemoryAllocator",
    "HipFileMemoryAllocator",
    "HostMemoryAllocator",
    "MixedMemoryAllocator",
    "PagedCpuGpuMemoryAllocator",
    "PagedTensorMemoryAllocator",
    "PinMemoryAllocator",
    "TensorMemoryAllocator",
]
