# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj


# The LazyMemoryAllocator registers its host buffer with cudaHostRegister /
# hipHostRegister in PIN_CHUNK_SIZE-sized chunks. The HIP runtime will reject
# a single async memcpy that crosses two independently-registered regions, so
# any transfer touching more than one pin chunk must be split at the chunk
# boundaries. The split is implemented in Python (rather than inside the C++
# kernel) so that the arithmetic uses Python's arbitrary-precision integers
# and is safe for offsets past the 2 GB mark.
def _lazy_split_copy(
    host_view: torch.Tensor,
    gpu_view: torch.Tensor,
    host_offset: int,
    h2d: bool,
) -> None:
    pin_size = LazyMemoryAllocator.PIN_CHUNK_SIZE
    mask = pin_size - 1
    nbytes = host_view.nbytes
    offset = 0
    while offset < nbytes:
        aligned_end = ((offset + host_offset) & ~mask) + pin_size
        real_end = min(host_offset + nbytes, aligned_end)
        chunk_nbytes = real_end - offset - host_offset
        host_chunk = host_view[offset : offset + chunk_nbytes]
        gpu_chunk = gpu_view[offset : offset + chunk_nbytes]
        if h2d:
            gpu_chunk.copy_(host_chunk, non_blocking=True)
        else:
            host_chunk.copy_(gpu_chunk, non_blocking=True)
        offset += chunk_nbytes


# Helper functions
def lmcache_memcpy_async_h2d(
    memory_obj: MemoryObj,
    gpu_buffer: torch.Tensor,
):
    """Helper function to copy memory object allocated by different
    allocators to GPU buffer.

    This function is non-blocking and won't do stream synchronization.

    :param MemoryObj memory_obj: The memory object to be copied.
    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data to.
    """
    src_tensor = memory_obj.raw_tensor
    if src_tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    mem_obj_size = memory_obj.get_size()
    if mem_obj_size != gpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
            f"gpu_buffer nbytes={gpu_buffer.nbytes}"
        )
    src_view = src_tensor.view(torch.uint8)[:mem_obj_size]
    gpu_view = gpu_buffer.view(torch.uint8)
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        # _lazy_split_copy slices in bytes, so it needs 1-D byte views.
        _lazy_split_copy(
            src_view.reshape(-1),
            gpu_view.reshape(-1),
            memory_obj.meta.address,
            h2d=True,
        )
    else:
        gpu_view.copy_(src_view, non_blocking=True)


def lmcache_memcpy_async_d2h(
    gpu_buffer: torch.Tensor,
    memory_obj: MemoryObj,
):
    """Helper function to copy memory object allocated by different
    allocators from GPU buffer.

    This function is non-blocking and won't do stream synchronization.

    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data from.
    :param MemoryObj memory_obj: The memory object to be copied to.
    """
    dst_tensor = memory_obj.raw_tensor
    if dst_tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    mem_obj_size = memory_obj.get_size()
    if mem_obj_size != gpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
            f"gpu_buffer nbytes={gpu_buffer.nbytes}"
        )
    dst_view = dst_tensor.view(torch.uint8)[:mem_obj_size]
    gpu_view = gpu_buffer.view(torch.uint8)
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        # _lazy_split_copy slices in bytes, so it needs 1-D byte views.
        _lazy_split_copy(
            dst_view.reshape(-1),
            gpu_view.reshape(-1),
            memory_obj.meta.address,
            h2d=False,
        )
    else:
        dst_view.copy_(gpu_view, non_blocking=True)
