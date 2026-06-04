# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.distributed.gds_l1 import GdsMemoryObj, GdsScratchAllocator
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
import lmcache.c_ops as lmc_ops


# Helper functions
def lmcache_memcpy_async_h2d(
    memory_obj: MemoryObj,
    gpu_buffer: torch.Tensor,
):
    """Helper function to copy memory object allocated by different
    allocators to GPU buffer.

    This function is non-blocking and won't do stream synchronization
    on the CPU-pinned and lazy-pinned paths. For the GDS L1 path it
    invokes a synchronous cuFile read into ``gpu_buffer`` — see
    docs/design/v1/distributed/gds_l1_backend.md for why this is
    still framed as an H2D.

    :param MemoryObj memory_obj: The memory object to be copied.
    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data to.
    """
    parent = memory_obj.parent()
    if isinstance(parent, GdsScratchAllocator):
        if not isinstance(memory_obj, GdsMemoryObj):
            raise TypeError(
                "GdsScratchAllocator parent requires a GdsMemoryObj, got "
                f"{type(memory_obj).__name__}"
            )
        parent.cufile_read_into(memory_obj, gpu_buffer)
        return

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
    if isinstance(parent, LazyMemoryAllocator):
        lmc_ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            memory_obj.data_ptr,
            mem_obj_size,
            lmc_ops.TransferDirection.H2D,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        gpu_buffer.view(torch.uint8).copy_(
            src_tensor.view(torch.uint8)[:mem_obj_size], non_blocking=True
        )


def lmcache_memcpy_async_d2h(
    gpu_buffer: torch.Tensor,
    memory_obj: MemoryObj,
):
    """Helper function to copy memory object allocated by different
    allocators from GPU buffer.

    This function is non-blocking and won't do stream synchronization
    on the CPU-pinned and lazy-pinned paths. For the GDS L1 path it
    invokes a synchronous cuFile write from ``gpu_buffer`` — see
    docs/design/v1/distributed/gds_l1_backend.md for why this is
    still framed as a D2H.

    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data from.
    :param MemoryObj memory_obj: The memory object to be copied to.
    """
    parent = memory_obj.parent()
    if isinstance(parent, GdsScratchAllocator):
        if not isinstance(memory_obj, GdsMemoryObj):
            raise TypeError(
                "GdsScratchAllocator parent requires a GdsMemoryObj, got "
                f"{type(memory_obj).__name__}"
            )
        parent.cufile_write_from(memory_obj, gpu_buffer)
        return

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
    if isinstance(parent, LazyMemoryAllocator):
        lmc_ops.lmcache_memcpy_async(
            memory_obj.data_ptr,
            gpu_buffer.data_ptr(),
            mem_obj_size,
            lmc_ops.TransferDirection.D2H,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        dst_tensor.view(torch.uint8)[:mem_obj_size].copy_(
            gpu_buffer.view(torch.uint8), non_blocking=True
        )
