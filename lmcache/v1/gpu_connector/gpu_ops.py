# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.distributed.gds_l1 import GdsMemoryObj, GdsSlabAllocator
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
import lmcache.c_ops as lmc_ops


# Helper functions
def _validate_obj_tensor(
    memory_obj: MemoryObj,
    gpu_buffer: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return the memory object's raw tensor and byte size, validated
    against ``gpu_buffer``.

    :param MemoryObj memory_obj: The memory object; must be allocated.
    :param torch.Tensor gpu_buffer: The GPU buffer it transfers with.
    :return: ``(raw_tensor, size_in_bytes)``.
    :raises ValueError: If ``memory_obj`` is unallocated or its size
        does not match ``gpu_buffer``.
    """
    tensor = memory_obj.raw_tensor
    if tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    size = memory_obj.get_size()
    if size != gpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={size}, "
            f"gpu_buffer nbytes={gpu_buffer.nbytes}"
        )
    return tensor, size


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
    parent = memory_obj.parent()
    if isinstance(parent, GdsSlabAllocator):
        if not isinstance(memory_obj, GdsMemoryObj):
            raise TypeError(
                "GdsSlabAllocator parent requires a GdsMemoryObj, got "
                f"{type(memory_obj).__name__}"
            )
        parent.cufile_read_into(memory_obj, gpu_buffer)
    elif isinstance(parent, LazyMemoryAllocator):
        _, mem_obj_size = _validate_obj_tensor(memory_obj, gpu_buffer)
        lmc_ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            memory_obj.data_ptr,
            mem_obj_size,
            lmc_ops.TransferDirection.H2D,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        src_tensor, mem_obj_size = _validate_obj_tensor(memory_obj, gpu_buffer)
        gpu_buffer.view(torch.uint8).copy_(
            src_tensor.view(torch.uint8)[:mem_obj_size], non_blocking=True
        )


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
    parent = memory_obj.parent()
    if isinstance(parent, GdsSlabAllocator):
        if not isinstance(memory_obj, GdsMemoryObj):
            raise TypeError(
                "GdsSlabAllocator parent requires a GdsMemoryObj, got "
                f"{type(memory_obj).__name__}"
            )
        parent.cufile_write_from(memory_obj, gpu_buffer)
    elif isinstance(parent, LazyMemoryAllocator):
        _, mem_obj_size = _validate_obj_tensor(memory_obj, gpu_buffer)
        lmc_ops.lmcache_memcpy_async(
            memory_obj.data_ptr,
            gpu_buffer.data_ptr(),
            mem_obj_size,
            lmc_ops.TransferDirection.D2H,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        dst_tensor, mem_obj_size = _validate_obj_tensor(memory_obj, gpu_buffer)
        dst_tensor.view(torch.uint8)[:mem_obj_size].copy_(
            gpu_buffer.view(torch.uint8), non_blocking=True
        )
