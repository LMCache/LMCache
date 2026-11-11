# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.v1.gpu_connector.gds_context import SlabDirection, get_l1_gds_context
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    DeviceStagedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)
from lmcache.v1.platform import stream as platform_stream
from lmcache.v1.platform.ops_types import StagingCopy
import lmcache.lmcache_native as lmcache_native


# Helper functions
def _copy_through_lent_buffer(
    memory_obj: DeviceStagedMemoryObj,
    gpu_buffer: torch.Tensor,
    store: bool,
) -> None:
    """Copy one chunk between a staging buffer and a device-staged object.

    The object lends a device buffer and persists it after a store, or fills
    it before a retrieve; this copies between that buffer and ``gpu_buffer``
    on the current stream. Unlike the other branches it blocks: the object
    may only persist, or reuse, its buffer once the copy completed.

    Args:
        memory_obj: The device-staged object.
        gpu_buffer: The chunk's staging buffer, filled before a store and
            read after a retrieve.
        store: True to copy ``gpu_buffer`` into the object (D2H direction),
            False to copy the object into ``gpu_buffer`` (H2D direction).

    Raises:
        ValueError: If the sizes differ, or the lent buffer is too small or on
            another device.
    """
    nbytes = gpu_buffer.nbytes
    if memory_obj.get_size() != nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={memory_obj.get_size()}, "
            f"gpu_buffer nbytes={nbytes}"
        )
    staging = gpu_buffer.view(torch.uint8).reshape(-1)
    device = gpu_buffer.device

    def move(area: torch.Tensor) -> None:
        if area.device != device or area.nbytes < nbytes:
            raise ValueError(
                f"Lent buffer must hold {nbytes} bytes on {device}, got "
                f"{area.nbytes} bytes on {area.device}"
            )
        lent = area.view(torch.uint8).reshape(-1)[:nbytes]
        if store:
            lent.copy_(staging, non_blocking=True)
        else:
            staging.copy_(lent, non_blocking=True)
        # A CPU copy has completed when copy_ returns; there is no stream.
        if device.type != "cpu":
            platform_stream.synchronize_stream(
                device, platform_stream.current_stream(device)
            )

    memory_obj.transfer_through_device(device, store, move)


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
    if isinstance(memory_obj, GDSMemoryObject):
        get_l1_gds_context(memory_obj.get_l1_manager()).transfer_async(
            memory_obj, gpu_buffer, SlabDirection.READ
        )
        return
    if isinstance(memory_obj, DeviceStagedMemoryObj):
        _copy_through_lent_buffer(memory_obj, gpu_buffer, store=False)
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
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        device_ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            memory_obj.data_ptr,
            mem_obj_size,
            lmcache_native.TransferDirection.H2D,
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

    This function is non-blocking and won't do stream synchronization.

    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data from.
    :param MemoryObj memory_obj: The memory object to be copied to.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        get_l1_gds_context(memory_obj.get_l1_manager()).transfer_async(
            memory_obj, gpu_buffer, SlabDirection.WRITE
        )
        return
    if isinstance(memory_obj, DeviceStagedMemoryObj):
        _copy_through_lent_buffer(memory_obj, gpu_buffer, store=True)
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
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        device_ops.lmcache_memcpy_async(
            memory_obj.data_ptr,
            gpu_buffer.data_ptr(),
            mem_obj_size,
            lmcache_native.TransferDirection.D2H,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        dst_tensor.view(torch.uint8)[:mem_obj_size].copy_(
            gpu_buffer.view(torch.uint8), non_blocking=True
        )


def build_staging_copies(
    memory_objs: Sequence[MemoryObj],
    gpu_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
) -> list[StagingCopy]:
    """Build native ``StagingCopy`` descriptors for one batch of lazy objects.

    The H2D/D2H direction decides which side is source vs. destination; the host
    side is always the lazy memory object. Callers must ensure every object is
    lazy-allocator-backed.

    Args:
        memory_objs: Lazy-allocator memory objects, one per chunk in the batch.
        gpu_buffers: GPU staging buffers, aligned element-wise with
            ``memory_objs``.
        is_h2d: True for retrieve (CPU->GPU), False for store (GPU->CPU).

    Returns:
        One ``device_ops.StagingCopy`` per object, in input order.

    Raises:
        ValueError: If an object has not been allocated (``raw_tensor`` is None)
            or its size does not match its GPU buffer.
    """
    copies: list[StagingCopy] = []
    for memory_obj, gpu_buffer in zip(memory_objs, gpu_buffers, strict=True):
        if memory_obj.raw_tensor is None:
            raise ValueError(
                "memory_obj.raw_tensor is None; ensure the MemoryObj has been "
                "allocated."
            )
        mem_obj_size = memory_obj.get_size()
        if mem_obj_size != gpu_buffer.nbytes:
            raise ValueError(
                f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
                f"gpu_buffer nbytes={gpu_buffer.nbytes}"
            )
        host_ptr = memory_obj.data_ptr
        gpu_ptr = gpu_buffer.data_ptr()
        host_offset = memory_obj.meta.address
        if is_h2d:
            copies.append(
                device_ops.StagingCopy(gpu_ptr, host_ptr, mem_obj_size, host_offset)
            )
        else:
            copies.append(
                device_ops.StagingCopy(host_ptr, gpu_ptr, mem_obj_size, host_offset)
            )
    return copies
