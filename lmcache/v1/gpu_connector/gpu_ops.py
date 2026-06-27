# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_context import SlabDirection, get_gds_context
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
import lmcache.c_ops as lmc_ops

# Whether the loaded native extension exposes the batched async-copy entry
# point. Older c_ops builds only have the per-object ``lmcache_memcpy_async``;
# the batched helpers below fall back to it so the Python side works against
# either build (the GIL win only materializes once the extension is rebuilt).
_HAS_BATCHED_MEMCPY_ASYNC = hasattr(lmc_ops, "lmcache_memcpy_async_batched")


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
    if isinstance(memory_obj, GDSMemoryObject):
        get_gds_context().transfer_async(memory_obj, gpu_buffer, SlabDirection.READ)
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

    This function is non-blocking and won't do stream synchronization.

    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data from.
    :param MemoryObj memory_obj: The memory object to be copied to.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        get_gds_context().transfer_async(memory_obj, gpu_buffer, SlabDirection.WRITE)
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


def _issue_lazy_memcpy_async_batched(
    dests: list[int],
    srcs: list[int],
    sizes: list[int],
    direction: "lmc_ops.TransferDirection",
    host_offsets: list[int],
) -> None:
    """Issue the accumulated lazy-allocator copies on the current CUDA stream.

    When the native extension exposes ``lmcache_memcpy_async_batched`` the whole
    list is issued in one call, releasing the GIL once for the batch instead of
    once per copy. Otherwise it falls back to one ``lmcache_memcpy_async`` per
    copy so the helper works against older extension builds.

    ``dests`` and ``srcs`` are already ordered for ``direction`` (the host side
    is the source for H2D and the destination for D2H); ``host_offsets`` is the
    host buffer offset of each copy regardless of direction.

    Args:
        dests: Destination pointers, one per copy.
        srcs: Source pointers, one per copy.
        sizes: Byte counts, one per copy.
        direction: H2D or D2H, applied to every copy.
        host_offsets: Per-copy host-buffer offset in the lmcache allocator.
    """
    if not dests:
        return
    if _HAS_BATCHED_MEMCPY_ASYNC:
        lmc_ops.lmcache_memcpy_async_batched(
            dests,
            srcs,
            sizes,
            direction,
            host_offsets,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
        return
    for dest, src, size, host_offset in zip(
        dests, srcs, sizes, host_offsets, strict=True
    ):
        lmc_ops.lmcache_memcpy_async(
            dest,
            src,
            size,
            direction,
            host_offset,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )


def lmcache_memcpy_async_h2d_batched(
    memory_objs: Sequence[MemoryObj],
    gpu_buffers: Sequence[torch.Tensor],
) -> None:
    """Stage a batch of memory objects into GPU buffers (H2D) in one call.

    Equivalent to calling :func:`lmcache_memcpy_async_h2d` once per
    ``(memory_obj, gpu_buffer)`` pair, except every copy that uses the
    lazy-allocator native path is issued through a single
    ``lmc_ops.lmcache_memcpy_async_batched`` call. That collapses the per-chunk
    GIL release/re-acquire handoffs into a single handoff for the whole batch,
    removing the dominant source of GIL contention between the per-instance
    transfer worker threads. GDS- and plain-tensor-backed objects still copy one
    at a time, since they do not use the native staging path.

    This function is non-blocking and won't do stream synchronization.

    Args:
        memory_objs: The memory objects to copy from, one per chunk in the
            batch. Must not contain None.
        gpu_buffers: The GPU buffers to copy into, aligned element-wise with
            ``memory_objs``.

    Raises:
        ValueError: If the two sequences differ in length, if a memory object
            has not been allocated (``raw_tensor`` is None), or if a memory
            object's size does not match its GPU buffer.
    """
    if len(memory_objs) != len(gpu_buffers):
        raise ValueError(
            "memory_objs and gpu_buffers must have the same length, got "
            f"{len(memory_objs)} and {len(gpu_buffers)}"
        )

    dests: list[int] = []
    srcs: list[int] = []
    sizes: list[int] = []
    host_offsets: list[int] = []
    for memory_obj, gpu_buffer in zip(memory_objs, gpu_buffers, strict=True):
        if isinstance(memory_obj, GDSMemoryObject) or not isinstance(
            memory_obj.parent(), LazyMemoryAllocator
        ):
            # Non-lazy objects do not use the native staging path.
            lmcache_memcpy_async_h2d(memory_obj, gpu_buffer)
            continue
        src_tensor = memory_obj.raw_tensor
        if src_tensor is None:
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
        dests.append(gpu_buffer.data_ptr())
        srcs.append(memory_obj.data_ptr)
        sizes.append(mem_obj_size)
        host_offsets.append(memory_obj.meta.address)

    _issue_lazy_memcpy_async_batched(
        dests, srcs, sizes, lmc_ops.TransferDirection.H2D, host_offsets
    )


def objects_all_lazy(memory_objs: Sequence[MemoryObj | None]) -> bool:
    """Return True if every non-None object uses the lazy-allocator path.

    The native object-group transfer executor can only stage lazy-allocator
    objects (raw host pointer + alignment window). GDS- and plain-tensor-backed
    objects need their own copy mechanisms, so a group containing any of them
    must fall back to the per-object Python path. None entries are ignored (they
    only occur for D2H, where the batch is skipped anyway).

    Args:
        memory_objs: The memory objects of an object group.

    Returns:
        True if every non-None object is lazy-allocator-backed.
    """
    for memory_obj in memory_objs:
        if memory_obj is None:
            continue
        if isinstance(memory_obj, GDSMemoryObject) or not isinstance(
            memory_obj.parent(), LazyMemoryAllocator
        ):
            return False
    return True


def build_staging_copies(
    memory_objs: Sequence[MemoryObj],
    gpu_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
) -> list["lmc_ops.StagingCopy"]:
    """Build native ``StagingCopy`` descriptors for one batch of lazy objects.

    The H2D/D2H direction decides which side is source vs. destination; the host
    side is always the lazy memory object. Callers must ensure every object is
    lazy-allocator-backed (see :func:`objects_all_lazy`).

    Args:
        memory_objs: Lazy-allocator memory objects, one per chunk in the batch.
        gpu_buffers: GPU staging buffers, aligned element-wise with
            ``memory_objs``.
        is_h2d: True for retrieve (CPU->GPU), False for store (GPU->CPU).

    Returns:
        One ``lmc_ops.StagingCopy`` per object, in input order.

    Raises:
        ValueError: If an object has not been allocated (``raw_tensor`` is None)
            or its size does not match its GPU buffer.
    """
    copies: list["lmc_ops.StagingCopy"] = []
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
                lmc_ops.StagingCopy(gpu_ptr, host_ptr, mem_obj_size, host_offset)
            )
        else:
            copies.append(
                lmc_ops.StagingCopy(host_ptr, gpu_ptr, mem_obj_size, host_offset)
            )
    return copies


def lmcache_memcpy_async_d2h_batched(
    gpu_buffers: Sequence[torch.Tensor],
    memory_objs: Sequence[MemoryObj],
) -> None:
    """Copy a batch of GPU buffers back into memory objects (D2H) in one call.

    The D2H counterpart of :func:`lmcache_memcpy_async_h2d_batched`: copies that
    use the lazy-allocator native path are issued through a single
    ``lmc_ops.lmcache_memcpy_async_batched`` call, collapsing the per-chunk GIL
    handoffs into one. GDS- and plain-tensor-backed objects still copy one at a
    time.

    This function is non-blocking and won't do stream synchronization.

    Args:
        gpu_buffers: The GPU buffers to copy from, one per chunk in the batch.
        memory_objs: The memory objects to copy into, aligned element-wise with
            ``gpu_buffers``. Must not contain None.

    Raises:
        ValueError: If the two sequences differ in length, if a memory object
            has not been allocated (``raw_tensor`` is None), or if a memory
            object's size does not match its GPU buffer.
    """
    if len(memory_objs) != len(gpu_buffers):
        raise ValueError(
            "gpu_buffers and memory_objs must have the same length, got "
            f"{len(gpu_buffers)} and {len(memory_objs)}"
        )

    dests: list[int] = []
    srcs: list[int] = []
    sizes: list[int] = []
    host_offsets: list[int] = []
    for gpu_buffer, memory_obj in zip(gpu_buffers, memory_objs, strict=True):
        if isinstance(memory_obj, GDSMemoryObject) or not isinstance(
            memory_obj.parent(), LazyMemoryAllocator
        ):
            # Non-lazy objects do not use the native staging path.
            lmcache_memcpy_async_d2h(gpu_buffer, memory_obj)
            continue
        dst_tensor = memory_obj.raw_tensor
        if dst_tensor is None:
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
        dests.append(memory_obj.data_ptr)
        srcs.append(gpu_buffer.data_ptr())
        sizes.append(mem_obj_size)
        host_offsets.append(memory_obj.meta.address)

    _issue_lazy_memcpy_async_batched(
        dests, srcs, sizes, lmc_ops.TransferDirection.D2H, host_offsets
    )
