# SPDX-License-Identifier: Apache-2.0
"""XPU copy helpers for LMCache-driven transfer."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
import lmcache.c_ops as lmc_ops


def _pointer_to_i64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def build_staging_copies(
    memory_objs: Sequence[MemoryObj],
    gpu_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
) -> list["lmc_ops.StagingCopy"]:
    """Build native staging descriptors.

    Raises:
        RuntimeError: XPU transfer currently uses dpctl tensor interop instead of
            the native CUDA staging-copy executor.
    """
    del memory_objs, gpu_buffers, is_h2d
    raise RuntimeError("Native staging-copy execution is not supported for XPU")


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs: torch.Tensor | Sequence[int],
    lmcache_objects_ptrs: list[int],
    block_ids: torch.Tensor,
    device: torch.device,
    direction: "lmc_ops.TransferDirection",
    shape_desc: "lmc_ops.PageBufferShapeDesc",
    lmcache_chunk_size: int,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    skip_prefix_n_blocks: int,
) -> None:
    """Launch the XPU multi-layer block KV transfer kernel."""
    if isinstance(paged_buffer_ptrs, torch.Tensor):
        paged_buffer_ptrs_tensor = paged_buffer_ptrs
    else:
        paged_buffer_ptrs_tensor = torch.tensor(
            [_pointer_to_i64(ptr) for ptr in paged_buffer_ptrs],
            dtype=torch.long,
            device=device,
        )
    lmc_ops.multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        [_pointer_to_i64(ptr) for ptr in lmcache_objects_ptrs],
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


def lmcache_memcpy_async_h2d(
    memory_obj: MemoryObj,
    xpu_buffer: torch.Tensor,
) -> None:
    """Copy a CPU memory object into an XPU tensor.

    Args:
        memory_obj: CPU memory object containing source bytes.
        xpu_buffer: XPU destination tensor.

    Raises:
        ValueError: If source memory is unavailable or sizes/devices mismatch.
        RuntimeError: If the memory object type is unsupported for XPU transfer.
    """
    src_tensor = memory_obj.raw_tensor
    if src_tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    if src_tensor.device.type != "cpu":
        raise ValueError(f"Expected CPU memory object tensor, got {src_tensor.device}")
    mem_obj_size = memory_obj.get_size()
    if mem_obj_size != xpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
            f"xpu_buffer nbytes={xpu_buffer.nbytes}"
        )
    if isinstance(memory_obj, GDSMemoryObject):
        raise RuntimeError("GDS memory objects are not supported for XPU transfer")
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        raise RuntimeError(
            "LazyMemoryAllocator XPU transfers require addressable memory"
        )

    src = src_tensor.view(torch.uint8)[:mem_obj_size].to(
        device=xpu_buffer.device,
        non_blocking=False,
    )
    xpu_buffer.view(torch.uint8).copy_(src, non_blocking=False)


def lmcache_memcpy_async_d2h(
    xpu_buffer: torch.Tensor,
    memory_obj: MemoryObj,
) -> None:
    """Copy an XPU tensor into a CPU memory object.

    Args:
        xpu_buffer: XPU source tensor.
        memory_obj: CPU memory object receiving destination bytes.

    Raises:
        ValueError: If destination memory is unavailable or sizes/devices mismatch.
        RuntimeError: If the memory object type is unsupported for XPU transfer.
    """
    dst_tensor = memory_obj.raw_tensor
    if dst_tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    if dst_tensor.device.type != "cpu":
        raise ValueError(f"Expected CPU memory object tensor, got {dst_tensor.device}")
    mem_obj_size = memory_obj.get_size()
    if mem_obj_size != xpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
            f"xpu_buffer nbytes={xpu_buffer.nbytes}"
        )
    if isinstance(memory_obj, GDSMemoryObject):
        raise RuntimeError("GDS memory objects are not supported for XPU transfer")
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        raise RuntimeError(
            "LazyMemoryAllocator XPU transfers require addressable memory"
        )

    src = xpu_buffer.view(torch.uint8).cpu()
    dst_tensor.view(torch.uint8)[:mem_obj_size].copy_(src, non_blocking=False)
