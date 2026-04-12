# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Standard
from enum import Enum, IntEnum
from multiprocessing import shared_memory
import ctypes

# Third Party
import torch

# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}
_shm_registry: dict[int, shared_memory.SharedMemory] = {}
_buf_registry: dict[int, ctypes.Array] = {}


class TransferDirection(Enum):
    """Specifies the direction of a memory transfer."""

    H2D = 0
    D2H = 1


class GPUKVFormat(IntEnum):
    """Enumeration of different GPU KV cache memory layouts."""

    # used by: vLLM CROSS_LAYER mode
    NB_NL_TWO_BS_NH_HS = 0

    # used by: vLLM non-MLA flash attention
    NL_X_TWO_NB_BS_NH_HS = 1

    # used by: vLLM non-MLA flash infer
    NL_X_NB_TWO_BS_NH_HS = 2

    # used by: vLLM MLA
    NL_X_NB_BS_HS = 3

    # used by: SGLang MHA (flash attention and flash infer)
    TWO_X_NL_X_NBBS_NH_HS = 4

    # used by: SGLang MLA
    NL_X_NBBS_ONE_HS = 5

    # used by: vLLM non-MLA flash attention (HND layout)
    NL_X_TWO_NB_NH_BS_HS = 6

    # used by: vLLM non-MLA flash infer (HND layout)
    NL_X_NB_TWO_NH_BS_HS = 7


# On XPU (Intel GPU), PyTorch 2.4+ supports pin_memory=True via SYCL USM
# host allocation, enabling fast DMA for XPU<->CPU transfers.
_XPU_PIN_MEMORY = hasattr(torch, "xpu") and torch.xpu.is_available()


def alloc_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory with NUMA awareness.
    On XPU, uses pin_memory=True (SYCL USM host allocation) for fast transfers.
    Note: NUMA node selection is not supported on non-CUDA."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=_XPU_PIN_MEMORY)

    # First-touch initialization (forces physical allocation)
    tensor.fill_(0)

    # Get a pointer to the start of the tensor object as this is what is
    # returned by the CUDA equivalent function
    ptr = tensor.data_ptr()

    # Store the tensor so it can be accessed outide this function scope
    _tensor_registry[ptr] = tensor

    return ptr


def free_pinned_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory and returning pointer
    to it. On XPU, uses pin_memory=True (SYCL USM host allocation) for
    fast DMA transfers. On other non-CUDA platforms, pinning is not supported."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=_XPU_PIN_MEMORY)

    # First-touch initialization (forces physical allocation)
    tensor.fill_(0)

    # Get a pointer to the start of the tensor object as this is what is
    # returned by the CUDA equivalent function
    ptr = tensor.data_ptr()

    # Store the tensor so it can be accessed outide this function scope
    _tensor_registry[ptr] = tensor

    return ptr


def free_pinned_ptr(ptr: int) -> None:
    """Non-CUDA equivalent of freeing a previously allocated pinned pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_shm_pinned_ptr(size: int, shm_name: str = "") -> int:
    """Non-CUDA equivalent of allocating shared memory pinned pointer.
    Uses multiprocessing.shared_memory for cross-platform POSIX shm."""

    # Strip leading '/' for SharedMemory name
    name = shm_name.lstrip("/") if shm_name else None

    # Clean up stale shm segment if it exists
    if name:
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass

    shm = shared_memory.SharedMemory(name=name, create=True, size=size)

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_buffer(shm.buf)
    ptr = ctypes.addressof(buf)

    # Store references to keep them alive
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    _tensor_registry[ptr] = tensor
    _buf_registry[ptr] = buf
    _shm_registry[ptr] = shm
    return ptr


def free_shm_pinned_ptr(ptr: int, size: int = 0, shm_name: str = "") -> None:
    """Non-CUDA equivalent of freeing a shared memory
    pinned pointer."""

    # Release in order: tensor -> ctypes buf -> shm
    _tensor_registry.pop(ptr, None)
    _buf_registry.pop(ptr, None)
    shm = _shm_registry.pop(ptr, None)
    if shm is not None:
        shm.close()
        shm.unlink()


# ------------------------------------------------------------------
# GPU kernel stubs (no-op on CPU-only platforms)
# ------------------------------------------------------------------


class PageBufferShapeDesc:
    """No-op stand-in for the CUDA PageBufferShapeDesc."""

    kv_size: int = 2
    nl: int = 0
    nb: int = 0
    bs: int = 0
    nh: int = 0
    hs: int = 0
    element_size: int = 2


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: torch.Tensor,
    lmcache_objects_ptrs: list[int],
    block_ids: torch.Tensor,
    device: torch.device,  # noqa: ARG001
    direction: TransferDirection,
    shape_desc: "PageBufferShapeDesc",
    lmcache_chunk_size: int,
    gpu_kv_format: "GPUKVFormat",  # noqa: ARG001
    skip_prefix_n_blocks: int = 0,
) -> None:
    """CPU replacement for the CUDA block KV transfer kernel.

    Transfers data between paged KV cache tensors and
    contiguous LMCache memory objects.  Only the
    ``NL_X_TWO_NB_BS_NH_HS`` layout is supported (the
    default for ``CpuCacheContext``).

    LMCache memory layout (contiguous, "2LTD"):
        ``[2, NL, chunk_size, NH * HS]``  (in element units)
    Paged buffer layout per layer (NL_X_TWO_NB_BS_NH_HS):
        ``[2, NB, BS, NH, HS]``
    """
    num_objects = len(lmcache_objects_ptrs)
    total_blocks = block_ids.shape[0]
    blocks_per_obj = total_blocks // num_objects
    nl = shape_desc.nl
    bs = shape_desc.bs
    nh = shape_desc.nh
    hs = shape_desc.hs
    nb = shape_desc.nb
    elem_size = shape_desc.element_size
    dtype = {2: torch.float16, 4: torch.float32}[elem_size]

    # Reconstruct per-layer paged tensors from raw pointers
    layer_ptrs = paged_buffer_ptrs_tensor.tolist()
    paged_tensors: list[torch.Tensor] = []
    paged_nbytes = 2 * nb * bs * nh * hs * elem_size
    for ptr in layer_ptrs:
        buf = (ctypes.c_uint8 * paged_nbytes).from_address(int(ptr))
        t = torch.frombuffer(buf, dtype=dtype).view(2, nb, bs, nh, hs)
        paged_tensors.append(t)

    block_ids_list = block_ids.tolist()
    is_d2h = direction == TransferDirection.D2H

    for obj_idx in range(num_objects):
        obj_ptr = lmcache_objects_ptrs[obj_idx]
        obj_numel = 2 * nl * lmcache_chunk_size * nh * hs
        obj_nbytes = obj_numel * elem_size
        obj_buf = (ctypes.c_uint8 * obj_nbytes).from_address(obj_ptr)
        obj_tensor = torch.frombuffer(obj_buf, dtype=dtype).view(
            2, nl, lmcache_chunk_size, nh * hs
        )

        blk_start = obj_idx * blocks_per_obj
        for local_blk in range(blocks_per_obj):
            flat_blk = blk_start + local_blk
            if flat_blk < skip_prefix_n_blocks:
                continue
            engine_blk = block_ids_list[flat_blk]
            token_off = local_blk * bs

            for layer_idx in range(nl):
                paged = paged_tensors[layer_idx]
                for kv in range(2):
                    # paged: [2, NB, BS, NH, HS]
                    src_p = paged[kv, engine_blk, :, :, :]
                    # obj: [2, NL, chunk_size, NH*HS]
                    t_st = token_off
                    t_ed = token_off + bs
                    if is_d2h:
                        obj_tensor[kv, layer_idx, t_st:t_ed, :] = src_p.reshape(
                            bs, nh * hs
                        )
                    else:
                        paged[kv, engine_blk, :, :, :] = obj_tensor[
                            kv, layer_idx, t_st:t_ed, :
                        ].reshape(bs, nh, hs)


def lmcache_memcpy_async(*args, **kwargs) -> None:  # noqa: ARG001
    """No-op replacement for the CUDA async memcpy."""


def record_event_on_stream(*args, **kwargs) -> None:  # noqa: ARG001
    """No-op replacement for CUDA stream event recording."""
