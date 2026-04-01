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
