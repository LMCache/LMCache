# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Third Party
import torch

# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}


def alloc_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory with NUMA awareness.
    Note: NUMA and pinned memory are not supported on non-CUDA."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)

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
    to it. Note: Pinned memory is not supported on non-CUDA."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)

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


def mmap_host_ptr(size: int) -> int:
    """Non-CUDA equivalent of reserving a contiguous host range (mmap).
    Implemented as a plain CPU tensor allocation."""
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)
    ptr = tensor.data_ptr()
    _tensor_registry[ptr] = tensor
    return ptr


def mmap_host_numa_ptr(size: int, node: int = 0) -> int:
    """Non-CUDA equivalent of reserving a NUMA-bound mmap range.
    NUMA binding is not supported here; behaves like mmap_host_ptr()."""
    return mmap_host_ptr(size)


def munmap_host_ptr(ptr: int, size: int) -> None:
    """Non-CUDA equivalent of munmap: release the backing tensor."""
    _tensor_registry.pop(ptr, None)


def cuda_host_register(ptr: int, size: int, flags: int = 0) -> None:
    """Non-CUDA equivalent of cudaHostRegister (no-op)."""
    return


def cuda_host_unregister(ptr: int) -> None:
    """Non-CUDA equivalent of cudaHostUnregister (no-op)."""
    return
