# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Third Party
import torch


def alloc_pinned_numa_ptr(size: int) -> torch.Tensor:
    # Create a 1D uint8 CPU tensor in pinned memory, as uint8 == 1 byte
    # Requires pin_memory=False for non-CUDA
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)

    # First-touch initialization (forces physical allocation)
    tensor.fill_(0.0)

    return tensor


def alloc_pinned_ptr(size: int) -> torch.Tensor:
    # Create a 1D uint8 tensor in pinned memory, as uint8 == 1 byte
    # Requires pin_memory=False for non-CUDA
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)

    # First-touch initialization (forces physical allocation)
    tensor.fill_(0.0)

    return tensor
