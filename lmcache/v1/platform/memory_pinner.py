# SPDX-License-Identifier: Apache-2.0
"""Cross-platform memory pinning abstraction.

Provides a ``MemoryPinner`` interface for CUDA host memory
registration.  On CUDA platforms, uses ``cudart.cudaHostRegister``;
on CPU-only platforms, all operations are no-ops.

Usage::

    from lmcache.v1.platform import create_memory_pinner

    pinner = create_memory_pinner()
    pinner.pin(ptr, size)
    pinner.unpin(ptr)
    pinner.close()       # unpin all tracked regions
"""

# Standard
from abc import ABC, abstractmethod

# Third Party
import torch

# First Party
from lmcache.v1.platform.capabilities import HAS_CUDA


class MemoryPinner(ABC):
    """Abstract base class for memory pinning operations."""

    @abstractmethod
    def pin(self, ptr: int, size: int, flags: int = 2) -> None:
        """Pin a memory region for DMA access.

        Args:
            ptr: Host memory pointer.
            size: Size in bytes.
            flags: CUDA host register flags
                (default 2 = cudaHostRegisterMapped).
        """

    @abstractmethod
    def unpin(self, ptr: int) -> None:
        """Unpin a previously pinned memory region."""

    @abstractmethod
    def close(self) -> None:
        """Unpin all tracked regions. Idempotent."""


class CudaMemoryPinner(MemoryPinner):
    """CUDA-based memory pinner using cudart APIs."""

    def __init__(self) -> None:
        self._cudart = torch.cuda.cudart()
        self._pin_record: list[tuple[int, int]] = []

    def pin(self, ptr: int, size: int, flags: int = 2) -> None:
        self._cudart.cudaHostRegister(ptr, size, flags)
        self._pin_record.append((ptr, size))

    def unpin(self, ptr: int) -> None:
        self._cudart.cudaHostUnregister(ptr)
        self._pin_record = [(p, s) for p, s in self._pin_record if p != ptr]

    def close(self) -> None:
        for ptr, _ in self._pin_record:
            self._cudart.cudaHostUnregister(ptr)
        self._pin_record.clear()


class NoOpMemoryPinner(MemoryPinner):
    """No-op pinner for CPU-only platforms."""

    def pin(self, ptr: int, size: int, flags: int = 2) -> None:
        pass

    def unpin(self, ptr: int) -> None:
        pass

    def close(self) -> None:
        pass


def create_memory_pinner() -> MemoryPinner:
    """Create a platform-appropriate MemoryPinner.

    Returns ``CudaMemoryPinner`` when CUDA is available,
    ``NoOpMemoryPinner`` otherwise.
    """
    if HAS_CUDA:
        return CudaMemoryPinner()
    return NoOpMemoryPinner()
