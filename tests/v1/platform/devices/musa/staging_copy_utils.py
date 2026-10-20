# SPDX-License-Identifier: Apache-2.0
"""Shared lazy-memory test double for MUSA transfer tests."""

# Standard
from types import SimpleNamespace

# Third Party
import torch

# First Party
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat


class FakeMemoryObj:
    """Minimal lazy-memory object surface consumed by ``gpu_ops``."""

    def __init__(
        self,
        ptr: int,
        size: int,
        host_offset: int,
        tensor: torch.Tensor | None = None,
        fmt: MemoryFormat | None = None,
    ) -> None:
        self.data_ptr = ptr
        self.raw_tensor = tensor if tensor is not None else object()
        self.tensor = tensor
        self.meta = SimpleNamespace(address=host_offset, fmt=fmt)
        self.metadata = self.meta
        self._size = size
        self._parent = LazyMemoryAllocator.__new__(LazyMemoryAllocator)

    def get_size(self) -> int:
        """Return the represented host span in bytes."""
        return self._size

    def parent(self) -> LazyMemoryAllocator:
        """Return a lazy allocator marker for the production branch."""
        return self._parent
