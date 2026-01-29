# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
    MixedMemoryAllocator,
)
from lmcache.v1.multiprocess.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.distributed.config import L1MemoryManagerConfig
from lmcache.v1.multiprocess.distributed.error import L1MemoryManagerError

logger = init_logger(__name__)


# HELPER FUNCTIONS
def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.
    """
    if config.use_lazy:
        return LazyMemoryAllocator(
            config.init_size_in_bytes, config.size_in_bytes, config.align_bytes
        )
    else:
        logger.warning(
            "MixedMemoryAllocator does not support explicit alignment configuration."
        )
        return MixedMemoryAllocator(
            config.size_in_bytes,
        )


# MAIN CLASS
class L1MemoryManager:
    """
    L1MemoryManager manages the allocation and deallocation of L1 memory.

    Observability metrics to emit:
    1. Memory usage
    2. Active allocations
    """

    def __init__(self, config: L1MemoryManagerConfig):
        self._allocator = create_memory_allocator(config)

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1MemoryManagerError, list[MemoryObj]]:
        """
        Allocate memory objects based on the provided layout description and count.
        This function should be thread-safe

        Args:
            layout_desc (MemoryLayoutDesc): Description of the memory layout.
            count (int): Number of memory objects to allocate.

        Returns:
            tuple[L1MemoryManagerError, list[MemoryObj]]: Error code and list of
            allocated memory objects.
            Error code will be `L1MemoryManagerError.OUT_OF_MEMORY` if allocation
            fails; otherwise, it will be `L1MemoryManagerError.SUCCESS`.

        Note:
            If the allocation fails, the memory object list will be empty.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1MemoryManagerError.OUT_OF_MEMORY, []
        return L1MemoryManagerError.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1MemoryManagerError:
        """
        Free the provided memory objects.
        This function should be thread-safe.

        Args:
            mem_objs (list[MemoryObj]): List of memory objects to free.

        Returns:
            L1MemoryManagerError: Error code indicating the result of the operation.
            It will be `L1MemoryManagerError.SUCCESS` if the operation succeeds.
        """
        self._allocator.batched_free(mem_objs)
        return L1MemoryManagerError.SUCCESS

    def get_vm_space(self) -> torch.Tensor:
        """
        Used by RDMA communication to get the underlying virtual memory space.

        Returns:
            the underlying virtual memory space as a torch.Tensor.

        Raises:
            NotImplementedError: If the allocator type does not support this operation.
        """
        if isinstance(self._allocator, MixedMemoryAllocator):
            return self._allocator.buffer
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            return self._allocator.get_underlying_buffer()
        else:
            raise NotImplementedError(
                "get_vm_space is not implemented for this allocator type."
            )

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()
