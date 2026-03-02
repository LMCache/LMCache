# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
    PagedCpuMemoryAllocator,
)

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
        assert not config.use_page, (
            "Page mem allocator is not supported with lazy initialization."
        )
        return LazyMemoryAllocator(
            config.init_size_in_bytes, config.size_in_bytes, config.align_bytes
        )
    elif config.use_page:
        logger.warning(
            "PagedCpuMemoryAllocator does not support explicit alignment configuration."
        )
        return PagedCpuMemoryAllocator(config.size_in_bytes)
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
        self._size_in_bytes = config.size_in_bytes
        if isinstance(self._allocator, PagedCpuMemoryAllocator):
            self._memory_initialized = False
        else:
            self._memory_initialized = True

    def lazy_init_memory(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> None:
        """
        Lazily initialize the underlying memory allocator with KV cache
        shape and dtype information. This is only needed for paged allocators
        (PagedCpuMemoryAllocator, PagedCpuMemoryAllocator) whose CPU
        memory cannot be allocated until shapes/dtypes are known.

        This method is idempotent — repeated calls after the first are no-ops.

        Args:
            shapes: KV cache tensor shapes.
            dtypes: KV cache tensor dtypes.
            fmt: Memory format. Default is KV_2LTD.
        """
        if self._memory_initialized:
            return

        if isinstance(self._allocator, PagedCpuMemoryAllocator):
            self._allocator.init_cpu_memory_allocator(
                self._size_in_bytes, shapes, dtypes, fmt
            )
            logger.info(
                "Lazily initialized CPU memory for %s "
                "(size=%d bytes, shapes=%s, dtypes=%s)",
                self._allocator,
                self._size_in_bytes,
                shapes,
                dtypes,
            )

        self._memory_initialized = True

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """
        Allocate memory objects based on the provided layout description and count.
        This function should be thread-safe

        Args:
            layout_desc (MemoryLayoutDesc): Description of the memory layout.
            count (int): Number of memory objects to allocate.

        Returns:
            tuple[L1Error, list[MemoryObj]]: Error code and list of
            allocated memory objects.
            Error code will be `L1Error.OUT_OF_MEMORY` if allocation
            fails; otherwise, it will be `L1Error.SUCCESS`.

        Note:
            If the allocation fails, the memory object list will be empty.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1Error.OUT_OF_MEMORY, []
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """
        Free the provided memory objects.
        This function should be thread-safe.

        Args:
            mem_objs (list[MemoryObj]): List of memory objects to free.

        Returns:
            L1Error: Error code indicating the result of the operation.
            It will be `L1Error.SUCCESS` if the operation succeeds.
        """
        self._allocator.batched_free(mem_objs)
        return L1Error.SUCCESS

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
        elif isinstance(self._allocator, PagedCpuMemoryAllocator):
            return self._allocator.cpu_buffer
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            return self._allocator.get_underlying_buffer()
        else:
            raise NotImplementedError(
                "get_vm_space is not implemented for this allocator type."
            )

    def get_memory_usage(self) -> tuple[int, int]:
        """
        Get the current memory usage. This function will mainly be used to support
        eviction decision.

        Returns:
            tuple[int, int]: A tuple containing used memory in bytes and total memory
            in bytes.

        Note:
            In the future, we may want to make a "callback" based mechanism to
            trigger eviction when the memory usage reaches a watermark.
        """

        if not self._memory_initialized:
            return 0, 0

        # HACK: now trying to read this from the address manager in a ad-hoc
        # manner
        def get_address_manager(allocator: MemoryAllocatorInterface):
            if isinstance(allocator, MixedMemoryAllocator) and hasattr(
                allocator.pin_allocator, "address_manager"
            ):
                return allocator.pin_allocator.address_manager
            elif isinstance(allocator, LazyMemoryAllocator):
                return allocator.get_address_manager()
            elif isinstance(allocator, PagedCpuMemoryAllocator):
                return allocator.allocator.address_manager
            else:
                raise NotImplementedError(
                    "get_memory_usage is not implemented for this allocator type."
                )

        address_manager = get_address_manager(self._allocator)
        free_size = address_manager.get_free_size()
        total_size = address_manager.get_heap_size()
        used_size = total_size - free_size
        return used_size, total_size

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
