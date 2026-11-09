# SPDX-License-Identifier: Apache-2.0
"""CPU pinned-DRAM L1 memory manager."""

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    L1_POOL_SHM_NAME_ATTEMPTS,
    L1MemoryManagerConfig,
    generated_l1_pool_shm_name,
    is_generated_l1_pool_shm_name,
    normalize_l1_pool_shm_name,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_allocators.lazy_memory_allocator import (
    LazyMemoryAllocator,
    mlock_on_fault,
)
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
)
from lmcache.v1.multiprocess.posix_shm import (
    is_shm_name_collision,
    shm_segment_is_live,
    shm_unlink,
)

logger = init_logger(__name__)


# HELPER FUNCTIONS
def _unlink_stale_shm(shm_name: str) -> None:
    """Remove a stale LMCache shm segment if it exists.

    A segment whose advisory liveness lock is held by a live process is
    never unlinked: the subsequent exclusive create surfaces the
    collision instead. A stale segment (its holder crashed, so the
    kernel released the lock) is reclaimed. Kernels that reject
    ``flock`` on SHM fds (macOS) cannot express liveness and always
    reclaim the name.
    """
    normalized = shm_name.lstrip("/")
    if "/" in normalized or "\\" in normalized:
        logger.warning("Refusing to unlink invalid shm name %s", shm_name)
        return
    if not normalized.startswith("lmcache_l1_pool_"):
        return
    if shm_segment_is_live(normalized):
        logger.info(
            "Skipping unlink of live shm segment %s; creation must "
            "surface the collision",
            normalized,
        )
        return
    shm_unlink(normalized)


def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.

    Notes:
        When SHM is enabled and the name was auto-generated (matches the
        ``lmcache_l1_pool_<L1_POOL_SHM_NAME_TOKEN_HEX_LEN hex>`` pattern),
        a name collision causes
        ``config.shm_name`` to be updated in-place to the name actually
        used by ``MixedMemoryAllocator`` before returning.
    """
    if config.use_lazy:
        logger.debug(
            "use lazy memory allocator, init size is %d bytes, "
            "final size is %d bytes, align bytes is %d bytes",
            config.init_size_in_bytes,
            config.size_in_bytes,
            config.align_bytes,
        )
        return LazyMemoryAllocator(
            config.init_size_in_bytes,
            config.size_in_bytes,
            config.align_bytes,
            mlock=config.mlock,
        )
    else:
        logger.debug(
            "use mixed memory allocator, total size is %d bytes, "
            "align bytes is %d bytes",
            config.size_in_bytes,
            config.align_bytes,
        )
        if config.shm_name:
            requested = normalize_l1_pool_shm_name(config.shm_name)
            generated = is_generated_l1_pool_shm_name(requested)
            shm_name = requested
            for attempt in range(L1_POOL_SHM_NAME_ATTEMPTS):
                if attempt > 0:
                    shm_name = generated_l1_pool_shm_name()
                if attempt == 0 and not generated:
                    _unlink_stale_shm(shm_name)
                try:
                    allocator = MixedMemoryAllocator(
                        config.size_in_bytes,
                        align_bytes=config.align_bytes,
                        shm_name=shm_name,
                        use_hugepages=config.use_hugepages,
                    )
                    config.shm_name = shm_name
                    return allocator
                except OSError as exc:
                    if not is_shm_name_collision(exc):
                        raise
                    if attempt == L1_POOL_SHM_NAME_ATTEMPTS - 1 or not generated:
                        raise
        return MixedMemoryAllocator(
            config.size_in_bytes,
            align_bytes=config.align_bytes,
            use_hugepages=config.use_hugepages,
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
        self._align_bytes = config.align_bytes
        if config.mlock and isinstance(self._allocator, MixedMemoryAllocator):
            mlock_on_fault(self._allocator.buffer)

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

        if hasattr(self._allocator, "get_memory_usage"):
            return self._allocator.get_memory_usage()

        def get_address_manager(allocator: MemoryAllocatorInterface):
            if isinstance(allocator, MixedMemoryAllocator) and hasattr(
                allocator.pin_allocator, "address_manager"
            ):
                return allocator.pin_allocator.address_manager
            if isinstance(allocator, LazyMemoryAllocator):
                return allocator.get_address_manager()
            raise NotImplementedError(
                "get_memory_usage is not implemented for this allocator type."
            )

        address_manager = get_address_manager(self._allocator)
        free_size = address_manager.get_free_size()
        total_size = address_manager.get_heap_size()
        used_size = total_size - free_size
        return used_size, total_size

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """
        Return an L1MemoryDesc describing the underlying memory buffer.

        Returns:
            Descriptor containing the final L1 arena and its stable-size
            snapshot.

        Raises:
            NotImplementedError: If the allocator type does not support this operation.
        """
        if isinstance(self._allocator, MixedMemoryAllocator):
            buffer = self._allocator.buffer
            # POSIX SHM is file-backed and is not eligible for io_uring
            # fixed-buffer registration, so expose no stable prefix for it.
            stable_registration_size = (
                None if self._allocator.shm_name else self._size_in_bytes
            )
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            buffer = self._allocator.get_underlying_buffer()
            # Lazy expansion pins each new prefix before publishing it through
            # AddressManager.sbrk(), so the current heap size is a stable
            # registration snapshot even though ``size`` remains final capacity.
            stable_registration_size = min(
                self._size_in_bytes,
                self._allocator.get_address_manager().get_heap_size(),
            )
            if stable_registration_size <= 0:
                stable_registration_size = None
        else:
            raise NotImplementedError(
                "get_l1_memory_desc is not implemented for this allocator type."
            )
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
            stable_registration_size=stable_registration_size,
        )

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
