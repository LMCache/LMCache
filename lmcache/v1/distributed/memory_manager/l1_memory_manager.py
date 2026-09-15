# SPDX-License-Identifier: Apache-2.0
"""CPU pinned-DRAM L1 memory manager."""

# Standard
from multiprocessing import shared_memory

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.system_detection import SystemMemoryDetector
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
)

logger = init_logger(__name__)


# HELPER FUNCTIONS
def _unlink_stale_shm(shm_name: str) -> None:
    """Remove a stale LMCache shm segment if it exists."""
    normalized = shm_name.lstrip("/")
    if "/" in normalized or "\\" in normalized:
        logger.warning("Refusing to unlink invalid shm name %s", shm_name)
        return
    if not normalized.startswith("lmcache_l1_pool_"):
        return
    try:
        shm = shared_memory.SharedMemory(name=normalized, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "Failed to remove stale shm segment %s", normalized, exc_info=True
        )


GIB = 1024**3


def _clamp_to_available_memory(size_in_bytes: int) -> int:
    """
    Hold the L1 pool to what the host can actually give.

    Unlike the per-rank local CPU backend, MP allocates this pool once per
    cache server, so the configured size is the host footprint directly. It
    was still unguarded: asking for more than the node has got the server
    killed by the OOM killer during startup, with nothing in the log saying
    how big the request had been.

    Args:
        size_in_bytes: The configured L1 pool size

    Returns:
        The size to actually allocate, in bytes
    """
    available_gb = SystemMemoryDetector.get_available_memory_gb()
    requested_gb = size_in_bytes / GIB

    if available_gb <= 0:
        logger.warning(
            "L1 pool: requesting %.2f GB; could not determine system available "
            "memory, allocating the configured size unchecked",
            requested_gb,
        )
        return size_in_bytes

    logger.info(
        "L1 pool: requesting %.2f GB of host memory (system available: %.2f GB)",
        requested_gb,
        available_gb,
    )
    if requested_gb <= available_gb:
        return size_in_bytes

    clamped = int(available_gb * GIB)
    logger.warning(
        "L1 pool: reducing %.2f GB to %.2f GB — the configured size exceeds "
        "system available memory and would be killed by the OOM killer. "
        "Lower the configured L1 size, or move to a host with more memory.",
        requested_gb,
        clamped / GIB,
    )
    return clamped


def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.
    """
    # Write the effective size back onto the config, the way __post_init__
    # already normalizes init_size. L1MemoryManager reports config.size_in_bytes
    # in L1MemoryDesc, which consumers use to address the shared pool, so a
    # clamp that did not reach it would hand out a length past the end of the
    # mapping.
    config.size_in_bytes = _clamp_to_available_memory(config.size_in_bytes)
    size_in_bytes = config.size_in_bytes
    # Lazy allocation only reserves init_size up front, but it grows to the
    # full size later, so the cap has to apply to both ends.
    config.init_size_in_bytes = min(config.init_size_in_bytes, size_in_bytes)
    init_size_in_bytes = config.init_size_in_bytes

    if config.use_lazy:
        logger.debug(
            "use lazy memory allocator, init size is %d bytes, "
            "final size is %d bytes, align bytes is %d bytes",
            init_size_in_bytes,
            size_in_bytes,
            config.align_bytes,
        )
        return LazyMemoryAllocator(
            init_size_in_bytes, size_in_bytes, config.align_bytes
        )
    else:
        logger.debug(
            "use mixed memory allocator, total size is %d bytes, "
            "align bytes is %d bytes",
            size_in_bytes,
            config.align_bytes,
        )
        shm_name = config.shm_name
        if shm_name:
            # Keep the lmcache_l1_pool_ prefix in normalized SHM names so
            # stale-segment cleanup can recognize and unlink user-provided names.
            bare = shm_name.lstrip("/")
            if not bare.startswith("lmcache_l1_pool_"):
                shm_name = f"lmcache_l1_pool_{bare}"
            _unlink_stale_shm(shm_name)
            return MixedMemoryAllocator(
                size_in_bytes,
                align_bytes=config.align_bytes,
                shm_name=shm_name,
                use_hugepages=config.use_hugepages,
            )
        return MixedMemoryAllocator(
            size_in_bytes,
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
        # Read after the call: create_memory_allocator may have clamped it.
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

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

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return the storage medium backing ``memory_obj``.

        Args:
            memory_obj: An object allocated by this manager.

        Returns:
            ``L1BackendType.DRAM`` — the CPU tier is pinned DRAM only.
        """
        return L1BackendType.DRAM

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
