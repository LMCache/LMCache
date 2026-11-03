# SPDX-License-Identifier: Apache-2.0
"""CPU pinned-DRAM L1 memory manager."""

# Standard
from multiprocessing import shared_memory
from typing import Optional
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
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


# Read at call time so tests can point them at a temporary directory.
_PROC_MEMINFO = "/proc/meminfo"
_CGROUP_ROOT = "/sys/fs/cgroup"

# cgroup v1 reports "no limit" as a page-rounded 2**63 - 1.
_CGROUP_V1_UNLIMITED = 1 << 62


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> Optional[int]:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _stat_field(path: str, field: str) -> int:
    text = _read_text(path) or ""
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name == field:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _host_available_bytes() -> Optional[int]:
    """MemAvailable from /proc/meminfo, in bytes."""
    text = _read_text(_PROC_MEMINFO) or ""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            try:
                return int(fields[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def _cgroup_available_bytes() -> Optional[int]:
    """
    Headroom under this process's cgroup memory limit, in bytes.

    Usage excludes inactive file cache, the same working-set figure the
    kubelet compares against the limit. Returns None when there is no
    limit or it cannot be read.
    """
    v2_limit = _read_text(os.path.join(_CGROUP_ROOT, "memory.max"))
    if v2_limit is not None:
        if v2_limit == "max":
            return None
        current = _read_int(os.path.join(_CGROUP_ROOT, "memory.current"))
        try:
            limit = int(v2_limit)
        except ValueError:
            return None
        if current is None:
            return None
        inactive = _stat_field(
            os.path.join(_CGROUP_ROOT, "memory.stat"), "inactive_file"
        )
        return limit - max(current - inactive, 0)

    v1_dir = os.path.join(_CGROUP_ROOT, "memory")
    limit = _read_int(os.path.join(v1_dir, "memory.limit_in_bytes"))
    if limit is None or limit >= _CGROUP_V1_UNLIMITED:
        return None
    usage = _read_int(os.path.join(v1_dir, "memory.usage_in_bytes"))
    if usage is None:
        return None
    inactive = _stat_field(os.path.join(v1_dir, "memory.stat"), "total_inactive_file")
    return limit - max(usage - inactive, 0)


def _available_memory_bytes() -> Optional[int]:
    """
    Memory the L1 pool can take without an OOM kill, in bytes.

    The smaller of the host's MemAvailable and the cgroup headroom, since a
    container is killed at its cgroup limit even when the node has free
    memory. None when neither can be read, or the reading is not positive.
    """
    readings = [
        r for r in (_host_available_bytes(), _cgroup_available_bytes()) if r is not None
    ]
    if not readings:
        return None
    available = min(readings)
    return available if available > 0 else None


def _clamp_to_available_memory(size_in_bytes: int) -> int:
    """
    Hold the L1 pool to the memory available to this process.

    MP allocates this pool once per cache server, so the configured size is
    its memory footprint directly. A pool larger than the host's available
    memory or the container's cgroup limit gets the server OOM-killed during
    startup.

    Args:
        size_in_bytes: The configured L1 pool size

    Returns:
        The size to actually allocate, in bytes
    """
    available = _available_memory_bytes()
    requested_gb = size_in_bytes / GIB

    if available is None:
        logger.warning(
            "L1 pool: requesting %.2f GB; could not determine available "
            "memory, allocating the configured size unchecked",
            requested_gb,
        )
        return size_in_bytes

    logger.info(
        "L1 pool: requesting %.2f GB of host memory (available: %.2f GB)",
        requested_gb,
        available / GIB,
    )
    if size_in_bytes <= available:
        return size_in_bytes

    logger.warning(
        "L1 pool: reducing %.2f GB to %.2f GB because the configured size "
        "exceeds the memory available to this process (host MemAvailable or "
        "the container's cgroup limit) and would be OOM-killed. Lower the "
        "configured L1 size or raise the memory limit.",
        requested_gb,
        available / GIB,
    )
    return available


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
            init_size_in_bytes,
            size_in_bytes,
            config.align_bytes,
            mlock=config.mlock,
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
