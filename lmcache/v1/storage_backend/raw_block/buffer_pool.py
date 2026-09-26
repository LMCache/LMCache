# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING
import ctypes
import threading

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.raw_block.spdk_ffi import SpdkIoEngineFFI

logger = init_logger(__name__)


class HeaderBufferPool:
    """Pool of SPDK DMA-allocated header buffers.

    Used exclusively by the SPDK I/O engine to eliminate per-write
    allocations in ``_encode_header``.

    Attributes:
        buffer_size: Size of each pooled buffer in bytes.
        pool_size: Number of pre-allocated buffers.
    """

    def __init__(
        self,
        buffer_size: int,
        pool_size: int = 8,
        spdk_engine: "SpdkIoEngineFFI | None" = None,
    ) -> None:
        """Initialize the SPDK header buffer pool.

        Args:
            buffer_size: Size of each header buffer in bytes.
            pool_size: Number of pre-allocated DMA buffers.
            spdk_engine: ``SpdkIoEngineFFI`` instance for SPDK DMA allocation.

        Raises:
            ValueError: If ``buffer_size`` is not positive or not aligned to
                4096, or if ``pool_size`` is non-positive.
            RuntimeError: If any SPDK DMA allocation fails.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if buffer_size % 4096 != 0:
            raise ValueError(f"buffer_size must be aligned to 4096, got {buffer_size}")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        if spdk_engine is None:
            raise ValueError("spdk_engine is required for HeaderBufferPool")

        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._spdk_engine: "SpdkIoEngineFFI" = spdk_engine  # type: ignore[assignment]  # noqa: F821
        self._lock = threading.Lock()

        self._available: list[object] = []
        self._in_use: list[object] = []
        self._spdk_ptrs: list[tuple[int, int]] = []

        self._allocate_all()

    def _allocate_all(self) -> None:
        """Pre-allocate all buffers using SPDK DMA memory."""
        align = 4096
        for _ in range(self.pool_size):
            ptr = self._spdk_engine.allocate_spdk_memory(
                self.buffer_size, align, numa_id=-1
            )
            if ptr == 0:
                self._free_all_allocated()
                raise RuntimeError(
                    f"Failed to allocate SPDK DMA buffer "
                    f"({self.buffer_size} bytes, align={align})"
                )
            buf = (ctypes.c_ubyte * self.buffer_size).from_address(ptr)
            self._available.append(buf)
            self._spdk_ptrs.append((ptr, self.buffer_size))

    def _free_all_allocated(self) -> None:
        """Free all SPDK-allocated buffers (used on partial failure)."""
        for ptr, size in self._spdk_ptrs:
            try:
                self._spdk_engine.free_spdk_memory(ptr)
            except Exception:
                pass
        self._spdk_ptrs.clear()
        self._available.clear()
        self._in_use.clear()

    def acquire_many(self, count: int) -> list[object]:
        """Acquire ``count`` pooled buffers, growing the pool on demand.

        Returns ``count`` buffers from the available pool.  When the pool does
        not yet hold enough free buffers, fresh SPDK DMA buffers are allocated
        and added to the pool.

        Args:
            count: Number of buffers to acquire. Must be greater than zero.

        Returns:
            A list of ``count`` ctypes ``Array`` objects wrapping SPDK DMA
            memory, each writable in-place.

        Raises:
            ValueError: If ``count`` is not greater than zero.
            RuntimeError: If an SPDK DMA allocation fails while growing the
                pool to satisfy the request.
        """
        if count <= 0:
            raise ValueError("count must be > 0")
        with self._lock:
            while len(self._available) < count:
                ptr = self._spdk_engine.allocate_spdk_memory(
                    self.buffer_size, 4096, numa_id=-1
                )
                if ptr == 0:
                    raise RuntimeError(
                        f"Failed to allocate SPDK DMA buffer "
                        f"({self.buffer_size} bytes, align=4096) to satisfy "
                        f"acquire_many({count})"
                    )
                buf = (ctypes.c_ubyte * self.buffer_size).from_address(ptr)
                self._available.append(buf)
                self._spdk_ptrs.append((ptr, self.buffer_size))
                self.pool_size += 1
            buffers = [self._available.pop() for _ in range(count)]
            self._in_use.extend(buffers)
            return buffers

    def release_many(self, buffers: list[object]) -> None:
        """Return a list of pooled buffers to the pool.

        Args:
            buffers: The buffers to return. Each must have been acquired from
                this pool (via ``acquire_many``).

        Raises:
            ValueError: If any buffer was not acquired from this pool.
        """
        with self._lock:
            for buf in buffers:
                try:
                    self._in_use.remove(buf)
                    self._available.append(buf)
                except ValueError:
                    raise ValueError("Buffer not from this HeaderBufferPool") from None

    def stats(self) -> dict[str, int]:
        """Return pool utilization statistics.

        Returns:
            Dictionary with ``available``, ``in_use``, and ``total`` counts.
        """
        with self._lock:
            return {
                "available": len(self._available),
                "in_use": len(self._in_use),
                "total": len(self._available) + len(self._in_use),
            }

    def cleanup(self) -> None:
        """Free all SPDK-allocated buffers and clear the pool."""
        for ptr, size in self._spdk_ptrs:
            try:
                self._spdk_engine.free_spdk_memory(ptr)
            except Exception as e:
                logger.warning(
                    "HeaderBufferPool: error freeing SPDK memory ptr=0x%x: %s",
                    ptr,
                    e,
                )
        self._spdk_ptrs.clear()
        with self._lock:
            self._available.clear()
            self._in_use.clear()
