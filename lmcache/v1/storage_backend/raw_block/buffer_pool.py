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
    allocations in ``_encode_header``.  Each buffer is allocated via
    ``spdk_dma_zmalloc`` and wrapped with ``ctypes.cast`` so that
    header data can be written directly into DMA memory without any
    intermediate copy.

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
            ValueError: If ``buffer_size`` or ``pool_size`` is non-positive.
            RuntimeError: If any SPDK DMA allocation fails.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        if spdk_engine is None:
            raise ValueError("spdk_engine is required for HeaderBufferPool")

        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._spdk_engine: "SpdkIoEngineFFI" = spdk_engine  # type: ignore[assignment]  # noqa: F821
        self._lock = threading.Lock()

        # ctypes char arrays wrapping SPDK DMA pointers
        self._available: list[object] = []
        # Buffers currently in use
        self._in_use: list[object] = []
        # (ptr, size) pairs for SPDK DMA memory cleanup
        self._spdk_ptrs: list[tuple[int, int]] = []

        self._allocate_all()

    def _allocate_all(self) -> None:
        """Pre-allocate all buffers using SPDK DMA memory."""
        align = 4096  # Page-aligned for DMA
        for _ in range(self.pool_size):
            ptr = self._spdk_engine.allocate_spdk_memory(
                self.buffer_size, align, numa_id=-1
            )
            if ptr == 0:
                # Clean up any already-allocated buffers on failure
                self._free_all_allocated()
                raise RuntimeError(
                    f"Failed to allocate SPDK DMA buffer "
                    f"({self.buffer_size} bytes, align={align})"
                )
            # This creates a view over the existing DMA memory with no copy
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

    def acquire(self) -> object:
        """Acquire a pooled buffer for header encoding.

        Returns:
            A ctypes ``Array`` wrapping the SPDK DMA memory, writable in-place.
        """
        with self._lock:
            buf = self._available.pop()
            self._in_use.append(buf)
            return buf

    def release(self, buf: object) -> None:
        """Return a pooled buffer to the pool.

        Args:
            buf: The buffer to return. Must have been acquired from this pool.

        Raises:
            ValueError: If ``buf`` was not acquired from this pool.
        """
        with self._lock:
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
                "total": self.pool_size,
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


class CheckPointPayloadBufferPool:
    """Pool of SPDK DMA-allocated payload buffers for checkpoint writes.

    Used exclusively by the SPDK I/O engine in ``_write_checkpoint`` to
    eliminate per-checkpoint allocations and the implicit SPDK bounce-buffer
    copy.  Each buffer is allocated via ``spdk_dma_zmalloc`` and wrapped with
    ``ctypes.cast`` so that checkpoint payload data can be written directly
    into DMA memory without any intermediate copy.

    Unlike ``HeaderBufferPool`` which handles fixed-size headers, this pool
    pre-allocates buffers at the maximum payload capacity so that variable-
    size checkpoint payloads can be copied directly into DMA memory and then
    passed to SPDK for true zero-copy NVMe writes.

    Attributes:
        buffer_size: Size of each pooled buffer in bytes (maximum payload).
        pool_size: Number of pre-allocated buffers.
    """

    def __init__(
        self,
        buffer_size: int,
        pool_size: int = 2,
        spdk_engine: "SpdkIoEngineFFI | None" = None,
    ) -> None:
        """Initialize the SPDK checkpoint payload buffer pool.

        Args:
            buffer_size: Size of each payload buffer in bytes (max payload).
            pool_size: Number of pre-allocated DMA buffers.
            spdk_engine: ``SpdkIoEngineFFI`` instance for SPDK DMA allocation.

        Raises:
            ValueError: If ``buffer_size`` or ``pool_size`` is non-positive.
            RuntimeError: If any SPDK DMA allocation fails.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        if spdk_engine is None:
            raise ValueError("spdk_engine is required for CheckPointPayloadBufferPool")

        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._spdk_engine: "SpdkIoEngineFFI" = spdk_engine  # type: ignore[assignment]  # noqa: F821
        self._lock = threading.Lock()

        # ctypes char arrays wrapping SPDK DMA pointers
        self._available: list[object] = []
        # Buffers currently in use
        self._in_use: list[object] = []
        # (ptr, size) pairs for SPDK DMA memory cleanup
        self._spdk_ptrs: list[tuple[int, int]] = []

        self._allocate_all()

    def _allocate_all(self) -> None:
        """Pre-allocate all buffers using SPDK DMA memory."""
        align = 4096  # Page-aligned for DMA
        for _ in range(self.pool_size):
            ptr = self._spdk_engine.allocate_spdk_memory(
                self.buffer_size, align, numa_id=-1
            )
            if ptr == 0:
                # Clean up any already-allocated buffers on failure
                self._free_all_allocated()
                raise RuntimeError(
                    f"Failed to allocate SPDK DMA buffer "
                    f"({self.buffer_size} bytes, align={align})"
                )
            # This creates a view over the existing DMA memory with no copy
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

    def acquire(self) -> object:
        """Acquire a pooled buffer for checkpoint payload.

        Returns:
            A ctypes ``Array`` wrapping the SPDK DMA memory, writable in-place.
            The buffer has ``buffer_size`` bytes available.
        """
        with self._lock:
            buf = self._available.pop()
            self._in_use.append(buf)
            return buf

    def release(self, buf: object) -> None:
        """Return a pooled buffer to the pool.

        Args:
            buf: The buffer to return. Must have been acquired from this pool.

        Raises:
            ValueError: If ``buf`` was not acquired from this pool.
        """
        with self._lock:
            try:
                self._in_use.remove(buf)
                self._available.append(buf)
            except ValueError:
                raise ValueError(
                    "Buffer not from this CheckPointPayloadBufferPool"
                ) from None

    def stats(self) -> dict[str, int]:
        """Return pool utilization statistics.

        Returns:
            Dictionary with ``available``, ``in_use``, and ``total`` counts.
        """
        with self._lock:
            return {
                "available": len(self._available),
                "in_use": len(self._in_use),
                "total": self.pool_size,
            }

    def cleanup(self) -> None:
        """Free all SPDK-allocated buffers and clear the pool."""
        for ptr, size in self._spdk_ptrs:
            try:
                self._spdk_engine.free_spdk_memory(ptr)
            except Exception as e:
                logger.warning(
                    "CheckPointPayloadBufferPool: error freeing SPDK memory "
                    "ptr=0x%x: %s",
                    ptr,
                    e,
                )
        self._spdk_ptrs.clear()
        with self._lock:
            self._available.clear()
            self._in_use.clear()
