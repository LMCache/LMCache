# SPDX-License-Identifier: Apache-2.0

"""SPDK I/O engine for raw-block storage.

This module implements :class:`SpdkBlockIO`, a :class:`RawBlockIO`
subclass backed by the SPDK C++ library (via :class:`SpdkIoEngineFFI`). It
enables zero-copy NVMe writes through DMA-allocated buffers and a pooled
header buffer scheme.

The engine owns the :class:`SpdkIoEngineFFI` handle and a
:class:`HeaderBufferPool`. It exposes the :class:`RawBlockIO` surface
(``write_buffers``/``read_buffers``/``acquire_buffer``/``release_buffer``/...)
with zero-copy NVMe writes through DMA-allocated buffers and a pooled header
buffer scheme. It also exposes the SPDK-specific ``read_batch_keys`` primitive
used by :class:`RawBlockCore` for its batched slot-header recovery path.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional
import ctypes
import os
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
        SpdkBlockIOConfig,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    PlacementId,
    RawBlockIO,
)
from lmcache.v1.distributed.l2_adapters.raw_block.spdk_ffi import (
    IO_READ,
    IO_WRITE,
    SpdkIoEngineFFI,
)

logger = init_logger(__name__)

# SPDK requires DMA buffers aligned to a 4 KiB boundary.
_SPDK_DMA_ALIGNMENT = 4096

# Default read-batch size used when the controller's max I/O depth is unknown.
_DEFAULT_SPDK_READ_BATCH_KEYS = 256

# Hard cap on the number of keys submitted per SPDK read batch.
_MAX_SPDK_READ_BATCH_KEYS_CAP = 1024

# Default concurrency factor: each SPDK submit targets this fraction of the
# controller's I/O queue depth so it stays below the submission-queue capacity.
_SPDK_IO_CONCURRENCY_FACTOR_DEFAULT = 2


def _resolve_concurrency_factor() -> int:
    """Return the SPDK submit concurrency factor from the environment.

    Returns:
        The concurrency factor, always positive.
    """
    raw = os.environ.get("LMCACHE_SPDK_IO_CONCURRENCY_FACTOR")
    if raw is None:
        return _SPDK_IO_CONCURRENCY_FACTOR_DEFAULT
    try:
        factor = int(raw)
    except ValueError:
        logger.warning(
            "SpdkBlockIO: ignoring invalid LMCACHE_SPDK_IO_CONCURRENCY_FACTOR=%r",
            raw,
        )
        return _SPDK_IO_CONCURRENCY_FACTOR_DEFAULT
    return factor if factor > 0 else _SPDK_IO_CONCURRENCY_FACTOR_DEFAULT


_SPDK_IO_CONCURRENCY_FACTOR = _resolve_concurrency_factor()


def spdk_max_chunk_bytes(
    queue_depth: int,
    max_transfer: int,
    concurrency_factor: int = _SPDK_IO_CONCURRENCY_FACTOR,
) -> int:
    """Compute the largest payload a single SPDK submit may carry in bytes.

    Args:
        queue_depth: I/O submission queue depth of the connected controller.
        max_transfer: Namespace max transfer size (MDTS) in bytes.
        concurrency_factor: Divisor applied to ``queue_depth`` to size each
            call; must be positive. Defaults to the module concurrency factor.

    Returns:
        The maximum bytes to submit in one call, or ``0`` when any argument is
        non-positive (callers must then skip chunking and rely on the device
        reservation gate).
    """
    if queue_depth <= 0 or max_transfer <= 0 or concurrency_factor <= 0:
        return 0
    commands_per_call = queue_depth // concurrency_factor
    if commands_per_call == 0:
        commands_per_call = 1
    return commands_per_call * max_transfer


def _chunk_io_inputs(
    offsets: Sequence[int],
    total_lens: Sequence[int],
    buf_ptrs: Sequence[int],
    max_chunk_bytes: int,
    alignment: int = _SPDK_DMA_ALIGNMENT,
) -> tuple[list[int], list[int], list[int]]:
    """Split a batch so no single transfer exceeds ``max_chunk_bytes`` bytes.

    Args:
        offsets: Device byte offsets for each operation.
        total_lens: Physical transfer sizes for each operation.
        buf_ptrs: Base buffer pointer for each operation.
        max_chunk_bytes: Upper bound on a single transfer's byte count. Values
            ``<= 0`` disable splitting (inputs are returned unchanged).
        alignment: DMA alignment used to round each chunk down; must be positive.

    Returns:
        A tuple of expanded ``(offsets, total_lens, buf_ptrs)`` lists whose
        combined byte count equals the input's and whose every transfer is
        ``<= max_chunk_bytes``.
    """
    if max_chunk_bytes <= 0:
        return list(offsets), list(total_lens), list(buf_ptrs)
    out_offsets: list[int] = []
    out_lens: list[int] = []
    out_ptrs: list[int] = []
    for offset, buf_ptr, total_len in zip(offsets, buf_ptrs, total_lens, strict=True):
        pos = 0
        total_len = int(total_len)
        start_offset = int(offset)
        base_ptr = int(buf_ptr)
        while pos < total_len:
            room = total_len - pos
            chunk = min(max_chunk_bytes, room)
            chunk -= chunk % alignment
            if chunk <= 0:
                # ``max_chunk_bytes`` is below ``alignment``: keep the bound over
                # the alignment so the reservation gate still bounds children.
                chunk = min(max_chunk_bytes, room)
            out_offsets.append(start_offset + pos)
            out_ptrs.append(base_ptr + pos)
            out_lens.append(chunk)
            pos += chunk
    return out_offsets, out_lens, out_ptrs


def _round_up(x: int, align: int) -> int:
    """Round ``x`` up to the nearest multiple of ``align``.

    Args:
        x: Value to align.
        align: Positive alignment in bytes.

    Returns:
        ``x`` rounded up to a multiple of ``align``.
    """
    return ((x + align - 1) // align) * align


class HeaderBufferPool:
    """Pool of SPDK DMA-allocated header buffers.

    Used exclusively by the SPDK I/O engine to eliminate per-write
    allocations in header encoding.

    Attributes:
        header_size: Size of each pooled header buffer in bytes.
        pool_size: Number of pre-allocated buffers.
    """

    def __init__(
        self,
        header_size: int,
        pool_size: int = 8,
        spdk_engine: "SpdkIoEngineFFI | None" = None,
    ) -> None:
        """Initialize the SPDK header buffer pool.

        Args:
            header_size: Size of each header buffer in bytes.
            pool_size: Number of pre-allocated DMA buffers.
            spdk_engine: ``SpdkIoEngineFFI`` instance for SPDK DMA allocation.

        Raises:
            ValueError: If ``header_size`` is not positive or not aligned to
                4096, or if ``pool_size`` is non-positive.
            RuntimeError: If any SPDK DMA allocation fails.
        """
        if header_size <= 0:
            raise ValueError("header_size must be > 0")
        if header_size % _SPDK_DMA_ALIGNMENT != 0:
            raise ValueError(
                f"header_size must be aligned to {_SPDK_DMA_ALIGNMENT}, "
                f"got {header_size}"
            )
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        if spdk_engine is None:
            raise ValueError("spdk_engine is required for HeaderBufferPool")

        self.header_size = header_size
        self.pool_size = pool_size
        self._spdk_engine: "SpdkIoEngineFFI" = spdk_engine  # type: ignore[assignment]
        self._lock = threading.Lock()

        self._available: list[object] = []
        self._in_use: list[object] = []
        self._spdk_ptrs: list[int] = []

        self._allocate_all()

    def _allocate_all(self) -> None:
        """Pre-allocate all buffers using SPDK DMA memory."""
        align = _SPDK_DMA_ALIGNMENT
        for _ in range(self.pool_size):
            ptr = self._spdk_engine.allocate_spdk_memory(
                self.header_size, align, numa_id=-1
            )
            if ptr == 0:
                self._free_all_allocated()
                raise RuntimeError(
                    f"Failed to allocate SPDK DMA buffer "
                    f"({self.header_size} bytes, align={align})"
                )
            buf = (ctypes.c_ubyte * self.header_size).from_address(ptr)
            self._available.append(buf)
            self._spdk_ptrs.append(ptr)

    def _free_all_allocated(self) -> None:
        """Free all SPDK-allocated buffers (used on partial failure and cleanup)."""
        for ptr in self._spdk_ptrs:
            try:
                self._spdk_engine.free_spdk_memory(ptr)
            except Exception:
                pass
        self._spdk_ptrs.clear()
        self._available.clear()
        self._in_use.clear()

    def acquire(self) -> object:
        """Acquire one pooled buffer, allocating a fresh one when none are free.

        Returns:
            A ctypes ``Array`` object wrapping an SPDK DMA buffer, writable
            in-place.

        Raises:
            RuntimeError: If an SPDK DMA allocation fails while growing the
                pool to satisfy the request.
        """
        with self._lock:
            if not self._available:
                ptr = self._spdk_engine.allocate_spdk_memory(
                    self.header_size, _SPDK_DMA_ALIGNMENT, numa_id=-1
                )
                if ptr == 0:
                    raise RuntimeError(
                        f"Failed to allocate SPDK DMA buffer "
                        f"({self.header_size} bytes, "
                        f"align={_SPDK_DMA_ALIGNMENT})"
                    )
                buf = (ctypes.c_ubyte * self.header_size).from_address(ptr)
                self._available.append(buf)
                self._spdk_ptrs.append(ptr)
                self.pool_size += 1
            acquired = self._available.pop()
            self._in_use.append(acquired)
            return acquired

    def release(self, buf: object) -> None:
        """Return a pooled buffer to the pool.

        Args:
            buf: The buffer to return. It must have been acquired from this
                pool (via :meth:`acquire`).

        Raises:
            ValueError: If ``buf`` was not acquired from this pool.
        """
        with self._lock:
            try:
                self._in_use.remove(buf)
            except ValueError:
                raise ValueError("Buffer not from this HeaderBufferPool") from None
            self._available.append(buf)

    def contains_ptr(self, ptr: int) -> bool:
        """Return whether ``ptr`` lies within a registered DMA buffer region.

        Args:
            ptr: Base address to check.

        Returns:
            ``True`` when ``ptr`` falls in the half-open range of one of the
            pool's allocated ``(ptr, header_size)`` regions.
        """
        ptr = int(ptr)
        with self._lock:
            return any(
                reg_ptr <= ptr < reg_ptr + self.header_size
                for reg_ptr in self._spdk_ptrs
            )

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
        with self._lock:
            self._free_all_allocated()


class SpdkBlockIO(RawBlockIO):
    """SPDK-backed raw-block engine with zero-copy NVMe DMA.

    The engine connects to an NVMe device over PCIe or NVMe-oF (TCP/RDMA) and
    performs all I/O through SPDK's DMA-allocated buffers. Buffers written or
    read through :meth:`write_buffers`/:meth:`read_buffers` must be registered
    with SPDK (external hugepage memory, a pooled header buffer, or a freshly
    DMA-allocated buffer); unregistered buffers are rejected because there is no
    temporary DMA copy and no fallback to ordinary I/O.

    Args:
        transport_type: Transport used to reach the device (``"pcie"``,
            ``"tcp"``, or ``"rdma"``).
        target_ip: For PCIe, the device address (for example ``"0000:01:00.0"``);
            for NVMe-oF, the target IP address.
        target_port: NVMe-oF target port (ignored for PCIe).
        target_nqn: NVMe-oF target NQN (ignored for PCIe).
        core_mask: Hex core mask reserving cores for the SPDK workers (for
            example ``"0x3f"`` for cores 0-5).
        mem_size_mb: Hugepage memory in MB to reserve for the SPDK environment.
        block_align: Raw-block block alignment in bytes.
        header_bytes: Size of one raw-block slot header in bytes.
        header_pool_size: Number of pre-allocated header DMA buffers.
        spdk_ffi: Optional injected :class:`SpdkIoEngineFFI` for tests. When
            provided, the real SPDK init/launch/affinity steps are skipped.

    Raises:
        RuntimeError: If the real SPDK environment cannot be initialized.
    """

    def __init__(
        self,
        *,
        transport_type: str = "tcp",
        target_ip: str = "127.0.0.1",
        target_port: str = "4420",
        target_nqn: str = "nqn.2016-06.io.spdk:cnode1",
        core_mask: str = "",
        mem_size_mb: int = 4096,
        block_align: int,
        header_bytes: int,
        header_pool_size: int = 256,
        spdk_ffi: "SpdkIoEngineFFI | None" = None,
    ) -> None:
        """Initialize the SPDK engine and its header buffer pool.

        Args:
            transport_type: Transport used to reach the device.
            target_ip: Device address (PCIe) or target IP (NVMe-oF).
            target_port: NVMe-oF target port (ignored for PCIe).
            target_nqn: NVMe-oF target NQN (ignored for PCIe).
            core_mask: Hex core mask reserving cores for the SPDK workers.
            mem_size_mb: Hugepage memory in MB to reserve for SPDK.
            block_align: Raw-block block alignment in bytes.
            header_bytes: Size of one raw-block slot header in bytes.
            header_pool_size: Number of pre-allocated header DMA buffers.
            spdk_ffi: Optional injected FFI for tests.

        Raises:
            RuntimeError: If the real SPDK environment cannot be initialized.
        """
        self._transport_type = transport_type
        self._target_ip = target_ip
        self._target_port = target_port
        self._target_nqn = target_nqn
        self._core_mask = core_mask
        self._mem_size_mb = mem_size_mb
        self._block_align = block_align
        self._header_bytes = header_bytes
        self._header_pool_size = header_pool_size

        self._ffi: "SpdkIoEngineFFI | None" = None
        self._header_pool: "HeaderBufferPool | None" = None
        self._registered_regions: list[tuple[int, int]] = []
        self._max_io_depth_value: int | None = None
        self._max_transfer_size_value: int | None = None
        self._spdk_max_chunk_bytes: int = 0
        self._closed = False
        # Byte address of each freshly DMA-allocated buffer, keyed by the buffer
        # object id, so release_buffer can free exactly the memory it leased.
        self._dma_buffers: dict[int, int] = {}
        # Byte address and size of each freshly DMA-allocated buffer, so
        # is_buffer_registered can recognize them. SPDK DMA memory is already
        # DMA-accessible and needs no separate registration.
        self._fresh_dma_regions: list[tuple[int, int]] = []
        # Largest pooled buffer size; buffers no larger than this are leased from
        # the header pool, larger ones (variable-size payloads) are DMA-allocated.
        self._header_buffer_size = 0

        if spdk_ffi is not None:
            # Test injection: reuse the provided FFI and skip the real
            # init/launch/affinity steps that require hardware and hugepages.
            self._ffi = spdk_ffi
            self._query_device_capabilities()
        else:
            self._init_spdk_engine()

        # Header buffers must be at least as large as a header and aligned to
        # the SPDK DMA boundary so the pooled buffers can hold a full slot
        # header, and at least one block so the raw-block checkpoint path can
        # pack a metadata header into a pooled buffer for a block-sized write.
        header_size = max(_round_up(header_bytes, _SPDK_DMA_ALIGNMENT), block_align)
        self._header_buffer_size = header_size
        self._header_pool = HeaderBufferPool(
            header_size=header_size,
            pool_size=header_pool_size,
            spdk_engine=self._ffi,
        )

    @classmethod
    def from_config(cls, cfg: "SpdkBlockIOConfig") -> "SpdkBlockIO":
        """Build an SPDK engine from its normalized config object.

        Args:
            cfg: An ``SpdkBlockIOConfig`` describing the transport, device, and
                header-pool parameters.

        Returns:
            A ``SpdkBlockIO`` configured from ``cfg``.
        """
        return cls(
            transport_type=cfg.transport_type,
            target_ip=cfg.target_ip,
            target_port=cfg.target_port,
            target_nqn=cfg.target_nqn,
            core_mask=cfg.core_mask,
            mem_size_mb=cfg.mem_size_mb,
            block_align=cfg.block_align,
            header_bytes=cfg.header_bytes,
            header_pool_size=cfg.header_pool_size,
        )

    def _init_spdk_engine(self) -> None:
        """Initialize SPDK, connect to the NVMe device, and start the workers.

        Initializes the SPDK environment, connects to the NVMe device (PCIe or
        NVMe-oF TCP/RDMA), and derives the admin/I/O worker cores from the
        configured core mask. The calling thread's affinity is then restricted
        to the producer cores so the SPDK workers make progress.

        Raises:
            RuntimeError: If the core mask is invalid, memory is unavailable, or
                the SPDK environment or workers fail to start.
        """
        try:
            if not self._core_mask:
                raise RuntimeError(
                    "spdk_core_mask must be set to a non-empty hex core mask "
                    "(for example '0x3f'), but got an empty value"
                )
            if self._mem_size_mb <= 0:
                raise RuntimeError(
                    f"spdk_mem_size_mb must be a positive value in MB, but got "
                    f"{self._mem_size_mb}"
                )

            # Base affinity on the cores the process may actually use (this
            # respects cgroup cpuset/container restrictions); fall back to the
            # raw host count only where the affinity API is unavailable.
            try:
                all_cores = set(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                all_cores = set(range(os.cpu_count() or 1))
            total_cores = len(all_cores)
            reserved_cores: set[int] = set()
            try:
                mask_value = int(self._core_mask, 16)
                for bit in all_cores:
                    if mask_value & (1 << bit):
                        reserved_cores.add(bit)
            except ValueError:
                raise RuntimeError(
                    f"SpdkBlockIO: invalid spdk_core_mask '{self._core_mask}'"
                ) from None

            self._ffi = SpdkIoEngineFFI()

            rc = self._ffi.init(
                core_mask=self._core_mask,
                mem_size_mb=self._mem_size_mb,
            )
            if rc != 0:
                raise RuntimeError("Failed to initialize SPDK environment")

            producer_cores = all_cores - reserved_cores
            if not producer_cores:
                raise RuntimeError(
                    "No producer cores available after excluding reserved cores "
                    f"(total={total_cores}, reserved={sorted(reserved_cores)})"
                )

            logger.info(
                "SpdkBlockIO: CPU affinity set for SPDK workers "
                "(total_cores=%d, reserved=%s)",
                total_cores,
                sorted(reserved_cores),
            )
            # Best-effort only: the SPDK worker threads pin themselves via
            # pthread_setaffinity_np, so an affinity change the environment
            # refuses is not fatal.
            try:
                os.sched_setaffinity(0, producer_cores)
            except OSError as e:
                logger.warning(
                    "SpdkBlockIO: could not set producer affinity on %s: %s",
                    sorted(producer_cores),
                    e,
                )

            rc = self._ffi.launch_spdk_workers(
                transport_type=self._transport_type,
                addr=self._target_ip,
                port=self._target_port,
                nqn=self._target_nqn,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Failed to launch SPDK I/O worker (type={self._transport_type})"
                )

            if self._transport_type in ("tcp", "rdma"):
                logger.info(
                    "SpdkBlockIO: engine initialized successfully "
                    "(NVMe-oF target=%s:%s)",
                    self._target_ip,
                    self._target_port,
                )
            else:
                logger.info(
                    "SpdkBlockIO: engine initialized successfully (PCIe device=%s)",
                    self._target_ip,
                )

            self._query_device_capabilities()
            if self._max_io_depth_value:
                logger.info(
                    "SpdkBlockIO: max I/O queue depth=%d, max transfer size=%d "
                    "(read batches capped at %d, writes/splits capped at %d "
                    "bytes per call)",
                    self._max_io_depth_value,
                    self._max_transfer_size_value,
                    min(self._max_io_depth_value, _MAX_SPDK_READ_BATCH_KEYS_CAP),
                    self._spdk_max_chunk_bytes,
                )
        except Exception as e:
            self._ffi = None
            raise RuntimeError(f"Failed to initialize SPDK engine: {e}") from e

    def _query_device_capabilities(self) -> None:
        """Record the controller queue depth and namespace max transfer size."""
        if self._ffi is None:
            raise RuntimeError("SPDK engine not initialized")

        self._max_io_depth_value = self._ffi.get_max_io_depth()
        self._max_transfer_size_value = self._ffi.get_max_transfer_size()
        self._spdk_max_chunk_bytes = self._resolve_spdk_max_chunk_bytes()

    def _resolve_spdk_max_chunk_bytes(self) -> int:
        """Return the bytes to submit per SPDK call, or ``0`` when unknown."""
        return spdk_max_chunk_bytes(
            self._max_io_depth_value or 0,
            self._max_transfer_size_value or 0,
            _SPDK_IO_CONCURRENCY_FACTOR,
        )

    # ------------------------------------------------------------------
    # RawBlockIO surface
    # ------------------------------------------------------------------
    def write_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> None:
        """Write buffers through SPDK with batched zero-copy DMA.

        All buffers must be registered with SPDK; unregistered buffers raise a
        ``RuntimeError`` because there is no temporary DMA copy and no fallback
        to ordinary I/O.

        Args:
            offsets: Device byte offsets for each write.
            buffers: Python buffers to write.
            payload_lens: Logical payload lengths for each buffer.
            total_lens: Physical I/O byte counts for each buffer.
            placement_ids: Accepted for ABC parity; SPDK batched I/O does not
                carry FDP placement directives.

        Raises:
            RuntimeError: If the SPDK engine is not initialized, a buffer is not
                registered, submission fails, or a batch reports failure.
        """
        if self._ffi is None:
            raise RuntimeError("SPDK engine not initialized")

        ffi = self._ffi
        n = len(offsets)
        if n == 0:
            return

        buf_ptrs: list[int] = []
        for offset, buf, _payload_len, _total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            buf_ptr = self._buffer_ptr(buf)
            if not self.is_buffer_registered(buf_ptr):
                raise RuntimeError(
                    f"Buffer at offset={offset} is not registered with SPDK; "
                    "zero-copy I/O requires all buffers to be registered"
                )
            buf_ptrs.append(buf_ptr)

        offsets_list = list(offsets)
        total_lens_list = list(total_lens)
        if self._spdk_max_chunk_bytes > 0:
            offsets_list, total_lens_list, buf_ptrs = _chunk_io_inputs(
                offsets_list,
                total_lens_list,
                buf_ptrs,
                self._spdk_max_chunk_bytes,
            )

        rc, batch_id = ffi.batch_io_submit(
            offsets_list,
            total_lens_list,
            buf_ptrs,
            len(offsets_list),
            IO_WRITE,
        )
        if rc != 0:
            raise RuntimeError("SPDK batched write submission failed")

        status = ffi.wait_batch(batch_id)
        if status != 0:
            raise RuntimeError(
                f"SPDK write failed for batch_id={batch_id} (status={status})"
            )

    def read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> list[bool]:
        """Read buffers through SPDK with batched zero-copy DMA.

        Reads are bounded two ways: by ``read_batch_keys`` (the number of
        operations per submit) and by byte size, so no single transfer exceeds
        ``_spdk_max_chunk_bytes`` (when the queue depth and max transfer size are
        known). Large payloads are split into same-buffer sub-reads.

        Args:
            offsets: Device byte offsets for each read.
            buffers: Destination Python buffers.
            payload_lens: Logical payload lengths to expose to callers.
            total_lens: Physical I/O byte counts for each read.

        Returns:
            A per-read success boolean list aligned with ``offsets``.

        Raises:
            RuntimeError: If the SPDK engine is not initialized, a buffer is not
                registered, submission fails, or a batch reports failure.
            ValueError: If an output buffer is shorter than its payload length.
        """
        if self._ffi is None:
            raise RuntimeError("SPDK engine not initialized")

        ffi = self._ffi
        n = len(offsets)
        if n == 0:
            return [True] * n

        buf_ptrs: list[int] = []
        for offset, buf, payload_len, _total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            buf_ptr = self._buffer_ptr(buf)
            if not self.is_buffer_registered(buf_ptr):
                raise RuntimeError(
                    f"Buffer at offset={offset} is not registered with SPDK; "
                    "zero-copy I/O requires all buffers to be registered"
                )
            if len(memoryview(buf)) < payload_len:
                raise ValueError("output buffer shorter than payload_len")
            buf_ptrs.append(buf_ptr)

        batch_keys = self.read_batch_keys()
        batch_bytes = self._spdk_max_chunk_bytes
        for start in range(0, n, batch_keys):
            end = min(start + batch_keys, n)
            offsets_slice = list(offsets[start:end])
            total_lens_slice = list(total_lens[start:end])
            if batch_bytes > 0:
                # Cap each transfer's byte count in addition to the per-submit
                # operation count.
                offsets_slice, total_lens_slice, buf_ptrs_slice = _chunk_io_inputs(
                    offsets_slice,
                    total_lens_slice,
                    buf_ptrs[start:end],
                    batch_bytes,
                )
            else:
                buf_ptrs_slice = buf_ptrs[start:end]

            rc, batch_id = ffi.batch_io_submit(
                offsets_slice,
                total_lens_slice,
                buf_ptrs_slice,
                len(offsets_slice),
                IO_READ,
            )
            if rc != 0:
                raise RuntimeError("SPDK batched read submission failed")

            status = ffi.wait_batch(batch_id)
            if status != 0:
                raise RuntimeError(
                    f"SPDK read failed for batch_id={batch_id} (status={status})"
                )

        return [True] * n

    def register_io_buffers(
        self,
        buffer_ptrs: Sequence[int],
        buffer_sizes: Sequence[int],
    ) -> None:
        """Register stable memory ranges with SPDK for zero-copy I/O.

        Args:
            buffer_ptrs: Base addresses of the stable ranges.
            buffer_sizes: Sizes of the stable ranges in bytes.

        Raises:
            RuntimeError: If the SPDK engine is not initialized or registration
                fails for any region.
        """
        if self._ffi is None:
            raise RuntimeError("SPDK engine not initialized")

        for ptr, size in zip(buffer_ptrs, buffer_sizes, strict=True):
            rc = self._ffi.register_external_memory(int(ptr), int(size))
            if rc != 0:
                raise RuntimeError(
                    "Failed to register external memory with SPDK "
                    "(ptr=0x%x, size=%d, rc=%d)" % (int(ptr), int(size), rc)
                )
            self._registered_regions.append((int(ptr), int(size)))

    @property
    def use_odirect(self) -> bool:
        """Return the effective O_DIRECT setting.

        Returns:
            ``False``; SPDK performs DMA I/O and does not use O_DIRECT.
        """
        return False

    @property
    def max_data_transfer_size(self) -> int:
        """Return the resolved SPDK submit-size bound in bytes.

        This is the largest payload per submitted SPDK command after chunking by
        ``queue_depth / concurrency_factor`` times the namespace max transfer
        size. ``0`` means the queue depth or max transfer size was unknown, so
        no byte-level bound is applied and the device splits transfers itself.
        """
        return self._spdk_max_chunk_bytes

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of headers read in one batch.

        Returns:
            The SPDK read-batch size (``read_batch_keys``), so the core's
            batched slot-header recovery path submits the same number of reads
            per call as :meth:`read_buffers`.
        """
        return self.read_batch_keys()

    def device_size(self) -> int:
        """Return the usable NVMe device size in bytes.

        Returns:
            The device size reported by SPDK.

        Raises:
            RuntimeError: If the device size cannot be determined (the device is
                not connected or reports an invalid size).
        """
        if self._ffi is None:
            raise RuntimeError("SPDK engine not initialized")

        size = self._ffi.get_device_size()
        if size <= 0:
            raise RuntimeError(
                f"SPDK device reported an invalid size ({size}); ensure the "
                "NVMe device is connected"
            )
        return size

    def capacity_bytes(self) -> int:
        """Return the usable NVMe device size in bytes.

        Returns:
            The device size reported by SPDK via :meth:`device_size`.
        """
        return int(self.device_size())

    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        """Return FDP placement/RUH status from the device.

        Returns:
            An empty list; SPDK's NVMe path does not expose FDP status through
            this engine.
        """
        return []

    def worker_error(self) -> Optional[str]:
        """Return the worker's terminal error, or ``None`` when healthy.

        Returns:
            ``None``; the SPDK engine does not report a terminal worker error
            through this surface.
        """
        return None

    def fixed_buffer_status(self) -> tuple[bool, int]:
        """Return ``(registered, registered_bytes)`` for fixed-buffer I/O.

        Returns:
            A tuple indicating whether any external buffers are registered and
            the total registered byte count.
        """
        registered_bytes = sum(size for _, size in self._registered_regions)
        return registered_bytes > 0, registered_bytes

    def supports_buffer_registration(self) -> bool:
        """Return whether this engine can register stable zero-copy buffers.

        Returns:
            ``True``; the SPDK engine registers host memory for zero-copy I/O.
        """
        return True

    def close(self) -> None:
        """Release the header buffer pool and shut down the SPDK engine."""
        if self._closed:
            return
        self._closed = True

        if self._header_pool is not None:
            try:
                self._header_pool.cleanup()
            except Exception as e:
                logger.warning(
                    "SpdkBlockIO: error cleaning up header buffer pool: %s", e
                )
            self._header_pool = None

        # Free any DMA buffers still outstanding (a caller that never released
        # them would otherwise leak hugepage memory).
        if self._ffi is not None:
            for dma_ptr in self._dma_buffers.values():
                try:
                    self._ffi.free_spdk_memory(dma_ptr)
                except Exception:
                    pass
        self._dma_buffers.clear()
        self._fresh_dma_regions.clear()

        if self._ffi is not None:
            try:
                self._ffi.shutdown_spdk_workers()
                self._ffi.deinit()
            except Exception as e:
                logger.warning("SpdkBlockIO: error shutting down SPDK engine: %s", e)
            self._ffi = None

        self._registered_regions.clear()
        self._max_io_depth_value = None
        self._max_transfer_size_value = None
        self._spdk_max_chunk_bytes = 0

    # ------------------------------------------------------------------
    # SPDK-specific primitives used by RawBlockCore
    # ------------------------------------------------------------------
    def read_batch_keys(self) -> int:
        """Compute the number of keys to submit per SPDK read batch.

        Returns:
            ``min(max_io_depth, _MAX_SPDK_READ_BATCH_KEYS_CAP)`` when the
            controller reports a max I/O depth, otherwise
            ``_DEFAULT_SPDK_READ_BATCH_KEYS``.
        """
        max_depth = self._max_io_depth_value
        if not max_depth:
            return _DEFAULT_SPDK_READ_BATCH_KEYS
        return min(max_depth, _MAX_SPDK_READ_BATCH_KEYS_CAP)

    def _acquire_buffer(self, size: int) -> Any:
        """Lease a writable DMA buffer of at least ``size`` bytes.

        Buffers no larger than the pooled header size are leased from the header
        pool; larger buffers (variable-size payloads such as checkpoints) are
        freshly DMA-allocated and tracked so :meth:`release_buffer` can free
        them. The returned object is a ctypes ``c_ubyte`` array, which supports
        the in-place slice assignment core uses to encode headers.

        Args:
            size: Minimum number of writable bytes to expose.

        Returns:
            A ctypes ``c_ubyte`` array exposing at least ``size`` writable bytes.

        Raises:
            RuntimeError: If the engine is unavailable or allocation fails.
        """
        if self._ffi is None:
            raise RuntimeError("SPDK engine not initialized")
        if self._header_pool is not None and size <= self._header_buffer_size:
            return self._header_pool.acquire()
        dma_ptr = self._ffi.allocate_spdk_memory(
            int(size), _SPDK_DMA_ALIGNMENT, numa_id=-1
        )
        if dma_ptr == 0:
            raise RuntimeError("Failed to allocate SPDK DMA buffer")
        buf = (ctypes.c_ubyte * int(size)).from_address(dma_ptr)
        self._dma_buffers[id(buf)] = dma_ptr
        self._fresh_dma_regions.append((dma_ptr, int(size)))
        return buf

    def _release_buffer(self, buf: Any) -> None:
        """Return a previously acquired buffer to its owner.

        Freshly DMA-allocated buffers are freed; pooled buffers are returned to
        the header pool.

        Args:
            buf: A buffer previously returned by :meth:`acquire_buffer`.
        """
        buf_id = id(buf)
        if buf_id in self._dma_buffers:
            dma_ptr = self._dma_buffers.pop(buf_id)
            self._remove_dma_region(dma_ptr)
            try:
                if self._ffi is not None:
                    self._ffi.free_spdk_memory(dma_ptr)
            except Exception as e:
                logger.warning(
                    "SpdkBlockIO: error freeing DMA buffer on release: %s", e
                )
            return
        if self._header_pool is not None:
            try:
                self._header_pool.release(buf)
            except Exception as e:
                logger.warning(
                    "SpdkBlockIO: error returning pooled buffer on release: %s",
                    e,
                )

    def _acquire_buffers(self, count: int, size: int) -> list[Any]:
        """Lease ``count`` writable DMA buffers for a batched read.

        Args:
            count: Number of buffers to lease.
            size: Minimum number of writable bytes per buffer.

        Returns:
            A list of ``count`` ctypes ``c_ubyte`` arrays.
        """
        return [self._acquire_buffer(size) for _ in range(count)]

    def _release_buffers(self, bufs: Sequence[Any]) -> None:
        """Return previously acquired batched buffers to their owner.

        Args:
            bufs: Buffers previously returned by :meth:`_acquire_buffers`.
        """
        for buf in bufs:
            self._release_buffer(buf)

    def is_buffer_registered(self, ptr: int) -> bool:
        """Return whether ``ptr`` falls inside a registered memory region.

        A buffer counts as registered when its address lies within an external
        hugepage region registered via :meth:`register_io_buffers`, within one
        of the header pool's DMA buffers, or within one of the freshly
        DMA-allocated buffer regions leased via :meth:`_acquire_buffer`.

        Args:
            ptr: Base address to check.

        Returns:
            ``True`` when the address is inside a registered region.
        """
        ptr = int(ptr)
        for reg_ptr, reg_size in self._registered_regions:
            if reg_ptr <= ptr < reg_ptr + reg_size:
                return True
        if self._header_pool is not None and self._header_pool.contains_ptr(ptr):
            return True
        if any(
            reg_ptr <= ptr < reg_ptr + reg_size
            for reg_ptr, reg_size in self._fresh_dma_regions
        ):
            return True
        return False

    @staticmethod
    def _buffer_ptr(buf: Any) -> int:
        """Return the byte address of a buffer, or ``0`` when unavailable.

        Args:
            buf: A memoryview, bytearray, or ctypes Array.

        Returns:
            The buffer's base address, or ``0`` when ``buf`` is not a supported
            buffer type.
        """
        if isinstance(buf, (memoryview, bytearray, ctypes.Array)):
            return ctypes.addressof((ctypes.c_ubyte * len(buf)).from_buffer(buf))
        return 0

    def _remove_dma_region(self, ptr: int) -> None:
        """Drop a freshly DMA-allocated buffer's region from the tracking list.

        Args:
            ptr: Base address of the buffer whose region should be removed.
        """
        for i, (reg_ptr, _reg_size) in enumerate(self._fresh_dma_regions):
            if reg_ptr == ptr:
                del self._fresh_dma_regions[i]
                return
