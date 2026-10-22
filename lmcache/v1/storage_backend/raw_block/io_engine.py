# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
import ctypes
import os
import re
import stat

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

RAW_BLOCK_IO_ENGINES = frozenset({"posix", "io_uring"})
DEFAULT_IOURING_QUEUE_DEPTH = 256
_MAX_FDP_PLACEMENT_ID = 0xFFFF

# FDP placement ID semantics are shared by design across raw-block write paths.
# None omits the directive. Explicit identifiers must be positive because
# default writes already use the RUH mapping associated with Placement
# Identifier 0. RawBlockCore rejects explicit identifier 0 so KV data never sends
# an FDP directive for the default placement identifier. Non-zero identifiers are
# encoded as 16-bit NVMe directive-specific values.
# Metadata checkpoint placement is optional. ``None`` keeps the historical
# default NVMe write behavior; a positive identifier emits an FDP directive.
PlacementId = int | None


def normalize_raw_block_placement_ids(
    placement_ids: Sequence[PlacementId] | None,
    expected_len: int,
    *,
    field_name: str = "placement_ids",
    allow_none: bool = True,
) -> list[PlacementId]:
    """Validate FDP placement identifiers and preserve omitted directives."""
    if placement_ids is None:
        return [None] * expected_len
    if len(placement_ids) != expected_len:
        raise ValueError(f"{field_name} must have length {expected_len}")

    normalized: list[PlacementId] = []
    for placement_id in placement_ids:
        if placement_id is None:
            if not allow_none:
                raise ValueError(f"{field_name} must contain integers")
            normalized.append(None)
            continue
        if not isinstance(placement_id, int) or isinstance(placement_id, bool):
            raise ValueError(f"{field_name} must contain integers or None")
        if placement_id == 0:
            raise ValueError(f"{field_name} must not contain placement identifier 0")
        if placement_id < 0:
            raise ValueError(f"{field_name} must contain positive integers or None")
        if placement_id > _MAX_FDP_PLACEMENT_ID:
            raise ValueError(
                f"{field_name} must contain placement identifiers in range "
                f"1..={_MAX_FDP_PLACEMENT_ID}"
                f"{' or None' if allow_none else ''}"
            )
        normalized.append(int(placement_id))
    return normalized


def normalize_raw_block_io_engine(
    io_engine: Any = None,
    *,
    use_iouring: Any = None,
    use_uring: Any = None,
) -> str:
    """Normalize raw-block I/O engine config with legacy compatibility.

    Args:
        io_engine: Explicit engine string. Valid values are ``"posix"``,
            and ``"io_uring"``.
        use_iouring: Legacy boolean knob. Used only when ``io_engine`` is not
            set.
        use_uring: Legacy boolean alias. Used only when ``io_engine`` is not
            set.

    Returns:
        The normalized engine string.

    Raises:
        ValueError: If ``io_engine`` names an unsupported engine.
    """
    if io_engine is None or io_engine == "":
        if bool(use_iouring) or bool(use_uring):
            return "io_uring"
        return "posix"
    normalized = str(io_engine).lower()
    if normalized not in RAW_BLOCK_IO_ENGINES:
        allowed = ", ".join(sorted(RAW_BLOCK_IO_ENGINES))
        raise ValueError(f"io_engine must be one of: {allowed}")
    return normalized


def _byte_view(buf: Any) -> memoryview:
    """Return a byte-addressable memoryview over a Python buffer.

    Args:
        buf: Object implementing the Python buffer protocol.

    Returns:
        A memoryview with one-byte elements.

    Raises:
        TypeError: If ``buf`` does not expose a compatible contiguous buffer.
    """
    view = buf if isinstance(buf, memoryview) else memoryview(buf)
    if view.itemsize == 1 and view.format in ("B", "b", "c"):
        return view
    return view.cast("B")


@dataclass(frozen=True)
class RawBlockIoEngineConfig:
    """Shared raw-block I/O engine configuration.

    Attributes:
        device_path: Raw-device path.
        block_align: Block alignment in bytes.
        io_engine: Normalized engine name (``"posix"`` or ``"io_uring"``).
        use_odirect: Whether O_DIRECT I/O is required.
    """

    device_path: str
    block_align: int
    io_engine: str
    use_odirect: bool = False

    def __post_init__(self) -> None:
        """Validate the engine's field-level (device-independent) options."""
        if not self.device_path:
            raise ValueError(f"{type(self).__name__}: device_path must not be empty")
        if self.block_align <= 0 or (self.block_align & (self.block_align - 1)) != 0:
            raise ValueError(
                f"{type(self).__name__}: block_align ({self.block_align}) "
                "must be a positive power of two"
            )
        if self.io_engine not in RAW_BLOCK_IO_ENGINES:
            raise ValueError(
                f"{type(self).__name__}: unsupported io_engine "
                f"({self.io_engine}); expected one of "
                f"{sorted(RAW_BLOCK_IO_ENGINES)}"
            )


@dataclass(frozen=True)
class PosixIoEngineConfig(RawBlockIoEngineConfig):
    """Configuration for the synchronous posix engine.

    The posix engine ignores the io_uring-specific knobs, so validation falls
    back to the shared :class:`RawBlockIoEngineConfig` ``__post_init__``.
    """


@dataclass(frozen=True)
class IoUringIoEngineConfig(RawBlockIoEngineConfig):
    """Configuration for the batched io_uring engine.

    Attributes:
        use_uring_cmd: Whether to use NVMe passthrough (io_uring_cmd).
        iouring_queue_depth: io_uring queue depth for batched I/O.
        max_data_transfer_size: Max single-transfer size for io_uring_cmd (0 to
            auto-detect from device limits).
    """

    use_uring_cmd: bool = False
    iouring_queue_depth: int = DEFAULT_IOURING_QUEUE_DEPTH
    max_data_transfer_size: int = 0

    def __post_init__(self) -> None:
        """Validate the io_uring engine's divergent field-level options."""
        super().__post_init__()
        if self.use_uring_cmd and self.io_engine != "io_uring":
            raise ValueError(
                "IoUringIoEngineConfig: use_uring_cmd requires io_uring as io_engine"
            )
        if self.iouring_queue_depth <= 0:
            raise ValueError("iouring_queue_depth must be > 0")
        if (
            self.max_data_transfer_size > 0
            and self.max_data_transfer_size % self.block_align != 0
        ):
            raise ValueError(
                f"max_data_transfer_size ({self.max_data_transfer_size}) must be "
                f"a multiple of block_align ({self.block_align})"
            )


class RawBlockIoEngine(ABC):
    """Thin abstraction over one raw-block I/O engine."""

    @abstractmethod
    def write_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> None:
        """Write one or more buffers through the engine's I/O path.

        Args:
            offsets: Device offsets for each write.
            buffers: Python buffers to write.
            payload_lens: Logical payload lengths for each buffer.
            total_lens: Physical I/O lengths for each buffer.
            placement_ids: Optional FDP placement identifiers; ``None`` omits the
                directive and explicit identifier 0 is rejected.
        """

    @abstractmethod
    def read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> list[bool]:
        """Read one or more buffers through the engine's I/O path.

        Args:
            offsets: Device offsets for each read.
            buffers: Destination Python buffers.
            payload_lens: Logical payload lengths to expose to callers.
            total_lens: Physical I/O lengths for each read.

        Returns:
            A per-read success boolean list aligned with ``offsets``.
        """

    @abstractmethod
    def acquire_buffer(self, size: int) -> Any:
        """Lease a writable buffer of at least ``size`` bytes for I/O.

        The returned buffer must stay valid and unchanged until it is returned
        through :meth:`release_buffer`. Engines backed by durable DMA memory
        lease a buffer from a pool and expect it back; engines backed by
        transient host memory hand out a fresh buffer whose release is a no-op.

        Args:
            size: Minimum number of writable bytes to expose.

        Returns:
            A buffer exposing the Python buffer protocol with at least ``size``
            bytes.
        """

    @abstractmethod
    def release_buffer(self, buf: Any) -> None:
        """Return a previously acquired buffer to its owner.

        Args:
            buf: A buffer previously returned by :meth:`acquire_buffer`.
        """

    @abstractmethod
    def acquire_buffers(self, count: int, size: int) -> list[Any]:
        """Lease ``count`` writable buffers for a batched read.

        This is the batched counterpart of :meth:`acquire_buffer` so a batch of
        headers can be allocated together. Every returned buffer must be given
        back through :meth:`release_buffers`.

        Args:
            count: Number of buffers to lease.
            size: Minimum number of writable bytes per buffer.

        Returns:
            A list of ``count`` buffers exposing the Python buffer protocol.
        """

    @abstractmethod
    def release_buffers(self, bufs: Sequence[Any]) -> None:
        """Return previously acquired batched buffers to their owner.

        Args:
            bufs: Buffers previously returned by :meth:`acquire_buffers`.
        """

    @abstractmethod
    def register_io_buffers(
        self,
        buffer_ptrs: Sequence[int],
        buffer_sizes: Sequence[int],
    ) -> None:
        """Register stable memory ranges for zero-copy I/O.

        Args:
            buffer_ptrs: Base addresses of the stable ranges.
            buffer_sizes: Sizes of the stable ranges in bytes.
        """

    @abstractmethod
    def device_size(self) -> int:
        """Return the usable raw-device size in bytes."""

    @abstractmethod
    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        """Return ``(placement_id, ruh_id)`` tuples from the device."""

    @abstractmethod
    def worker_error(self) -> Optional[str]:
        """Return the worker's terminal error, or ``None`` when healthy."""

    @abstractmethod
    def fixed_buffer_status(self) -> tuple[bool, int]:
        """Return ``(registered, registered_bytes)`` for fixed-buffer I/O."""

    def raise_if_failed(self) -> None:
        """Reject work after a terminal worker failure.

        Raises:
            RuntimeError: When ``worker_error`` reports a terminal error.
        """
        worker_error = self.worker_error()
        if worker_error is not None:
            raise RuntimeError(worker_error)

    @property
    @abstractmethod
    def use_odirect(self) -> bool:
        """Return the effective O_DIRECT setting after engine coercion."""

    @property
    @abstractmethod
    def max_data_transfer_size(self) -> int:
        """Return the resolved transfer split size in bytes (0 when none)."""

    @abstractmethod
    def supports_io_buffers(self) -> bool:
        """Return whether this engine can register stable zero-copy buffers."""

    @abstractmethod
    def wait_for_batch(
        self,
        batch_id: int,
        expected_count: int,
        operation: str,
    ) -> list[bool]:
        """Wait for a submitted batch and return its completion bitmap.

        Args:
            batch_id: Identifier of the submitted batch to wait on.
            expected_count: Number of individual I/O entries submitted in the
                batch. For io_engines that support splitting this is the
                post-splitting chunk count, not the number of logical reads
                or writes.
            operation: Human-readable operation name used in diagnostics.

        Returns:
            A per-entry success boolean list aligned with the batch.
        """

    @abstractmethod
    def device(self) -> Any:
        """Return the lazily opened raw device handle."""

    @abstractmethod
    def set_device(self, device: Any) -> None:
        """Replace the raw device handle (used for testing)."""

    @abstractmethod
    def close(self) -> None:
        """Release the raw device and any associated resources."""


class _RustRawBlockEngine(RawBlockIoEngine):
    """Common base for engines backed by the Rust ``RawBlockDevice``."""

    def __init__(
        self,
        device_path: str,
        *,
        use_odirect: bool,
        use_uring_cmd: bool,
        io_engine: str,
        iouring_queue_depth: int,
        block_align: int,
        max_data_transfer_size: int = 0,
    ) -> None:
        """Initialize the engine configuration.

        Args:
            device_path: Raw-device path.
            use_odirect: Whether O_DIRECT I/O is required.
            use_uring_cmd: Whether NVMe passthrough (io_uring_cmd) is enabled.
            io_engine: Engine name used when opening the Rust device.
            iouring_queue_depth: io_uring queue depth for batched I/O.
            block_align: Block alignment in bytes.
            max_data_transfer_size: Max single-transfer size for io_uring_cmd.

        Raises:
            ValueError: If a device-level check fails during device construction
                (see the concrete engine ``__init__``).
        """
        self._device_path = device_path
        # io_uring_cmd (NVMe passthrough) does not support O_DIRECT; coerce the
        # effective setting so it matches what the Rust binding will actually do.
        self._use_odirect = use_odirect and not use_uring_cmd
        if use_uring_cmd and use_odirect:
            logger.warning(
                "RawBlockCore: use_odirect is ignored for NVMe namespace "
                "character devices when use_uring_cmd=true"
            )
        self._use_uring_cmd = use_uring_cmd
        self._io_engine_name = io_engine
        self._iouring_queue_depth = iouring_queue_depth
        self._block_align = block_align
        self._max_data_transfer_size = max_data_transfer_size
        self._raw: Any = None

    @property
    def use_odirect(self) -> bool:
        """Return the effective O_DIRECT setting after uring_cmd coercion."""
        return self._use_odirect

    @property
    def max_data_transfer_size(self) -> int:
        """Return the resolved NVMe io_uring_cmd transfer split size in bytes.

        For io_uring_cmd engines this is the resolved split size; for engines
        that do not split transfers (posix) it is the configured value, which
        defaults to 0.
        """
        return self._max_data_transfer_size

    def acquire_buffer(self, size: int) -> Any:
        """Lease a transient host buffer for I/O.

        Args:
            size: Number of writable bytes to expose.

        Returns:
            A fresh ``bytearray`` of ``size`` bytes.
        """
        return bytearray(size)

    def release_buffer(self, buf: Any) -> None:
        """Release a transient host buffer; nothing to reclaim for posix/io_uring."""
        return

    def acquire_buffers(self, count: int, size: int) -> list[Any]:
        """Lease ``count`` transient host buffers for a batched read.

        Args:
            count: Number of buffers to lease.
            size: Number of writable bytes per buffer.

        Returns:
            A list of ``count`` fresh ``bytearray`` buffers.
        """
        return [bytearray(size) for _ in range(count)]

    def release_buffers(self, bufs: Sequence[Any]) -> None:
        """Release transient host buffers; nothing to reclaim for posix/io_uring."""
        return

    def device(self) -> Any:
        """Return the lazily opened raw device, rejecting a failed worker."""
        if self._raw is None:
            self._open()
        self.raise_if_failed()
        return self._raw

    def set_device(self, device: Any) -> None:
        """Replace the raw device handle (used for testing)."""
        self._raw = device

    def _open(self) -> None:
        """Open the Rust ``RawBlockDevice`` binding."""
        try:
            # Third Party
            from lmcache_rust_raw_block_io import RawBlockDevice  # type: ignore
        except Exception as e:
            # First Party
            from lmcache import __version__

            # Releases publish it at lmcache's version; dev builds have
            # no matching package, so leave those unpinned.
            is_release = re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", __version__)
            pin = f"=={__version__}" if is_release else ""
            raise RuntimeError(
                "Rust raw-block extension is not installed. Install it with "
                f"`pip install lmcache-rust-raw-block-io{pin}` and retry."
            ) from e
        self._raw = RawBlockDevice(
            self._device_path,
            writable=True,
            use_odirect=self._use_odirect,
            alignment=self._block_align,
            io_engine=self._io_engine_name,
            iouring_queue_depth=self._iouring_queue_depth,
            use_uring_cmd=self._use_uring_cmd,
        )

    def device_size(self) -> int:
        """Return the usable raw-device size in bytes."""
        return int(self.device().size_bytes())

    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        """Return ``(placement_id, ruh_id)`` tuples from the device."""
        return [
            (int(pid), int(ruhid)) for pid, ruhid in self.device().fetch_fdp_status()
        ]

    def worker_error(self) -> Optional[str]:
        """Return the worker's terminal error, or ``None`` when healthy."""
        if self._raw is None:
            return None
        get_worker_error = getattr(self._raw, "worker_error", None)
        return get_worker_error() if get_worker_error is not None else None

    def fixed_buffer_status(self) -> tuple[bool, int]:
        """Return ``(registered, registered_bytes)`` for fixed-buffer I/O."""
        if self._raw is None:
            return False, 0
        return self._raw.fixed_buffer_status()

    def wait_for_batch(
        self,
        batch_id: int,
        expected_count: int,
        operation: str,
    ) -> list[bool]:
        """Wait for an io_uring batch, log failures, and return its bitmap.

        ``expected_count`` is the number of individual I/O entries submitted
        in the Rust batch. For io_uring_cmd, this is the post-splitting chunk
        count, not the number of logical reads or writes.

        Args:
            batch_id: Identifier of the submitted batch to wait on.
            expected_count: Number of individual I/O entries submitted in the
                batch.
            operation: Human-readable operation name used in diagnostics.

        Returns:
            A per-entry success boolean list aligned with the batch. When the
            completion count mismatches ``expected_count`` every entry is
            reported as false.
        """
        raw_dev = self.device()
        results, completion_errors = raw_dev.wait_iouring(batch_id)
        results = list(results)
        for operation_index, error in completion_errors:
            logger.error(
                "RawBlockCore %s batch %d operation %d failed: %s",
                operation,
                batch_id,
                operation_index,
                error,
            )
        if len(results) != expected_count:
            logger.error(
                "RawBlockCore %s completion count mismatch: expected %d, got %d",
                operation,
                expected_count,
                len(results),
            )
            return [False] * expected_count
        return [bool(result) for result in results]

    def close(self) -> None:
        """Release the raw device and any associated resources."""
        if self._raw is None:
            return
        try:
            self._raw.close()
        except Exception as e:
            logger.warning(
                "Failed to close raw block device %s: %s", self._device_path, e
            )
        finally:
            self._raw = None


class PosixIoEngine(_RustRawBlockEngine):
    """Synchronous pread/pwrite engine used for ``io_engine='posix'``."""

    @classmethod
    def from_config(cls, cfg: RawBlockIoEngineConfig) -> "PosixIoEngine":
        """Build a posix engine from its normalized config object.

        Args:
            cfg: Engine config; the posix engine ignores the io_uring-specific
                knobs.

        Returns:
            A ``PosixIoEngine`` configured from ``cfg``.
        """
        return cls(
            device_path=cfg.device_path,
            use_odirect=cfg.use_odirect,
            use_uring_cmd=False,
            io_engine=cfg.io_engine,
            iouring_queue_depth=DEFAULT_IOURING_QUEUE_DEPTH,
            block_align=cfg.block_align,
            max_data_transfer_size=0,
        )

    def write_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> None:
        """Write each buffer synchronously with ``pwrite_from_buffer``."""
        raw_dev = self.device()
        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            raw_dev.pwrite_from_buffer(offset, buf, payload_len, total_len)

    def read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> list[bool]:
        """Read each buffer synchronously with ``pread_into``.

        Returns:
            A per-read success boolean list aligned with ``offsets``.
        """
        raw_dev = self.device()
        results: list[bool] = []
        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            try:
                raw_dev.pread_into(offset, buf, payload_len, total_len)
                results.append(True)
            except Exception:
                results.append(False)
        return results

    def register_io_buffers(
        self,
        buffer_ptrs: Sequence[int],
        buffer_sizes: Sequence[int],
    ) -> None:
        """posix does not support zero-copy registration; nothing to register."""
        return

    def supports_io_buffers(self) -> bool:
        """posix cannot register host memory for zero-copy I/O."""
        return False


class IoUringIoEngine(_RustRawBlockEngine):
    """Batched io_uring engine used for ``io_engine='io_uring'``."""

    def __init__(
        self,
        device_path: str,
        *,
        use_odirect: bool,
        use_uring_cmd: bool,
        io_engine: str,
        iouring_queue_depth: int,
        block_align: int,
        max_data_transfer_size: int = 0,
    ) -> None:
        """Initialize the io_uring engine and validate its NVMe configuration."""
        super().__init__(
            device_path,
            use_odirect=use_odirect,
            use_uring_cmd=use_uring_cmd,
            io_engine=io_engine,
            iouring_queue_depth=iouring_queue_depth,
            block_align=block_align,
            max_data_transfer_size=max_data_transfer_size,
        )
        if self._use_uring_cmd:
            IoUringIoEngine.validate_uring_cmd_device(device_path)
            self._max_data_transfer_size = (
                IoUringIoEngine.resolve_max_data_transfer_size(
                    device_path,
                    self._block_align,
                    max_data_transfer_size,
                )
            )

    @classmethod
    def from_config(cls, cfg: IoUringIoEngineConfig) -> "IoUringIoEngine":
        """Build an io_uring engine from its normalized config object.

        Args:
            cfg: Engine config including the io_uring-specific knobs.

        Returns:
            An ``IoUringIoEngine`` configured from ``cfg``.
        """
        return cls(
            device_path=cfg.device_path,
            use_odirect=cfg.use_odirect,
            use_uring_cmd=cfg.use_uring_cmd,
            io_engine=cfg.io_engine,
            iouring_queue_depth=cfg.iouring_queue_depth,
            block_align=cfg.block_align,
            max_data_transfer_size=cfg.max_data_transfer_size,
        )

    @staticmethod
    def validate_uring_cmd_device(device_path: str) -> None:
        """Validate that device_path is an NVMe generic namespace character device.

        Args:
            device_path: Raw-device path for the io_uring_cmd engine.

        Raises:
            ValueError: If the path is missing, not a character device, or does
                not match the ``ng<ctrl>n<ns>`` naming pattern.
        """
        try:
            mode = os.stat(device_path).st_mode
        except OSError as e:
            raise ValueError(
                "use_uring_cmd requires an existing NVMe namespace "
                f"character device path, got {device_path!r}"
            ) from e
        if not stat.S_ISCHR(mode):
            raise ValueError(
                "use_uring_cmd requires an NVMe namespace character device "
                f"(for example /dev/ng0n1), got {device_path!r}"
            )
        if not re.match(r"^ng\d+n\d+$", os.path.basename(device_path)):
            raise ValueError(
                "use_uring_cmd requires an NVMe generic namespace character device "
                f"with naming pattern ng<ctrl>n<ns> (for example /dev/ng0n1), "
                f"got {device_path!r}"
            )

    @staticmethod
    def resolve_max_data_transfer_size(
        device_path: str,
        block_align: int,
        configured_size: int,
    ) -> int:
        """Resolve the NVMe io_uring_cmd transfer split size.

        When ``configured_size`` is positive it is used directly (and must be a
        multiple of ``block_align``). Otherwise the size is auto-detected from
        the device's sysfs queue limits, bounded by both ``max_hw_sectors_kb``
        (total transfer size) and ``max_segments`` (scatter-gather limit). The
        io_uring_cmd passthrough path builds one iovec segment per page, so a
        single transfer can consume up to ``ceil(len / page_size)`` segments;
        exceeding ``max_segments`` makes the kernel reject the command with
        ``EINVAL``. This is an io_uring_cmd concern, so it lives on the io_uring
        engine rather than as a global helper.

        Args:
            device_path: NVMe generic namespace character device path.
            block_align: Block alignment in bytes; the result is a multiple of it.
            configured_size: Explicitly configured size in bytes. If > 0 it is
                used directly; if <= 0 the size is auto-detected from device
                limits.

        Returns:
            The resolved max data transfer size in bytes, a multiple of
            ``block_align``.

        Raises:
            ValueError: If ``configured_size`` is > 0 but not a multiple of
                ``block_align``.
            RuntimeError: If sysfs queue limits cannot be resolved during
                auto-detection.
        """
        if configured_size > 0:
            if configured_size % block_align != 0:
                raise ValueError(
                    f"max_data_transfer_size ({configured_size}) must be a "
                    f"multiple of block_align ({block_align})"
                )
            return configured_size

        queue_dir = IoUringIoEngine._resolve_sysfs_queue_dir(device_path)
        if queue_dir is None:
            raise RuntimeError(
                "RustRawBlockBackend: unable to derive NVMe sysfs queue path from "
                "NVMe character device path "
                f"{device_path} for auto max_data_transfer_size"
            )

        max_hw_sectors_kb = IoUringIoEngine._read_sysfs_int(
            f"{queue_dir}/max_hw_sectors_kb"
        )
        if max_hw_sectors_kb is None or max_hw_sectors_kb <= 0:
            raise RuntimeError(
                "RustRawBlockBackend: failed to read max_hw_sectors_kb from "
                f"{queue_dir} for auto max_data_transfer_size"
            )

        resolved_bytes = max_hw_sectors_kb * 1024

        # The io_uring_cmd passthrough path builds one iovec segment per page,
        # so cap the transfer at the device's scatter-gather segment limit.
        max_segments = IoUringIoEngine._read_sysfs_int(f"{queue_dir}/max_segments")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if max_segments is not None and max_segments > 0:
            segment_limit_bytes = max_segments * page_size
            resolved_bytes = min(resolved_bytes, segment_limit_bytes)

        aligned_bytes = (resolved_bytes // block_align) * block_align
        if aligned_bytes <= 0:
            aligned_bytes = block_align

        logger.info(
            "RustRawBlockBackend: auto max_data_transfer_size=%d bytes "
            "(device=%s, max_hw_sectors_kb=%s, max_segments=%s, page_size=%d)",
            aligned_bytes,
            device_path,
            max_hw_sectors_kb,
            max_segments,
            page_size,
        )
        return aligned_bytes

    @staticmethod
    def _resolve_sysfs_queue_dir(device_path: str) -> Optional[str]:
        """Resolve sysfs queue directory for NVMe character device paths."""
        base_name = os.path.basename(device_path)
        match = re.fullmatch(r"ng(\d+)n(\d+)", base_name)
        if match:
            ctrl, nsid = match.groups()
            return f"/sys/block/nvme{ctrl}n{nsid}/queue"
        return None

    @staticmethod
    def _read_sysfs_int(path: str) -> Optional[int]:
        """Read an integer value from sysfs and return None on failure."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            return None

    def write_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> None:
        """Write buffers through the batched or io_uring_cmd path."""
        raw_dev = self.device()
        per_write_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(offsets),
            field_name="placement_ids",
        )

        if self._use_uring_cmd:
            self._write_uring_cmd_buffers(
                offsets,
                buffers,
                payload_lens,
                total_lens,
                per_write_placement_ids,
            )
            return

        batch_id = raw_dev.batched_write(
            [int(offset) for offset in offsets],
            list(buffers),
            [int(total_len) for total_len in total_lens],
            per_write_placement_ids,
            [int(payload_len) for payload_len in payload_lens],
        )
        if not all(
            self.wait_for_batch(
                batch_id,
                len(offsets),
                "io_uring write",
            )
        ):
            raise RuntimeError("raw-block io_uring write failed")

    def read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> list[bool]:
        """Read buffers through the batched or io_uring_cmd path.

        Returns:
            A per-read success boolean list aligned with ``offsets``.
        """
        raw_dev = self.device()
        if self._use_uring_cmd:
            return self._read_uring_cmd_buffers(
                offsets,
                buffers,
                payload_lens,
                total_lens,
            )

        batch_id = raw_dev.batched_read(
            [int(offset) for offset in offsets],
            list(buffers),
            [int(total_len) for total_len in total_lens],
        )
        return self.wait_for_batch(
            batch_id,
            len(offsets),
            "io_uring read",
        )

    def register_io_buffers(
        self,
        buffer_ptrs: Sequence[int],
        buffer_sizes: Sequence[int],
    ) -> None:
        """Register stable memory ranges for io_uring zero-copy I/O."""
        self.device().register_fixed_buffers(list(buffer_ptrs), list(buffer_sizes))

    def supports_io_buffers(self) -> bool:
        """io_uring registers host memory for zero-copy I/O."""
        return True

    def acquire_buffers(self, count: int, size: int) -> list[Any]:
        """Lease ``count`` block-aligned buffers for a batched read.

        io_uring batched reads require block-aligned destination buffers under
        O_DIRECT, so each buffer is allocated aligned rather than as a plain
        ``bytearray``.

        Args:
            count: Number of buffers to lease.
            size: Number of writable bytes per buffer.

        Returns:
            A list of ``count`` block-aligned memoryviews.
        """
        return [self._allocate_aligned_buffer(size) for _ in range(count)]

    def _validate_uring_cmd_chunk(self, offset: int, total_len: int) -> None:
        """Validate one NVMe raw-command transfer range.

        Args:
            offset: Device byte offset for the transfer.
            total_len: Transfer size in bytes.

        Raises:
            ValueError: If either value is not block aligned.
        """
        if offset % self._block_align != 0:
            raise ValueError("io_uring_cmd requires aligned offsets")
        if total_len % self._block_align != 0:
            raise ValueError("io_uring_cmd requires aligned transfer lengths")

    def _allocate_aligned_buffer(self, length: int) -> memoryview:
        """Allocate a writable byte buffer aligned to the block boundary.

        Args:
            length: Number of bytes to expose through the returned memoryview.

        Returns:
            A memoryview whose starting address is aligned to ``block_align``.

        Raises:
            ValueError: If ``length`` is negative.
        """
        if length < 0:
            raise ValueError("length must be >= 0")
        if length == 0:
            return memoryview(bytearray())

        backing = bytearray(length + self._block_align - 1)
        ptr = ctypes.addressof((ctypes.c_byte * 1).from_buffer(backing))
        offset = (-ptr) % self._block_align
        return memoryview(backing)[offset : offset + length]

    def _write_uring_cmd_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> None:
        """Write buffers as bounded NVMe raw-command chunks.

        Args:
            offsets: Device offsets for each logical write.
            buffers: Source buffers.
            payload_lens: Logical source byte counts.
            total_lens: Physical transfer sizes, including padding.
            placement_ids: Optional FDP placement identifiers; ``None`` omits the
                directive and explicit identifier 0 is rejected.

        Raises:
            ValueError: If lengths are inconsistent or unaligned.
            Exception: Propagates Rust raw-device write errors.
        """
        raw_dev = self.device()
        chunk_offsets: list[int] = []
        chunk_buffers: list[memoryview] = []
        chunk_lens: list[int] = []
        chunk_placement_ids: list[PlacementId] = []
        keepalive: list[memoryview] = []
        per_write_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(offsets),
            field_name="placement_ids",
        )

        for offset, buf, payload_len, total_len, placement_id in zip(
            offsets,
            buffers,
            payload_lens,
            total_lens,
            per_write_placement_ids,
            strict=True,
        ):
            offset = int(offset)
            payload_len = int(payload_len)
            total_len = int(total_len)
            self._validate_uring_cmd_chunk(offset, total_len)

            view = _byte_view(buf)
            if len(view) < total_len:
                if len(view) < payload_len:
                    raise ValueError("input buffer shorter than payload_len")
                padded = self._allocate_aligned_buffer(total_len)
                padded[:payload_len] = view[:payload_len]
                view = padded
            else:
                view = view[:total_len]
            keepalive.append(view)

            cursor = 0
            while cursor < total_len:
                chunk_len = min(self._max_data_transfer_size, total_len - cursor)
                self._validate_uring_cmd_chunk(offset + cursor, chunk_len)
                chunk_offsets.append(offset + cursor)
                chunk_buffers.append(view[cursor : cursor + chunk_len])
                chunk_lens.append(chunk_len)
                chunk_placement_ids.append(placement_id)
                cursor += chunk_len

        if not chunk_offsets:
            return
        batch_id = raw_dev.batched_write(
            chunk_offsets,
            chunk_buffers,
            chunk_lens,
            chunk_placement_ids,
        )
        if not all(
            self.wait_for_batch(
                batch_id,
                len(chunk_offsets),
                "io_uring_cmd write",
            )
        ):
            raise RuntimeError("raw-block io_uring_cmd write failed")
        keepalive.clear()

    def _read_uring_cmd_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> list[bool]:
        """Read buffers as bounded NVMe raw-command chunks.

        Args:
            offsets: Device offsets for each logical read.
            buffers: Destination buffers.
            payload_lens: Logical bytes to expose to callers.
            total_lens: Physical transfer sizes, including padding.

        Returns:
            A per-logical-read success boolean list aligned with ``offsets``. If
            a submitted batch returns too few or too many completions, all
            submitted logical reads are reported as false.
        """
        raw_dev = self.device()
        results = [False] * len(offsets)
        chunk_offsets: list[int] = []
        chunk_buffers: list[memoryview] = []
        chunk_lens: list[int] = []
        chunk_logical_indices: list[int] = []
        chunk_statuses: list[list[bool]] = [[] for _ in offsets]
        copy_back_targets: dict[int, tuple[memoryview, memoryview, int]] = {}
        keepalive: list[Any] = []

        for logical_idx, (offset, buf, payload_len, total_len) in enumerate(
            zip(offsets, buffers, payload_lens, total_lens, strict=True)
        ):
            try:
                offset = int(offset)
                payload_len = int(payload_len)
                total_len = int(total_len)
                self._validate_uring_cmd_chunk(offset, total_len)

                dst = _byte_view(buf)
                if len(dst) < total_len:
                    if len(dst) < payload_len:
                        raise ValueError("output buffer shorter than payload_len")
                    target = self._allocate_aligned_buffer(total_len)
                    copy_back = True
                else:
                    target = dst[:total_len]
                    copy_back = False
                keepalive.append(target)

                cursor = 0
                max_chunk_len = (
                    self._max_data_transfer_size
                    if self._max_data_transfer_size > 0
                    else total_len
                )
                while cursor < total_len:
                    chunk_len = min(max_chunk_len, total_len - cursor)
                    self._validate_uring_cmd_chunk(offset + cursor, chunk_len)
                    chunk_offsets.append(offset + cursor)
                    chunk_buffers.append(target[cursor : cursor + chunk_len])
                    chunk_lens.append(chunk_len)
                    chunk_logical_indices.append(logical_idx)
                    cursor += chunk_len

                if copy_back:
                    copy_back_targets[logical_idx] = (dst, target, payload_len)
            except Exception:
                continue

        if not chunk_offsets:
            return results

        try:
            batch_id = raw_dev.batched_read(
                chunk_offsets,
                chunk_buffers,
                chunk_lens,
            )
            chunk_results = self.wait_for_batch(
                batch_id,
                len(chunk_offsets),
                "io_uring_cmd read",
            )
        except Exception:
            return results

        for chunk_idx, logical_idx in enumerate(chunk_logical_indices):
            ok = chunk_idx < len(chunk_results) and bool(chunk_results[chunk_idx])
            chunk_statuses[logical_idx].append(ok)

        for logical_idx, statuses in enumerate(chunk_statuses):
            if not statuses or not all(statuses):
                continue
            if logical_idx in copy_back_targets:
                dst, target, payload_len = copy_back_targets[logical_idx]
                dst[:payload_len] = target[:payload_len]
            results[logical_idx] = True

        keepalive.clear()
        return results


def make_raw_block_io_engine(cfg: RawBlockIoEngineConfig) -> RawBlockIoEngine:
    """Create the configured raw-block I/O engine from its config object.

    The engine is selected by the concrete config type, so adding a new engine
    requires only a new ``RawBlockIoEngineConfig`` subclass, a matching engine
    implementing :class:`RawBlockIoEngine`, and one ``isinstance`` branch here.

    Args:
        cfg: Normalized engine config describing the target engine and its
            parameters.

    Returns:
        A ``PosixIoEngine`` for a ``PosixIoEngineConfig`` or an
        ``IoUringIoEngine`` for an ``IoUringIoEngineConfig``.

    Raises:
        ValueError: If ``cfg`` does not describe a supported engine.
    """
    if isinstance(cfg, PosixIoEngineConfig):
        return PosixIoEngine.from_config(cfg)
    if isinstance(cfg, IoUringIoEngineConfig):
        return IoUringIoEngine.from_config(cfg)
    raise ValueError(f"Unsupported raw-block io engine config: {type(cfg).__name__}")
