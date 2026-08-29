# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
import ctypes

if TYPE_CHECKING:
    from lmcache.v1.storage_backend.raw_block.spdk_ffi import SpdkIoEngineFFI

# Standard
import json
import os
import re
import stat
import struct
import threading
import time
import zlib

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    DiskCacheMetadata,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.raw_block.buffer_pool import (
    CheckPointPayloadBufferPool,
    HeaderBufferPool,
)
from lmcache.v1.storage_backend.raw_block.key_codec import (
    RawBlockKeyNamespace,
    RawBlockKeySpec,
    decode_legacy_key,
    slot_identity_from_encoded_key,
)

logger = init_logger(__name__)


_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1
_META_HEADER_STRUCT = struct.Struct("<8sIQQI")
RAW_BLOCK_IO_ENGINES = frozenset({"posix", "io_uring", "spdk"})
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


# Module-level lock for serializing SPDK ctypes calls.
# SPDK's internal state may not be fully concurrent-safe, so we serialize
# access to C functions while still allowing GIL release for parallelism.
_spdk_call_lock = threading.Lock()


def _spdk_call_with_gil_released(func, *args):
    """Call an SPDK C function synchronously.

    This helper function invokes a SPDK ctypes call directly in the
    current thread, propagating any exceptions raised by the underlying
    C function.

    Args:
        func: ctypes CFunctype to invoke.
        *args: Arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        Exception: Re-raises any exception raised by the function.
    """
    result_container = [None]
    error_container = [None]

    def _call_wrapper():
        try:
            result_container[0] = func(*args)
        except Exception as e:
            error_container[0] = e

    _call_wrapper()

    if error_container[0] is not None:
        raise error_container[0]
    return result_container[0]


def round_up(x: int, align: int) -> int:
    """Round a value up to the nearest alignment boundary.

    Args:
        x: Value to align.
        align: Positive alignment in bytes.

    Returns:
        ``x`` rounded up to a multiple of ``align``.
    """
    return ((x + align - 1) // align) * align


def normalize_raw_block_io_engine(
    io_engine: Any = None,
    *,
    use_iouring: Any = None,
    use_uring: Any = None,
    use_spdk: Any = None,
) -> str:
    """Normalize raw-block I/O engine config with legacy compatibility.

    Args:
        io_engine: Explicit engine string. Valid values are ``"posix"``,
            ``"io_uring"``, and ``"spdk"``.
        use_iouring: Legacy boolean knob. Used only when ``io_engine`` is not
            set.
        use_uring: Legacy boolean alias. Used only when ``io_engine`` is not
            set.
        use_spdk: Legacy boolean knob. Used only when ``io_engine`` is not
            set. When True, returns ``"spdk"``.

    Returns:
        The normalized engine string.

    Raises:
        ValueError: If ``io_engine`` names an unsupported engine.
    """
    if io_engine is None or io_engine == "":
        if bool(use_spdk):
            return "spdk"
        if bool(use_iouring) or bool(use_uring):
            return "io_uring"
        return "posix"
    normalized = str(io_engine).lower()
    if normalized not in RAW_BLOCK_IO_ENGINES:
        allowed = ", ".join(sorted(RAW_BLOCK_IO_ENGINES))
        raise ValueError(f"io_engine must be one of: {allowed}")
    return normalized


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


def validate_raw_block_io_options(
    *,
    iouring_queue_depth: int,
) -> None:
    """Validate numeric raw-block I/O engine options.

    Args:
        iouring_queue_depth: Queue depth for the Rust io_uring path.

    Raises:
        ValueError: If any numeric option is not positive.
    """
    if int(iouring_queue_depth) <= 0:
        raise ValueError("iouring_queue_depth must be > 0")


def _resolve_sysfs_queue_dir(device_path: str) -> Optional[str]:
    """Resolve sysfs queue directory for NVMe character device paths."""
    base_name = os.path.basename(device_path)
    match = re.fullmatch(r"ng(\d+)n(\d+)", base_name)
    if match:
        ctrl, nsid = match.groups()
        return f"/sys/block/nvme{ctrl}n{nsid}/queue"
    return None


def _read_sysfs_int(path: str) -> Optional[int]:
    """Read an integer value from sysfs and return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


@dataclass(frozen=True)
class RawBlockCoreConfig:
    """Configuration for RawBlockCore device layout, I/O, and checkpoints."""

    device_path: str
    capacity_bytes: int
    block_align: int
    header_bytes: int
    slot_bytes: int
    use_odirect: bool
    enable_zero_copy: bool
    meta_total_bytes: int
    meta_magic: bytes
    meta_version: int
    meta_checkpoint_interval_sec: int
    meta_idle_quiet_ms: int
    meta_enable_periodic: bool
    meta_verify_on_load: bool
    max_data_transfer_size: int = 0
    load_checkpoint_on_init: bool = True
    io_engine: str = "posix"
    iouring_queue_depth: int = DEFAULT_IOURING_QUEUE_DEPTH
    use_uring_cmd: bool = False
    meta_checkpoint_placement_id: PlacementId = None
    fdp_slot_affinity_enabled: bool = False

    # SPDK-specific configuration (consumed when io_engine="spdk")
    spdk_transport_type: str = "tcp"  # "pcie" for local NVMe, "tcp" for NVMe-oF
    spdk_target_ip: str = "127.0.0.1"  # For PCIe: device address (e.g., "0000:01:00.0")
    spdk_target_port: str = "4420"
    spdk_target_nqn: str = "nqn.2019-04.pos:subsystem1"
    spdk_core_mask: str = ""  # Hex core mask for SPDK (e.g., "0x3f" for cores 0-5)
    spdk_mem_size_mb: int = 4096  # MB for SPDK hugepage memory allocation


@dataclass
class _Entry:
    offset: int
    size: int
    meta: DiskCacheMetadata


@dataclass
class _Inflight:
    offset: int
    meta: DiskCacheMetadata
    canceled: bool = False


@dataclass(frozen=True)
class RawBlockPutManyResult:
    """Result of a RawBlockCore batched write."""

    results: list[bool]
    stored_keys: list[str]


class RawBlockCore:
    """
    Shared raw-block storage engine used by both legacy non-MP and MP L2 paths.

    This class owns the raw-device I/O path, slot allocation, checkpoint/recovery,
    and lock refcounts that protect slots from deletion while in use.
    """

    def __init__(
        self,
        config: RawBlockCoreConfig,
        *,
        key_namespace: RawBlockKeyNamespace,
    ):
        """Initialize the raw-block storage engine.

        Args:
            config: Raw-block device, layout, I/O, and checkpoint settings.
            key_namespace: Encoding namespace used by keys stored in this core.

        Raises:
            ValueError: If the supplied configuration is invalid.
            RuntimeError: If the raw device cannot be opened or the computed
                layout cannot fit metadata and at least one data slot.

        Notes:
            If initialization opens the device and a later recovery step fails,
            the partially opened resources are closed before the exception is
            re-raised.
        """
        self.device_path = config.device_path
        self.capacity_bytes = int(config.capacity_bytes)
        self.block_align = int(config.block_align)
        self.header_bytes = int(config.header_bytes)
        self.slot_bytes = int(config.slot_bytes)
        self.use_odirect = bool(config.use_odirect)
        self.enable_zero_copy = bool(config.enable_zero_copy)

        self.meta_total_bytes = int(config.meta_total_bytes)
        self.meta_magic = bytes(config.meta_magic)
        self.meta_version = int(config.meta_version)
        self.meta_checkpoint_interval_sec = int(config.meta_checkpoint_interval_sec)
        self.meta_idle_quiet_ms = int(config.meta_idle_quiet_ms)
        self.meta_enable_periodic = bool(config.meta_enable_periodic)
        self.load_checkpoint_on_init = bool(config.load_checkpoint_on_init)
        self.meta_verify_on_load = bool(config.meta_verify_on_load)
        self.io_engine = normalize_raw_block_io_engine(config.io_engine)
        self.iouring_queue_depth = int(config.iouring_queue_depth)
        self.use_uring_cmd = bool(config.use_uring_cmd)

        # SPDK-specific configuration (consumed when io_engine="spdk")
        self.spdk_transport_type = str(config.spdk_transport_type)
        self.spdk_target_ip = str(config.spdk_target_ip)
        self.spdk_target_port = str(config.spdk_target_port)
        self.spdk_target_nqn = str(config.spdk_target_nqn)
        self.spdk_core_mask = str(config.spdk_core_mask)
        self.spdk_mem_size_mb = int(config.spdk_mem_size_mb)

        self.fdp_slot_affinity_enabled = bool(config.fdp_slot_affinity_enabled)
        self.meta_checkpoint_placement_id = normalize_raw_block_placement_ids(
            [config.meta_checkpoint_placement_id],
            1,
            field_name="meta_checkpoint_placement_id",
        )[0]
        if self.meta_checkpoint_placement_id is not None and (
            self.io_engine != "io_uring" or not self.use_uring_cmd
        ):
            raise ValueError(
                "meta_checkpoint_placement_id requires "
                "io_engine='io_uring' and use_uring_cmd=true"
            )
        if self.use_uring_cmd and self.use_odirect:
            logger.warning(
                "RawBlockCore: use_odirect is ignored for NVMe namespace "
                "character devices when use_uring_cmd=true"
            )
            self.use_odirect = False
        self.key_namespace = key_namespace

        # For SPDK mode, device_path is not required (SPDK manages NVMe connection)
        if not self.device_path and self.io_engine != "spdk":
            raise ValueError(
                "RawBlockCore requires a non-empty device_path when io_engine != 'spdk'"
            )

        if self.block_align <= 0 or (self.block_align & (self.block_align - 1)) != 0:
            raise ValueError(
                f"block_align must be a power of 2, got {self.block_align}"
            )
        if self.header_bytes < 24:
            raise ValueError("header_bytes must be >= 24")
        if self.header_bytes % self.block_align != 0:
            raise ValueError("header_bytes must be a multiple of block_align")
        if self.slot_bytes < self.header_bytes + 1:
            raise ValueError("slot_bytes must be >= header_bytes + 1")
        if self.slot_bytes % self.block_align != 0:
            raise ValueError("slot_bytes must be a multiple of block_align")
        if self.meta_total_bytes <= self.block_align:
            raise ValueError("meta_total_bytes must provide room for metadata header")
        if self.meta_total_bytes % self.block_align != 0:
            raise ValueError("meta_total_bytes must be a multiple of block_align")
        if len(self.meta_magic) != 8:
            raise ValueError("meta_magic must be exactly 8 bytes")
        if self.meta_version <= 0:
            raise ValueError("meta_version must be > 0")
        validate_raw_block_io_options(
            iouring_queue_depth=self.iouring_queue_depth,
        )
        if self.use_uring_cmd and self.io_engine != "io_uring":
            raise ValueError("use_uring_cmd requires io_uring as io_engine")
        if self.use_uring_cmd:
            try:
                mode = os.stat(self.device_path).st_mode
            except OSError as e:
                raise ValueError(
                    "use_uring_cmd requires an existing NVMe namespace "
                    f"character device path, got {self.device_path!r}"
                ) from e
            if not stat.S_ISCHR(mode):
                raise ValueError(
                    "use_uring_cmd requires an NVMe namespace character device "
                    f"(for example /dev/ng0n1), got {self.device_path!r}"
                )
            # Validate NVMe generic namespace naming pattern (ng<ctrl>n<ns>)
            basename = os.path.basename(self.device_path)
            if not re.match(r"^ng\d+n\d+$", basename):
                raise ValueError(
                    "use_uring_cmd requires an NVMe generic namespace character device "
                    f"with naming pattern ng<ctrl>n<ns> (for example /dev/ng0n1), "
                    f"got {self.device_path!r}"
                )

        # Maximum data transfer size for a single I/O request.
        # Default is 0 (no splitting).
        # > 0 : explicit manual split size
        # <= 0: opt-in auto-detect from device queue limits
        if self.use_uring_cmd:
            self.max_data_transfer_size = self._resolve_max_data_transfer_size(
                config.max_data_transfer_size
            )

        try:
            self.meta_magic_text = self.meta_magic.decode("ascii")
        except UnicodeDecodeError as e:
            raise ValueError("meta_magic must be ASCII bytes") from e

        self._meta_copy_count: int = 2
        self._meta_container_bytes: int = (
            (self.meta_total_bytes // self._meta_copy_count) // self.block_align
        ) * self.block_align
        if self._meta_container_bytes <= self.block_align:
            raise ValueError(
                "meta_total_bytes must provide room for at least two metadata copies"
            )

        self._lock = threading.Lock()
        self._index: dict[str, _Entry] = {}
        self._lock_refcnt: dict[str, int] = {}
        self._inflight: dict[str, _Inflight] = {}

        self._next_slot: int = 0
        self._free_slots: dict[int, None] = {}
        self._free_slots_by_placement_id: dict[int, dict[int, None]] = {}
        self._slot_placement_ids: dict[int, int] = {}
        self._fdp_slot_affinity_hit_count: int = 0
        self._fdp_slot_affinity_fallback_count: int = 0
        self._max_slots: int = 0
        self._effective_capacity_bytes: int = 0
        self._data_base_offset: int = 0

        self._raw = None
        self._closed = False

        # SPDK engine (initialized when io_engine="spdk")
        self._spdk_engine: Optional["SpdkIoEngineFFI"] = None
        # Registered external buffer for SPDK zero-copy DMA
        self._spdk_ext_buf_ptr: int = 0
        self._spdk_ext_buf_size: int = 0
        # Registered external buffer regions for _is_buffer_spdk_registered()
        self._registered_external_buffers: list[tuple[int, int]] = []

        self._meta_seq: int = 0
        self._meta_dirty_total: int = 0
        self._meta_persisted: int = 0
        self._inflight_io_count: int = 0
        self._last_io_ts: float = time.monotonic()
        self._meta_stop_evt = threading.Event()
        self._meta_thread: Optional[threading.Thread] = None

        try:
            # Initialize SPDK engine before capacity check if enabled
            # (capacity auto-detection requires the SPDK engine)
            if self.io_engine == "spdk":
                if self.spdk_transport_type == "tcp":
                    logger.debug(
                        "RawBlockCore: initializing SPDK engine for NVMe-oF "
                        "target %s:%s (NQN: %s)",
                        self.spdk_target_ip,
                        self.spdk_target_port,
                        self.spdk_target_nqn,
                    )
                else:
                    logger.debug(
                        "RawBlockCore: initializing SPDK engine for PCIe device %s",
                        self.spdk_target_ip,
                    )
                self._init_spdk_engine()
                # Create header buffer pool for zero-copy DMA writes
                self._header_pool: Optional[HeaderBufferPool] = HeaderBufferPool(
                    buffer_size=self.block_align,
                    pool_size=32,
                    spdk_engine=self._spdk_engine,
                )
                logger.debug(
                    "RawBlockCore: SPDK header buffer pool created "
                    "(buffers=%d size=%d)",
                    self._header_pool.pool_size,
                    self._header_pool.buffer_size,
                )
                # Create checkpoint payload buffer pool for zero-copy DMA writes
                self._checkpoint_pool: Optional[CheckPointPayloadBufferPool] = (
                    CheckPointPayloadBufferPool(
                        buffer_size=self._meta_payload_capacity(),
                        pool_size=2,
                        spdk_engine=self._spdk_engine,
                    )
                )
                logger.debug(
                    "RawBlockCore: SPDK checkpoint payload buffer pool created "
                    "(buffers=%d size=%d)",
                    self._checkpoint_pool.pool_size,
                    self._checkpoint_pool.buffer_size,
                )

            self._ensure_capacity_and_layout()
            if self.load_checkpoint_on_init:
                self._load_checkpoint_from_device()
            else:
                logger.info("RawBlockCore: skipping on-device metadata checkpoint load")

            if self.meta_enable_periodic:
                self._meta_thread = threading.Thread(
                    target=self._checkpoint_loop,
                    daemon=True,
                    name="raw-block-core-checkpoint",
                )
                self._meta_thread.start()

            # SPDK buffer pools are already created above
            # For non-SPDK modes, set pools to None
            if self.io_engine != "spdk":
                self._header_pool = None
                self._checkpoint_pool = None
        except Exception:
            self._cleanup_after_init_failure()
            raise

    @property
    def _requires_transfer_alignment(self) -> bool:
        """Return whether I/O transfers require block alignment.

        Returns:
            True when transfers must be aligned to ``self.block_align``.
            This is required for O_DIRECT I/O and for io_uring_cmd operations.
        """
        return self.use_odirect or self.use_uring_cmd

    def _resolve_max_data_transfer_size(self, configured_size: int) -> int:
        """Resolve transfer split size from config or NVMe sysfs queue limits.

        When auto-detecting, the size is bounded by both the device's
        ``max_hw_sectors_kb`` (total transfer size) and its ``max_segments``
        scatter-gather limit. The NVMe ``io_uring_cmd`` passthrough path builds
        one iovec segment per page, so a single transfer can consume up to
        ``ceil(len / page_size)`` segments. Exceeding ``max_segments`` makes the
        kernel reject the command with ``EINVAL``, so the resolved size is
        capped at ``max_segments * page_size``.

        Args:
            configured_size: Explicitly configured max data transfer size in bytes.
                If > 0, this value is used directly. If <= 0, the size is
                auto-detected from device queue limits.

        Returns:
            The resolved max data transfer size in bytes, guaranteed to be
            a multiple of ``self.block_align``.

        Raises:
            ValueError: If ``configured_size`` is > 0 but not a multiple of
                ``self.block_align``.
            RuntimeError: If sysfs queue limits cannot be resolved during
                auto-detection.
        """
        if configured_size > 0:
            if configured_size % self.block_align != 0:
                raise ValueError(
                    f"max_data_transfer_size ({configured_size}) must be a "
                    f"multiple of block_align ({self.block_align})"
                )
            return configured_size

        queue_dir = _resolve_sysfs_queue_dir(self.device_path)
        if queue_dir is None:
            raise RuntimeError(
                "RustRawBlockBackend: unable to derive NVMe sysfs queue path from "
                "NVMe character device path "
                f"{self.device_path} for auto max_data_transfer_size"
            )

        max_hw_sectors_kb = _read_sysfs_int(f"{queue_dir}/max_hw_sectors_kb")
        if max_hw_sectors_kb is None or max_hw_sectors_kb <= 0:
            raise RuntimeError(
                "RustRawBlockBackend: failed to read max_hw_sectors_kb from "
                f"{queue_dir} for auto max_data_transfer_size"
            )

        resolved_bytes = max_hw_sectors_kb * 1024

        # The io_uring_cmd passthrough path builds one iovec segment per page,
        # so cap the transfer at the device's scatter-gather segment limit.
        max_segments = _read_sysfs_int(f"{queue_dir}/max_segments")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if max_segments is not None and max_segments > 0:
            segment_limit_bytes = max_segments * page_size
            resolved_bytes = min(resolved_bytes, segment_limit_bytes)

        aligned_bytes = (resolved_bytes // self.block_align) * self.block_align
        if aligned_bytes <= 0:
            aligned_bytes = self.block_align

        logger.info(
            "RustRawBlockBackend: auto max_data_transfer_size=%d bytes "
            "(device=%s, max_hw_sectors_kb=%s, max_segments=%s, page_size=%d)",
            aligned_bytes,
            self.device_path,
            max_hw_sectors_kb,
            max_segments,
            page_size,
        )
        return aligned_bytes

    def _rawdev(self):
        """Return the lazily opened Rust raw-block device binding.

        Note: When SPDK is enabled (`io_engine="spdk"`), this returns None
        because SPDK manages the NVMe connection directly without
        needing the Rust raw-block device.
        """
        if self._raw is None:
            # For SPDK mode, skip RawBlockDevice - SPDK handles I/O directly
            if self.io_engine == "spdk":
                return None
            try:
                # Third Party
                from lmcache_rust_raw_block_io import RawBlockDevice  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Rust raw-block extension is not installed. "
                    "Install / build `rust_raw_block_io` and retry."
                ) from e
            # For SPDK mode, use posix since SPDK handles I/O directly
            # via its own NVMe driver layer, not through the Rust device
            raw_io_engine = "posix" if self.io_engine == "spdk" else self.io_engine
            self._raw = RawBlockDevice(
                self.device_path,
                writable=True,
                use_odirect=self.use_odirect,
                alignment=self.block_align,
                io_engine=raw_io_engine,
                iouring_queue_depth=self.iouring_queue_depth,
                use_uring_cmd=self.use_uring_cmd,
            )
        return self._raw

    def raw_device(self) -> Any:
        """Return the lazily opened Rust raw-block device.

        Returns:
            The underlying Rust ``RawBlockDevice`` object.

        Raises:
            Exception: Propagates raw-device open errors from the Rust binding.
        """
        return self._rawdev()

    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        """Fetch NVMe FDP placement/RUH status from the raw device.

        Returns:
            List of ``(placement_id, ruh_id)`` tuples.

        Raises:
            RuntimeError: If the raw device binding or target device cannot
                provide FDP status.
        """
        return [
            (int(pid), int(ruhid)) for pid, ruhid in self._rawdev().fetch_fdp_status()
        ]

    def set_raw_device_for_testing(self, raw_device: Any) -> None:
        """Replace the raw device handle used by this core.

        Args:
            raw_device: Object implementing the Rust raw-device methods.
        """
        self._raw = raw_device

    def _init_spdk_engine(self) -> None:
        """Initialize SPDK engine.

        Initializes the SPDK environment, connects to the NVMe device
        (either via PCIe or NVMe-oF TCP transport), and sets up
        the admin and I/O worker threads.

        Raises:
            RuntimeError: If SPDK engine initialization fails.
        """
        try:
            # First Party
            from lmcache.v1.storage_backend.raw_block.spdk_ffi import SpdkIoEngineFFI

            self._spdk_engine = SpdkIoEngineFFI()

            # Set SPDK memory size for hugepage allocation (must be called
            # before init() so SPDK reserves the correct amount of memory)
            self._spdk_engine.set_mem_size(self.spdk_mem_size_mb)

            # Set SPDK core mask if provided
            if self.spdk_core_mask:
                rc = self._spdk_engine.set_dpdk_core_mask(self.spdk_core_mask)
                if rc != 0:
                    raise RuntimeError(
                        f"Failed to set SPDK DPDK core mask to {self.spdk_core_mask}"
                    )

            # Initialize SPDK
            rc = self._spdk_engine.init()
            if rc != 0:
                raise RuntimeError("Failed to initialize SPDK environment")

            # Set CPU affinity for SPDK - critical for proper SPDK operation
            # Dynamically determine producer cores based on system topology
            total_cores = os.cpu_count() or 1
            all_cores = set(range(total_cores))

            # Determine cores to exclude (reserved for SPDK)
            reserved_cores: set[int] = set()

            if self.spdk_core_mask:
                # Parse the hex core mask to get reserved core indices
                try:
                    mask_value = int(self.spdk_core_mask, 16)
                    for bit in range(total_cores):
                        if mask_value & (1 << bit):
                            reserved_cores.add(bit)
                except ValueError:
                    logger.warning(
                        "RawBlockCore: invalid spdk_core_mask '%s', "
                        "using default reserved cores",
                        self.spdk_core_mask,
                    )
                    reserved_cores.update([max(1, total_cores - 1)])
            else:
                # Default: exclude only the last core (total_cores - 1)
                # Ensure at least core 0 is reserved to avoid negative index
                reserved_cores.update([max(1, total_cores - 1)])

            # Producer cores = all system cores minus reserved SPDK cores
            producer_cores = all_cores - reserved_cores

            if not producer_cores:
                raise RuntimeError(
                    "No producer cores available after excluding reserved cores "
                    f"(total={total_cores}, reserved={reserved_cores})"
                )

            logger.info(
                "RawBlockCore: CPU affinity set for SPDK workers "
                "(total_cores=%d, reserved=%s)",
                total_cores,
                sorted(reserved_cores),
            )
            os.sched_setaffinity(0, producer_cores)

            # Launch the I/O worker thread with connection parameters.
            # launch_io_worker internally calls core_set_connection_params
            # and core_launch_io_worker to handle both PCIe and TCP transport.
            rc = self._spdk_engine.launch_io_worker(
                transport_type=self.spdk_transport_type,
                addr=self.spdk_target_ip,
                port=self.spdk_target_port,
                nqn=self.spdk_target_nqn,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Failed to launch SPDK I/O worker "
                    f"(type={self.spdk_transport_type})"
                )

            if self.spdk_transport_type == "tcp":
                logger.info(
                    "RawBlockCore: SPDK engine initialized successfully "
                    "(NVMe-oF target=%s:%s)",
                    self.spdk_target_ip,
                    self.spdk_target_port,
                )
            else:
                logger.info(
                    "RawBlockCore: SPDK engine initialized successfully "
                    "(PCIe device=%s)",
                    self.spdk_target_ip,
                )
        except Exception as e:
            self._spdk_engine = None
            raise RuntimeError(f"Failed to initialize SPDK engine: {e}") from e

    def _cleanup_spdk_engine(self) -> None:
        """Clean up SPDK engine resources.

        Shuts down the I/O worker thread, disconnects from the NVMe device,
        and deinitializes the SPDK environment.
        """
        if self._spdk_engine is None:
            return

        try:
            # Shutdown I/O worker
            self._spdk_engine.shutdown_io_worker()

            # Deinitialize SPDK
            self._spdk_engine.deinit()

            logger.debug("RawBlockCore: SPDK engine cleaned up")
        except Exception as e:
            logger.warning("RawBlockCore: error cleaning up SPDK engine: %s", e)
        finally:
            self._spdk_engine = None

    def register_external_memory(self, ptr: int, size: int) -> bool:
        """Register an external memory buffer with SPDK for DMA.

        Args:
            ptr: Physical/virtual address of the buffer.
            size: Size of the buffer in bytes.

        Returns:
            True if registration succeeded, False otherwise.
        """
        if self._spdk_engine is None:
            logger.warning(
                "SPDK engine not initialized. Cannot register external memory."
            )
            return False

        try:
            rc = self._spdk_engine.register_external_memory(ptr, size)
            if rc == 0:
                self._registered_external_buffers.append((ptr, size))
                logger.info(
                    "RawBlockCore: registered external buffer with SPDK "
                    "(ptr=0x%x, size=%d)",
                    ptr,
                    size,
                )
                return True
            else:
                logger.error("RawBlockCore: SPDK registration failed with rc=%d", rc)
                return False
        except Exception as e:
            logger.error("RawBlockCore: SPDK registration exception: %s", e)
            return False

    def register_spdk_external_buffers(self, memory_allocator: Any) -> None:
        """Register LocalCPUBackend's hugepage-allocated buffer with SPDK.

        This method retrieves the main CPU buffer from the LocalCPUBackend's
        allocator (which was allocated with hugepages when SPDK is enabled)
        and registers it with SPDK for zero-copy DMA operations.

        Args:
            memory_allocator: Local CPU allocator that exposes
                ``get_spdk_buffer()`` and returns the hugepage-allocated buffer.

        Raises:
            RuntimeError: If SPDK engine is not initialized or buffer
                registration fails.
        """
        if not self.io_engine == "spdk":
            return
        if self._spdk_engine is None:
            raise RuntimeError(
                "SPDK engine not initialized. Cannot register external buffer."
            )

        get_spdk_buffer = getattr(memory_allocator, "get_spdk_buffer", None)
        if not callable(get_spdk_buffer):
            logger.warning(
                "RawBlockCore: allocator does not expose get_spdk_buffer(); "
                "SPDK external-buffer zero-copy is disabled"
            )
            return

        buffer = get_spdk_buffer()
        if buffer is None:
            logger.warning(
                "RawBlockCore: allocator returned None for get_spdk_buffer(); "
                "SPDK external-buffer zero-copy is disabled"
            )
            return

        ptr = int(buffer.data_ptr())
        size = buffer.numel() * buffer.element_size()

        if not self.register_external_memory(ptr, size):
            raise RuntimeError(
                f"Failed to register external buffer with SPDK (ptr=0x{ptr:x}, "
                f"size={size}). SPDK DMA operations will not use "
                "zero-copy mode."
            )

    def register_fixed_buffers_from_allocator(self, memory_allocator: Any) -> None:
        """Register allocator pages with io_uring when the allocator exposes them.

        Args:
            memory_allocator: Local CPU allocator that may expose
                ``get_paged_buffers()``.

        Raises:
            Exception: Propagates Rust registration errors after logging.
        """
        if self.io_engine != "io_uring":
            return
        paged_buffers = getattr(memory_allocator, "get_paged_buffers", None)
        if not callable(paged_buffers):
            logger.warning(
                "RawBlockCore: allocator does not expose paged buffers; "
                "io_uring fixed-buffer zero-copy is disabled"
            )
            return
        buffers = paged_buffers()
        if not buffers:
            logger.warning(
                "RawBlockCore: allocator returned no paged buffers; "
                "io_uring fixed-buffer zero-copy is disabled"
            )
            return
        buffer_ptrs = [buf.data_ptr() for buf in buffers]
        buffer_sizes = [buf.numel() * buf.element_size() for buf in buffers]
        self._rawdev().register_fixed_buffers(buffer_ptrs, buffer_sizes)
        logger.info(
            "RawBlockCore: registered %d paged buffers for io_uring fixed I/O",
            len(buffers),
        )

    def contains_key(self, encoded_key: str, *, lock: bool = False) -> bool:
        """Return whether one encoded key is present in the raw-block index.

        Args:
            encoded_key: Encoded raw-block key string.
            lock: If true, increment the key's L2 lock refcount on hit.

        Returns:
            True when the key is indexed and available for load.
        """
        return self.exists_many([encoded_key], lock=lock)[0]

    def exists_inflight(self, encoded_key: str) -> bool:
        """Return whether a key currently has an in-flight write.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            True when the key is being written but not committed yet.
        """
        with self._lock:
            return encoded_key in self._inflight

    def get_metadata_many(
        self, encoded_keys: Sequence[str]
    ) -> list[DiskCacheMetadata | None]:
        """Return metadata for encoded keys without loading payload bytes.

        Args:
            encoded_keys: Ordered encoded raw-block keys to inspect.

        Returns:
            A metadata-or-None list aligned with ``encoded_keys``.
        """
        with self._lock:
            metas: list[DiskCacheMetadata | None] = []
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                metas.append(entry.meta if entry is not None else None)
            return metas

    def get_metadata_prefix(
        self,
        encoded_keys: Sequence[str],
        *,
        lock: bool = False,
        skip_locked: set[str] | None = None,
    ) -> list[DiskCacheMetadata]:
        """Return leading-hit metadata and optionally lock those entries.

        Args:
            encoded_keys: Ordered encoded raw-block keys to inspect.
            lock: If true, increment L2 lock refcounts for every returned
                metadata entry while holding the index lock.
            skip_locked: Encoded keys that are already protected by the caller
                and should not receive an additional lock refcount.

        Returns:
            Metadata for the contiguous leading hit prefix. The returned list
            stops at the first missing key.
        """
        with self._lock:
            metas: list[DiskCacheMetadata] = []
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                if entry is None:
                    break
                metas.append(entry.meta)
                if lock and (skip_locked is None or encoded_key not in skip_locked):
                    self._lock_refcnt[encoded_key] = (
                        self._lock_refcnt.get(encoded_key, 0) + 1
                    )
            return metas

    def first_encoded_key(self) -> str | None:
        """Return one indexed encoded key for diagnostics.

        Returns:
            The first indexed key according to dictionary iteration order, or
            None if the recovered/indexed metadata is empty.
        """
        with self._lock:
            return next(iter(self._index), None)

    def lock_refcount(self, encoded_key: str) -> int:
        """Return the L2 lock refcount for an encoded key.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            Current lock refcount, or zero when the key is unlocked or absent.
        """
        with self._lock:
            return int(self._lock_refcnt.get(encoded_key, 0))

    def inflight_io_count(self) -> int:
        """Return the number of currently active raw-device I/O operations."""
        with self._lock:
            return int(self._inflight_io_count)

    def indexed_key_count(self) -> int:
        """Return the number of entries currently present in the key index."""
        with self._lock:
            return len(self._index)

    def snapshot_indexed_keys(self) -> list[str]:
        """Return a detached snapshot of encoded keys currently in the index."""
        with self._lock:
            return list(self._index.keys())

    def entry_offset(self, encoded_key: str) -> int | None:
        """Return the raw-device slot offset for an indexed key.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            Slot offset in bytes, or None when the key is not indexed.
        """
        with self._lock:
            entry = self._index.get(encoded_key)
            return None if entry is None else int(entry.offset)

    def metadata_container_offsets(self) -> list[int]:
        """Return checkpoint metadata container offsets in bytes."""
        return self._meta_container_offsets()

    def data_base_offset(self) -> int:
        """Return the byte offset where raw-block data slots begin."""
        return int(self._data_base_offset)

    def put_many(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> RawBlockPutManyResult:
        """Persist a batch of memory objects into raw-block slots.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``.
            objs: Memory objects whose byte buffers should be written.
            placement_ids: Optional per-key FDP placement identifiers for
                raw-block writes. ``None`` omits the directive; explicit identifier
                0 is rejected because default writes already use that mapping.

        Returns:
            Per-key success results and newly stored encoded keys. If no free
            raw-block slot is available, that key is reported as failed; slot
            reclamation is owned by the adapter/controller calling
            ``delete_many``.

        Raises:
            ValueError: If either sequence is empty, sequence lengths do not
                match, or a placement identifier is 0.
        """
        if not keys or not objs:
            raise ValueError("keys and objs must be non-empty")
        if len(keys) != len(objs):
            raise ValueError("keys and objs must have the same length")
        per_key_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(keys),
            field_name="placement_ids",
        )

        results = [False] * len(keys)
        stored_keys: list[str] = []

        for i, (key, obj) in enumerate(zip(keys, objs, strict=False)):
            placement_id = per_key_placement_ids[i]
            if self._closed:
                break

            with self._lock:
                if key.encoded in self._index:
                    results[i] = True
                    continue
                if key.encoded in self._inflight:
                    continue

                try:
                    offset = self._allocate_slot_locked(placement_id)
                except RuntimeError:
                    logger.warning(
                        "RawBlockCore: no free slot available for key %s",
                        key.encoded,
                    )
                    continue

                meta = DiskCacheMetadata(
                    path=f"{self.device_path}@{offset}",
                    size=len(obj.byte_array),
                    shape=obj.metadata.shape,
                    dtype=obj.metadata.dtype,
                    cached_positions=obj.metadata.cached_positions,
                    fmt=obj.metadata.fmt,
                    pin_count=0,
                )
                self._inflight[key.encoded] = _Inflight(offset=offset, meta=meta)

            success = self._write_one(key, obj, offset, placement_id=placement_id)

            with self._lock:
                inflight = self._inflight.pop(key.encoded, None)
                if inflight is None:
                    results[i] = False
                    continue
                if inflight.canceled or not success:
                    self._append_free_slot_locked(
                        self._offset_to_slot(int(inflight.offset))
                    )
                    self._meta_dirty_total += 1
                    results[i] = False
                    continue

                self._index[key.encoded] = _Entry(
                    offset=inflight.offset,
                    size=inflight.meta.size,
                    meta=inflight.meta,
                )
                self._meta_dirty_total += 1
                results[i] = True
                stored_keys.append(key.encoded)

        return RawBlockPutManyResult(
            results=results,
            stored_keys=stored_keys,
        )

    def exists_many(
        self,
        encoded_keys: Sequence[str],
        *,
        lock: bool = False,
    ) -> list[bool]:
        """Return a full hit bitmap as booleans for encoded keys.

        Args:
            encoded_keys: Ordered encoded raw-block keys to check.
            lock: If true, increment L2 lock refcounts for every hit.

        Returns:
            A list of booleans aligned with ``encoded_keys``.
        """
        results: list[bool] = []
        with self._lock:
            for encoded_key in encoded_keys:
                found = encoded_key in self._index
                results.append(found)
                if found and lock:
                    self._lock_refcnt[encoded_key] = (
                        self._lock_refcnt.get(encoded_key, 0) + 1
                    )
        return results

    def load_many_into(
        self,
        encoded_keys: Sequence[str],
        objs: Sequence[MemoryObj],
        *,
        raise_on_error: bool = False,
    ) -> list[bool]:
        """Load raw-block payloads into caller-provided memory objects.

        Args:
            encoded_keys: Ordered encoded raw-block keys to load.
            objs: Destination memory objects. Buffers must remain valid until
                this method returns.
            raise_on_error: If true, re-raise the first load exception instead
                of logging it and returning ``False`` for that key.

        Returns:
            A list of per-key load success booleans aligned with
            ``encoded_keys``.

        Raises:
            ValueError: If either sequence is empty or the sequence lengths do
                not match.
            Exception: Re-raises load errors when ``raise_on_error`` is true.
        """
        if not encoded_keys or not objs:
            raise ValueError("encoded_keys and objs must be non-empty")
        if len(encoded_keys) != len(objs):
            raise ValueError("encoded_keys and objs must have the same length")

        with self._lock:
            items = [
                (encoded_key, self._index.get(encoded_key))
                for encoded_key in encoded_keys
            ]
            self._inflight_io_count += 1

        results = [False] * len(encoded_keys)
        try:
            for i, (encoded_key, entry) in enumerate(items):
                if entry is None:
                    continue
                try:
                    payload_len = int(entry.size)
                    total_len = (
                        round_up(payload_len, self.block_align)
                        if self._requires_transfer_alignment
                        else payload_len
                    )
                    buf = memoryview(objs[i].byte_array)
                    try:
                        buf = buf.cast("B")
                    except Exception:
                        pass

                    direct_view = self._build_direct_odirect_view(
                        memory_obj=objs[i],
                        payload_len=payload_len,
                        total_len=total_len,
                        buffer_len=len(buf),
                        zero_tail=False,
                    )
                    if direct_view is not None:
                        self._read_buffers(
                            [entry.offset + self.header_bytes],
                            [direct_view],
                            [
                                total_len
                                if len(direct_view) >= total_len
                                else payload_len
                            ],
                            [total_len],
                        )
                    else:
                        self._read_buffers(
                            [entry.offset + self.header_bytes],
                            [buf],
                            [payload_len],
                            [total_len],
                        )
                    objs[i].metadata.cached_positions = entry.meta.cached_positions
                    results[i] = True
                except Exception as e:
                    if raise_on_error:
                        raise
                    logger.error("RawBlockCore load failed for %s: %s", encoded_key, e)
        finally:
            with self._lock:
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()
        return results

    def unlock_many(self, encoded_keys: Sequence[str]) -> None:
        """Release L2 lock references for encoded keys.

        Args:
            encoded_keys: Encoded raw-block keys whose lock refcounts should be
                decremented. Missing keys and underflow are treated as no-ops.
        """
        with self._lock:
            for encoded_key in encoded_keys:
                refcnt = self._lock_refcnt.get(encoded_key, 0)
                if refcnt <= 1:
                    self._lock_refcnt.pop(encoded_key, None)
                else:
                    self._lock_refcnt[encoded_key] = refcnt - 1

    def delete_many(
        self,
        encoded_keys: Sequence[str],
        *,
        force: bool = False,
    ) -> list[bool]:
        """Delete indexed keys and recycle their slots when allowed.

        Args:
            encoded_keys: Ordered encoded raw-block keys to delete.
            force: If true, delete locked keys as well. Normal MP eviction uses
                false so locked entries are preserved.

        Returns:
            A list of per-key deletion booleans aligned with ``encoded_keys``.
        """
        deleted: list[bool] = []
        with self._lock:
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                locked = self._lock_refcnt.get(encoded_key, 0) > 0
                if entry is not None and locked and not force:
                    deleted.append(False)
                    continue

                removed_entry = self._index.pop(encoded_key, None)
                inflight = self._inflight.get(encoded_key)
                if inflight is not None:
                    inflight.canceled = True
                self._lock_refcnt.pop(encoded_key, None)
                if removed_entry is not None:
                    self._append_free_slot_locked(
                        self._offset_to_slot(int(removed_entry.offset))
                    )
                    self._meta_dirty_total += 1
                deleted.append(removed_entry is not None or inflight is not None)
        return deleted

    def usage(self) -> tuple[float, float]:
        """Return current raw-block slot usage fractions.

        Returns:
            ``(current_usage, projected_usage)``. Raw-block has no separate
            projected value, so both values are identical. ``(-1.0, -1.0)``
            indicates that usable capacity is unknown.
        """
        with self._lock:
            usable_capacity = self._max_slots * self.slot_bytes
            if usable_capacity <= 0:
                return (-1.0, -1.0)
            used_slots = len(self._index) + len(self._inflight)
            usage = (used_slots * self.slot_bytes) / usable_capacity
            return (usage, usage)

    def checkpoint_now(self) -> None:
        """Synchronously write a metadata checkpoint."""
        self._checkpoint_once(force=True)

    def apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Validate and apply a recovered metadata checkpoint payload.

        Args:
            data: Decoded checkpoint dictionary.

        Returns:
            True when the payload shape and layout match this core and all
            valid entries were applied. Invalid per-entry records are skipped.
        """
        return self._apply_loaded_state(data)

    def report_status(self) -> dict:
        """Return raw-block health, layout, metadata, and in-flight counters."""
        with self._lock:
            return {
                "is_healthy": not self._closed,
                "type": "RawBlockCore",
                "key_namespace": self.key_namespace,
                "device_path": self.device_path,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "slot_bytes": self.slot_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "usable_capacity_bytes": self._max_slots * self.slot_bytes,
                "indexed_key_count": len(self._index),
                "inflight_key_count": len(self._inflight),
                "locked_key_count": sum(
                    1 for refcnt in self._lock_refcnt.values() if refcnt > 0
                ),
                "free_slot_count": len(self._free_slots),
                "next_slot": self._next_slot,
                "max_slots": self._max_slots,
                "metadata_seq": self._meta_seq,
                "metadata_dirty_total": self._meta_dirty_total,
                "metadata_persisted": self._meta_persisted,
                "inflight_io_count": self._inflight_io_count,
                "use_odirect": self.use_odirect,
                "enable_zero_copy": self.enable_zero_copy,
                "io_engine": self.io_engine,
                "iouring_queue_depth": self.iouring_queue_depth,
                "use_uring_cmd": self.use_uring_cmd,
                "fdp_slot_affinity_enabled": self.fdp_slot_affinity_enabled,
                "fdp_slot_affinity_hit_count": (self._fdp_slot_affinity_hit_count),
                "fdp_slot_affinity_fallback_count": (
                    self._fdp_slot_affinity_fallback_count
                ),
            }

    def close(self) -> None:
        """Stop checkpointing, write a final checkpoint, and close the device."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None

        try:
            self._checkpoint_once(force=True)
        except Exception as e:
            logger.warning("RawBlockCore final checkpoint failed: %s", e)

        if self._raw is not None:
            try:
                self._raw.close()
            except Exception as e:
                logger.warning(
                    "Failed to close raw block device %s: %s", self.device_path, e
                )
            finally:
                self._raw = None

    def _cleanup_after_init_failure(self) -> None:
        """Clean up resources when initialization fails.

        This is called from the except block in ``__init__`` to ensure
        partial resources are cleaned up before re-raising the exception.

        For SPDK mode, only SPDK resources are cleaned up.
        For non-SPDK mode, raw device and thread resources are cleaned up.
        """
        if self.io_engine == "spdk":
            if self._spdk_engine is not None:
                try:
                    self._cleanup_spdk_engine()
                except Exception:
                    pass

            for attr in ("_header_pool", "_checkpoint_pool"):
                pool = getattr(self, attr, None)
                if pool is not None and hasattr(pool, "cleanup"):
                    try:
                        pool.cleanup()
                    except Exception:
                        pass
            return

        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception as e:
                logger.warning(
                    "Failed to close raw block device %s: %s", self.device_path, e
                )
            finally:
                self._raw = None
        self._closed = True

    def _byte_view(self, buf: Any) -> memoryview:
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

    def _is_buffer_aligned(self, buf: Any) -> bool:
        """Check if a buffer is aligned to the block alignment boundary.

        Args:
            buf: Object implementing the Python buffer protocol.

        Returns:
            True if the buffer is aligned, False otherwise.
        """
        if not self.use_odirect:
            return True
        view = self._byte_view(buf)
        # Check if the buffer pointer is aligned
        ptr = ctypes.addressof((ctypes.c_byte * 1).from_buffer(view))
        return ptr % self.block_align == 0

    def _allocate_aligned_buffer(self, length: int) -> memoryview:
        """Allocate a writable byte buffer aligned to ``self.block_align``.

        Args:
            length: Number of bytes to expose through the returned memoryview.

        Returns:
            A memoryview whose starting address is aligned to ``self.block_align``.

        Raises:
            ValueError: If ``length`` is negative.
        """
        if length < 0:
            raise ValueError("length must be >= 0")
        if length == 0:
            return memoryview(bytearray())

        backing = bytearray(length + self.block_align - 1)
        ptr = ctypes.addressof((ctypes.c_byte * 1).from_buffer(backing))
        offset = (-ptr) % self.block_align
        return memoryview(backing)[offset : offset + length]

    def _build_direct_odirect_view(
        self,
        memory_obj: MemoryObj,
        payload_len: int,
        total_len: int,
        buffer_len: int,
        *,
        zero_tail: bool,
    ) -> Optional[memoryview]:
        """Build an aligned memoryview for direct O_DIRECT I/O when possible.

        Args:
            memory_obj: Memory object whose backing allocation may be aligned.
            payload_len: Logical payload length in bytes.
            total_len: I/O length after any O_DIRECT padding.
            buffer_len: Available buffer length in bytes.
            zero_tail: Whether to zero any padded tail bytes before writing.

        Returns:
            A direct memoryview over the allocation, or None when the memory
            object is unsuitable for direct I/O.
        """
        if not self.use_odirect or not self.enable_zero_copy:
            return None

        ptr_val = getattr(memory_obj, "data_ptr", None)
        if callable(ptr_val):
            try:
                ptr_val = ptr_val()
            except Exception:
                ptr_val = None
        if ptr_val is None:
            return None
        if buffer_len <= 0:
            return None

        ptr = int(ptr_val)
        if ptr <= 0 or ptr % self.block_align != 0:
            return None
        if buffer_len < payload_len:
            return None

        view_len = min(buffer_len, total_len)
        if view_len < payload_len:
            return None

        try:
            raw = (ctypes.c_ubyte * view_len).from_address(ptr)
            view = memoryview(raw)
            if zero_tail and total_len > payload_len and view_len >= total_len:
                ctypes.memset(ptr + payload_len, 0, total_len - payload_len)
            return view
        except Exception:
            return None

    def _prepare_write_payload(self, memory_obj: MemoryObj) -> tuple[Any, int, int]:
        """Prepare the payload buffer and lengths for a raw-block write.

        Args:
            memory_obj: Source object to persist.

        Returns:
            A tuple of ``(buffer, payload_len, total_len)`` where ``total_len``
            includes any O_DIRECT padding.

        Raises:
            RuntimeError: If the aligned payload would exceed slot capacity.
        """
        buf = memory_obj.byte_array
        if hasattr(buf, "cast"):
            buf = buf.cast("B")
        payload_len = len(memory_obj.byte_array)
        payload_capacity = self.slot_bytes - self.header_bytes
        if payload_len > payload_capacity:
            raise RuntimeError(
                f"RawBlockCore payload {payload_len} exceeds slot capacity "
                f"{payload_capacity}"
            )
        total_len = payload_len
        if self._requires_transfer_alignment:
            total_len = round_up(payload_len, self.block_align)
            if total_len > payload_capacity:
                raise RuntimeError(
                    f"Aligned payload {total_len} exceeds slot capacity "
                    f"{payload_capacity}"
                )
            direct_view = self._build_direct_odirect_view(
                memory_obj=memory_obj,
                payload_len=payload_len,
                total_len=total_len,
                buffer_len=len(buf),
                zero_tail=True,
            )
            if direct_view is not None:
                buf = direct_view
        return buf, payload_len, total_len

    def _validate_uring_cmd_chunk(self, offset: int, total_len: int) -> None:
        """Validate one NVMe raw-command transfer range.

        Args:
            offset: Device byte offset for the transfer.
            total_len: Transfer size in bytes.

        Raises:
            ValueError: If either value is not block aligned.
        """
        if offset % self.block_align != 0:
            raise ValueError("io_uring_cmd requires aligned offsets")
        if total_len % self.block_align != 0:
            raise ValueError("io_uring_cmd requires aligned transfer lengths")

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
            placement_ids: Optional FDP placement identifiers for each logical
                write. ``None`` omits the directive; explicit identifier 0 is
                rejected.

        Raises:
            ValueError: If lengths are inconsistent or unaligned.
            Exception: Propagates Rust raw-device write errors.
        """
        raw_dev = self._rawdev()
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

            view = self._byte_view(buf)
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
                chunk_len = min(self.max_data_transfer_size, total_len - cursor)
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
        raw_dev.wait_iouring(batch_id)
        keepalive.clear()

    def _read_uring_cmd_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> None:
        """Read buffers as bounded NVMe raw-command chunks.

        Args:
            offsets: Device offsets for each logical read.
            buffers: Destination buffers.
            payload_lens: Logical bytes to expose to callers.
            total_lens: Physical transfer sizes, including padding.

        Raises:
            ValueError: If lengths are inconsistent or unaligned.
            Exception: Propagates Rust raw-device read errors.
        """
        raw_dev = self._rawdev()
        read_uring = raw_dev.read_uring

        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            offset = int(offset)
            payload_len = int(payload_len)
            total_len = int(total_len)
            self._validate_uring_cmd_chunk(offset, total_len)

            dst = self._byte_view(buf)
            if len(dst) < total_len:
                if len(dst) < payload_len:
                    raise ValueError("output buffer shorter than payload_len")
                target = self._allocate_aligned_buffer(total_len)
                copy_back = True
            else:
                target = dst[:total_len]
                copy_back = False

            cursor = 0
            while cursor < total_len:
                chunk_len = min(self.max_data_transfer_size, total_len - cursor)
                self._validate_uring_cmd_chunk(offset + cursor, chunk_len)
                read_uring(
                    offset + cursor,
                    target[cursor : cursor + chunk_len],
                    chunk_len,
                    chunk_len,
                )
                cursor += chunk_len

            if copy_back:
                dst[:payload_len] = target[:payload_len]

    def _is_buffer_spdk_registered(self, buf_ptr: int) -> bool:
        """Check if a buffer pointer is within registered memory regions.

        This checks both externally-registered buffers (hugepage memory from
        LocalCPUBackend) and internally-allocated SPDK DMA buffers (header
        pool, checkpoint pool).

        Args:
            buf_ptr: Buffer pointer to check.

        Returns:
            True if the buffer is within a registered region, False otherwise.
        """
        if hasattr(self, "_registered_external_buffers"):
            for reg_ptr, reg_size in self._registered_external_buffers:
                if reg_ptr <= buf_ptr < reg_ptr + reg_size:
                    return True

        if hasattr(self, "_header_pool") and self._header_pool is not None:
            for reg_ptr, reg_size in self._header_pool._spdk_ptrs:
                if reg_ptr <= buf_ptr < reg_ptr + reg_size:
                    return True

        if hasattr(self, "_checkpoint_pool") and self._checkpoint_pool is not None:
            for reg_ptr, reg_size in self._checkpoint_pool._spdk_ptrs:
                if reg_ptr <= buf_ptr < reg_ptr + reg_size:
                    return True

        return False

    def _write_spdk_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> None:
        """Write buffers using SPDK engine.

        The GIL is released during each SPDK I/O call to allow other Python
        threads to execute in parallel.

        Args:
            offsets: Device byte offsets for each write.
            buffers: Python buffers to write.
            payload_lens: Logical payload lengths for each buffer.
            total_lens: Physical I/O byte counts for each buffer.

        Raises:
            RuntimeError: If SPDK engine is not initialized.
            Exception: Propagates SPDK I/O errors.
        """
        if self._spdk_engine is None:
            raise RuntimeError("SPDK engine not initialized")

        ffi = self._spdk_engine

        # Perform writes using SPDK (pass byte offsets and byte counts)
        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            # Get the raw pointer from the buffer for SPDK DMA
            if hasattr(buf, "data_ptr"):
                buf_ptr = int(buf.data_ptr())
            elif hasattr(buf, "__array_interface__"):
                buf_ptr = int(buf.__array_interface__["data"][0])
            elif isinstance(buf, ctypes._Pointer):
                buf_ptr = ctypes.addressof(buf)
            elif isinstance(buf, (memoryview, bytearray, ctypes.Array)):
                buf_ptr = ctypes.addressof((ctypes.c_ubyte * len(buf)).from_buffer(buf))
            else:
                buf_ptr = 0

            # Check if buffer is registered with SPDK for zero-copy I/O
            is_registered = self._is_buffer_spdk_registered(buf_ptr)

            if is_registered:
                # Zero-copy path: buffer is already registered with SPDK
                # Release GIL during SPDK write I/O
                rc = _spdk_call_with_gil_released(
                    ffi.spdk_write_external,
                    offset,
                    total_len,
                    buf_ptr,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"SPDK write failed at byte offset {offset}, "
                        f"byte_count={total_len}, rc={rc}"
                    )
            else:
                # Temporary DMA buffer path: copy data to DMA memory first
                dma_ptr = ffi.allocate_spdk_memory(total_len, 4096, numa_id=-1)
                if dma_ptr == 0:
                    raise RuntimeError(
                        f"Failed to allocate SPDK DMA buffer for write "
                        f"(offset={offset}, size={total_len})"
                    )

                try:
                    # Copy data from source buffer to DMA memory
                    if hasattr(buf, "data_ptr"):
                        src_ptr = int(buf.data_ptr())
                    elif hasattr(buf, "__array_interface__"):
                        src_ptr = int(buf.__array_interface__["data"][0])
                    elif isinstance(buf, ctypes._Pointer):
                        src_ptr = ctypes.addressof(buf)
                    else:
                        # Both Python buffers and ctypes.Array support len()
                        buf_len = len(buf)
                        src_ptr = ctypes.addressof(
                            (ctypes.c_ubyte * min(buf_len, payload_len)).from_buffer(
                                self._byte_view(buf)
                            )
                        )

                    # Copy data to DMA buffer
                    dma_buf = ctypes.cast(
                        dma_ptr, ctypes.POINTER(ctypes.c_ubyte * payload_len)
                    )
                    dma_buf.contents[:] = list(  # type: ignore[index]
                        ctypes.cast(
                            src_ptr, ctypes.POINTER(ctypes.c_ubyte * payload_len)
                        ).contents
                    )

                    # Release GIL during SPDK write I/O
                    rc = _spdk_call_with_gil_released(
                        ffi.spdk_write_external,
                        offset,
                        total_len,
                        dma_ptr,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"SPDK write failed at byte offset {offset}, "
                            f"byte_count={total_len}, rc={rc}"
                        )
                finally:
                    ffi.free_spdk_memory(dma_ptr)

    def _read_spdk_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> None:
        """Read buffers using SPDK engine.

        The GIL is released during each SPDK I/O call to allow other Python
        threads to execute in parallel.

        Args:
            offsets: Device byte offsets for each read.
            buffers: Destination Python buffers.
            payload_lens: Logical payload lengths to expose to callers.
            total_lens: Physical I/O byte counts for each read.

        Raises:
            RuntimeError: If SPDK engine is not initialized.
            Exception: Propagates SPDK I/O errors.
        """
        if self._spdk_engine is None:
            raise RuntimeError("SPDK engine not initialized")

        ffi = self._spdk_engine

        # Perform reads using SPDK (one at a time, synchronous)
        # Pass byte offsets and byte counts - C++ code handles LBA conversion
        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            # Get buffer pointer to determine if it's registered
            if hasattr(buf, "data_ptr"):
                buf_ptr = int(buf.data_ptr())
            elif hasattr(buf, "__array_interface__"):
                buf_ptr = int(buf.__array_interface__["data"][0])
            elif isinstance(buf, ctypes._Pointer):
                buf_ptr = ctypes.addressof(buf)
            elif isinstance(buf, (memoryview, bytearray)):
                buf_ptr = ctypes.addressof((ctypes.c_ubyte * len(buf)).from_buffer(buf))
            else:
                buf_ptr = 0

            # Check if buffer is registered with SPDK for zero-copy I/O
            is_registered = self._is_buffer_spdk_registered(buf_ptr)

            if is_registered:
                # Zero-copy path: buffer is already registered with SPDK
                if len(self._byte_view(buf)) < payload_len:
                    raise ValueError("output buffer shorter than payload_len")

                rc = _spdk_call_with_gil_released(
                    ffi.spdk_read_external,
                    offset,
                    total_len,
                    buf_ptr,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"SPDK read failed at byte offset {offset}, "
                        f"byte_count={total_len}, rc={rc}"
                    )
            else:
                # Temporary DMA buffer path: allocate, read, copy back
                dma_ptr = ffi.allocate_spdk_memory(total_len, 4096, numa_id=-1)
                if dma_ptr == 0:
                    raise RuntimeError(
                        f"Failed to allocate SPDK DMA buffer for read "
                        f"(offset={offset}, size={total_len})"
                    )

                try:
                    rc = _spdk_call_with_gil_released(
                        ffi.spdk_read_external,
                        offset,
                        total_len,
                        dma_ptr,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"SPDK read failed at byte offset {offset}, "
                            f"byte_count={total_len}, rc={rc}"
                        )

                    # Copy data from DMA buffer to destination
                    dst = self._byte_view(buf)
                    data_ptr = ctypes.cast(
                        dma_ptr, ctypes.POINTER(ctypes.c_ubyte * payload_len)
                    )
                    dst[:payload_len] = bytes(data_ptr.contents)
                finally:
                    ffi.free_spdk_memory(dma_ptr)

    def _write_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> None:
        """Write one or more buffers through the configured Rust I/O path.

        Args:
            offsets: Device offsets for each write.
            buffers: Python buffers to write.
            payload_lens: Logical payload lengths for each buffer.
            total_lens: Physical I/O lengths for each buffer.
            placement_ids: Optional FDP placement identifiers for raw-block writes.
                ``None`` omits the directive; explicit identifier 0 is rejected.

        Raises:
            RuntimeError: If the requested io_uring mode is unavailable.
            Exception: Propagates Rust raw-device or SPDK write errors.
        """
        # Route to SPDK if enabled
        if self.io_engine == "spdk":
            self._write_spdk_buffers(offsets, buffers, payload_lens, total_lens)
            return

        raw_dev = self._rawdev()
        per_write_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(offsets),
            field_name="placement_ids",
        )

        if self.io_engine != "io_uring":
            for offset, buf, payload_len, total_len in zip(
                offsets, buffers, payload_lens, total_lens, strict=True
            ):
                raw_dev.pwrite_from_buffer(offset, buf, payload_len, total_len)
            return

        if self.use_uring_cmd:
            self._write_uring_cmd_buffers(
                offsets,
                buffers,
                payload_lens,
                total_lens,
                per_write_placement_ids,
            )
            return

        can_batch = all(
            int(payload_len) == int(total_len)
            for payload_len, total_len in zip(payload_lens, total_lens, strict=True)
        )
        if can_batch:
            batch_id = raw_dev.batched_write(
                [int(offset) for offset in offsets],
                list(buffers),
                [int(total_len) for total_len in total_lens],
                per_write_placement_ids,
            )
            raw_dev.wait_iouring(batch_id)
            return

        for offset, buf, payload_len, total_len, placement_id in zip(
            offsets,
            buffers,
            payload_lens,
            total_lens,
            per_write_placement_ids,
            strict=True,
        ):
            raw_dev.write_uring(
                int(offset), buf, int(payload_len), int(total_len), placement_id
            )

    def _read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> None:
        """Read one or more buffers through the configured Rust I/O path.

        Args:
            offsets: Device offsets for each read.
            buffers: Destination Python buffers.
            payload_lens: Logical payload lengths to expose to callers.
            total_lens: Physical I/O lengths for each read.

        Raises:
            RuntimeError: If the requested io_uring mode is unavailable.
            Exception: Propagates Rust raw-device or SPDK read errors.
        """
        # Route to SPDK if enabled
        if self.io_engine == "spdk":
            self._read_spdk_buffers(offsets, buffers, payload_lens, total_lens)
            return

        raw_dev = self._rawdev()
        if self.io_engine != "io_uring":
            for offset, buf, payload_len, total_len in zip(
                offsets, buffers, payload_lens, total_lens, strict=True
            ):
                raw_dev.pread_into(offset, buf, payload_len, total_len)
            return

        if self.use_uring_cmd:
            self._read_uring_cmd_buffers(offsets, buffers, payload_lens, total_lens)
            return

        can_batch = all(
            int(payload_len) == int(total_len)
            for payload_len, total_len in zip(payload_lens, total_lens, strict=True)
        )
        # batched_read requires aligned buffers when O_DIRECT is enabled
        # Check alignment before using batched_read
        if can_batch and all(self._is_buffer_aligned(buf) for buf in buffers):
            batch_id = raw_dev.batched_read(
                [int(offset) for offset in offsets],
                list(buffers),
                [int(total_len) for total_len in total_lens],
            )
            raw_dev.wait_iouring(batch_id)
            return

        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            raw_dev.read_uring(int(offset), buf, int(payload_len), int(total_len))

    def _write_one(
        self,
        key: RawBlockKeySpec,
        memory_obj: MemoryObj,
        offset: int,
        *,
        placement_id: PlacementId = None,
    ) -> bool:
        """Write one object header and payload into a raw-block slot.

        Args:
            key: Raw-block key spec with the slot-header identity.
            memory_obj: Source object to write.
            offset: Slot byte offset on the raw device.
            placement_id: FDP placement identifier for this raw-block write.
                ``None`` omits the directive; explicit identifier 0 is rejected.

        Returns:
            True when both header and payload writes complete; false otherwise.
        """
        pool_header: Any = None
        try:
            # For SPDK IO engine, use pooled DMA buffer for header
            if self.io_engine == "spdk":
                pool_header = self._encode_header_using_pool(
                    key.slot_identity, len(memory_obj.byte_array)
                )
            else:
                header = self._encode_header(
                    key.slot_identity, len(memory_obj.byte_array)
                )

            buf, payload_len, total_len = self._prepare_write_payload(memory_obj)

            with self._lock:
                self._inflight_io_count += 1
            try:
                header_buf: Any
                if self.io_engine == "spdk":
                    header_buf = pool_header
                    hdr_total = self.block_align
                else:
                    hdr_total = (
                        round_up(len(header), self.block_align)
                        if self._requires_transfer_alignment
                        else len(header)
                    )
                    header_buf = header
                    if self.io_engine != "io_uring" and len(header) < hdr_total:
                        padded_header = bytearray(header)
                        padded_header.extend(b"\x00" * (hdr_total - len(header)))
                        header_buf = padded_header
                if self.io_engine == "io_uring" or self.io_engine == "spdk":
                    header_len = hdr_total
                else:
                    header_len = len(header)

                self._write_buffers(
                    [offset, offset + self.header_bytes],
                    [header_buf, buf],
                    [header_len, payload_len],
                    [hdr_total, payload_len],
                    [placement_id, placement_id],
                )
            finally:
                with self._lock:
                    self._inflight_io_count -= 1
                    self._last_io_ts = time.monotonic()
                if pool_header is not None and self._header_pool is not None:
                    self._header_pool.release(pool_header)
                    pool_header = None
            return True
        except Exception as e:
            if pool_header is not None and self._header_pool is not None:
                self._header_pool.release(pool_header)
                pool_header = None
            logger.error("RawBlockCore write failed for %s: %s", key.encoded, e)
            return False

    def _encode_header(self, slot_identity: int, payload_len: int) -> bytes:
        """Encode a fixed-size raw-block slot header."""
        hdr = bytearray(self.header_bytes)
        hdr[0:8] = b"LMCBLK01"
        hdr[8:16] = int(slot_identity & ((1 << 64) - 1)).to_bytes(
            8,
            "little",
            signed=False,
        )
        hdr[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
        return bytes(hdr)

    def _encode_header_using_pool(self, slot_identity: int, payload_len: int) -> object:
        """Encode a fixed-size raw-block header into a pooled SPDK DMA buffer.

        Acquires a buffer from ``_header_pool``, writes the header directly
        into the DMA memory, and returns the **same** buffer object so it
        can be released later via ``_header_pool.release()``.

        Args:
            slot_identity: The slot identity to encode.
            payload_len: The payload length to encode.

        Returns:
            The ctypes array from the pool (to be released via
            ``_header_pool.release()``).  The header data has already been
            written into the buffer in-place.

        Raises:
            RuntimeError: If ``_header_pool`` is not available (SPDK not
                enabled or pool not created).
        """
        if not hasattr(self, "_header_pool") or self._header_pool is None:
            raise RuntimeError("_encode_header_using_pool requires SPDK header pool")

        pool = self._header_pool
        assert pool is not None
        buf: Any = pool.acquire()
        buf_casted = ctypes.cast(
            buf, ctypes.POINTER(ctypes.c_ubyte * self.header_bytes)
        )
        ctypes.memset(buf_casted.contents, 0, self.header_bytes)
        buf_casted.contents[0:8] = list(b"LMCBLK01")
        identity_bytes = int(slot_identity & ((1 << 64) - 1)).to_bytes(
            8,
            "little",
            signed=False,
        )
        buf_casted.contents[8 : 8 + len(identity_bytes)] = list(identity_bytes)
        payload_bytes = int(payload_len).to_bytes(8, "little", signed=False)
        buf_casted.contents[16 : 16 + len(payload_bytes)] = list(payload_bytes)

        return buf

    def _decode_slot_header(self, hdr: bytes) -> Optional[tuple[int, int]]:
        """Decode a raw-block slot header into identity and payload length."""
        if len(hdr) < 24 or hdr[0:8] != b"LMCBLK01":
            return None
        slot_identity = int.from_bytes(hdr[8:16], "little", signed=False)
        payload_len = int.from_bytes(hdr[16:24], "little", signed=False)
        return slot_identity, payload_len

    def _read_slot_header(self, offset: int) -> Optional[tuple[int, int]]:
        """Read and decode the slot header at a raw-device offset."""
        buf = bytearray(self.header_bytes)
        try:
            with self._lock:
                self._inflight_io_count += 1
            self._read_buffers(
                [offset],
                [buf],
                [self.header_bytes],
                [self.header_bytes],
            )
            return self._decode_slot_header(buf)
        except Exception:
            return None
        finally:
            with self._lock:
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()

    def _ensure_capacity_and_layout(self) -> None:
        """Open the device if needed and compute metadata/data layout."""
        if self._effective_capacity_bytes > 0 and self._max_slots > 0:
            return

        # For SPDK mode, capacity is set explicitly or auto-detected from NVMe device
        if self.io_engine == "spdk":
            if self.capacity_bytes <= 0:
                if hasattr(self, "_spdk_engine") and self._spdk_engine is not None:
                    try:
                        device_size = self._spdk_engine.get_device_size()
                        if device_size > 0:
                            self._effective_capacity_bytes = device_size
                            self.capacity_bytes = device_size
                            logger.info(
                                "RawBlockCore: auto-detected SPDK NVMe device "
                                "capacity: %d bytes (%.2f GB)",
                                device_size,
                                device_size / (1024**3),
                            )
                        else:
                            raise RuntimeError(
                                "SPDK get_device_size returned invalid size: "
                                + str(device_size)
                            )
                    except Exception as e:
                        raise RuntimeError(
                            f"SPDK mode failed to auto-detect device size: {e}. "
                            "Set capacity_bytes explicitly in extra_config to override."
                        ) from e
                else:
                    raise RuntimeError(
                        "SPDK mode requires explicit capacity_bytes configuration "
                        "or SPDK engine to be initialized"
                    )
            else:
                self._effective_capacity_bytes = self.capacity_bytes
            self.capacity_bytes = self._effective_capacity_bytes
        else:
            device_size = int(self._rawdev().size_bytes())
            requested = self.capacity_bytes if self.capacity_bytes > 0 else device_size
            self._effective_capacity_bytes = min(requested, device_size)
            self.capacity_bytes = self._effective_capacity_bytes

        if self.meta_total_bytes >= self._effective_capacity_bytes:
            raise RuntimeError("metadata region exceeds usable device capacity")

        self._data_base_offset = self.meta_total_bytes
        data_bytes = self._effective_capacity_bytes - self._data_base_offset
        self._max_slots = data_bytes // self.slot_bytes
        if self._max_slots <= 0:
            raise RuntimeError(
                "raw block capacity too small for slot size after metadata"
            )

    def _slot_to_offset(self, slot: int) -> int:
        """Convert a data-slot index to its byte offset."""
        return self._data_base_offset + slot * self.slot_bytes

    def _offset_to_slot(self, offset: int) -> int:
        """Convert a data-slot byte offset to its slot index."""
        return (offset - self._data_base_offset) // self.slot_bytes

    def _allocate_slot_locked(self, placement_id: PlacementId = None) -> int:
        """Allocate a slot offset while ``self._lock`` is held."""
        self._ensure_capacity_and_layout()

        if self.fdp_slot_affinity_enabled and placement_id is not None:
            affinity_slots = self._free_slots_by_placement_id.get(placement_id)
            if affinity_slots:
                slot, _ = affinity_slots.popitem()
                if not affinity_slots:
                    self._free_slots_by_placement_id.pop(placement_id, None)
                self._free_slots.pop(slot, None)
                self._fdp_slot_affinity_hit_count += 1
                self._set_slot_placement_id_locked(slot, placement_id)
                return self._slot_to_offset(slot)

        if self._free_slots:
            slot, _ = self._free_slots.popitem()
            self._remove_slot_from_affinity_pool_locked(slot)
            if self.fdp_slot_affinity_enabled and placement_id is not None:
                self._fdp_slot_affinity_fallback_count += 1
            self._set_slot_placement_id_locked(slot, placement_id)
            return self._slot_to_offset(slot)

        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            self._set_slot_placement_id_locked(slot, placement_id)
            return self._slot_to_offset(slot)
        raise RuntimeError("No free slots available")

    def _append_free_slot_locked(self, slot: int) -> None:
        """Add a slot to the free list while ``self._lock`` is held."""
        if slot < 0 or slot >= self._max_slots:
            return
        if slot in self._free_slots:
            return
        self._free_slots[slot] = None
        if not self.fdp_slot_affinity_enabled:
            return
        placement_id = self._slot_placement_ids.get(slot)
        if placement_id is not None:
            self._free_slots_by_placement_id.setdefault(placement_id, {})[slot] = None

    def _remove_slot_from_affinity_pool_locked(self, slot: int) -> None:
        """Remove an allocated slot from its PID-specific free-slot pool."""
        placement_id = self._slot_placement_ids.get(slot)
        if placement_id is None:
            return
        affinity_slots = self._free_slots_by_placement_id.get(placement_id)
        if affinity_slots is None:
            return
        affinity_slots.pop(slot, None)
        if not affinity_slots:
            self._free_slots_by_placement_id.pop(placement_id, None)

    def _set_slot_placement_id_locked(
        self,
        slot: int,
        placement_id: PlacementId,
    ) -> None:
        """Record the latest runtime-only placement identifier for a slot."""
        if not self.fdp_slot_affinity_enabled or placement_id is None:
            self._slot_placement_ids.pop(slot, None)
            return
        self._slot_placement_ids[slot] = placement_id

    def _checkpoint_loop(self) -> None:
        """Periodically checkpoint dirty metadata until shutdown."""
        interval = max(1, self.meta_checkpoint_interval_sec)
        while not self._meta_stop_evt.wait(interval):
            try:
                self._checkpoint_once(force=False)
            except Exception as e:
                logger.warning("Periodic raw-block metadata checkpoint failed: %s", e)

    def _meta_payload_capacity(self) -> int:
        """Return usable bytes in one metadata checkpoint payload area."""
        return self._meta_container_bytes - self.block_align

    def _meta_container_offsets(self) -> list[int]:
        """Return byte offsets for mirrored metadata checkpoint containers."""
        return [
            idx * self._meta_container_bytes for idx in range(self._meta_copy_count)
        ]

    def _read_meta_header(self, container_offset: int) -> Optional[dict[str, int]]:
        """Read and validate a metadata checkpoint header."""
        if self.io_engine == "spdk" and self._spdk_engine is not None:
            return self._read_meta_header_spdk(container_offset)

        buf = bytearray(self.block_align)
        try:
            self._read_buffers(
                [container_offset],
                [buf],
                [self.block_align],
                [self.block_align],
            )
        except Exception:
            return None

        hdr = bytes(buf[: _META_HEADER_STRUCT.size])
        magic, version, seq, payload_len, crc = _META_HEADER_STRUCT.unpack(hdr)
        if magic != self.meta_magic or version != self.meta_version:
            return None

        payload_cap = self._meta_payload_capacity()
        if payload_len <= 0 or payload_len > payload_cap:
            return None
        return {
            "seq": int(seq),
            "payload_len": int(payload_len),
            "crc": int(crc),
            "container_offset": int(container_offset),
        }

    def _read_meta_header_spdk(self, container_offset: int) -> Optional[dict[str, int]]:
        """Read metadata checkpoint header using SPDK DMA-allocated buffer.

        Allocates a DMA-safe buffer via SPDK, reads the header, then frees
        the buffer. This avoids vtophys failures that occur when reading
        from unregistered memory.

        Args:
            container_offset: Byte offset of the metadata container on device.

        Returns:
            Parsed header dictionary, or None on failure.
        """
        ffi = self._spdk_engine
        if ffi is None:
            return None

        # Allocate DMA buffer for reading header
        dma_ptr = ffi.allocate_spdk_memory(self.block_align, 4096, numa_id=-1)
        if dma_ptr == 0:
            logger.error(
                "RawBlockCore: failed to allocate DMA buffer for header read "
                "(offset=%d, size=%d)",
                container_offset,
                self.block_align,
            )
            return None

        try:
            rc = _spdk_call_with_gil_released(
                ffi.spdk_read_external, container_offset, self.block_align, dma_ptr
            )
            if rc != 0:
                logger.debug(
                    "RawBlockCore: SPDK read failed for header at offset %d: rc=%d",
                    container_offset,
                    rc,
                )
                return None

            # Copy header data from DMA buffer to Python struct
            header_data = bytes(
                ctypes.cast(
                    dma_ptr, ctypes.POINTER(ctypes.c_ubyte * _META_HEADER_STRUCT.size)
                ).contents
            )
            magic, version, seq, payload_len, crc = _META_HEADER_STRUCT.unpack(
                header_data
            )

            if magic != self.meta_magic or version != self.meta_version:
                return None

            payload_cap = self._meta_payload_capacity()
            if payload_len <= 0 or payload_len > payload_cap:
                return None

            return {
                "seq": int(seq),
                "payload_len": int(payload_len),
                "crc": int(crc),
                "container_offset": int(container_offset),
            }
        finally:
            ffi.free_spdk_memory(dma_ptr)

    def _load_meta_payload(self, header: dict[str, int]) -> Optional[bytes]:
        """Load and CRC-validate a checkpoint payload for a metadata header."""
        if self.io_engine == "spdk" and self._spdk_engine is not None:
            return self._load_meta_payload_spdk(header)

        payload_len = int(header["payload_len"])
        payload_off = int(header["container_offset"]) + self.block_align
        total_len = round_up(payload_len, self.block_align)
        buf = bytearray(total_len)
        try:
            self._read_buffers([payload_off], [buf], [payload_len], [total_len])
        except Exception:
            return None

        payload = bytes(buf[:payload_len])
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != int(header["crc"]):
            return None
        return payload

    def _load_meta_payload_spdk(self, header: dict[str, int]) -> Optional[bytes]:
        """Load checkpoint payload using SPDK DMA-allocated buffer.

        Allocates a DMA-safe buffer via SPDK, reads the payload, validates
        CRC, then frees the buffer.

        Args:
            header: Metadata header dictionary with payload_len and container_offset.

        Returns:
            Payload bytes on success, None on failure.
        """
        ffi = self._spdk_engine
        if ffi is None:
            return None

        payload_len = int(header["payload_len"])
        payload_off = int(header["container_offset"]) + self.block_align
        total_len = round_up(payload_len, self.block_align)

        # Allocate DMA buffer for reading payload
        dma_ptr = ffi.allocate_spdk_memory(total_len, 4096, numa_id=-1)
        if dma_ptr == 0:
            logger.error(
                "RawBlockCore: failed to allocate DMA buffer for payload read "
                "(offset=%d, size=%d)",
                payload_off,
                total_len,
            )
            return None

        try:
            rc = _spdk_call_with_gil_released(
                ffi.spdk_read_external, payload_off, total_len, dma_ptr
            )
            if rc != 0:
                logger.debug(
                    "RawBlockCore: SPDK read failed for payload at offset %d: rc=%d",
                    payload_off,
                    rc,
                )
                return None

            # Copy payload data from DMA buffer
            payload_ptr = ctypes.cast(
                dma_ptr, ctypes.POINTER(ctypes.c_ubyte * payload_len)
            )
            payload = bytes(payload_ptr.contents)

            # Validate CRC
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            if crc != int(header["crc"]):
                logger.debug(
                    "RawBlockCore: CRC mismatch for payload at offset %d: "
                    "expected=%d got=%d",
                    payload_off,
                    header["crc"],
                    crc,
                )
                return None

            return payload
        finally:
            ffi.free_spdk_memory(dma_ptr)

    def _select_latest_checkpoint(
        self,
    ) -> tuple[Optional[dict[str, int]], Optional[bytes]]:
        """Return the newest valid checkpoint header and payload."""
        best_header: Optional[dict[str, int]] = None
        best_payload: Optional[bytes] = None
        for offset in self._meta_container_offsets():
            header = self._read_meta_header(offset)
            if header is None:
                continue
            payload = self._load_meta_payload(header)
            if payload is None:
                continue
            if best_header is None or int(header["seq"]) > int(best_header["seq"]):
                best_header = header
                best_payload = payload
        return best_header, best_payload

    def _snapshot_state(self) -> tuple[dict[str, Any], int]:
        """Build a JSON-serializable checkpoint state snapshot."""
        with self._lock:
            dirty_total = self._meta_dirty_total
            snapshot = {
                "version": 1,
                "device_path": self.device_path,
                "capacity_bytes": self.capacity_bytes,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "slot_bytes": self.slot_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "meta_magic": self.meta_magic_text,
                "meta_version": self.meta_version,
                "data_base_offset": self._data_base_offset,
                "next_slot": self._next_slot,
                "entries": {
                    encoded_key: {
                        "offset": entry.offset,
                        "size": entry.meta.size,
                        "shape": list(entry.meta.shape)
                        if entry.meta.shape is not None
                        else None,
                        "dtype": self._checkpoint_dtype_name(entry.meta.dtype),
                        "fmt": (
                            entry.meta.fmt.name
                            if entry.meta.fmt is not None
                            and hasattr(entry.meta.fmt, "name")
                            else str(entry.meta.fmt)
                            if entry.meta.fmt is not None
                            else None
                        ),
                        "cached_positions": (
                            entry.meta.cached_positions.tolist()
                            if entry.meta.cached_positions is not None
                            and hasattr(entry.meta.cached_positions, "tolist")
                            else None
                        ),
                    }
                    for encoded_key, entry in self._index.items()
                },
            }
        return snapshot, dirty_total

    def _checkpoint_dtype_name(self, dtype: torch.dtype | None) -> str | None:
        """Return a durable checkpoint string for a torch dtype.

        Args:
            dtype: Torch dtype from recovered or live memory metadata.

        Returns:
            Stable LMCache dtype name when known, ``str(dtype)`` for unknown
            torch dtypes, or None when no dtype is available.
        """
        if dtype is None:
            return None
        return TORCH_DTYPE_TO_STR_DTYPE.get(dtype, str(dtype))

    def _write_checkpoint(self, payload: bytes, dirty_total_snapshot: int) -> bool:
        """Write one checkpoint copy and advance persisted metadata counters."""
        payload_cap = self._meta_payload_capacity()
        if len(payload) > payload_cap:
            logger.warning(
                "RawBlockCore metadata payload too large (%d > %d), "
                "skipping checkpoint",
                len(payload),
                payload_cap,
            )
            return False

        next_seq = self._meta_seq + 1
        target_idx = int((next_seq - 1) % self._meta_copy_count)
        target = self._meta_container_offsets()[target_idx]

        payload_len = len(payload)
        payload_total_len = round_up(payload_len, self.block_align)
        payload_off = target + self.block_align
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        # Use SPDK DMA pools for zero-copy writes when available
        if self.io_engine == "spdk":
            checkpoint_pool = self._checkpoint_pool
            assert checkpoint_pool is not None
            dma_payload_raw: Any = checkpoint_pool.acquire()
            dma_payload = ctypes.cast(
                dma_payload_raw, ctypes.POINTER(ctypes.c_ubyte * payload_total_len)
            )
            try:
                # Copy payload directly into DMA memory (one explicit copy)
                dma_payload.contents[:payload_len] = list(payload)

                header_pool = self._header_pool
                assert header_pool is not None
                dma_header_raw: Any = header_pool.acquire()
                dma_header = ctypes.cast(
                    dma_header_raw,
                    ctypes.POINTER(ctypes.c_ubyte * self.block_align),
                )
                try:
                    # Write header directly into DMA memory
                    dma_header.contents[: _META_HEADER_STRUCT.size] = list(
                        _META_HEADER_STRUCT.pack(
                            self.meta_magic,
                            self.meta_version,
                            int(next_seq),
                            int(payload_len),
                            int(crc),
                        )
                    )

                    # Write both buffers using DMA pointers (zero-copy to NVMe)
                    self._write_buffers(
                        [payload_off, target],
                        [dma_payload_raw, dma_header_raw],
                        [payload_len, self.block_align],
                        [payload_total_len, self.block_align],
                    )
                finally:
                    header_pool.release(dma_header_raw)
            finally:
                checkpoint_pool.release(dma_payload_raw)
        else:
            header_block = bytearray(self.block_align)
            header_block[: _META_HEADER_STRUCT.size] = _META_HEADER_STRUCT.pack(
                self.meta_magic,
                self.meta_version,
                int(next_seq),
                int(payload_len),
                int(crc),
            )

            placement_id = self.meta_checkpoint_placement_id
            self._write_buffers(
                [payload_off, target],
                [payload, header_block],
                [payload_len, self.block_align],
                [payload_total_len, self.block_align],
                [placement_id, placement_id],
            )

        with self._lock:
            self._meta_seq = int(next_seq)
            self._meta_persisted = max(self._meta_persisted, int(dirty_total_snapshot))
        return True

    def _checkpoint_once(self, force: bool) -> bool:
        """Write a metadata checkpoint when dirty and sufficiently idle."""
        with self._lock:
            dirty = self._meta_dirty_total > self._meta_persisted
            idle_ok = self._inflight_io_count == 0 and (
                time.monotonic() - self._last_io_ts
            ) >= (self.meta_idle_quiet_ms / 1000.0)

        if not dirty:
            return False
        if not force and not idle_ok:
            return False

        snapshot, dirty_total_snapshot = self._snapshot_state()
        payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
        return self._write_checkpoint(payload, dirty_total_snapshot)

    def _is_valid_checkpoint_entry(self, offset: int, size: int) -> bool:
        """Return whether a checkpoint entry references a valid data slot."""
        if offset < self._data_base_offset:
            return False
        rel = offset - self._data_base_offset
        if rel % self.slot_bytes != 0:
            return False
        slot = rel // self.slot_bytes
        if slot >= self._max_slots:
            return False
        return 0 < size <= (self.slot_bytes - self.header_bytes)

    def _apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Apply decoded checkpoint state after validating layout fields."""
        if not isinstance(data, dict):
            return False
        if int(data.get("version", 0)) != 1:
            return False
        checkpoint_device_path = data.get("device_path")
        if checkpoint_device_path and checkpoint_device_path != self.device_path:
            logger.warning("Device metadata device_path mismatch; ignoring metadata")
            return False
        if int(data.get("block_align", self.block_align)) != self.block_align:
            logger.warning("Device metadata block_align mismatch; ignoring metadata")
            return False
        if int(data.get("header_bytes", self.header_bytes)) != self.header_bytes:
            logger.warning("Device metadata header_bytes mismatch; ignoring metadata")
            return False
        if int(data.get("slot_bytes", self.slot_bytes)) != self.slot_bytes:
            logger.warning("Device metadata slot_bytes mismatch; ignoring metadata")
            return False
        if (
            int(data.get("meta_total_bytes", self.meta_total_bytes))
            != self.meta_total_bytes
        ):
            logger.warning(
                "Device metadata meta_total_bytes mismatch; ignoring metadata"
            )
            return False
        if str(data.get("meta_magic", self.meta_magic_text)) != self.meta_magic_text:
            logger.warning("Device metadata meta_magic mismatch; ignoring metadata")
            return False
        if int(data.get("meta_version", self.meta_version)) != self.meta_version:
            logger.warning("Device metadata meta_version mismatch; ignoring metadata")
            return False

        try:
            next_slot = int(data.get("next_slot", 0))
        except Exception:
            logger.warning("Device metadata next_slot is invalid; ignoring metadata")
            return False
        if next_slot < 0 or next_slot > self._max_slots:
            logger.warning(
                "Device metadata next_slot out of range (%d); ignoring metadata",
                next_slot,
            )
            return False

        with self._lock:
            self._next_slot = next_slot
            self._free_slots = {}
            self._free_slots_by_placement_id.clear()
            self._slot_placement_ids.clear()
            self._index.clear()
            self._lock_refcnt.clear()

            entries = data.get("entries", {})
            if isinstance(entries, dict):
                for encoded_key, entry in entries.items():
                    if not isinstance(entry, dict):
                        continue

                    offset = int(entry.get("offset", 0))
                    size = int(entry.get("size", 0))
                    shape_list = entry.get("shape")
                    fmt_name = entry.get("fmt")
                    cached_positions_list = entry.get("cached_positions")
                    dtype_name = entry.get("dtype")

                    if not self._is_valid_checkpoint_entry(offset, size):
                        continue

                    shape = (
                        torch.Size(list(shape_list)) if shape_list is not None else None
                    )
                    fmt = (
                        MemoryFormat[fmt_name]
                        if isinstance(fmt_name, str)
                        and fmt_name in MemoryFormat.__members__
                        else MemoryFormat.UNDEFINED
                    )
                    cached_positions = (
                        torch.tensor(cached_positions_list, dtype=torch.long)
                        if cached_positions_list is not None
                        else None
                    )
                    dtype = self._recover_checkpoint_dtype(
                        str(encoded_key),
                        dtype_name,
                    )

                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=size,
                        shape=shape,
                        dtype=dtype,
                        cached_positions=cached_positions,
                        fmt=fmt,
                        pin_count=0,
                    )
                    self._index[encoded_key] = _Entry(
                        offset=offset, size=size, meta=meta
                    )

            used_slots = {
                self._offset_to_slot(int(entry.offset))
                for entry in self._index.values()
            }
            # Rebuild from committed entries instead of trusting checkpoint
            # free_slots. A crash-time checkpoint can otherwise preserve a slot
            # reserved by an uncommitted in-flight write as neither used nor free.
            self._free_slots = {
                slot: None for slot in range(self._next_slot) if slot not in used_slots
            }

            self._meta_dirty_total = 0
            self._meta_persisted = 0

        if self.meta_verify_on_load:
            self._validate_loaded_entries()
        return True

    def _recover_checkpoint_dtype(
        self,
        encoded_key: str,
        dtype_name: Any,
    ) -> torch.dtype | None:
        """Recover checkpoint dtype from entry metadata or legacy key strings.

        Args:
            encoded_key: Encoded raw-block key from the checkpoint entry.
            dtype_name: Raw dtype value stored in the checkpoint entry.

        Returns:
            A torch dtype when recovery succeeds, otherwise None.
        """
        if isinstance(dtype_name, str):
            dtype = STR_DTYPE_TO_TORCH_DTYPE.get(dtype_name)
            if dtype is not None:
                return dtype

            torch_prefix = "torch."
            if dtype_name.startswith(torch_prefix):
                dtype_attr = dtype_name.removeprefix(torch_prefix)
                dtype = STR_DTYPE_TO_TORCH_DTYPE.get(dtype_attr)
                if dtype is not None:
                    return dtype
                torch_dtype = getattr(torch, dtype_attr, None)
                if isinstance(torch_dtype, torch.dtype):
                    return torch_dtype

        if self.key_namespace != "legacy":
            return None

        try:
            return decode_legacy_key(encoded_key).dtype
        except Exception:
            logger.debug(
                "Unable to recover dtype from legacy raw-block key %s",
                encoded_key,
                exc_info=True,
            )
            return None

    def _validate_loaded_entries(self) -> None:
        """Drop recovered entries whose slot headers do not match metadata."""
        to_drop: list[str] = []
        with self._lock:
            items = list(self._index.items())

        for encoded_key, entry in items:
            slot_hdr = self._read_slot_header(int(entry.offset))
            if slot_hdr is None:
                to_drop.append(encoded_key)
                continue
            try:
                expected_identity = slot_identity_from_encoded_key(
                    encoded_key,
                    self.key_namespace,
                )
            except Exception:
                to_drop.append(encoded_key)
                continue
            slot_identity, payload_len = slot_hdr
            if int(slot_identity) != int(expected_identity):
                to_drop.append(encoded_key)
                continue
            if int(payload_len) != int(entry.size):
                to_drop.append(encoded_key)

        if not to_drop:
            return

        with self._lock:
            for encoded_key in to_drop:
                removed_entry = self._index.pop(encoded_key, None)
                self._lock_refcnt.pop(encoded_key, None)
                if removed_entry is not None:
                    self._append_free_slot_locked(
                        self._offset_to_slot(int(removed_entry.offset))
                    )
            self._meta_dirty_total += 1

        logger.warning(
            "RawBlockCore dropped %d stale metadata entries after "
            "slot-header validation",
            len(to_drop),
        )

    def _load_checkpoint_from_device(self) -> None:
        """Load the newest valid checkpoint from the raw device if present."""
        header, payload = self._select_latest_checkpoint()
        if header is None:
            logger.info("RawBlockCore: no valid on-device metadata checkpoint found")
            return
        if payload is None:
            logger.warning("RawBlockCore: checkpoint header had no payload")
            return
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning("RawBlockCore: failed to decode metadata payload")
            return
        if not self.apply_loaded_state(data):
            logger.warning("RawBlockCore: metadata payload rejected by checks")
            return
        self._meta_seq = int(header["seq"])
        logger.info(
            "RawBlockCore loaded checkpoint (entries=%d next_slot=%d seq=%d device=%s)",
            len(self._index),
            self._next_slot,
            self._meta_seq,
            self.device_path,
        )
