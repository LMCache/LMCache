# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
import ctypes
import json
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
from lmcache.v1.storage_backend.raw_block.key_codec import (
    RawBlockKeyNamespace,
    RawBlockKeySpec,
    decode_legacy_key,
    slot_identity_from_encoded_key,
)

logger = init_logger(__name__)


_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1
_META_VERSION_V1 = 1
_META_VERSION_V2 = 2
_META_HEADER_V1_STRUCT = struct.Struct("<8sIQQI")
_META_HEADER_V2_STRUCT = struct.Struct("<8sIQQIQQ")
_DELTA_RECORD_MAGIC = b"LMCD"
_DELTA_RECORD_HEADER_STRUCT = struct.Struct("<4sIQQIIIII")
_DELTA_RECORD_VERSION = 1
# Default share of meta_total_bytes given to each base mirror in v2 layout.
# Two mirrors plus a delta region must fit within meta_total_bytes, so this
# defaults to 40% per mirror leaving ~20% for the delta region.
_AUTO_BASE_FRACTION = 0.4
RAW_BLOCK_IO_ENGINES = frozenset({"posix", "io_uring"})
DEFAULT_IOURING_QUEUE_DEPTH = 256


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
    load_checkpoint_on_init: bool = True
    io_engine: str = "posix"
    iouring_queue_depth: int = DEFAULT_IOURING_QUEUE_DEPTH
    # Incremental checkpoint controls.
    meta_incremental_enabled: bool = True
    meta_full_checkpoint_interval_sec: int = 600
    meta_full_checkpoint_max_deltas: int = 1024
    meta_delta_high_watermark_pct: int = 90
    # 0 means "auto": v2 layout uses ~40% per base copy, ~20% delta.
    meta_base_copy_bytes: int = 0


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


@dataclass(frozen=True)
class _MetaLayout:
    """On-device metadata region geometry.

    Two mirrored base copies are followed by an optional delta region used by
    incremental checkpoints. ``delta_region_bytes == 0`` falls back to the
    legacy v1 full-snapshot-only layout.
    """

    block_align: int
    base_copy_bytes: int
    base_copy_offsets: tuple[int, ...]
    delta_region_off: int
    delta_region_bytes: int

    @property
    def base_payload_capacity(self) -> int:
        return self.base_copy_bytes - self.block_align


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
        self.key_namespace = key_namespace
        self.meta_incremental_enabled = bool(config.meta_incremental_enabled)
        self.meta_full_checkpoint_interval_sec = int(
            config.meta_full_checkpoint_interval_sec
        )
        self.meta_full_checkpoint_max_deltas = int(
            config.meta_full_checkpoint_max_deltas
        )
        self.meta_delta_high_watermark_pct = int(config.meta_delta_high_watermark_pct)
        self._configured_base_copy_bytes = int(config.meta_base_copy_bytes)

        if not self.device_path:
            raise ValueError("RawBlockCore requires a non-empty device_path")
        if self.block_align <= 0:
            raise ValueError("block_align must be > 0")
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

        try:
            self.meta_magic_text = self.meta_magic.decode("ascii")
        except UnicodeDecodeError as e:
            raise ValueError("meta_magic must be ASCII bytes") from e

        self._meta_copy_count: int = 2
        # If incremental mode is requested but the meta region is too small for
        # both base mirrors and a delta region, fall back to v1 layout.
        self._use_incremental: bool = (
            self.meta_incremental_enabled and self._can_fit_incremental_layout()
        )
        self._meta_layout: _MetaLayout = self._build_meta_layout()
        if self.meta_incremental_enabled and not self._use_incremental:
            logger.warning(
                "RawBlockCore: incremental checkpoint disabled — "
                "meta_total_bytes=%d cannot host two base mirrors and a "
                "delta region; falling back to full-snapshot layout",
                self.meta_total_bytes,
            )

        self._lock = threading.Lock()
        # Serializes checkpoint writes (full + delta) against each other so
        # concurrent ``_checkpoint_once`` calls (background loop + manual
        # flush + close) cannot race on the delta tail offset or mirror seq.
        # Held only across a single checkpoint attempt; never held during
        # put/get critical sections (those use ``self._lock`` instead).
        self._writer_lock = threading.Lock()
        self._index: dict[str, _Entry] = {}
        self._lock_refcnt: dict[str, int] = {}
        self._inflight: dict[str, _Inflight] = {}

        self._next_slot: int = 0
        self._free_slots: list[int] = []
        self._max_slots: int = 0
        self._effective_capacity_bytes: int = 0
        self._data_base_offset: int = 0

        self._raw = None
        self._closed = False

        self._meta_seq: int = 0
        self._meta_dirty_total: int = 0
        self._meta_persisted: int = 0
        self._inflight_io_count: int = 0
        self._last_io_ts: float = time.monotonic()
        self._meta_stop_evt = threading.Event()
        self._meta_thread: Optional[threading.Thread] = None

        # Pending mutations since the last successful checkpoint write. Each
        # entry is a JSON-serializable op dict consumed by the delta encoder.
        self._dirty_log: list[dict[str, Any]] = []
        self._delta_seq: int = 0
        self._delta_tail_off: int = 0
        self._active_base_seq: int = 0
        self._active_base_is_v2: bool = False
        self._last_full_ts: float = time.monotonic()

        try:
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
        except Exception:
            self._cleanup_after_init_failure()
            raise

    def _rawdev(self):
        """Return the lazily opened Rust raw-block device binding."""
        if self._raw is None:
            try:
                # Third Party
                from lmcache_rust_raw_block_io import RawBlockDevice  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Rust raw-block extension is not installed. "
                    "Install / build `rust_raw_block_io` and retry."
                ) from e
            self._raw = RawBlockDevice(
                self.device_path,
                writable=True,
                use_odirect=self.use_odirect,
                alignment=self.block_align,
                io_engine=self.io_engine,
                iouring_queue_depth=self.iouring_queue_depth,
            )
        return self._raw

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
    ) -> RawBlockPutManyResult:
        """Persist a batch of memory objects into raw-block slots.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``.
            objs: Memory objects whose byte buffers should be written.

        Returns:
            Per-key success results and newly stored encoded keys. If no free
            raw-block slot is available, that key is reported as failed; slot
            reclamation is owned by the adapter/controller calling
            ``delete_many``.

        Raises:
            ValueError: If either sequence is empty or the sequence lengths do
                not match.
        """
        if not keys or not objs:
            raise ValueError("keys and objs must be non-empty")
        if len(keys) != len(objs):
            raise ValueError("keys and objs must have the same length")

        results = [False] * len(keys)
        stored_keys: list[str] = []

        for i, (key, obj) in enumerate(zip(keys, objs, strict=False)):
            if self._closed:
                break

            with self._lock:
                if key.encoded in self._index:
                    results[i] = True
                    continue
                if key.encoded in self._inflight:
                    continue

                try:
                    offset = self._allocate_slot_locked()
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

            success = self._write_one(key, obj, offset)

            with self._lock:
                inflight = self._inflight.pop(key.encoded, None)
                if inflight is None:
                    results[i] = False
                    continue
                if inflight.canceled or not success:
                    freed_slot = self._offset_to_slot(int(inflight.offset))
                    self._append_free_slot_locked(freed_slot)
                    self._record_freelist_push_locked(freed_slot)
                    self._meta_dirty_total += 1
                    results[i] = False
                    continue

                entry = _Entry(
                    offset=inflight.offset,
                    size=inflight.meta.size,
                    meta=inflight.meta,
                )
                self._index[key.encoded] = entry
                self._record_put_op_locked(key.encoded, entry)
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
            raw_dev = self._rawdev()
            for i, (encoded_key, entry) in enumerate(items):
                if entry is None:
                    continue
                try:
                    payload_len = int(entry.size)
                    total_len = (
                        round_up(payload_len, self.block_align)
                        if self.use_odirect
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
                        raw_dev.pread_into(
                            entry.offset + self.header_bytes,
                            direct_view,
                            total_len if len(direct_view) >= total_len else payload_len,
                            total_len,
                        )
                    else:
                        raw_dev.pread_into(
                            entry.offset + self.header_bytes,
                            buf,
                            payload_len,
                            total_len,
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
                existed = encoded_key in self._index or encoded_key in self._inflight
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
                    freed_slot = self._offset_to_slot(int(removed_entry.offset))
                    self._append_free_slot_locked(freed_slot)
                    self._record_delete_op_locked(encoded_key, freed_slot)
                    self._meta_dirty_total += 1
                deleted.append(
                    existed and (removed_entry is not None or inflight is not None)
                )
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
        """Synchronously persist pending metadata mutations.

        Bypasses the idle-quiet check so callers can flush from non-quiet
        code paths. Picks an incremental delta record when the v2 layout
        is active and the dirty log fits; otherwise rewrites a full
        snapshot mirror. No-op when nothing is dirty.
        """
        self._checkpoint_once(force=True)

    def apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Validate and apply a recovered metadata checkpoint payload.

        Args:
            data: Decoded checkpoint dictionary.

        Returns:
            True when the payload shape and layout match this core and all
            valid entries were applied. Invalid per-entry records are skipped.
        """
        if not self._apply_loaded_state(data):
            return False
        if self.meta_verify_on_load:
            self._validate_loaded_entries()
        return True

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
                "incremental_enabled": self._use_incremental,
                "delta_seq": self._delta_seq,
                "delta_tail_off": self._delta_tail_off,
                "delta_region_bytes": self._meta_layout.delta_region_bytes,
                "active_base_seq": self._active_base_seq,
                "inflight_io_count": self._inflight_io_count,
                "use_odirect": self.use_odirect,
                "enable_zero_copy": self.enable_zero_copy,
                "io_engine": self.io_engine,
                "iouring_queue_depth": self.iouring_queue_depth,
            }

    def close(self) -> None:
        """Stop checkpointing, write a final checkpoint, and close the device.

        The final write follows the same trigger logic as ``checkpoint_now``:
        a small dirty log becomes one delta record, otherwise a full base
        compaction. Either is sufficient for crash-consistent recovery on
        the next mount.
        """
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
        """Close resources that may have been opened before init failed."""
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
        if self.use_odirect:
            total_len = round_up(payload_len, self.block_align)
            if total_len > payload_capacity:
                raise RuntimeError(
                    f"O_DIRECT payload {total_len} exceeds slot capacity "
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

    def _write_one(
        self, key: RawBlockKeySpec, memory_obj: MemoryObj, offset: int
    ) -> bool:
        """Write one object header and payload into a raw-block slot.

        Args:
            key: Raw-block key spec with the slot-header identity.
            memory_obj: Source object to write.
            offset: Slot byte offset on the raw device.

        Returns:
            True when both header and payload writes complete; false otherwise.
        """
        try:
            header = self._encode_header(key.slot_identity, len(memory_obj.byte_array))
            buf, payload_len, total_len = self._prepare_write_payload(memory_obj)

            with self._lock:
                self._inflight_io_count += 1
            try:
                raw_dev = self._rawdev()
                hdr_total = (
                    round_up(len(header), self.block_align)
                    if self.use_odirect
                    else len(header)
                )
                raw_dev.pwrite_from_buffer(offset, header, len(header), hdr_total)
                raw_dev.pwrite_from_buffer(
                    offset + self.header_bytes,
                    buf,
                    payload_len,
                    total_len,
                )
            finally:
                with self._lock:
                    self._inflight_io_count -= 1
                    self._last_io_ts = time.monotonic()
            return True
        except Exception as e:
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
            self._rawdev().pread_into(offset, buf, self.header_bytes, self.header_bytes)
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

    def _allocate_slot_locked(self) -> int:
        """Allocate a slot offset while ``self._lock`` is held."""
        self._ensure_capacity_and_layout()
        if self._free_slots:
            return self._slot_to_offset(self._free_slots.pop())
        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            return self._slot_to_offset(slot)
        raise RuntimeError("No free slots available")

    def _append_free_slot_locked(self, slot: int) -> None:
        """Add a slot to the free list while ``self._lock`` is held.

        Fast path for trusted callers (delete, put-failure, validation drop)
        that have just removed the entry that owned ``slot``. Skips the
        index scan that ``_safe_append_free_slot_locked`` performs because
        the caller already knows the slot is unowned.
        """
        if slot < 0 or slot >= self._max_slots:
            return
        if slot in self._free_slots:
            return
        self._free_slots.append(slot)

    def _safe_append_free_slot_locked(self, slot: int) -> None:
        """Add a slot to the free list, refusing slots an indexed entry owns.

        Used when the slot index comes from external/untrusted input
        (delta replay). Scans the live index to guard against malformed
        ``freed_slot`` values that point at a re-allocated slot.
        """
        if slot < 0 or slot >= self._max_slots:
            return
        if slot in self._free_slots:
            return
        slot_offset = self._slot_to_offset(slot)
        for entry in self._index.values():
            if int(entry.offset) == slot_offset:
                return
        self._free_slots.append(slot)

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
        return self._meta_layout.base_payload_capacity

    def _entry_to_op_payload(self, entry: "_Entry") -> dict[str, Any]:
        """Serialize an in-memory entry into the on-disk JSON form.

        Used both by full-snapshot ``entries`` and by delta ``put`` ops so
        the field schema stays in lockstep across both encodings.
        """
        meta = entry.meta
        return {
            "offset": int(entry.offset),
            "size": int(meta.size),
            "shape": list(meta.shape) if meta.shape is not None else None,
            "dtype": self._checkpoint_dtype_name(meta.dtype),
            "fmt": (
                meta.fmt.name
                if meta.fmt is not None and hasattr(meta.fmt, "name")
                else str(meta.fmt)
                if meta.fmt is not None
                else None
            ),
            "cached_positions": (
                meta.cached_positions.tolist()
                if meta.cached_positions is not None
                and hasattr(meta.cached_positions, "tolist")
                else None
            ),
        }

    def _record_put_op_locked(self, encoded_key: str, entry: "_Entry") -> None:
        """Append a put op to the dirty log; caller holds ``self._lock``."""
        op: dict[str, Any] = {"op": "put", "k": str(encoded_key)}
        op.update(self._entry_to_op_payload(entry))
        op["next_slot"] = int(self._next_slot)
        self._dirty_log.append(op)

    def _record_delete_op_locked(
        self, encoded_key: str, freed_slot: Optional[int]
    ) -> None:
        """Append a delete op to the dirty log; caller holds ``self._lock``."""
        op: dict[str, Any] = {
            "op": "delete",
            "k": str(encoded_key),
            "next_slot": int(self._next_slot),
        }
        if freed_slot is not None:
            op["freed_slot"] = int(freed_slot)
        self._dirty_log.append(op)

    def _record_freelist_push_locked(self, freed_slot: int) -> None:
        """Append a freelist_push op (put-failure path); caller holds the lock."""
        op: dict[str, Any] = {
            "op": "freelist_push",
            "freed_slot": int(freed_slot),
            "next_slot": int(self._next_slot),
        }
        self._dirty_log.append(op)

    def _clear_dirty_log_locked(self) -> None:
        """Drop pending ops; caller holds ``self._lock``."""
        self._dirty_log = []

    def _meta_container_offsets(self) -> list[int]:
        """Return byte offsets for mirrored metadata checkpoint containers."""
        return list(self._meta_layout.base_copy_offsets)

    def _build_meta_layout(self) -> _MetaLayout:
        """Compute the on-device metadata region layout.

        v1 layout reserves the whole region for two mirrored base copies. v2
        layout carves a delta region from the same budget so incremental
        checkpoints can append small records between full snapshots. Total
        bytes are ``meta_total_bytes`` either way; the data slot region begins
        at ``meta_total_bytes`` regardless of mode, preserving cross-version
        slot offsets.
        """
        if self._use_incremental:
            base_copy_bytes = self._auto_base_copy_bytes()
            offsets = tuple(
                idx * base_copy_bytes for idx in range(self._meta_copy_count)
            )
            delta_region_off = offsets[-1] + base_copy_bytes
            delta_region_bytes = self.meta_total_bytes - delta_region_off
            delta_region_bytes = (
                delta_region_bytes // self.block_align
            ) * self.block_align
            return _MetaLayout(
                block_align=self.block_align,
                base_copy_bytes=base_copy_bytes,
                base_copy_offsets=offsets,
                delta_region_off=delta_region_off,
                delta_region_bytes=delta_region_bytes,
            )

        copy_bytes = self._v1_base_copy_bytes()
        if copy_bytes <= self.block_align:
            raise ValueError(
                "meta_total_bytes must provide room for at least two metadata copies"
            )
        offsets = tuple(idx * copy_bytes for idx in range(self._meta_copy_count))
        return _MetaLayout(
            block_align=self.block_align,
            base_copy_bytes=copy_bytes,
            base_copy_offsets=offsets,
            delta_region_off=offsets[-1] + copy_bytes,
            delta_region_bytes=0,
        )

    def _v1_base_copy_bytes(self) -> int:
        """Return the legacy v1 50/50 base-mirror size, block-aligned."""
        return (
            (self.meta_total_bytes // self._meta_copy_count) // self.block_align
        ) * self.block_align

    def _proposed_base_copy_bytes(self) -> int:
        """Return the block-aligned base-mirror size proposed for v2 layout.

        Honors ``meta_base_copy_bytes`` when explicitly set, otherwise
        targets ``_AUTO_BASE_FRACTION`` of ``meta_total_bytes`` so the
        remainder can host the delta region. Always block-aligned.
        """
        if self._configured_base_copy_bytes > 0:
            raw = self._configured_base_copy_bytes
        else:
            raw = int(self.meta_total_bytes * _AUTO_BASE_FRACTION)
        return (raw // self.block_align) * self.block_align

    def _auto_base_copy_bytes(self) -> int:
        """Return the per-base-mirror size used to construct the v2 layout."""
        base_copy_bytes = self._proposed_base_copy_bytes()
        if base_copy_bytes <= self.block_align:
            base_copy_bytes = 2 * self.block_align
        return base_copy_bytes

    def _can_fit_incremental_layout(self) -> bool:
        """Check that meta_total_bytes can host two bases plus a delta region.

        Each base copy needs at least 2 * block_align (header + at least one
        payload block). The delta region needs at least one delta record block.
        """
        base_copy_bytes = self._proposed_base_copy_bytes()
        if base_copy_bytes < 2 * self.block_align:
            return False
        bases_total = base_copy_bytes * self._meta_copy_count
        delta_total = self.meta_total_bytes - bases_total
        return delta_total >= self.block_align

    @staticmethod
    def _pack_base_header_v1(
        magic: bytes, version: int, seq: int, payload_len: int, crc: int
    ) -> bytes:
        return _META_HEADER_V1_STRUCT.pack(
            magic, int(version), int(seq), int(payload_len), int(crc)
        )

    @staticmethod
    def _pack_base_header_v2(
        magic: bytes,
        seq: int,
        payload_len: int,
        crc: int,
        delta_log_start_off: int,
    ) -> bytes:
        return _META_HEADER_V2_STRUCT.pack(
            magic,
            _META_VERSION_V2,
            int(seq),
            int(payload_len),
            int(crc),
            int(delta_log_start_off),
            0,
        )

    @staticmethod
    def _unpack_base_header(buf: bytes) -> Optional[dict[str, int]]:
        """Decode a base header buffer; returns None on malformed magic/version."""
        if len(buf) < _META_HEADER_V1_STRUCT.size:
            return None
        magic_v1, version, seq, payload_len, crc = _META_HEADER_V1_STRUCT.unpack(
            buf[: _META_HEADER_V1_STRUCT.size]
        )
        if version == _META_VERSION_V2:
            if len(buf) < _META_HEADER_V2_STRUCT.size:
                return None
            (
                magic_v2,
                version,
                seq,
                payload_len,
                crc,
                delta_log_start_off,
                _reserved,
            ) = _META_HEADER_V2_STRUCT.unpack(buf[: _META_HEADER_V2_STRUCT.size])
            return {
                "magic": magic_v2,
                "version": int(version),
                "seq": int(seq),
                "payload_len": int(payload_len),
                "crc": int(crc),
                "delta_log_start_off": int(delta_log_start_off),
            }
        return {
            "magic": magic_v1,
            "version": int(version),
            "seq": int(seq),
            "payload_len": int(payload_len),
            "crc": int(crc),
            "delta_log_start_off": 0,
        }

    def _read_meta_header(
        self,
        container_offset: int,
        payload_capacity: int,
    ) -> Optional[dict[str, int]]:
        """Read and validate a metadata checkpoint header.

        ``payload_capacity`` is the maximum payload-byte length permitted for
        this candidate offset. Different candidate layouts (active v2 vs
        legacy v1 fallback) carry different capacities, so the caller must
        pass the value matching the layout being probed.
        """
        buf = bytearray(self.block_align)
        try:
            self._rawdev().pread_into(
                container_offset, buf, self.block_align, self.block_align
            )
        except Exception:
            return None

        decoded = self._unpack_base_header(bytes(buf))
        if decoded is None:
            return None
        if decoded["magic"] != self.meta_magic:
            return None
        if decoded["version"] not in (_META_VERSION_V1, _META_VERSION_V2):
            return None
        if decoded["payload_len"] <= 0 or decoded["payload_len"] > payload_capacity:
            return None
        decoded["container_offset"] = int(container_offset)
        return decoded

    def _load_meta_payload(self, header: dict[str, int]) -> Optional[bytes]:
        """Load and CRC-validate a checkpoint payload for a metadata header."""
        payload_len = int(header["payload_len"])
        payload_off = int(header["container_offset"]) + self.block_align
        total_len = round_up(payload_len, self.block_align)
        buf = bytearray(total_len)
        try:
            self._rawdev().pread_into(payload_off, buf, payload_len, total_len)
        except Exception:
            return None

        payload = bytes(buf[:payload_len])
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != int(header["crc"]):
            return None
        return payload

    def _select_latest_checkpoint(
        self,
    ) -> tuple[Optional[dict[str, int]], Optional[bytes]]:
        """Return the newest valid checkpoint header and payload.

        When the active layout is v2, also probes the legacy v1 50/50
        offsets so a freshly-mounted v2 core can pick up either v1 mirror
        from a disk written by older code.
        """
        best_header: Optional[dict[str, int]] = None
        best_payload: Optional[bytes] = None
        for offset, capacity in self._candidate_base_offsets():
            header = self._read_meta_header(offset, payload_capacity=capacity)
            if header is None:
                continue
            payload = self._load_meta_payload(header)
            if payload is None:
                continue
            if best_header is None or int(header["seq"]) > int(best_header["seq"]):
                best_header = header
                best_payload = payload
        return best_header, best_payload

    def _candidate_base_offsets(self) -> list[tuple[int, int]]:
        """Yield ``(container_offset, payload_capacity)`` candidates to probe.

        Always includes the active layout. When in v2 mode, also includes
        the legacy v1 50/50 offsets so old disks remain readable. Duplicate
        offsets are removed (v2's base_0 always coincides with v1's mirror_0).
        """
        seen: set[int] = set()
        candidates: list[tuple[int, int]] = []
        active_capacity = self._meta_layout.base_payload_capacity
        for off in self._meta_layout.base_copy_offsets:
            if off in seen:
                continue
            seen.add(off)
            candidates.append((off, active_capacity))

        if self._use_incremental:
            v1_copy_bytes = self._v1_base_copy_bytes()
            if v1_copy_bytes > self.block_align:
                v1_capacity = v1_copy_bytes - self.block_align
                for idx in range(self._meta_copy_count):
                    off = idx * v1_copy_bytes
                    if off in seen:
                        continue
                    seen.add(off)
                    candidates.append((off, v1_capacity))
        return candidates

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
                "free_slots": list(self._free_slots),
                "entries": {
                    encoded_key: self._entry_to_op_payload(entry)
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
        target = self._meta_layout.base_copy_offsets[target_idx]

        payload_len = len(payload)
        payload_total_len = round_up(payload_len, self.block_align)
        payload_off = target + self.block_align
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        header_block = bytearray(self.block_align)
        if self._use_incremental:
            header_bytes = self._pack_base_header_v2(
                self.meta_magic, next_seq, payload_len, crc, delta_log_start_off=0
            )
        else:
            header_bytes = self._pack_base_header_v1(
                self.meta_magic, self.meta_version, next_seq, payload_len, crc
            )
        header_block[: len(header_bytes)] = header_bytes

        raw = self._rawdev()
        raw.pwrite_from_buffer(payload_off, payload, payload_len, payload_total_len)
        raw.pwrite_from_buffer(target, header_block, self.block_align, self.block_align)
        # In v2 mode, invalidate any stale delta records that may still carry
        # the new base's seq number. Replay walks ``delta_log_start_off`` first
        # and stops on a non-magic header, so a single zeroed block is enough
        # to cap the log without touching the rest of the region.
        if self._use_incremental and self._meta_layout.delta_region_bytes > 0:
            zero_block = bytes(self.block_align)
            raw.pwrite_from_buffer(
                self._meta_layout.delta_region_off,
                zero_block,
                self.block_align,
                self.block_align,
            )

        with self._lock:
            self._meta_seq = int(next_seq)
            self._meta_persisted = max(self._meta_persisted, int(dirty_total_snapshot))
            self._clear_dirty_log_locked()
            self._delta_seq = 0
            self._delta_tail_off = 0
            self._active_base_seq = int(next_seq)
            self._active_base_is_v2 = self._use_incremental
            self._last_full_ts = time.monotonic()
        return True

    def _checkpoint_once(self, force: bool) -> bool:
        """Write a metadata checkpoint when dirty and sufficiently idle.

        v2 (incremental) path: when an active v2 base exists and the dirty
        log fits, append a delta record. Falls back to a full snapshot when
        compaction is needed (first checkpoint, region full, count/time
        thresholds, oversized record, or migrating from v1).

        Holds ``self._writer_lock`` for the full attempt so concurrent
        callers cannot collide on delta tail offset or mirror seq.
        """
        with self._writer_lock:
            with self._lock:
                dirty = self._meta_dirty_total > self._meta_persisted
                idle_ok = self._inflight_io_count == 0 and (
                    time.monotonic() - self._last_io_ts
                ) >= (self.meta_idle_quiet_ms / 1000.0)

            if not dirty:
                return False
            if not force and not idle_ok:
                return False

            if self._use_incremental:
                payload, op_count, dirty_total_snapshot = self._snapshot_dirty_log()
                if payload is not None and not self._need_full_for_delta(payload):
                    if self._write_delta(payload, op_count, dirty_total_snapshot):
                        return True
                    # Fall through to full compaction if delta append failed.

            snapshot, dirty_total_snapshot = self._snapshot_state()
            payload = json.dumps(
                snapshot, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            return self._write_checkpoint(payload, dirty_total_snapshot)

    def _snapshot_dirty_log(
        self,
    ) -> tuple[Optional[bytes], int, int]:
        """Atomically swap out the in-memory dirty log and JSON-encode it.

        Returns a triple ``(payload, op_count, dirty_total_snapshot)``. When
        no dirty ops are pending, ``payload`` is ``None``.
        """
        with self._lock:
            if not self._dirty_log:
                return None, 0, self._meta_dirty_total
            ops = self._dirty_log
            self._dirty_log = []
            dirty_total_snapshot = self._meta_dirty_total
        try:
            payload = json.dumps(
                {"ops": ops}, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        except Exception:
            # Restore the log so a later compaction can retry.
            with self._lock:
                self._dirty_log = ops + self._dirty_log
            raise
        return payload, len(ops), dirty_total_snapshot

    def _need_full_for_delta(self, payload: bytes) -> bool:
        """Decide whether the next checkpoint must be a full compaction."""
        if not self._active_base_is_v2:
            return True
        if self._meta_seq == 0:
            return True
        layout = self._meta_layout
        if layout.delta_region_bytes <= 0:
            return True
        record_bytes = self._delta_record_total_bytes(len(payload))
        if record_bytes > layout.delta_region_bytes:
            return True
        if self._delta_tail_off + record_bytes > layout.delta_region_bytes:
            return True
        if self._delta_seq + 1 > self.meta_full_checkpoint_max_deltas:
            return True
        if (
            time.monotonic() - self._last_full_ts
            > self.meta_full_checkpoint_interval_sec
        ):
            return True
        watermark = max(0, min(100, self.meta_delta_high_watermark_pct))
        used_after = self._delta_tail_off + record_bytes
        if used_after * 100 >= layout.delta_region_bytes * watermark:
            return True
        return False

    def _delta_record_total_bytes(self, payload_len: int) -> int:
        """Block-aligned record size for a delta payload of ``payload_len`` bytes."""
        return round_up(
            _DELTA_RECORD_HEADER_STRUCT.size + int(payload_len), self.block_align
        )

    def _replay_delta_log(self, start_off: int) -> int:
        """Walk the delta region starting at ``start_off`` and apply records.

        Stops at the first record that fails any check (bad magic, stale
        ``base_seq``, non-monotonic ``delta_seq``, oversized payload, CRC
        mismatch, or read error). Earlier records are durable and applied;
        anything past the stop point is treated as torn or stale.

        Returns the number of records successfully applied.
        """
        layout = self._meta_layout
        if start_off < 0 or start_off >= layout.delta_region_bytes:
            if start_off < 0:
                logger.warning(
                    "RawBlockCore: ignoring negative delta_log_start_off=%d",
                    start_off,
                )
            self._delta_tail_off = 0
            self._delta_seq = 0
            return 0

        off = start_off
        applied = 0
        expected_next = self._delta_seq + 1
        while off + _DELTA_RECORD_HEADER_STRUCT.size <= layout.delta_region_bytes:
            absolute = layout.delta_region_off + off
            header = self._read_delta_header(absolute, expected_next, layout, off)
            if header is None:
                break
            ops = self._load_delta_ops(absolute, header)
            if ops is None:
                break
            if not self._apply_delta_ops(ops):
                break

            applied += 1
            self._delta_seq = int(header["delta_seq"])
            off += int(header["record_bytes"])
            expected_next = self._delta_seq + 1

        self._delta_tail_off = off
        return applied

    def _read_delta_header(
        self,
        absolute_off: int,
        expected_next_seq: int,
        layout: "_MetaLayout",
        relative_off: int,
    ) -> Optional[dict[str, int]]:
        """Read and validate one delta record's header block.

        Returns a dict with the validated fields plus the derived
        ``record_bytes``, or None if any sanity check fails. The caller
        treats None as "stop replay here".
        """
        buf = bytearray(self.block_align)
        try:
            self._rawdev().pread_into(
                absolute_off, buf, self.block_align, self.block_align
            )
        except Exception:
            return None

        try:
            (
                magic,
                rec_version,
                base_seq,
                delta_seq,
                payload_len,
                payload_crc,
                total_blocks,
                op_count,
                _reserved,
            ) = _DELTA_RECORD_HEADER_STRUCT.unpack(
                bytes(buf[: _DELTA_RECORD_HEADER_STRUCT.size])
            )
        except struct.error:
            return None

        if magic != _DELTA_RECORD_MAGIC:
            return None
        if rec_version != _DELTA_RECORD_VERSION:
            return None
        if int(base_seq) != self._active_base_seq:
            return None
        if int(delta_seq) != expected_next_seq:
            return None
        if int(total_blocks) <= 0:
            return None
        record_bytes = int(total_blocks) * self.block_align
        if relative_off + record_bytes > layout.delta_region_bytes:
            return None
        max_payload = record_bytes - _DELTA_RECORD_HEADER_STRUCT.size
        if int(payload_len) <= 0 or int(payload_len) > max_payload:
            return None
        return {
            "delta_seq": int(delta_seq),
            "payload_len": int(payload_len),
            "payload_crc": int(payload_crc),
            "op_count": int(op_count),
            "record_bytes": int(record_bytes),
        }

    def _load_delta_ops(
        self, absolute_off: int, header: dict[str, int]
    ) -> Optional[list[Any]]:
        """Read, CRC-check, and decode the ops list of one delta record."""
        record_bytes = header["record_bytes"]
        payload_len = header["payload_len"]
        buf = bytearray(record_bytes)
        try:
            self._rawdev().pread_into(absolute_off, buf, record_bytes, record_bytes)
        except Exception:
            return None
        start = _DELTA_RECORD_HEADER_STRUCT.size
        payload = bytes(buf[start : start + payload_len])
        if (zlib.crc32(payload) & 0xFFFFFFFF) != header["payload_crc"]:
            return None
        try:
            doc = json.loads(payload.decode("utf-8"))
        except Exception:
            return None
        ops = doc.get("ops") if isinstance(doc, dict) else None
        if not isinstance(ops, list):
            return None
        if header["op_count"] != len(ops):
            return None
        return ops

    def _apply_delta_ops(self, ops: list[Any]) -> bool:
        """Apply a decoded list of delta ops to the live in-memory state."""
        with self._lock:
            for op in ops:
                if not isinstance(op, dict):
                    return False
                kind = op.get("op")
                next_slot = op.get("next_slot")
                if isinstance(next_slot, int) and next_slot > self._next_slot:
                    self._next_slot = next_slot
                if kind == "put":
                    if not self._apply_delta_put_locked(op):
                        return False
                elif kind == "delete":
                    self._apply_delta_delete_locked(op)
                elif kind == "freelist_push":
                    freed = op.get("freed_slot")
                    if isinstance(freed, int):
                        self._safe_append_free_slot_locked(int(freed))
                else:
                    logger.warning(
                        "RawBlockCore: ignoring unknown delta op kind=%r", kind
                    )
        return True

    def _apply_delta_put_locked(self, op: dict[str, Any]) -> bool:
        """Apply a put op while ``self._lock`` is held."""
        encoded_key = op.get("k")
        offset = op.get("offset")
        size = op.get("size")
        if not isinstance(encoded_key, str):
            return False
        if not isinstance(offset, int) or not isinstance(size, int):
            return False
        if not self._is_valid_checkpoint_entry(int(offset), int(size)):
            return False

        meta = self._decode_put_op_meta(encoded_key, int(offset), int(size), op)
        prior = self._index.get(encoded_key)
        self._index[encoded_key] = _Entry(offset=int(offset), size=int(size), meta=meta)

        slot = self._offset_to_slot(int(offset))
        if 0 <= slot < self._max_slots and slot in self._free_slots:
            self._free_slots = [s for s in self._free_slots if s != slot]
        if prior is not None and int(prior.offset) != int(offset):
            old_slot = self._offset_to_slot(int(prior.offset))
            self._append_free_slot_locked(old_slot)
        return True

    def _decode_put_op_meta(
        self, encoded_key: str, offset: int, size: int, op: dict[str, Any]
    ) -> DiskCacheMetadata:
        """Build a ``DiskCacheMetadata`` from a put-op dict's optional fields."""
        shape_list = op.get("shape")
        shape = torch.Size(list(shape_list)) if isinstance(shape_list, list) else None
        fmt_name = op.get("fmt")
        if isinstance(fmt_name, str) and fmt_name in MemoryFormat.__members__:
            fmt = MemoryFormat[fmt_name]
        else:
            fmt = MemoryFormat.UNDEFINED
        cached_positions_list = op.get("cached_positions")
        cached_positions = (
            torch.tensor(cached_positions_list, dtype=torch.long)
            if isinstance(cached_positions_list, list)
            else None
        )
        dtype = self._recover_checkpoint_dtype(encoded_key, op.get("dtype"))
        return DiskCacheMetadata(
            path=f"{self.device_path}@{offset}",
            size=size,
            shape=shape,
            dtype=dtype,
            cached_positions=cached_positions,
            fmt=fmt,
            pin_count=0,
        )

    def _apply_delta_delete_locked(self, op: dict[str, Any]) -> None:
        """Apply a delete op while ``self._lock`` is held."""
        encoded_key = op.get("k")
        if not isinstance(encoded_key, str):
            return
        removed = self._index.pop(encoded_key, None)
        self._lock_refcnt.pop(encoded_key, None)
        freed_slot = op.get("freed_slot")
        if isinstance(freed_slot, int):
            self._safe_append_free_slot_locked(int(freed_slot))
        elif removed is not None:
            self._append_free_slot_locked(self._offset_to_slot(int(removed.offset)))

    def _write_delta(
        self,
        payload: bytes,
        op_count: int,
        dirty_total_snapshot: int,
    ) -> bool:
        """Append a delta record to the delta log region.

        Caller must hold ``self._writer_lock``: this method reads
        ``_delta_tail_off`` / ``_delta_seq`` outside ``self._lock`` and
        relies on the writer lock to keep concurrent checkpoints from
        racing on those fields or on the underlying device offset.

        On success, advances ``_delta_seq`` / ``_delta_tail_off`` and the
        persisted-counter watermark. Returns ``False`` if the record cannot
        fit; the caller should fall back to a full compaction.
        """
        layout = self._meta_layout
        if layout.delta_region_bytes <= 0 or not self._active_base_is_v2:
            return False
        record_bytes = self._delta_record_total_bytes(len(payload))
        if record_bytes > layout.delta_region_bytes:
            return False
        if self._delta_tail_off + record_bytes > layout.delta_region_bytes:
            return False

        next_delta_seq = self._delta_seq + 1
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        total_blocks = record_bytes // self.block_align

        record = bytearray(record_bytes)
        _DELTA_RECORD_HEADER_STRUCT.pack_into(
            record,
            0,
            _DELTA_RECORD_MAGIC,
            _DELTA_RECORD_VERSION,
            int(self._active_base_seq),
            int(next_delta_seq),
            int(len(payload)),
            int(crc),
            int(total_blocks),
            int(op_count),
            0,
        )
        record[
            _DELTA_RECORD_HEADER_STRUCT.size : _DELTA_RECORD_HEADER_STRUCT.size
            + len(payload)
        ] = payload

        write_off = layout.delta_region_off + self._delta_tail_off
        try:
            self._rawdev().pwrite_from_buffer(
                write_off, bytes(record), record_bytes, record_bytes
            )
        except Exception as e:
            logger.warning(
                "RawBlockCore: delta write failed at off=%d: %s", write_off, e
            )
            return False

        with self._lock:
            self._delta_seq = next_delta_seq
            self._delta_tail_off += record_bytes
            self._meta_persisted = max(self._meta_persisted, int(dirty_total_snapshot))
        return True

    def _is_valid_checkpoint_entry(self, offset: int, size: int) -> bool:
        """Return whether a checkpoint entry references a valid data slot."""
        if offset < self._data_base_offset:
            return False
        rel = offset - self._data_base_offset
        if rel % self.slot_bytes != 0:
            return False
        slot = rel // self.slot_bytes
        if slot < 0 or slot >= self._max_slots:
            return False
        return 0 < size <= (self.slot_bytes - self.header_bytes)

    def _apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Apply decoded checkpoint state after validating layout fields."""
        if not isinstance(data, dict):
            return False
        if int(data.get("version", 0)) != 1:
            logger.warning(
                "Device metadata payload version mismatch (got %r, expected 1); "
                "ignoring metadata",
                data.get("version"),
            )
            return False
        if data.get("device_path") and data.get("device_path") != self.device_path:
            logger.warning("Device metadata device_path mismatch; ignoring metadata")
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

        raw_free_slots = data.get("free_slots", [])
        if not isinstance(raw_free_slots, list):
            logger.warning("Device metadata free_slots is invalid; ignoring metadata")
            return False
        free_slots: list[int] = []
        seen_slots: set[int] = set()
        for raw_slot in raw_free_slots:
            try:
                slot = int(raw_slot)
            except Exception:
                logger.warning(
                    "Device metadata free_slots contains non-integer; ignoring metadata"
                )
                return False
            if slot < 0 or slot >= self._max_slots:
                logger.warning(
                    "Device metadata free_slots contains out-of-range slot %d; "
                    "ignoring metadata",
                    slot,
                )
                return False
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            free_slots.append(slot)

        with self._lock:
            self._next_slot = next_slot
            self._free_slots = free_slots
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
            self._free_slots = [
                slot for slot in self._free_slots if slot not in used_slots
            ]

            self._meta_dirty_total = 0
            self._meta_persisted = 0
            self._clear_dirty_log_locked()
            self._delta_seq = 0
            self._delta_tail_off = 0
            # External callers may hand us an arbitrary state that does not
            # match what's on disk. Force the next checkpoint to write a
            # full snapshot so any stale on-disk delta records (which still
            # carry the previous active_base_seq) cannot be replayed on top
            # of the new state. ``_load_checkpoint_from_device`` overrides
            # these immediately after with the values from the just-loaded
            # base header.
            self._meta_seq = 0
            self._active_base_seq = 0
            self._active_base_is_v2 = False
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
                    freed_slot = self._offset_to_slot(int(removed_entry.offset))
                    self._append_free_slot_locked(freed_slot)
                    self._record_delete_op_locked(encoded_key, freed_slot)
            # One dirty event per dropped entry, matching the per-op accounting
            # used by put/delete so ``metadata_dirty_total`` stays in sync with
            # the dirty-log size.
            self._meta_dirty_total += len(to_drop)

        logger.warning(
            "RawBlockCore dropped %d stale metadata entries after "
            "slot-header validation",
            len(to_drop),
        )

    def _load_checkpoint_from_device(self) -> None:
        """Load the newest valid checkpoint from the raw device if present.

        Applies the base snapshot, replays any v2 delta records, and only
        then runs slot-header validation against the final state. Validating
        before replay would let a subsequent delta op resurrect an entry
        that the base validation just dropped.
        """
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
        if not self._apply_loaded_state(data):
            logger.warning("RawBlockCore: metadata payload rejected by checks")
            return
        self._meta_seq = int(header["seq"])
        self._active_base_seq = self._meta_seq
        version = int(header.get("version", _META_VERSION_V1))
        self._active_base_is_v2 = version == _META_VERSION_V2
        self._last_full_ts = time.monotonic()

        delta_count = 0
        if self._active_base_is_v2 and self._meta_layout.delta_region_bytes > 0:
            delta_count = self._replay_delta_log(
                int(header.get("delta_log_start_off", 0))
            )

        if self.meta_verify_on_load:
            self._validate_loaded_entries()

        logger.info(
            "RawBlockCore loaded checkpoint (entries=%d next_slot=%d seq=%d "
            "version=%d deltas_applied=%d device=%s)",
            len(self._index),
            self._next_slot,
            self._meta_seq,
            version,
            delta_count,
            self.device_path,
        )
