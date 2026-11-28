# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Optional
import bisect
import ctypes
import json
import os
import random
import struct
import threading
import time
import zlib

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.l2_adapters.raw_block.extent_allocator import (
    ExtentAllocator,
)
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    DEFAULT_IOURING_QUEUE_DEPTH,
    IoUringBlockIOConfig,
    PlacementId,
    PosixBlockIOConfig,
    RawBlockIO,
    RawBlockIOConfig,
    make_raw_block_io,
    normalize_raw_block_io_engine,
    normalize_raw_block_placement_ids,
)
from lmcache.v1.distributed.l2_adapters.raw_block.key_codec import (
    RawBlockKeySpec,
    object_key_slot_identity,
)
from lmcache.v1.distributed.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    DiskCacheMetadata,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)


_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1
_META_HEADER_STRUCT = struct.Struct("<8sIQQI")
_MAX_PUT_MANY_IO_URING_BATCH_KEYS = 64
_MAX_FIXED_BUFFER_REGION_BYTES = 1 << 30
# Checkpoint payload format. Version 1 is the fixed-slot layout; it is still
# accepted on load and converted.
_CHECKPOINT_FORMAT_VERSION = 2
# The first data block holds this marker. A fixed-slot binary that does not
# understand format 2 starts empty and writes its first object at that block,
# so a missing marker means such a run reused the space (see
# _prepare_data_region).
_MARKER_MAGIC = b"LMCVAR01"
# Write generations only need to differ across writes and runs, not resist
# guessing; an own instance (seeded from os.urandom) is several times cheaper
# than secrets and unaffected by random.seed() elsewhere in the process.
# Reseeded in a forked child, or every child of one parent would repeat them.
_GENERATION_RNG = random.Random()
os.register_at_fork(after_in_child=_GENERATION_RNG.seed)
# A header block that was read but holds no valid header (e.g. invalidated),
# as opposed to None: the read itself failed.
_NO_HEADER = (0, 0, 0)

# Slot-header validation issues many small pread calls during POSIX restart
# recovery. Use a conservative reader count to expose device parallelism without
# relying on high thread counts; io_uring recovery should use batched reads.
DEFAULT_RECOVERY_READ_THREADS = 8
# Warn once when a checkpoint payload uses this share of one metadata copy.
_META_USAGE_WARN_RATIO = 0.8


def _without_overlaps(
    entries: list[tuple[str, "_Entry"]],
) -> list[tuple[str, "_Entry"]]:
    """Drop every entry whose range overlaps another entry's.

    A consistent checkpoint never has overlapping entries, so an overlap means
    corruption and neither entry can be trusted.
    """
    order = sorted(range(len(entries)), key=lambda i: entries[i][1].offset)
    bad: set[int] = set()
    reach_end, reach_idx = -1, -1
    for i in order:
        entry = entries[i][1]
        if entry.offset < reach_end:
            bad.update((i, reach_idx))
        if entry.offset + entry.length > reach_end:
            reach_end, reach_idx = entry.offset + entry.length, i
    return [item for i, item in enumerate(entries) if i not in bad]


def round_up(x: int, align: int) -> int:
    """Round a value up to the nearest alignment boundary.

    Args:
        x: Value to align.
        align: Positive alignment in bytes.

    Returns:
        ``x`` rounded up to a multiple of ``align``.
    """
    return ((x + align - 1) // align) * align


def _allocation_span_bytes(memory_obj: MemoryObj) -> int:
    """Return how many bytes the object's allocation spans from its data start.

    Args:
        memory_obj: Memory object whose tensor starts at its data pointer.

    Returns:
        The smaller of the backing tensor size and the physical size, or 0 when
        the object has no local tensor.
    """
    tensor = memory_obj.raw_tensor
    if tensor is None:
        return 0
    return min(tensor.nbytes, memory_obj.get_physical_size())


@dataclass(frozen=True)
class RawBlockCoreConfig:
    """Configuration for RawBlockCore device layout, I/O, and checkpoints."""

    device_path: str
    capacity_bytes: int
    block_align: int
    header_bytes: int
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


@dataclass(slots=True)
class _Entry:
    """A committed object: header at ``offset``, then its payload."""

    offset: int
    length: int
    """Allocated bytes: header plus block-aligned payload."""
    size: int
    meta: DiskCacheMetadata
    # Runtime-only FDP placement identifier of the write, for reuse affinity.
    placement_id: PlacementId = None
    # Random and odd per write, also in the object header: recovery trusts an
    # entry only if the header on the device came from that very write. An
    # entry imported from fixed slots takes the header's bytes there (-1 until
    # the import reads them).
    generation: int = 0


class _CheckpointEntry(msgspec.Struct, gc=False):
    """A format-2 JSON record with only acyclic metadata fields."""

    offset: int
    size: int
    gen: int
    shape: torch.Size | None
    dtype: str | None
    fmt: str | None
    cached_positions: list[Any] | None


@dataclass(slots=True)
class _Inflight:
    offset: int
    length: int
    meta: DiskCacheMetadata
    placement_id: PlacementId = None
    canceled: bool = False
    generation: int = 0


@dataclass(frozen=True)
class RawBlockPutManyResult:
    """Result of a RawBlockCore batched write."""

    results: list[bool]
    stored_keys: list[str]
    stored_bytes: list[int]
    """Device bytes allocated to each of ``stored_keys``."""


class RawBlockCore:
    """
    Raw-block storage engine for the distributed L2 path.

    This class owns the raw-device I/O path, space allocation, checkpoint and
    recovery, and lock refcounts that protect objects from deletion while in
    use. Each object gets ``header_bytes`` plus its block-aligned payload, so
    no object size is configured.
    """

    def __init__(self, config: RawBlockCoreConfig):
        """Initialize the raw-block storage engine.

        Args:
            config: Raw-block device, layout, I/O, and checkpoint settings.

        Raises:
            ValueError: If the supplied configuration is invalid.
            RuntimeError: If the raw device cannot be opened, the computed
                layout cannot fit metadata and a data region, or a startup
                checkpoint or marker write fails.

        Notes:
            If initialization opens the device and a later recovery step fails,
            the partially opened resources are closed before the exception is
            re-raised.
        """
        self.device_path = config.device_path
        self.capacity_bytes = int(config.capacity_bytes)
        self.block_align = int(config.block_align)
        self.header_bytes = int(config.header_bytes)
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
        self.use_uring_cmd = bool(config.use_uring_cmd)
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
        self._recovery_read_threads = DEFAULT_RECOVERY_READ_THREADS
        # The posix engine writes one request after another; deletes spread
        # their header writes over these threads.
        self._invalidate_pool: Optional[ThreadPoolExecutor] = None
        if not self.device_path:
            raise ValueError("RawBlockCore requires a non-empty device_path")
        if self.block_align <= 0 or (self.block_align & (self.block_align - 1)) != 0:
            raise ValueError(
                f"block_align must be a power of 2, got {self.block_align}"
            )
        if self.header_bytes < 32:
            # Magic, key identity, payload length and write generation.
            raise ValueError("header_bytes must be >= 32")
        if self.header_bytes % self.block_align != 0:
            raise ValueError("header_bytes must be a multiple of block_align")
        if self.meta_total_bytes <= self.block_align:
            raise ValueError("meta_total_bytes must provide room for metadata header")
        if self.meta_total_bytes % self.block_align != 0:
            raise ValueError("meta_total_bytes must be a multiple of block_align")
        if len(self.meta_magic) != 8:
            raise ValueError("meta_magic must be exactly 8 bytes")
        if self.meta_version <= 0:
            raise ValueError("meta_version must be > 0")

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
        # Header invalidations in flight; close() waits for them to finish.
        self._invalidations_inflight = 0
        self._invalidations_done = threading.Condition(self._lock)
        self._index: dict[str, _Entry] = {}
        self._lock_refcnt: dict[str, int] = {}
        self._inflight: dict[str, _Inflight] = {}

        self._allocator: Optional[ExtentAllocator] = None
        # Retired (quarantined) ranges, sorted and disjoint, and their total.
        self._retired: list[tuple[int, int]] = []
        self._retired_bytes: int = 0
        self._last_payload_uecc: str | None = None
        # End of the highest range ever allocated: below it space was written
        # before, so FDP affinity counters only count reuse below it.
        self._high_water: int = 0
        self._fdp_slot_affinity_hit_count: int = 0
        self._fdp_slot_affinity_fallback_count: int = 0
        self._alloc_failure_count: int = 0
        self._invalidation_failures: int = 0
        self._header_invalidations: int = 0
        # Metadata containers of a rejected checkpoint's layout, voided with
        # this layout's at startup (see _note_foreign_layout).
        self._foreign_container_offsets: list[int] = []
        self._no_space_logged_at: float = float("-inf")
        self._no_space_since_log: int = 0
        self._effective_capacity_bytes: int = 0
        self._data_base_offset: int = 0
        self._data_bytes: int = 0

        self._io_engine: Optional[RawBlockIO] = None
        self._closed = False
        self._worker_error_logged: bool = False

        self._meta_seq: int = 0
        self._meta_usage_warned = False
        self._meta_dirty_total: int = 0
        self._meta_persisted: int = 0
        self._inflight_io_count: int = 0
        self._last_io_ts: float = time.monotonic()
        # Start of the wait for the next checkpoint: the last checkpoint's
        # snapshot, or when unpersisted metadata was first seen after a quiet
        # period.
        self._meta_dirty_since: Optional[float] = None
        self._meta_stop_evt = threading.Event()
        self._meta_thread: Optional[threading.Thread] = None

        self._io_engine = make_raw_block_io(
            self._build_io_engine_config(config),
        )
        self.use_odirect = self._io_engine.use_odirect
        self.max_data_transfer_size = self._io_engine.max_data_transfer_size
        if self.io_engine == "posix" and self._recovery_read_threads > 1:
            self._invalidate_pool = ThreadPoolExecutor(
                max_workers=self._recovery_read_threads,
                thread_name_prefix="rawblk-invalidate",
            )

        try:
            self._ensure_capacity_and_layout()
            # New checkpoints must supersede every copy on the device, also
            # one this run does not load (skipped or rejected); otherwise the
            # next restart would load that older copy instead.
            self._meta_seq = self._max_checkpoint_seq()
            loaded_format = None
            if self.load_checkpoint_on_init:
                loaded_format = self._load_checkpoint_from_device()
            else:
                logger.info("RawBlockCore: skipping on-device metadata checkpoint load")
            self._prepare_data_region(loaded_format)

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

    def _build_io_engine_config(self, config: RawBlockCoreConfig) -> RawBlockIOConfig:
        """Assemble the normalized config for the configured I/O engine.

        Args:
            config: The raw-block core configuration whose I/O fields describe
                the target engine.

        Returns:
            A ``PosixBlockIOConfig`` for ``io_engine='posix'`` or an
            ``IoUringBlockIOConfig`` otherwise.
        """
        if self.io_engine == "posix":
            return PosixBlockIOConfig(
                device_path=self.device_path,
                block_align=self.block_align,
                io_engine=self.io_engine,
                use_odirect=self.use_odirect,
            )
        return IoUringBlockIOConfig(
            device_path=self.device_path,
            block_align=self.block_align,
            io_engine=self.io_engine,
            use_odirect=self.use_odirect,
            use_uring_cmd=self.use_uring_cmd,
            iouring_queue_depth=config.iouring_queue_depth,
            max_data_transfer_size=config.max_data_transfer_size,
        )

    @property
    def _requires_transfer_alignment(self) -> bool:
        """Return whether I/O transfers require block alignment.

        Returns:
            True when transfers must be aligned to ``self.block_align``.
            This is required for O_DIRECT I/O and for io_uring_cmd operations.
        """
        return self.use_odirect or self.use_uring_cmd

    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        """Fetch NVMe FDP placement/RUH status from the raw device.

        Returns:
            List of ``(placement_id, ruh_id)`` tuples.

        Raises:
            RuntimeError: If the raw device binding or target device cannot
                provide FDP status.
        """
        assert self._io_engine is not None
        return self._io_engine.fetch_fdp_status()

    def register_io_buffers_from_allocator(self, memory_allocator: Any) -> None:
        """Register allocator pages for zero-copy I/O when the allocator exposes them.

        Args:
            memory_allocator: Local CPU allocator that may expose
                ``get_paged_buffers()``.

        Raises:
            Exception: Propagates Rust registration errors after logging.
        """
        assert self._io_engine is not None
        if not self._io_engine.supports_buffer_registration():
            return
        paged_buffers = getattr(memory_allocator, "get_paged_buffers", None)
        if not callable(paged_buffers):
            logger.warning(
                "RawBlockCore: allocator does not expose paged buffers; "
                "zero-copy I/O is disabled"
            )
            return
        buffers = paged_buffers()
        if not buffers:
            logger.warning(
                "RawBlockCore: allocator returned no paged buffers; "
                "zero-copy I/O is disabled"
            )
            return
        buffer_ptrs = [buf.data_ptr() for buf in buffers]
        buffer_sizes = [buf.numel() * buf.element_size() for buf in buffers]
        self._io_engine.register_io_buffers(buffer_ptrs, buffer_sizes)
        logger.info(
            "RawBlockCore: registered %d paged buffers for zero-copy I/O",
            len(buffers),
        )

    def register_io_buffer_range(self, ptr: int, size: int) -> None:
        """Register one stable memory range for zero-copy I/O.

        Args:
            ptr: Base address of the stable memory range.
            size: Size of the stable memory range in bytes.

        Raises:
            ValueError: If ``ptr`` or ``size`` is not positive.
            Exception: Propagates errors from the Rust registration API.

        Notes:
            The range is divided into regions no larger than 1 GiB, which is
            the maximum size supported for one registered buffer.
        """
        assert self._io_engine is not None
        if not self._io_engine.supports_buffer_registration():
            return

        ptr = int(ptr)
        size = int(size)
        if ptr <= 0:
            raise ValueError("fixed-buffer range pointer must be > 0")
        if size <= 0:
            raise ValueError("fixed-buffer range size must be > 0")

        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []
        next_ptr = ptr
        remaining = size
        while remaining > 0:
            region_size = min(remaining, _MAX_FIXED_BUFFER_REGION_BYTES)
            buffer_ptrs.append(next_ptr)
            buffer_sizes.append(region_size)
            next_ptr += region_size
            remaining -= region_size

        self._io_engine.register_io_buffers(buffer_ptrs, buffer_sizes)
        logger.info(
            "RawBlockCore: registered %d regions covering %d bytes for zero-copy I/O",
            len(buffer_ptrs),
            size,
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
            stops at the first missing key. A terminal worker failure returns
            an empty list without locking entries.
        """
        assert self._io_engine is not None
        worker_error = self._io_engine.worker_error()
        if worker_error is not None:
            self._log_worker_error_once(worker_error)
            return []
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
        """Return the raw-device offset of an indexed key's object.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            Offset of the object's header in bytes, or None when the key is
            not indexed.
        """
        with self._lock:
            entry = self._index.get(encoded_key)
            return None if entry is None else int(entry.offset)

    def metadata_container_offsets(self) -> list[int]:
        """Return checkpoint metadata container offsets in bytes."""
        return self._meta_container_offsets()

    def data_base_offset(self) -> int:
        """Return the byte offset where the raw-block data region begins."""
        return int(self._data_base_offset)

    def put_many(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
        placement_ids: Sequence[PlacementId] | None = None,
    ) -> RawBlockPutManyResult:
        """Persist a batch of memory objects into raw-block space.

        With ``io_engine='io_uring'`` and more than one key, writes are
        submitted in batches and failure is no longer independent per key: a
        device write failure rolls back every key submitted in the same batch.
        Batches are bounded, so a request larger than one batch can leave
        earlier batches committed while a later one rolls back.

        With ``io_engine='io_uring'`` and zero-copy enabled, a payload that is
        not block-aligned is written straight from its allocation: the padding
        after the logical payload, up to the next block boundary, is zeroed in
        place. The logical payload is unchanged.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``.
            objs: Memory objects whose byte buffers should be written.
            placement_ids: Optional per-key FDP placement identifiers for
                raw-block writes. ``None`` omits the directive; explicit identifier
                0 is rejected because default writes already use that mapping.

        Returns:
            Per-key success results, newly stored encoded keys and their
            allocated bytes. A key is reported as failed when no free range is
            long enough for it. Space reclamation is owned by the
            adapter/controller calling ``delete_many_with_sizes``.

        Raises:
            ValueError: If either sequence is empty, sequence lengths do not
                match, or a placement identifier is 0.
            RuntimeError: If the native io_uring worker has permanently failed.
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
        self.raise_if_failed()

        if self.io_engine == "io_uring" and len(keys) > 1:
            return self._put_many_batch_io(keys, objs, per_key_placement_ids)

        results = [False] * len(keys)
        stored_keys: list[str] = []
        stored_bytes: list[int] = []

        for i, (key, obj) in enumerate(zip(keys, objs, strict=False)):
            placement_id = per_key_placement_ids[i]
            if self._closed:
                break
            payload_len = len(obj.byte_array)

            with self._lock:
                if key.encoded in self._index:
                    results[i] = True
                    continue
                if key.encoded in self._inflight:
                    continue
                inflight = self._reserve_locked(key, payload_len, obj, placement_id)
                if inflight is None:
                    continue

            success = self._write_one(
                key,
                obj,
                inflight.offset,
                inflight.generation,
                placement_id=placement_id,
            )

            with self._lock:
                entry = self._finish_write_locked(key.encoded, success)
                if entry is not None:
                    results[i] = True
                    stored_keys.append(key.encoded)
                    stored_bytes.append(entry.length)

        return RawBlockPutManyResult(
            results=results,
            stored_keys=stored_keys,
            stored_bytes=stored_bytes,
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
            A list of booleans aligned with ``encoded_keys``. A permanently
            failed io_uring worker reports all misses without
            locking keys.
        """
        worker_error = self._worker_error()
        if worker_error is not None:
            self._log_worker_error_once(worker_error)
            return [False] * len(encoded_keys)
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
        on_quarantine: Callable[[list[tuple[str, int]]], None] | None = None,
    ) -> list[bool]:
        """Load raw-block payloads into caller-provided memory objects.

        With zero-copy enabled, a read may fill the destination up to the
        block-aligned end of the stored payload, including allocation padding
        beyond the logical destination.

        Args:
            encoded_keys: Ordered encoded raw-block keys to load.
            objs: Destination memory objects. Buffers must remain valid until
                this method returns. A destination shorter than its stored
                payload is reported as an unsuccessful load.
            on_quarantine: Called after retiring a failed object's space, with
                the encoded keys removed from the index and their allocated
                bytes, outside the core lock.

        Returns:
            A list of per-key load success booleans aligned with
            ``encoded_keys``. A confirmed passthrough payload UECC retires its
            object's space for this open handle. Other successfully loaded keys remain
            hits. A terminal worker failure reports all misses.

        Raises:
            ValueError: If either sequence is empty or the sequence lengths do
                not match.
        """
        if not encoded_keys or not objs:
            raise ValueError("encoded_keys and objs must be non-empty")
        if len(encoded_keys) != len(objs):
            raise ValueError("encoded_keys and objs must have the same length")
        assert self._io_engine is not None
        worker_error = self._io_engine.worker_error()
        if worker_error is not None:
            self._log_worker_error_once(worker_error)
            return [False] * len(encoded_keys)

        with self._lock:
            items = [
                (encoded_key, self._index.get(encoded_key))
                for encoded_key in encoded_keys
            ]
            self._inflight_io_count += 1

        results = [False] * len(encoded_keys)
        try:
            read_indices: list[int] = []
            read_offsets: list[int] = []
            read_buffers: list[Any] = []
            read_payload_lens: list[int] = []
            read_total_lens: list[int] = []

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
                    if len(buf) < payload_len:
                        raise ValueError(
                            "destination buffer shorter than stored payload"
                        )

                    direct_view = self._build_direct_odirect_view(
                        memory_obj=objs[i],
                        payload_len=payload_len,
                        total_len=total_len,
                        buffer_len=len(buf),
                        zero_tail=False,
                    )
                    if direct_view is not None:
                        read_buffer = direct_view
                        read_payload_len = (
                            total_len if len(direct_view) >= total_len else payload_len
                        )
                    else:
                        read_buffer = buf
                        read_payload_len = payload_len

                    read_indices.append(i)
                    read_offsets.append(entry.offset + self.header_bytes)
                    read_buffers.append(read_buffer)
                    read_payload_lens.append(read_payload_len)
                    read_total_lens.append(total_len)
                except Exception as e:
                    logger.error("RawBlockCore load failed for %s: %s", encoded_key, e)

            if read_indices:
                unrecovered_reads: dict[int, str] = {}
                try:
                    io_results = self._io_engine.read_buffers(
                        read_offsets,
                        read_buffers,
                        read_payload_lens,
                        read_total_lens,
                        unrecovered_reads=unrecovered_reads,
                    )
                except Exception as e:
                    logger.error("RawBlockCore batched load failed: %s", e)
                    io_results = [False] * len(read_indices)

                if unrecovered_reads:
                    removed = []
                    with self._lock:
                        for read_idx, error in unrecovered_reads.items():
                            key, entry = items[read_indices[read_idx]]
                            assert entry is not None
                            removed.extend(self._quarantine_locked(key, entry, error))
                    if on_quarantine is not None:
                        on_quarantine(removed)

                worker_error = self._io_engine.worker_error()
                if worker_error is not None:
                    self._log_worker_error_once(worker_error)
                    return results

                with self._lock:
                    for item_idx, ok in zip(read_indices, io_results, strict=True):
                        key, entry = items[item_idx]
                        if ok and entry is not None and self._index.get(key) is entry:
                            objs[
                                item_idx
                            ].metadata.cached_positions = entry.meta.cached_positions
                            results[item_idx] = True
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
        """Delete indexed keys and free their space when allowed.

        Same as ``delete_many_with_sizes`` without the byte counts.

        Args:
            encoded_keys: Ordered encoded raw-block keys to delete.
            force: If true, delete locked keys as well. Normal MP eviction uses
                false so locked entries are preserved.

        Returns:
            A list of per-key deletion booleans aligned with ``encoded_keys``.
        """
        return [
            size is not None
            for size in self.delete_many_with_sizes(encoded_keys, force=force)
        ]

    def delete_many_with_sizes(
        self,
        encoded_keys: Sequence[str],
        *,
        force: bool = False,
    ) -> list[int | None]:
        """Delete indexed keys and free their space when allowed.

        A deleted object's space is freed only after its header has been
        overwritten: the newest on-device checkpoint may still list the
        object, and recovery verifies headers only, so the object must fail
        that check before another object can reuse any of its bytes. A range
        whose header write fails is retired instead, like a quarantined one.
        In-flight writes of the keys are cancelled.

        Args:
            encoded_keys: Ordered encoded raw-block keys to delete.
            force: If true, delete locked keys as well. Normal MP eviction uses
                false so locked entries are preserved.

        Returns:
            Per key, aligned with ``encoded_keys``: the allocated bytes the
            key's committed object held, 0 when only an in-flight write was
            cancelled, or None when nothing was deleted.
        """
        sizes: list[int | None] = []
        invalidate: list[tuple[int, int, PlacementId]] = []
        with self._lock:
            for encoded_key in encoded_keys:
                locked = self._lock_refcnt.get(encoded_key, 0) > 0
                if locked and not force:
                    sizes.append(None)
                    continue

                removed_entry = self._index.pop(encoded_key, None)
                inflight = self._inflight.get(encoded_key)
                if inflight is not None:
                    inflight.canceled = True
                self._lock_refcnt.pop(encoded_key, None)
                if removed_entry is not None:
                    invalidate.append(
                        (
                            removed_entry.offset,
                            removed_entry.length,
                            removed_entry.placement_id,
                        )
                    )
                    self._meta_dirty_total += 1
                    sizes.append(removed_entry.length)
                else:
                    sizes.append(0 if inflight is not None else None)
        if invalidate:
            self._invalidate_and_free(invalidate)
        return sizes

    def usage(self) -> tuple[float, float]:
        """Return current raw-block space usage fractions.

        Returns:
            ``(current_usage, projected_usage)``. Raw-block has no separate
            projected value, so both values are identical. ``(-1.0, -1.0)``
            indicates that usable capacity is unknown.
        """
        with self._lock:
            usable_capacity = self._usable_capacity_bytes_locked()
            if usable_capacity <= 0:
                return (-1.0, -1.0)
            usage = self._allocated_bytes_locked() / usable_capacity
            return (usage, usage)

    def usable_capacity_bytes(self) -> int:
        """Return allocatable data bytes, less space retired after I/O errors."""
        with self._lock:
            return self._usable_capacity_bytes_locked()

    def snapshot_indexed_allocations(self) -> list[tuple[str, int]]:
        """Return every indexed key with the device bytes it holds."""
        with self._lock:
            return [(k, entry.length) for k, entry in self._index.items()]

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

    def raise_if_failed(self) -> None:
        """Reject work after a terminal native io_uring submission failure.

        Raises:
            RuntimeError: Includes the worker's terminal error. Recoverable
                submission errors and ordinary per-request I/O failures do not
                mark the worker as failed.
        """
        assert self._io_engine is not None
        self._io_engine.raise_if_failed()

    def report_status(self) -> dict:
        """Return health, registration, layout, and in-flight status.

        Returns:
            Status dictionary with ``is_healthy=False`` after close or terminal
            worker failure, and the failure reason in ``worker_error`` when
            available. Fixed-buffer fields describe successful kernel
            registration. Inspecting status never opens a new native device.
        """
        with self._lock:
            assert self._io_engine is not None
            assert self._allocator is not None
            worker_error = self._worker_error()
            fixed_buffers_registered, fixed_buffer_registered_bytes = (
                self._io_engine.fixed_buffer_status()
            )
            return {
                "is_healthy": not self._closed and worker_error is None,
                "worker_error": worker_error,
                "type": "RawBlockCore",
                "device_path": self.device_path,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "usable_capacity_bytes": self._usable_capacity_bytes_locked(),
                "allocated_bytes": self._allocated_bytes_locked(),
                "free_bytes": self._allocator.free_bytes,
                "free_extent_count": self._allocator.free_extent_count,
                "largest_free_extent_bytes": self._allocator.largest_free_extent,
                "alloc_failure_count": self._alloc_failure_count,
                "quarantined_bytes": self._retired_bytes,
                "quarantined_range_count": len(self._retired),
                "header_invalidation_count": self._header_invalidations,
                "header_invalidation_failure_count": self._invalidation_failures,
                "last_payload_uecc": self._last_payload_uecc,
                "indexed_key_count": len(self._index),
                "inflight_key_count": len(self._inflight),
                "locked_key_count": sum(
                    1 for refcnt in self._lock_refcnt.values() if refcnt > 0
                ),
                "metadata_seq": self._meta_seq,
                "metadata_dirty_total": self._meta_dirty_total,
                "metadata_persisted": self._meta_persisted,
                "inflight_io_count": self._inflight_io_count,
                "use_odirect": self.use_odirect,
                "enable_zero_copy": self.enable_zero_copy,
                "fixed_buffers_registered": fixed_buffers_registered,
                "fixed_buffer_registered_bytes": fixed_buffer_registered_bytes,
                "io_engine": self.io_engine,
                "max_batch_size": self._io_engine.max_batch_size,
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
            # Deletes started before close finish their header writes first;
            # later ones skip them (see _invalidate_and_free).
            if not self._invalidations_done.wait_for(
                lambda: self._invalidations_inflight == 0, timeout=30
            ):
                logger.warning(
                    "RawBlockCore: closing %s with header writes of deletes "
                    "still pending",
                    self.device_path,
                )

        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None

        try:
            self._checkpoint_once(force=True)
        except Exception as e:
            logger.warning("RawBlockCore final checkpoint failed: %s", e)

        if self._invalidate_pool is not None:
            self._invalidate_pool.shutdown(wait=True)
        assert self._io_engine is not None
        self._io_engine.close()

    def _worker_error(self) -> str | None:
        if self._closed:
            return None
        assert self._io_engine is not None
        return self._io_engine.worker_error()

    def _log_worker_error_once(self, worker_error: str) -> None:
        """Log the terminal worker error at ERROR the first time it is seen.

        After close, ``report_status()`` reports ``worker_error: None``, so
        this log line is what survives for operators. Called from the
        data paths that observe the error, not from status inspection.

        Args:
            worker_error: Terminal worker error observed by the data path.
        """
        if self._worker_error_logged:
            return
        with self._lock:
            if self._worker_error_logged:
                return
            self._worker_error_logged = True
        logger.error(
            "RawBlockCore terminal worker error on %s: %s",
            self.device_path,
            worker_error,
        )

    def _cleanup_after_init_failure(self) -> None:
        """Close resources that may have been opened before init failed."""
        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None
        if self._invalidate_pool is not None:
            self._invalidate_pool.shutdown(wait=True)
        assert self._io_engine is not None
        self._io_engine.close()
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
        """Build an aligned memoryview for direct block-aligned I/O when possible.

        With ``io_engine='io_uring'`` (regular or io_uring_cmd), the view of a
        padded payload extends into the object's physical allocation when that
        allocation covers ``total_len``, so the transfer needs no bounce
        buffer. ``[payload_len, total_len)`` is then zeroed in place when
        ``zero_tail`` is true; the logical payload is unchanged.

        Args:
            memory_obj: Memory object whose backing allocation may be aligned.
            payload_len: Logical payload length in bytes.
            total_len: I/O length after block-alignment padding.
            buffer_len: Logical buffer length exposed by the caller.
            zero_tail: Whether to zero any padded tail bytes before writing.

        Returns:
            A direct memoryview over the allocation, or None when the memory
            object is unsuitable for direct I/O.
        """
        if not self._requires_transfer_alignment or not self.enable_zero_copy:
            return None
        if self.use_uring_cmd and buffer_len >= total_len:
            # io_uring_cmd already submits a full-length logical buffer as is.
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
        # POSIX already writes the aligned prefix directly and pads only the
        # tail, so only io_uring gains from reaching into the allocation.
        if (
            buffer_len < total_len
            and self.io_engine == "io_uring"
            and _allocation_span_bytes(memory_obj) >= total_len
        ):
            buffer_len = total_len

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
            includes any O_DIRECT padding. The object's allocation always
            holds ``total_len`` bytes after the header.
        """
        buf = memory_obj.byte_array
        if hasattr(buf, "cast"):
            buf = buf.cast("B")
        payload_len = len(memory_obj.byte_array)
        total_len = payload_len
        if self._requires_transfer_alignment:
            total_len = round_up(payload_len, self.block_align)
            # byte_array exposes only payload_len bytes. Without a direct view
            # covering total_len, the I/O engine bounces the payload and zeroes
            # [payload_len, total_len) in its own buffer.
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
            Exception: Propagates Rust raw-device write errors.
        """
        assert self._io_engine is not None
        self._io_engine.write_buffers(
            offsets, buffers, payload_lens, total_lens, placement_ids
        )

    def _read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> list[bool]:
        """Read one or more buffers through the configured Rust I/O path.

        Args:
            offsets: Device offsets for each read.
            buffers: Destination Python buffers.
            payload_lens: Logical payload lengths to expose to callers.
            total_lens: Physical I/O lengths for each read.

        Returns:
            A list of per-read success booleans aligned with ``offsets``. The
            returned list always has the same length as ``offsets``; completion
            count mismatches are reported as false entries.

        Raises:
            RuntimeError: If the requested io_uring mode is unavailable before
                reads are submitted.
        """
        assert self._io_engine is not None
        return self._io_engine.read_buffers(offsets, buffers, payload_lens, total_lens)

    def _write_one(
        self,
        key: RawBlockKeySpec,
        memory_obj: MemoryObj,
        offset: int,
        generation: int,
        *,
        placement_id: PlacementId = None,
    ) -> bool:
        """Write one object header and payload at its allocated offset.

        Args:
            key: Raw-block key spec with the slot-header identity.
            memory_obj: Source object to write.
            offset: Byte offset of the object's allocation on the raw device.
            generation: The write's generation, stored in the header.
            placement_id: FDP placement identifier for this raw-block write.
                ``None`` omits the directive; explicit identifier 0 is rejected.

        Returns:
            True when both header and payload writes complete; false otherwise.
        """
        assert self._io_engine is not None
        try:
            with self._io_engine.lease_buffer(self.header_bytes) as header_buf:
                self._encode_header_into_buffer(
                    header_buf,
                    key.slot_identity,
                    len(memory_obj.byte_array),
                    generation,
                )
                buf, payload_len, total_len = self._prepare_write_payload(memory_obj)
                hdr_total = (
                    round_up(self.header_bytes, self.block_align)
                    if self._requires_transfer_alignment
                    else self.header_bytes
                )

                with self._lock:
                    self._inflight_io_count += 1
                try:
                    # Keep each slot header on the same placement identifier
                    # as its payload; future policy can split them if needed.
                    self._write_buffers(
                        [offset, offset + self.header_bytes],
                        [header_buf, buf],
                        [hdr_total, payload_len],
                        [hdr_total, total_len],
                        [placement_id, placement_id],
                    )
                finally:
                    with self._lock:
                        self._inflight_io_count -= 1
                        self._last_io_ts = time.monotonic()
                return True
        except Exception as e:
            logger.error("RawBlockCore write failed for %s: %s", key.encoded, e)
            return False

    def _put_many_batch_io(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
        placement_ids: Sequence[PlacementId],
    ) -> RawBlockPutManyResult:
        """Persist objects using bounded io_uring batch submissions.

        Large ``put_many`` calls are split into chunks so one caller cannot
        monopolize the RawBlockCore lock while planning space, and so the
        transient memory a single batch holds stays bounded.

        Each key contributes at least two write entries (header + payload).
        The io_uring_cmd path splits those further by
        ``max_data_transfer_size``, so one chunk can expand to many more
        entries there. Alignment and padding are handled by the existing write
        paths, which may allocate bounce buffers retained until I/O completes.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``.
            objs: Memory objects whose byte buffers should be written.
            placement_ids: Normalized per-key FDP placement identifiers, one
                entry per key. ``None`` entries omit the directive.

        Returns:
            Per-key success results aligned with ``keys``, the encoded keys
            newly committed to the index and their allocated bytes.

        Raises:
            ValueError: If ``keys``, ``objs``, and ``placement_ids`` do not all
                have the same length.
        """
        if len(keys) <= _MAX_PUT_MANY_IO_URING_BATCH_KEYS:
            return self._put_many_batch_io_chunk(keys, objs, placement_ids)

        results = [False] * len(keys)
        stored_keys: list[str] = []
        stored_bytes: list[int] = []
        first_occurrences: dict[str, int] = {}
        unique_plan: list[tuple[int, RawBlockKeySpec, MemoryObj, PlacementId]] = []
        duplicate_indices: list[tuple[int, int]] = []

        # Deduplicate before chunking so duplicates that cross chunk boundaries
        # still inherit the first occurrence result without being rewritten.
        for i, (key, obj, placement_id) in enumerate(
            zip(keys, objs, placement_ids, strict=True)
        ):
            first_index = first_occurrences.get(key.encoded)
            if first_index is not None:
                duplicate_indices.append((i, first_index))
                continue
            first_occurrences[key.encoded] = i
            unique_plan.append((i, key, obj, placement_id))

        chunk_size = _MAX_PUT_MANY_IO_URING_BATCH_KEYS
        for start in range(0, len(unique_plan), chunk_size):
            chunk = unique_plan[start : start + chunk_size]
            chunk_result = self._put_many_batch_io_chunk(
                [key for _, key, _, _ in chunk],
                [obj for _, _, obj, _ in chunk],
                [placement_id for _, _, _, placement_id in chunk],
            )
            for local_i, (global_i, _key, _obj, _placement_id) in enumerate(chunk):
                results[global_i] = chunk_result.results[local_i]
            stored_keys.extend(chunk_result.stored_keys)
            stored_bytes.extend(chunk_result.stored_bytes)

        for duplicate_i, first_i in duplicate_indices:
            results[duplicate_i] = results[first_i]

        return RawBlockPutManyResult(
            results=results, stored_keys=stored_keys, stored_bytes=stored_bytes
        )

    def _put_many_batch_io_chunk(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
        placement_ids: Sequence[PlacementId],
    ) -> RawBlockPutManyResult:
        """Persist one bounded chunk through a single ``_write_buffers`` call.

        Eligible new keys are submitted through one ``_write_buffers`` call so
        the io_uring path can submit those writes as one batch.

        Failures before submission are reported per key: already indexed keys
        report success without rewriting, duplicates of keys already reserved
        in this chunk share that key's final result, and keys with no free
        range long enough or buffer preparation errors fail individually.
        Once a combined device write is submitted, any write failure rolls back
        every submitted new key and commits none of them.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``. Must be
                the same length as ``objs``.
            objs: Memory objects whose byte buffers should be written. Must be
                the same length as ``keys``.
            placement_ids: Normalized per-key FDP placement identifiers. Must be
                the same length as ``keys``. Each key's header and payload write
                inherit that key's identifier; ``None`` omits the directive.

        Returns:
            Per-key success results aligned with ``keys``, the encoded keys
            newly committed to the index and their allocated bytes.

        Raises:
            ValueError: If ``keys``, ``objs``, and ``placement_ids`` do not all
                have the same length.
        """
        results = [False] * len(keys)
        stored_keys: list[str] = []
        stored_bytes: list[int] = []
        write_plan: list[
            tuple[int, RawBlockKeySpec, MemoryObj, _Inflight, PlacementId]
        ] = []
        planned_keys: set[str] = set()
        batch_duplicates: list[tuple[int, str]] = []
        payload_lens_in = [len(obj.byte_array) for obj in objs]

        # Reserve space for eligible first-occurrence keys under the lock.
        with self._lock:
            for i, (key, obj, placement_id) in enumerate(
                zip(keys, objs, placement_ids, strict=True)
            ):
                if self._closed:
                    break
                encoded_key = key.encoded
                if encoded_key in self._index:
                    results[i] = True
                    continue
                if encoded_key in planned_keys:
                    batch_duplicates.append((i, encoded_key))
                    continue
                if encoded_key in self._inflight:
                    continue
                inflight = self._reserve_locked(
                    key, payload_lens_in[i], obj, placement_id
                )
                if inflight is None:
                    continue
                planned_keys.add(encoded_key)
                write_plan.append((i, key, obj, inflight, placement_id))

        if not write_plan:
            return RawBlockPutManyResult(
                results=results, stored_keys=stored_keys, stored_bytes=stored_bytes
            )

        # Build header/payload write entries outside the lock. Preparation
        # failures happen before device submission and are isolated per key.
        assert self._io_engine is not None
        offsets: list[int] = []
        buffers: list[Any] = []
        payload_lens: list[int] = []
        total_lens: list[int] = []
        write_placement_ids: list[PlacementId] = []
        prepared_plan: list[tuple[int, RawBlockKeySpec]] = []
        write_succeeded = True
        with ExitStack() as released_buffers:
            for i, key, obj, inflight, placement_id in write_plan:
                offset = inflight.offset
                try:
                    header_buf = released_buffers.enter_context(
                        self._io_engine.lease_buffer(self.header_bytes)
                    )
                    self._encode_header_into_buffer(
                        header_buf,
                        key.slot_identity,
                        len(obj.byte_array),
                        inflight.generation,
                    )
                    hdr_total = (
                        round_up(self.header_bytes, self.block_align)
                        if self._requires_transfer_alignment
                        else self.header_bytes
                    )
                    buf, payload_len, total_len = self._prepare_write_payload(obj)
                except Exception as e:
                    logger.error(
                        "RawBlockCore batch buffer preparation failed for %s: %s",
                        key.encoded,
                        e,
                    )
                    with self._lock:
                        self._finish_write_locked(key.encoded, False)
                    continue

                # Queue the key only once every fallible step has succeeded, so a
                # failed key never leaves a header behind for space that the
                # rollback above just freed.
                offsets.extend((offset, offset + self.header_bytes))
                buffers.extend((header_buf, buf))
                payload_lens.extend((hdr_total, payload_len))
                total_lens.extend((hdr_total, total_len))
                write_placement_ids.extend((placement_id, placement_id))
                prepared_plan.append((i, key))

            if prepared_plan:
                with self._lock:
                    self._inflight_io_count += len(prepared_plan)
                try:
                    self._write_buffers(
                        offsets,
                        buffers,
                        payload_lens,
                        total_lens,
                        write_placement_ids,
                    )
                except Exception as e:
                    write_succeeded = False
                    logger.error("RawBlockCore batched write failed: %s", e)
                finally:
                    with self._lock:
                        self._inflight_io_count -= len(prepared_plan)
                        self._last_io_ts = time.monotonic()
            # released_buffers returns every leased header buffer here, after
            # the batch write, on every path.

        # Commit successful writes, or roll back submitted keys if the device
        # write failed.
        with self._lock:
            for i, key in prepared_plan:
                entry = self._finish_write_locked(key.encoded, write_succeeded)
                if entry is not None:
                    results[i] = True
                    stored_keys.append(key.encoded)
                    stored_bytes.append(entry.length)
            for i, encoded_key in batch_duplicates:
                results[i] = encoded_key in self._index

        return RawBlockPutManyResult(
            results=results, stored_keys=stored_keys, stored_bytes=stored_bytes
        )

    def _encode_header_into_buffer(
        self, buf: Any, slot_identity: int, payload_len: int, generation: int
    ) -> None:
        """Encode a raw-block object header into ``buf`` in place.

        Args:
            buf: Writable buffer of at least ``header_bytes`` bytes.
            slot_identity: Encoded key identity to store.
            payload_len: Logical payload length to store.
            generation: The write's generation (see ``_Entry.generation``).
        """
        buf[0:8] = b"LMCBLK01"
        buf[8:16] = int(slot_identity & ((1 << 64) - 1)).to_bytes(
            8,
            "little",
            signed=False,
        )
        buf[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
        buf[24:32] = int(generation).to_bytes(8, "little", signed=False)

    def _decode_slot_header(self, hdr: bytes) -> Optional[tuple[int, int, int]]:
        """Decode a raw-block object header into identity, payload length and
        generation; fixed-slot writers left the generation bytes unset."""
        if len(hdr) < 32 or hdr[0:8] != b"LMCBLK01":
            return None
        slot_identity = int.from_bytes(hdr[8:16], "little", signed=False)
        payload_len = int.from_bytes(hdr[16:24], "little", signed=False)
        generation = int.from_bytes(hdr[24:32], "little", signed=False)
        return slot_identity, payload_len, generation

    def _read_slot_header(self, offset: int) -> Optional[tuple[int, int, int]]:
        """Read and decode the slot header at a raw-device offset."""
        assert self._io_engine is not None
        with self._io_engine.lease_buffer(self.header_bytes) as buf:
            with self._lock:
                self._inflight_io_count += 1
            try:
                if not all(
                    self._read_buffers(
                        [offset],
                        [buf],
                        [self.header_bytes],
                        [self.header_bytes],
                    )
                ):
                    return None
                return self._decode_slot_header(bytes(buf)) or _NO_HEADER
            except Exception:
                return None
            finally:
                with self._lock:
                    self._inflight_io_count -= 1
                    self._last_io_ts = time.monotonic()

    def _ensure_capacity_and_layout(self) -> None:
        """Compute the metadata/data layout and start with all data space free.

        The first data block holds the format marker; objects are allocated
        after it.
        """
        if self._allocator is not None:
            return

        assert self._io_engine is not None
        device_capacity = self._io_engine.capacity_bytes()
        requested = self.capacity_bytes if self.capacity_bytes > 0 else device_capacity
        self._effective_capacity_bytes = min(requested, device_capacity)
        self.capacity_bytes = self._effective_capacity_bytes

        if self.meta_total_bytes >= self._effective_capacity_bytes:
            raise RuntimeError("metadata region exceeds usable device capacity")

        self._data_base_offset = self.meta_total_bytes
        data_bytes = self._effective_capacity_bytes - self._data_base_offset
        self._data_bytes = data_bytes // self.block_align * self.block_align
        if self._data_bytes < self.block_align + self.header_bytes + self.block_align:
            raise RuntimeError("raw block capacity too small for data after metadata")
        self._allocator = self._new_allocator()

    def _new_allocator(self, used: Sequence[tuple[int, int]] = ()) -> ExtentAllocator:
        """Return an allocator over the data region after the marker block."""
        return ExtentAllocator(
            self._data_base_offset + self.block_align,
            self._data_bytes - self.block_align,
            self.block_align,
            used,
        )

    def _alloc_len(self, payload_len: int) -> int:
        """Return the bytes an object with a ``payload_len`` payload occupies."""
        return self.header_bytes + round_up(payload_len, self.block_align)

    def _usable_capacity_bytes_locked(self) -> int:
        """Return allocatable data-region bytes not retired by quarantine."""
        assert self._allocator is not None
        return self._allocator.size - self._retired_bytes

    def _allocated_bytes_locked(self) -> int:
        """Return usable bytes that are not free: objects, writes in flight and
        deleted objects whose header is still being invalidated."""
        assert self._allocator is not None
        return self._usable_capacity_bytes_locked() - self._allocator.free_bytes

    def _reserve_locked(
        self,
        key: RawBlockKeySpec,
        payload_len: int,
        obj: MemoryObj,
        placement_id: PlacementId,
    ) -> _Inflight | None:
        """Allocate space for one object and record it as in flight.

        Args:
            key: Key being stored.
            payload_len: ``len(obj.byte_array)``, computed outside the lock.
            obj: Object being stored, for its metadata.
            placement_id: FDP placement identifier of the write.

        Returns:
            The reservation, or None when no free range is long enough.
        """
        length = self._alloc_len(payload_len)
        offset = self._allocate_locked(length, placement_id)
        if offset is None:
            self._warn_no_space_locked(length)
            return None
        meta = DiskCacheMetadata(
            path=f"{self.device_path}@{offset}",
            size=payload_len,
            shape=obj.metadata.shape,
            dtype=obj.metadata.dtype,
            cached_positions=obj.metadata.cached_positions,
            fmt=obj.metadata.fmt,
            pin_count=0,
        )
        inflight = _Inflight(
            offset=offset,
            length=length,
            meta=meta,
            placement_id=placement_id,
            generation=_GENERATION_RNG.getrandbits(64) | 1,
        )
        self._inflight[key.encoded] = inflight
        return inflight

    def _finish_write_locked(self, encoded_key: str, success: bool) -> _Entry | None:
        """Commit a finished write, or free its space if it failed or was cancelled.

        Returns:
            The committed entry, or None when nothing was committed.
        """
        inflight = self._inflight.pop(encoded_key, None)
        if inflight is None:
            return None
        if inflight.canceled or not success:
            # A header it left carries a generation no checkpoint lists.
            self._free_locked(inflight.offset, inflight.length, inflight.placement_id)
            return None
        self._meta_dirty_total += 1
        entry = _Entry(
            offset=inflight.offset,
            length=inflight.length,
            size=inflight.meta.size,
            meta=inflight.meta,
            placement_id=inflight.placement_id,
            generation=inflight.generation,
        )
        self._index[encoded_key] = entry
        return entry

    def _allocate_locked(self, nbytes: int, placement_id: PlacementId) -> int | None:
        """Allocate ``nbytes`` of device space while ``self._lock`` is held.

        With FDP slot affinity, a free range of exactly ``nbytes`` last written
        with the same placement identifier is preferred. Affinity counters
        count only allocations of space written before.

        Returns:
            The offset, or None when no free range is long enough.
        """
        assert self._allocator is not None
        tag = placement_id if self.fdp_slot_affinity_enabled else None
        hit = tag is not None and self._allocator.has_tagged_extent(nbytes, tag)
        offset = self._allocator.allocate(nbytes, tag=tag)
        if offset is None:
            self._alloc_failure_count += 1
            return None
        if tag is not None and offset < self._high_water:
            if hit:
                self._fdp_slot_affinity_hit_count += 1
            else:
                self._fdp_slot_affinity_fallback_count += 1
        self._high_water = max(self._high_water, offset + nbytes)
        return offset

    def _largest_unretired_span_locked(self) -> int:
        """Return the length of the longest data range with no retired byte."""
        assert self._allocator is not None
        cursor, longest = self._allocator.base, 0
        for start, length in self._retired:
            longest = max(longest, start - cursor)
            cursor = max(cursor, start + length)
        return max(longest, self._allocator.base + self._allocator.size - cursor)

    def _drop_retired_locked(self) -> None:
        """Forget retired ranges along with a dropped index: they were retired
        for its entries, and the rebuilt allocator holds them as free. The
        checkpoints are voided on that path, so nothing lists them."""
        self._retired.clear()
        self._retired_bytes = 0

    def _warn_no_space_locked(self, nbytes: int) -> None:
        """Log a failed allocation, at most once a minute (they come in runs)."""
        assert self._allocator is not None
        now = time.monotonic()
        self._no_space_since_log += 1
        if now - self._no_space_logged_at < 60:
            return
        never = nbytes > self._largest_unretired_span_locked()
        logger.warning(
            "RawBlockCore: %d stores found no free range long enough on %s "
            "since the last warning (latest needed %d bytes%s; %d bytes free, "
            "largest free range %d bytes)",
            self._no_space_since_log,
            self.device_path,
            nbytes,
            ", more than any range can ever hold" if never else "",
            self._allocator.free_bytes,
            self._allocator.largest_free_extent,
        )
        self._no_space_logged_at = now
        self._no_space_since_log = 0

    def _free_locked(self, offset: int, length: int, placement_id: PlacementId) -> None:
        """Return a range to free space, minus any retired parts of it."""
        assert self._allocator is not None
        tag = placement_id if self.fdp_slot_affinity_enabled else None
        self._free_unretired_locked(offset, length, tag)

    def _free_unretired_locked(self, offset: int, length: int, tag: int | None) -> None:
        """Free a range, minus any retired parts of it."""
        assert self._allocator is not None
        if not self._retired:
            self._allocator.free(offset, length, tag=tag)
            return
        end = offset + length
        i = bisect.bisect_left(self._retired, (offset,))
        if i and sum(self._retired[i - 1]) > offset:
            i -= 1
        cursor = offset
        while i < len(self._retired) and self._retired[i][0] < end:
            r_start, r_len = self._retired[i]
            if r_start > cursor:
                self._allocator.free(cursor, r_start - cursor, tag=tag)
            cursor = max(cursor, r_start + r_len)
            i += 1
        if cursor < end:
            self._allocator.free(cursor, end - cursor, tag=tag)

    def _retire_locked(self, offset: int, length: int, *, allocated: bool) -> None:
        """Exclude a range from allocation until the device is reopened.

        Free bytes in the range are withdrawn now; allocated bytes in it are
        dropped when their owner frees them (see ``_free_locked``).

        Args:
            offset: Start of the range.
            length: Its length.
            allocated: Whether the whole range is known to be allocated, so
                there are no free bytes to withdraw.
        """
        assert self._allocator is not None
        end = offset + length
        i = bisect.bisect_left(self._retired, (offset,))
        if i and sum(self._retired[i - 1]) >= offset:
            i -= 1
        j = i
        new_start, new_end, covered = offset, end, 0
        while j < len(self._retired) and self._retired[j][0] <= new_end:
            r_start, r_len = self._retired[j]
            covered += max(0, min(r_start + r_len, end) - max(r_start, offset))
            new_start = min(new_start, r_start)
            new_end = max(new_end, r_start + r_len)
            j += 1
        self._retired[i:j] = [(new_start, new_end - new_start)]
        self._retired_bytes += length - covered
        if allocated:
            return
        # shortcut: linear in free extents; fine while retiring free space
        # stays rare (a late read error), index free extents by start if not.
        self._allocator.remove_free_range(offset, length)

    def _quarantine_locked(
        self, key: str, observed: _Entry, error: str
    ) -> list[tuple[str, int]]:
        """Retire the space of an object whose payload read failed for good.

        The observed object's range is retired with every current owner of
        any byte of it: a force-delete or an unlocked reader can race reuse,
        so a new object may now hold part of the range. Owners are retired
        whole rather than freed, since their headers were not invalidated.
        In-flight writes into the range are cancelled.

        Returns:
            The removed keys with the bytes each held.
        """
        start, end = observed.offset, observed.offset + observed.length
        self._last_payload_uecc = error
        if self._index.get(key) is observed:
            owners = [key]
        else:
            # shortcut: scans the index under the lock; fine while payload
            # UECCs stay rare, index entries by offset if they are not.
            owners = [
                k
                for k, entry in self._index.items()
                if entry.offset < end and start < entry.offset + entry.length
            ]
            for pending in self._inflight.values():
                if pending.offset < end and start < pending.offset + pending.length:
                    pending.canceled = True
        # The observed object may be gone, its space partly free or reused.
        self._retire_locked(start, observed.length, allocated=False)
        removed: list[tuple[str, int]] = []
        for owner in owners:
            entry = self._index.pop(owner)
            self._retire_locked(entry.offset, entry.length, allocated=True)
            removed.append((owner, entry.length))
        self._meta_dirty_total += len(removed)
        logger.error(
            "RawBlockCore quarantined bytes [%d, %d) on %s: %s",
            start,
            end,
            self.device_path,
            error,
        )
        return removed

    def _invalidate_and_free(
        self, ranges: Sequence[tuple[int, int, PlacementId]]
    ) -> None:
        """Overwrite the header at the start of each range, then free it.

        Called outside the lock for deleted objects. Nothing is flushed, so
        this ordering holds for process crashes, not for power loss on drives
        with volatile write caches (issue #6568). A range whose header write
        fails is retired instead: its old header may still be valid, so
        reusing the rest of it could let recovery return another object's
        bytes for that key. The write happens with meta_verify_on_load off as
        well: this run's recovery reads no headers, but a later run with
        verification on may load the same checkpoint. After close() the space
        is left allocated; this run reuses nothing any more. Ranges are freed
        as their header writes end (each with posix, each io_uring batch
        otherwise), so stores can reuse them while the rest of a large delete
        is still being written.

        Args:
            ranges: ``(offset, length, placement_id)`` per range; the header
                write uses the range's placement identifier.
        """
        with self._lock:
            if self._closed:
                return
            self._invalidations_inflight += 1
        try:
            self._write_headers(ranges)
        finally:
            with self._lock:
                self._invalidations_inflight -= 1
                self._invalidations_done.notify_all()

    def _write_headers(self, ranges: Sequence[tuple[int, int, PlacementId]]) -> None:
        """Zero the header block at the start of each range, then settle it.

        The posix engine writes one request per call, so the ranges are
        spread over the invalidation threads, each write settling its own
        range. io_uring writes batches of ``max_batch_size``; a batch with a
        failed request is retried one request at a time.
        """
        assert self._io_engine is not None
        with self._io_engine.lease_buffers(1, self.header_bytes) as buffers:
            zeros = buffers[0]
            zeros[: self.header_bytes] = bytes(self.header_bytes)

            def one_by_one(part: Sequence[tuple[int, int, PlacementId]]) -> None:
                for r in part:
                    self._settle_invalidated([r], self._write_headers_once(zeros, [r]))

            if self.io_engine == "posix":
                if self._invalidate_pool is None or len(ranges) == 1:
                    one_by_one(ranges)
                    return
                parts = [
                    ranges[start:end]
                    for start, end in self._build_recovery_item_ranges(
                        len(ranges), self._recovery_read_threads
                    )
                ]
                # list() waits for every part and re-raises a worker's error.
                list(self._invalidate_pool.map(one_by_one, parts))
                return
            step = self._io_engine.max_batch_size
            for start in range(0, len(ranges), step):
                batch = ranges[start : start + step]
                if self._write_headers_once(zeros, batch):
                    self._settle_invalidated(batch, True)
                else:
                    one_by_one(batch)

    def _settle_invalidated(
        self, ranges: Sequence[tuple[int, int, PlacementId]], written: bool
    ) -> None:
        """Free ranges whose header was zeroed; retire them if it was not."""
        with self._lock:
            for offset, length, placement_id in ranges:
                if written:
                    self._header_invalidations += 1
                    self._free_locked(offset, length, placement_id)
                else:
                    self._invalidation_failures += 1
                    self._retire_locked(offset, length, allocated=True)

    def _write_headers_once(
        self, block: Any, ranges: Sequence[tuple[int, int, PlacementId]]
    ) -> bool:
        """Write ``block`` at the start of each range in one engine call."""
        n = len(ranges)
        with self._lock:
            self._inflight_io_count += 1
        try:
            self._write_buffers(
                [offset for offset, _, _ in ranges],
                [block] * n,
                [self.header_bytes] * n,
                [self.header_bytes] * n,
                [placement_id for _, _, placement_id in ranges],
            )
            return True
        except Exception:
            logger.exception("RawBlockCore could not invalidate %d object headers", n)
            return False
        finally:
            with self._lock:
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()

    def _read_marker(self) -> bool | None:
        """Return whether the data region's first block holds the format marker.

        Returns:
            True or False, or None when the block cannot be read, also on a
            retry: an unreadable marker says nothing about a downgrade.
        """
        assert self._io_engine is not None
        for _ in range(2):
            with self._io_engine.lease_buffers(1, self.block_align) as buffers:
                try:
                    ok = all(
                        self._read_buffers(
                            [self._data_base_offset],
                            buffers,
                            [self.block_align],
                            [self.block_align],
                        )
                    )
                except Exception:
                    logger.exception("RawBlockCore could not read the format marker")
                    continue
                if ok:
                    return bytes(buffers[0][: len(_MARKER_MAGIC)]) == _MARKER_MAGIC
        return None

    def _write_marker(self) -> None:
        """Write the format marker into the data region's first block.

        Raises:
            Exception: Propagates raw-device write errors.
        """
        assert self._io_engine is not None
        with self._io_engine.lease_buffer(self.block_align) as buf:
            buf[: self.block_align] = bytes(self.block_align)
            buf[: len(_MARKER_MAGIC)] = _MARKER_MAGIC
            self._write_buffers(
                [self._data_base_offset],
                [buf],
                [self.block_align],
                [self.block_align],
            )

    def _note_foreign_layout(self, data: Any) -> None:
        """Remember the containers of a rejected checkpoint's own layout.

        A checkpoint written with another ``meta_total_bytes`` has its other
        copy at an offset this layout does not read; a later run with that
        layout again would load it over space this run reused.
        """
        try:
            meta_total = int(data["meta_total_bytes"])
            align = int(data["block_align"])
        except (KeyError, TypeError, ValueError):
            return
        if meta_total <= 0 or align <= 0 or meta_total == self.meta_total_bytes:
            return
        container = (meta_total // self._meta_copy_count) // align * align
        self._foreign_container_offsets = [
            offset // self.block_align * self.block_align
            for offset in range(0, container * self._meta_copy_count, container)
            if offset + self.block_align <= self._effective_capacity_bytes
        ]

    def _void_checkpoints(self) -> None:
        """Zero the header block of every metadata container, also those of a
        rejected checkpoint's layout.

        Raises:
            Exception: Propagates raw-device write errors.
        """
        assert self._io_engine is not None
        offsets = sorted(
            set(self._meta_container_offsets()) | set(self._foreign_container_offsets)
        )
        with self._io_engine.lease_buffer(self.block_align) as buf:
            buf[: self.block_align] = bytes(self.block_align)
            self._write_buffers(
                offsets,
                [buf] * len(offsets),
                [self.block_align] * len(offsets),
                [self.block_align] * len(offsets),
            )

    def _prepare_data_region(self, loaded_format: int | None) -> None:
        """Make the device safe for this run's stores, before the first one.

        - A format-2 checkpoint with the data-region marker gone means a
          fixed-slot run started empty on this device and overwrote space the
          checkpoint lists, so the checkpoint is dropped.
        - A checkpoint that is not loaded (skipped, rejected, dropped, or
          written with another meta_magic or meta_version, which this run
          cannot see) may list space this run's stores will overwrite without
          touching those objects' headers, so every metadata container is
          voided first.
        - An imported fixed-slot checkpoint is rewritten in format 2, so a
          fixed-slot binary started later rejects it and starts empty. If the
          rewrite does not fit a container, the run starts empty instead.
        - The marker overwrites what was slot 0, before an imported
          checkpoint is rewritten, so a crash in between re-imports it.

        Args:
            loaded_format: Format of the checkpoint loaded at startup, or None.

        Raises:
            RuntimeError: If the marker cannot be read while a format-2
                checkpoint was loaded.
            Exception: Propagates raw-device write errors.
        """
        read = self._read_marker()
        if read is None and loaded_format == _CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                "RawBlockCore could not read the data-region format marker"
            )
        marked = bool(read)
        if loaded_format == _CHECKPOINT_FORMAT_VERSION and not marked:
            logger.warning(
                "RawBlockCore: the data-region marker is missing, so a "
                "fixed-slot run used %s after its last checkpoint; starting "
                "with an empty index",
                self.device_path,
            )
            with self._lock:
                self._index.clear()
                self._lock_refcnt.clear()
                self._drop_retired_locked()
                self._allocator = self._new_allocator()
            loaded_format = None
        if loaded_format == 1:
            if not marked:
                self._write_marker()
                marked = True
            with self._lock:
                self._meta_dirty_total += 1
            if not self._checkpoint_once(force=True):
                # Format 2 entries are longer, so the rewrite can outgrow a
                # container that held the fixed-slot checkpoint.
                logger.warning(
                    "RawBlockCore: the imported checkpoint does not fit a "
                    "metadata container in the new format; starting %s with "
                    "an empty index",
                    self.device_path,
                )
                with self._lock:
                    self._index.clear()
                    self._lock_refcnt.clear()
                    self._drop_retired_locked()
                    self._allocator = self._new_allocator()
                loaded_format = None
        if loaded_format is None:
            self._void_checkpoints()
            # A foreign container may sit where the marker goes.
            marked = marked and self._data_base_offset not in set(
                self._foreign_container_offsets
            )
        if not marked:
            self._write_marker()

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
        """Read and validate a metadata checkpoint header.

        Returns:
            The header, or None when the block holds no valid header.

        Raises:
            RuntimeError: If the block cannot be read. Startup would otherwise
                take an unreadable checkpoint for a missing one and void it.
        """
        assert self._io_engine is not None
        with self._io_engine.lease_buffer(self.block_align) as buf:
            self._read_meta_block(
                container_offset, buf, self.block_align, self.block_align
            )

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

    def _read_meta_block(
        self, offset: int, buf: Any, payload_len: int, total_len: int
    ) -> None:
        """Read part of the metadata region into ``buf``.

        Raises:
            RuntimeError: If the read fails.
        """
        try:
            ok = all(self._read_buffers([offset], [buf], [payload_len], [total_len]))
        except Exception as e:
            raise RuntimeError(
                f"RawBlockCore could not read the metadata region at {offset}"
            ) from e
        if not ok:
            raise RuntimeError(
                f"RawBlockCore could not read the metadata region at {offset}"
            )

    def _load_meta_payload(self, header: dict[str, int]) -> Optional[bytes]:
        """Load and CRC-validate a checkpoint payload for a metadata header."""
        payload_len = int(header["payload_len"])
        payload_off = int(header["container_offset"]) + self.block_align
        total_len = round_up(payload_len, self.block_align)
        assert self._io_engine is not None
        with self._io_engine.lease_buffer(total_len) as buf:
            self._read_meta_block(payload_off, buf, payload_len, total_len)
            payload = bytes(buf[:payload_len])
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            if crc != int(header["crc"]):
                return None
            return payload

    def _max_checkpoint_seq(self) -> int:
        """Return the highest sequence of a valid checkpoint header, or 0."""
        seqs = [
            header["seq"]
            for offset in self._meta_container_offsets()
            if (header := self._read_meta_header(offset)) is not None
        ]
        return max(seqs, default=0)

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
        """Build a JSON-serializable checkpoint state snapshot.

        Returns:
            The snapshot and the dirty counter it covers.
        """
        with self._lock:
            dirty_total = self._meta_dirty_total
            snapshot = {
                "version": _CHECKPOINT_FORMAT_VERSION,
                "device_path": self.device_path,
                "capacity_bytes": self.capacity_bytes,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "meta_magic": self.meta_magic_text,
                "meta_version": self.meta_version,
                "data_base_offset": self._data_base_offset,
                "entries": {
                    encoded_key: _CheckpointEntry(
                        offset=entry.offset,
                        size=entry.meta.size,
                        gen=entry.generation,
                        shape=entry.meta.shape,
                        dtype=self._checkpoint_dtype_name(entry.meta.dtype),
                        fmt=(
                            entry.meta.fmt.name
                            if entry.meta.fmt is not None
                            and hasattr(entry.meta.fmt, "name")
                            else str(entry.meta.fmt)
                            if entry.meta.fmt is not None
                            else None
                        ),
                        cached_positions=(
                            entry.meta.cached_positions.tolist()
                            if entry.meta.cached_positions is not None
                            and hasattr(entry.meta.cached_positions, "tolist")
                            else None
                        ),
                    )
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
        return TORCH_DTYPE_TO_STR_DTYPE.get(dtype) or str(dtype)

    def _write_checkpoint(self, payload: bytes, dirty_total_snapshot: int) -> bool:
        """Write one checkpoint copy and advance persisted metadata counters."""
        payload_cap = self._meta_payload_capacity()
        if len(payload) > payload_cap:
            logger.warning(
                "RawBlockCore metadata payload too large (%d > %d), "
                "skipping checkpoint; newer entries are lost on restart. "
                "Increase meta_total_bytes (this changes the device layout, "
                "so the existing cache is not recovered).",
                len(payload),
                payload_cap,
            )
            return False
        if (
            not self._meta_usage_warned
            and len(payload) >= _META_USAGE_WARN_RATIO * payload_cap
        ):
            self._meta_usage_warned = True
            logger.warning(
                "RawBlockCore metadata checkpoint uses %d of %d bytes (%.0f%%); "
                "checkpoints are skipped once the index no longer fits. "
                "Increase meta_total_bytes if the index keeps growing (this "
                "changes the device layout, so the existing cache is not "
                "recovered).",
                len(payload),
                payload_cap,
                100.0 * len(payload) / payload_cap,
            )

        next_seq = self._meta_seq + 1
        target_idx = int((next_seq - 1) % self._meta_copy_count)
        target = self._meta_container_offsets()[target_idx]

        payload_len = len(payload)
        payload_total_len = round_up(payload_len, self.block_align)
        payload_off = target + self.block_align
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        assert self._io_engine is not None
        with (
            self._io_engine.lease_buffer(self.block_align) as header_block,
            self._io_engine.lease_buffer(payload_total_len) as payload_buf,
        ):
            header_block[: _META_HEADER_STRUCT.size] = _META_HEADER_STRUCT.pack(
                self.meta_magic,
                self.meta_version,
                int(next_seq),
                int(payload_len),
                int(crc),
            )
            payload_buf[:payload_len] = payload

            placement_id = self.meta_checkpoint_placement_id
            self._write_buffers(
                [payload_off, target],
                [payload_buf, header_block],
                [payload_len, self.block_align],
                [payload_total_len, self.block_align],
                [placement_id, placement_id],
            )

            with self._lock:
                self._meta_seq = int(next_seq)
                self._meta_persisted = max(
                    self._meta_persisted, int(dirty_total_snapshot)
                )
            return True

    def _checkpoint_once(self, force: bool) -> bool:
        """Write a metadata checkpoint when dirty and sufficiently idle.

        Under steady traffic the device may never be idle for
        ``meta_idle_quiet_ms``, so metadata that has waited a full checkpoint
        interval is written anyway. Without that, a crash under load loses
        every store since startup.
        """
        now = time.monotonic()
        with self._lock:
            dirty = self._meta_dirty_total > self._meta_persisted
            idle_ok = self._inflight_io_count == 0 and (now - self._last_io_ts) >= (
                self.meta_idle_quiet_ms / 1000.0
            )
            overdue = False
            if not dirty:
                self._meta_dirty_since = None
            else:
                dirty_since = self._meta_dirty_since
                if dirty_since is None:
                    dirty_since = now
                    self._meta_dirty_since = now
                overdue = now - dirty_since >= max(1, self.meta_checkpoint_interval_sec)

        if not dirty:
            return False
        if not force and not idle_ok and not overdue:
            return False

        snapshot_ts = time.monotonic()
        snapshot, dirty_total_snapshot = self._snapshot_state()
        payload = msgspec.json.encode(snapshot)
        written = self._write_checkpoint(payload, dirty_total_snapshot)
        if written:
            with self._lock:
                # Changes after the snapshot are due one interval after it, so
                # steady traffic still gets one checkpoint per interval. The
                # next check resets this if nothing changed.
                self._meta_dirty_since = snapshot_ts
        return written

    def _apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Apply decoded checkpoint state after validating layout fields.

        Format 2 lists each object's offset; format 1 (fixed slots) is
        imported by giving each object only the space it needs, so the rest of
        its old slot becomes free. Entries outside the data region, off the
        block (or, in format 1, slot) grid, or overlapping another entry are
        dropped.
        """
        if not isinstance(data, dict):
            return False
        version = data.get("version")
        if version not in (1, _CHECKPOINT_FORMAT_VERSION):
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
        slot_bytes = 0
        if version == 1:
            raw_slot_bytes = data.get("slot_bytes")
            if (
                not isinstance(raw_slot_bytes, int)
                or raw_slot_bytes <= self.header_bytes
            ):
                logger.warning("Device metadata slot_bytes is invalid; ignoring it")
                return False
            slot_bytes = raw_slot_bytes

        assert self._allocator is not None
        region_start = self._allocator.base
        region_end = self._allocator.base + self._allocator.size
        loaded: list[tuple[str, _Entry]] = []
        entries = data.get("entries", {})
        if isinstance(entries, dict):
            for encoded_key, entry in entries.items():
                if not isinstance(entry, dict):
                    continue

                offset = int(entry.get("offset", 0))
                size = int(entry.get("size", 0))
                length = self._alloc_len(size)
                if (
                    size <= 0
                    or offset % self.block_align
                    or offset < region_start
                    or offset + length > region_end
                ):
                    continue
                if version == 1 and (
                    (offset - self._data_base_offset) % slot_bytes
                    or length > slot_bytes
                ):
                    continue
                shape_list = entry.get("shape")
                fmt_name = entry.get("fmt")
                cached_positions_list = entry.get("cached_positions")
                dtype_name = entry.get("dtype")

                shape = torch.Size(list(shape_list)) if shape_list is not None else None
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
                dtype = self._recover_checkpoint_dtype(dtype_name)

                meta = DiskCacheMetadata(
                    path=f"{self.device_path}@{offset}",
                    size=size,
                    shape=shape,
                    dtype=dtype,
                    cached_positions=cached_positions,
                    fmt=fmt,
                    pin_count=0,
                )
                # Fixed-slot headers have no generation: -1 adopts the one
                # the header has (see _validate_loaded_entries).
                generation = int(entry.get("gen", 0)) if version != 1 else -1
                if version != 1 and generation < 0:
                    continue
                loaded.append(
                    (
                        encoded_key,
                        _Entry(
                            offset=offset,
                            length=length,
                            size=size,
                            meta=meta,
                            generation=generation,
                        ),
                    )
                )
        kept = _without_overlaps(loaded)
        if len(kept) < len(loaded):
            logger.warning(
                "RawBlockCore dropped %d checkpoint entries with overlapping space",
                len(loaded) - len(kept),
            )

        with self._lock:
            self._index = dict(kept)
            self._lock_refcnt.clear()
            self._allocator = self._new_allocator(
                [(entry.offset, entry.length) for _, entry in kept]
            )
            self._high_water = max(
                (entry.offset + entry.length for _, entry in kept), default=0
            )
            self._meta_dirty_total = 0
            self._meta_persisted = 0

        # A fixed-slot import always reads its headers once: their generation
        # bytes are adopted, or a later verified run would drop the entries.
        if self.meta_verify_on_load or version == 1:
            self._validate_loaded_entries()
        with self._lock:
            for entry in self._index.values():
                entry.generation = max(entry.generation, 0)
        return True

    def _recover_checkpoint_dtype(
        self,
        dtype_name: Any,
    ) -> torch.dtype | None:
        """Recover checkpoint dtype from entry metadata.

        Args:
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

        return None

    def _validate_loaded_entries(self) -> None:
        """Drop recovered entries whose object headers do not match metadata."""
        with self._lock:
            items = list(self._index.items())

        if not items:
            return

        offsets = [int(entry.offset) for _, entry in items]
        headers = self._read_slot_headers(offsets)
        to_drop = [
            (encoded_key, slot_hdr is None)
            for (encoded_key, entry), slot_hdr in zip(items, headers, strict=True)
            if self._is_stale_header(encoded_key, entry, slot_hdr)
        ]
        with self._lock:
            # An entry imported from fixed slots takes the generation bytes its
            # header has, so a later rewrite of the key there is told apart.
            for (encoded_key, entry), slot_hdr in zip(items, headers, strict=True):
                if entry.generation < 0 and slot_hdr is not None:
                    entry.generation = slot_hdr[2]

        if not to_drop:
            return

        with self._lock:
            for encoded_key, unreadable in to_drop:
                removed_entry = self._index.pop(encoded_key, None)
                self._lock_refcnt.pop(encoded_key, None)
                if removed_entry is None:
                    continue
                if unreadable:
                    # The header may still be valid on the device, so the
                    # space cannot be reused without rewriting it first.
                    self._retire_locked(
                        removed_entry.offset, removed_entry.length, allocated=True
                    )
                else:
                    # Its header already fails the check, so no checkpoint can
                    # bring it back: the space is free at once.
                    self._free_locked(removed_entry.offset, removed_entry.length, None)
            self._meta_dirty_total += 1

        logger.warning(
            "RawBlockCore dropped %d stale metadata entries after "
            "slot-header validation",
            len(to_drop),
        )

    def _read_slot_headers(
        self,
        offsets: list[int],
    ) -> list[Optional[tuple[int, int, int]]]:
        """Dispatch slot-header reads to the appropriate engine path."""
        n = len(offsets)
        if self.io_engine == "posix" and self._recovery_read_threads > 1 and n > 1:
            return self._read_slot_headers_posix_parallel(offsets)
        if self.io_engine == "io_uring" and n > 1:
            return self._read_slot_headers_batched(offsets)
        return [self._read_slot_header(off) for off in offsets]

    def _read_slot_headers_posix_parallel(
        self,
        offsets: list[int],
    ) -> list[Optional[tuple[int, int, int]]]:
        """Return slot headers using a bounded POSIX reader thread pool."""
        n = len(offsets)
        max_workers = min(self._recovery_read_threads, n)
        ranges = self._build_recovery_item_ranges(n, max_workers)
        work_items = [(offsets, start, end) for start, end in ranges]

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="rawblk-recover",
        ) as pool:
            return [
                header
                for range_headers in pool.map(self._read_slot_header_range, work_items)
                for header in range_headers
            ]

    def _read_slot_header_range(
        self,
        work_item: tuple[list[int], int, int],
    ) -> list[Optional[tuple[int, int, int]]]:
        """Read one recovery range, preserving offset order and failed reads."""
        offsets, start, end = work_item
        return [self._read_slot_header(offsets[i]) for i in range(start, end)]

    def _read_slot_headers_batched(
        self,
        offsets: list[int],
    ) -> list[Optional[tuple[int, int, int]]]:
        """Return slot headers using the batched io_uring read path.

        Splits the reads into ``max_batch_size``-sized batches so a large
        checkpoint does not allocate one buffer for every entry at once, then
        reads each batch and concatenates results in input order. Failed
        completions are retried individually before returning their headers.

        Args:
            offsets: Device byte offsets for each slot header to read.

        Returns:
            Decoded (slot_identity, payload_len, generation) per header, or
            None on error, in the same order as ``offsets``.
        """
        n = len(offsets)
        assert self._io_engine is not None
        batch_size = self._io_engine.max_batch_size
        headers: list[Optional[tuple[int, int, int]]] = []
        for start in range(0, n, batch_size):
            headers.extend(
                self._read_slot_header_batch(offsets[start : start + batch_size])
            )
        return headers

    def _read_slot_header_batch(
        self,
        offsets: list[int],
    ) -> list[Optional[tuple[int, int, int]]]:
        """Read one bounded batch of slot headers through the engine read path.

        Submission, splitting, and completion are delegated to the engine's
        synchronous :meth:`read_buffers`, so every accepted read is complete
        before the leased buffers are returned to the pool. Headers are decoded
        while the leased buffers are still valid, then only failed slots are
        reread individually. Used for both regular io_uring block I/O and NVMe
        passthrough (``use_uring_cmd``).

        Args:
            offsets: Device byte offsets for this batch (non-empty).

        Returns:
            Decoded (slot_identity, payload_len, generation) per header, or
            None on error.
            Decode successful completions directly and reread only failed
            slots. Submission errors or a completion-count mismatch require
            rereading every slot because individual results are unavailable.
        """
        n = len(offsets)
        hdr = self.header_bytes
        assert self._io_engine is not None
        results: list[bool]
        with self._io_engine.lease_buffers(n, hdr) as bufs:
            with self._lock:
                self._inflight_io_count += 1
            try:
                results = self._read_buffers(
                    offsets,
                    bufs,
                    [hdr] * n,
                    [hdr] * n,
                )
            except Exception:
                logger.exception("RawBlockCore recovery header read failed")
                results = [False] * n
            finally:
                with self._lock:
                    self._inflight_io_count -= 1
                    self._last_io_ts = time.monotonic()
            return [
                (self._decode_slot_header(bytes(buf)) or _NO_HEADER)
                if success
                else self._read_slot_header(offset)
                for offset, buf, success in zip(offsets, bufs, results, strict=True)
            ]

    def _is_stale_header(
        self,
        encoded_key: str,
        entry: _Entry,
        slot_hdr: Optional[tuple[int, int, int]],
    ) -> bool:
        """Return True when the recovered object header does not match the
        entry: another key or size, or another write of the same key."""
        if slot_hdr is None:
            return True
        try:
            expected_identity = object_key_slot_identity(encoded_key)
        except Exception:
            return True
        slot_identity, payload_len, generation = slot_hdr
        return (
            int(slot_identity) != int(expected_identity)
            or int(payload_len) != int(entry.size)
            or (entry.generation >= 0 and generation != entry.generation)
        )

    def _build_recovery_item_ranges(
        self,
        item_count: int,
        num_ranges: int,
    ) -> list[tuple[int, int]]:
        """Build bounded work ranges for recovered checkpoint entries."""
        range_size = max(1, (item_count + num_ranges - 1) // num_ranges)
        return [
            (start, min(start + range_size, item_count))
            for start in range(0, item_count, range_size)
        ]

    def _load_checkpoint_from_device(self) -> int | None:
        """Load the newest valid checkpoint from the raw device if present.

        Returns:
            The loaded checkpoint's format version, or None when none was
            loaded.
        """
        header, payload = self._select_latest_checkpoint()
        if header is None:
            logger.info("RawBlockCore: no valid on-device metadata checkpoint found")
            return None
        if payload is None:
            logger.warning("RawBlockCore: checkpoint header had no payload")
            return None
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning("RawBlockCore: failed to decode metadata payload")
            return None
        if not self.apply_loaded_state(data):
            logger.warning("RawBlockCore: metadata payload rejected by checks")
            self._note_foreign_layout(data)
            return None
        # Continue after the copy loaded, not after a newer torn one: the next
        # write then replaces the torn copy and keeps this one.
        self._meta_seq = int(header["seq"])
        logger.info(
            "RawBlockCore loaded format-%d checkpoint (entries=%d seq=%d device=%s)",
            data["version"],
            len(self._index),
            self._meta_seq,
            self.device_path,
        )
        return int(data["version"])
