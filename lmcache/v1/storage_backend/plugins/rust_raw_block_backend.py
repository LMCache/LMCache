# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence
import asyncio
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
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StoragePluginInterface,
)

logger = init_logger(__name__)


_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1
_META_HEADER_STRUCT = struct.Struct("<8sIQQI")
TPRankKey = int | str
PerTPDevicePaths = Mapping[TPRankKey, str]


def _round_up(x: int, align: int) -> int:
    """Round up to nearest multiple of alignment (required for O_DIRECT)."""
    return ((x + align - 1) // align) * align


def _validate_per_tp_device_paths(per_tp_devices: PerTPDevicePaths) -> None:
    """Validate per-TP device mapping and enforce unique paths."""
    values = list(per_tp_devices.values())
    if len(values) != len(set(values)):
        raise ValueError(
            "Duplicate device path configured in rust_raw_block.per_tp_device_paths"
        )


def _get_per_tp_device_path(
    per_tp_devices: PerTPDevicePaths, tp_rank: int
) -> Optional[str]:
    """Return the configured device path for a TP rank.

    Looks up both string and integer forms of ``tp_rank`` so YAML mappings
    with either quoted or unquoted numeric keys are accepted.
    """
    return per_tp_devices.get(str(tp_rank), per_tp_devices.get(tp_rank))


@dataclass
class _Entry:
    """In-memory index entry for a stored chunk."""

    offset: int
    size: int
    meta: DiskCacheMetadata


@dataclass
class _Inflight:
    offset: int
    meta: DiskCacheMetadata
    canceled: bool = False


class RustRawBlockBackend(StoragePluginInterface):
    """
    A storage plugin backend that stores KV chunks into a block device (raw)
    using a Rust extension for pread/pwrite or io_uring.

    Features:
    - High-throughput I/O via direct block device access
    - O_DIRECT support to bypass page cache (requires aligned buffers)
    - On-device metadata checkpoint for restart recovery
    - Efficient buffer operations via Rust extension

    - TP > 1 support via per-TP partitions

    TP > 1 Support:
    ----------------
    When using Tensor Parallelism (TP > 1), each TP worker must use a
    separate partition to avoid metadata conflicts and data corruption.

    Configuration:
    For TP > 1, you must explicitly configure device paths for each TP worker:
       extra_config:
         rust_raw_block.per_tp_device_paths:
           "0": "/dev/nvme0n1p1"
           "1": "/dev/nvme0n1p2"
           "2": "/dev/nvme0n1p3"
           "3": "/dev/nvme0n1p4"

    Note: Partitions must be pre-created on the device before use.
    """

    def __init__(
        self,
        config=None,
        metadata=None,
        local_cpu_backend=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cpu",
    ):
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        if self.loop is None:
            raise ValueError("RustRawBlockBackend requires an asyncio event loop")
        if self.local_cpu_backend is None:
            raise ValueError("RustRawBlockBackend requires local_cpu_backend")
        if self.config is None:
            raise ValueError("RustRawBlockBackend requires config")

        extra = self.config.extra_config or {}

        # Support TP > 1 via per-TP device paths.
        # Each TP worker uses its own partition to avoid conflicts.
        self.device_path: str
        if self.metadata is not None and self.metadata.world_size > 1:
            tp_rank = self.metadata.worker_id
            per_tp_devices = extra.get("rust_raw_block.per_tp_device_paths", {})
            if not isinstance(per_tp_devices, Mapping):
                raise ValueError(
                    "rust_raw_block.per_tp_device_paths must be a mapping from "
                    "TP rank to device path"
                )

            if not per_tp_devices:
                raise ValueError(
                    "For TP > 1, rust_raw_block.per_tp_device_paths is required. "
                    "Each TP worker must have an explicit device path configured."
                )
            _validate_per_tp_device_paths(per_tp_devices)

            tp_rank_str = str(tp_rank)
            device_path = _get_per_tp_device_path(per_tp_devices, tp_rank)
            if not device_path:
                raise ValueError(
                    f"No device path configured for TP rank {tp_rank_str}. "
                    f"Available ranks: {list(per_tp_devices.keys())}"
                )
            self.device_path = device_path
            logger.info(
                f"RustRawBlockBackend: TP={self.metadata.world_size} mode, "
                f"using explicit device path for rank {tp_rank}: {self.device_path}"
            )
        else:
            self.device_path = extra.get("rust_raw_block.device_path", "")
            if not self.device_path:
                raise ValueError(
                    "extra_config['rust_raw_block.device_path'] is required"
                )

        self.capacity_bytes: int = int(extra.get("rust_raw_block.capacity_bytes", 0))
        self.block_align: int = int(extra.get("rust_raw_block.block_align", 4096))
        self.header_bytes: int = int(extra.get("rust_raw_block.header_bytes", 4096))
        self.use_odirect: bool = bool(extra.get("rust_raw_block.use_odirect", False))
        # Try direct aligned O_DIRECT I/O when allocator buffers permit.
        self.enable_zero_copy: bool = bool(
            extra.get("rust_raw_block.enable_zero_copy", True)
        )
        self.use_uring: bool = bool(extra.get("rust_raw_block.use_uring", False))

        # On-device metadata region config.
        self.meta_total_bytes: int = int(
            extra.get("rust_raw_block.meta_total_bytes", 128 * 1024 * 1024)
        )
        meta_magic_raw = extra.get("rust_raw_block.meta_magic", "LMCIDX01")
        if isinstance(meta_magic_raw, str):
            self.meta_magic: bytes = meta_magic_raw.encode("ascii")
        elif isinstance(meta_magic_raw, bytes):
            self.meta_magic = meta_magic_raw
        else:
            raise ValueError("rust_raw_block.meta_magic must be str or bytes")
        if len(self.meta_magic) != 8:
            raise ValueError("rust_raw_block.meta_magic must be exactly 8 bytes")
        try:
            self.meta_magic_text: str = self.meta_magic.decode("ascii")
        except UnicodeDecodeError as e:
            raise ValueError("rust_raw_block.meta_magic must be ASCII bytes") from e
        self.meta_version: int = int(
            extra.get("rust_raw_block.meta_version", _DEFAULT_META_VERSION)
        )
        if self.meta_version <= 0:
            raise ValueError("rust_raw_block.meta_version must be > 0")
        self.meta_checkpoint_interval_sec: int = int(
            extra.get("rust_raw_block.meta_checkpoint_interval_sec", 60)
        )
        self.meta_idle_quiet_ms: int = int(
            extra.get("rust_raw_block.meta_idle_quiet_ms", 100)
        )
        self.meta_enable_periodic: bool = bool(
            extra.get("rust_raw_block.meta_enable_periodic", True)
        )
        self.meta_verify_on_load: bool = bool(
            extra.get("rust_raw_block.meta_verify_on_load", True)
        )
        self._meta_copy_count: int = 2

        get_full_chunk_size_bytes = getattr(
            self.local_cpu_backend, "get_full_chunk_size_bytes", None
        )
        if callable(get_full_chunk_size_bytes):
            full_chunk_bytes = int(get_full_chunk_size_bytes())
        else:
            get_full_chunk_size = getattr(
                self.local_cpu_backend, "get_full_chunk_size", None
            )
            if not callable(get_full_chunk_size):
                raise ValueError(
                    "local_cpu_backend must expose get_full_chunk_size_bytes() "
                    "or get_full_chunk_size()"
                )
            full_chunk_bytes = int(get_full_chunk_size())
        default_slot_bytes = _round_up(
            self.header_bytes + full_chunk_bytes, self.block_align
        )
        self.slot_bytes: int = int(
            extra.get("rust_raw_block.slot_bytes", default_slot_bytes)
        )

        if self.slot_bytes < self.header_bytes + 1:
            raise ValueError("rust_raw_block.slot_bytes too small")
        if self.slot_bytes % self.block_align != 0:
            raise ValueError(
                "rust_raw_block.slot_bytes must be multiple of block_align"
            )
        if self.header_bytes % self.block_align != 0:
            raise ValueError(
                "rust_raw_block.header_bytes must be multiple of block_align"
            )
        if self.meta_total_bytes <= 0:
            raise ValueError("rust_raw_block.meta_total_bytes must be > 0")
        if self.meta_total_bytes % self.block_align != 0:
            raise ValueError(
                "rust_raw_block.meta_total_bytes must align to block_align"
            )
        if self.meta_total_bytes <= self.block_align:
            raise ValueError(
                "rust_raw_block.meta_total_bytes must be > block_align "
                "(room for metadata header + payload)"
            )
        self._meta_container_bytes: int = (
            (self.meta_total_bytes // self._meta_copy_count) // self.block_align
        ) * self.block_align
        if self._meta_container_bytes <= self.block_align:
            raise ValueError(
                "rust_raw_block.meta_total_bytes must provide room for at least "
                "two metadata copies (header + payload)"
            )

        self._lock = threading.Lock()
        self._index: dict[CacheEngineKey, _Entry] = {}
        self._pinned: set[CacheEngineKey] = set()
        self._inflight: dict[CacheEngineKey, _Inflight] = {}
        self._lru: "OrderedDict[CacheEngineKey, None]" = OrderedDict()

        self._next_slot: int = 0
        self._free_slots: list[int] = []
        self._max_slots: int = 0
        self._effective_capacity_bytes: int = 0
        self._data_base_offset: int = 0

        self._dbg_first_n: int = int(extra.get("rust_raw_block.debug_first_n", 4) or 0)
        self._dbg_every_n: int = int(
            extra.get("rust_raw_block.debug_every_n", 256) or 0
        )
        self._dbg_put_batches: int = 0
        self._dbg_put_keys: int = 0
        self._dbg_put_bytes: int = 0
        self._dbg_get_calls: int = 0
        self._dbg_get_bytes: int = 0

        self._raw = None

        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()

        # Metadata checkpoint state.
        self._meta_seq: int = 0
        self._meta_dirty_total: int = 0
        self._meta_persisted: int = 0
        self._inflight_io_count: int = 0
        self._last_io_ts: float = time.monotonic()
        self._meta_stop_evt = threading.Event()
        self._meta_thread: Optional[threading.Thread] = None

        self._ensure_capacity_and_layout()

        logger.info(
            "RustRawBlockBackend init: device=%s cap=%s slot=%d align=%d header=%d "
            "meta_total=%d data_base=%d zero_copy=%s",
            self.device_path,
            self.capacity_bytes,
            self.slot_bytes,
            self.block_align,
            self.header_bytes,
            self.meta_total_bytes,
            self._data_base_offset,
            self.enable_zero_copy,
        )

        # Register paged buffers with io_uring for fixed buffer support
        if self.use_uring:
            self._register_paged_buffers()

        # Load latest checkpoint from device (no JSON fallback).
        self._load_checkpoint_from_device()

        if self.meta_enable_periodic:
            self._meta_thread = threading.Thread(
                target=self._checkpoint_loop,
                name="rust-raw-block-meta-checkpoint",
                daemon=True,
            )
            self._meta_thread.start()

    def _dbg_should_log(self, n: int) -> bool:
        if not logger.isEnabledFor(10):
            return False
        if self._dbg_first_n and n <= self._dbg_first_n:
            return True
        if self._dbg_every_n and n % self._dbg_every_n == 0:
            return True
        return False

    def _dbg_key_short(self, key: CacheEngineKey) -> str:
        try:
            return f"chunk_hash={int(key.chunk_hash)}"
        except Exception:
            return "chunk_hash=?"

    def __str__(self) -> str:
        return "RustRawBlockBackend"

    def _rawdev(self):
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
                use_iouring=self.use_uring,
            )
        return self._raw

    def _register_paged_buffers(self) -> None:
        """Register paged buffers with io_uring for fixed buffer support."""
        try:
            assert self.local_cpu_backend is not None
            memory_allocator = self.local_cpu_backend.get_memory_allocator()
            paged_buffers = getattr(memory_allocator, "get_paged_buffers", None)

            if paged_buffers is not None and callable(paged_buffers):
                buffers = paged_buffers()
                if buffers is not None and len(buffers) > 0:
                    raw_dev = self._rawdev()
                    # Register buffers with io_uring
                    buffer_ptrs = [buf.data_ptr() for buf in buffers]
                    buffer_sizes = [buf.numel() * buf.element_size() for buf in buffers]

                    raw_dev.register_fixed_buffers(buffer_ptrs, buffer_sizes)

                    logger.info(
                        "RustRawBlockBackend: registered %d paged buffers with "
                        "io_uring for fixed buffer support (true zero copy)",
                        len(buffers),
                    )
                else:
                    logger.warning(
                        "RustRawBlockBackend: no paged buffers available for "
                        "io_uring fixed buffer registration"
                    )
            else:
                logger.warning(
                    "RustRawBlockBackend: memory allocator does not support "
                    "paged buffers for io_uring fixed buffer registration"
                )
        except Exception as e:
            logger.warning(
                "RustRawBlockBackend: failed to register paged buffers with "
                "io_uring: %s. Falling back to non-fixed buffer mode.",
                e,
            )

    def _build_direct_odirect_view(
        self,
        memory_obj: MemoryObj,
        payload_len: int,
        total_len: int,
        buffer_len: int,
        *,
        zero_tail: bool,
    ) -> Optional[memoryview]:
        """Build direct physical-memory view for O_DIRECT without staging copy."""
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

        try:
            ptr = int(ptr_val)
        except Exception:
            return None

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

    def _ensure_capacity_and_layout(self) -> None:
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
        return self._data_base_offset + slot * self.slot_bytes

    def _offset_to_slot(self, offset: int) -> int:
        return (offset - self._data_base_offset) // self.slot_bytes

    def _allocate_slot(self) -> int:
        self._ensure_capacity_and_layout()

        if self._free_slots:
            return self._slot_to_offset(self._free_slots.pop())

        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            return self._slot_to_offset(slot)

        raise RuntimeError("No free slots available; eviction required")

    def _touch(self, key: CacheEngineKey) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None

    def _append_free_slot_locked(self, slot: int) -> None:
        if slot < 0 or slot >= self._max_slots:
            return
        if slot in self._free_slots:
            return
        self._free_slots.append(slot)

    def _evict_one(self) -> bool:
        for victim in list(self._lru.keys()):
            if victim in self._pinned or victim in self._inflight:
                continue
            entry = self._index.pop(victim, None)
            if entry is None:
                self._lru.pop(victim, None)
                continue
            self._lru.pop(victim, None)
            self._pinned.discard(victim)
            self._append_free_slot_locked(self._offset_to_slot(int(entry.offset)))
            self._meta_dirty_total += 1
            return True
        return False

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self._lock:
            ok = key in self._index
            if ok and pin:
                self._pinned.add(key)
            return ok

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    def pin(self, key: CacheEngineKey) -> bool:
        with self._lock:
            if key in self._index:
                self._pinned.add(key)
                return True
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self._lock:
            if key in self._pinned:
                self._pinned.remove(key)
                return True
            return key in self._index

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:  # noqa: ARG002
        with self._lock:
            existed = key in self._index or key in self._inflight
            entry = self._index.pop(key, None)
            inflight = self._inflight.get(key)
            self._pinned.discard(key)
            self._lru.pop(key, None)
            if entry is not None:
                self._append_free_slot_locked(self._offset_to_slot(int(entry.offset)))
                self._meta_dirty_total += 1
            if inflight is not None:
                inflight.canceled = True
            return existed

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,  # noqa: ARG002
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        if logger.isEnabledFor(10):
            self._dbg_put_batches += 1
            self._dbg_put_keys += int(len(keys))
            try:
                self._dbg_put_bytes += int(sum(len(o.byte_array) for o in objs))
            except Exception:
                pass
            if self._dbg_should_log(self._dbg_put_batches):
                logger.debug(
                    "RustRawBlockBackend PUT: keys=%d inflight=%d indexed=%d",
                    len(keys),
                    len(self._inflight),
                    len(self._index),
                )

        # Use io_uring path if enabled
        if self.use_uring:
            # TODO(Ankit): Find a better way to handle this.
            for obj in objs:
                obj.ref_count_up()

            assert self.loop is not None
            fut = asyncio.run_coroutine_threadsafe(
                self._batched_submit_put_task_uring(keys, objs, on_complete_callback),
                self.loop,
            )
            return [fut]

        futures = []
        for key, obj in zip(keys, objs, strict=False):
            with self._put_lock:
                if key in self._put_tasks:
                    continue
                self._put_tasks.add(key)

            with self._lock:
                if key in self._index or key in self._inflight:
                    with self._put_lock:
                        self._put_tasks.discard(key)
                    continue
                while True:
                    try:
                        offset = self._allocate_slot()
                        break
                    except RuntimeError:
                        if not self._evict_one():
                            with self._put_lock:
                                self._put_tasks.discard(key)
                            raise

                meta = DiskCacheMetadata(
                    path=f"{self.device_path}@{offset}",
                    size=len(obj.byte_array),
                    shape=obj.metadata.shape,
                    dtype=obj.metadata.dtype,
                    cached_positions=obj.metadata.cached_positions,
                    fmt=obj.metadata.fmt,
                    pin_count=0,
                )
                self._inflight[key] = _Inflight(offset=offset, meta=meta)

            header = self._encode_header(key, meta.size)
            obj.ref_count_up()
            assert self.loop is not None
            fut = asyncio.run_coroutine_threadsafe(
                self._submit_write(
                    key=key,
                    offset=offset,
                    header=header,
                    memory_obj=obj,
                    on_complete_callback=on_complete_callback,
                ),
                self.loop,
            )
            futures.append(fut)
        return futures or None

    async def _batched_submit_put_task_uring(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        """Batched put using io_uring"""

        # Collect all write requests
        write_requests: list[tuple[CacheEngineKey, int, bytes, MemoryObj]] = []
        valid_keys: list[CacheEngineKey] = []
        valid_objs: list[MemoryObj] = []

        # Track which items had inflight_io_count incremented
        successfully_submitted: list[CacheEngineKey] = []

        # Track which objects have incremented ref counts
        objs_with_inc_ref = list(objs)

        if self._raw is None:
            raise RuntimeError("device is closed")

        write_error: Optional[Exception] = None
        try:
            for key, obj in zip(keys, objs, strict=False):
                with self._put_lock:
                    if key in self._put_tasks:
                        obj.ref_count_down()
                        objs_with_inc_ref.remove(obj)
                        continue
                    self._put_tasks.add(key)

                with self._lock:
                    if key in self._index or key in self._inflight:
                        with self._put_lock:
                            self._put_tasks.discard(key)
                        obj.ref_count_down()
                        objs_with_inc_ref.remove(obj)
                        continue
                    while True:
                        try:
                            offset = self._allocate_slot()
                            break
                        except RuntimeError:
                            if not self._evict_one():
                                with self._put_lock:
                                    self._put_tasks.discard(key)
                                raise

                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=len(obj.byte_array),
                        shape=obj.metadata.shape,
                        dtype=obj.metadata.dtype,
                        cached_positions=obj.metadata.cached_positions,
                        fmt=obj.metadata.fmt,
                        pin_count=0,
                    )
                    self._inflight[key] = _Inflight(offset=offset, meta=meta)

                header = self._encode_header(key, meta.size)
                write_requests.append((key, offset, header, obj))
                valid_keys.append(key)
                valid_objs.append(obj)

            if not write_requests:
                return None

            raw_dev = self._rawdev()
            offsets = []
            buffers = []
            total_lens = []

            for key, offset, header, obj in write_requests:
                # Prepare payload with proper O_DIRECT alignment and zero-copy handling.
                try:
                    buf, payload_len, total_len = self._prepare_write_payload(obj)
                    assert payload_len == total_len
                except Exception as e:
                    write_error = e
                    logger.error(f"Failed to prepare payload for key {key}: {e}")
                    raise

                hdr_total = (
                    _round_up(len(header), self.block_align)
                    if self.use_odirect
                    else len(header)
                )
                header_bytes = bytearray(header)
                if self.use_odirect and len(header_bytes) < hdr_total:
                    header_bytes.extend(b"\x00" * (hdr_total - len(header_bytes)))

                offsets.append(offset)
                buffers.append(header_bytes)
                total_lens.append(hdr_total)
                offsets.append(offset + self.header_bytes)
                buffers.append(buf)
                total_lens.append(total_len)

                with self._lock:
                    self._inflight_io_count += 1
                successfully_submitted.append(key)

            batch_id = raw_dev.batched_write(offsets, buffers, total_lens)
            # Wait for headers and payloads to complete. Buffer lifetime is managed by
            # Rust via Py_buffer views.
            # Pass batch_id to capture any error from this batch completions.
            # TODO(Ankit): Add a way to capture specific write failures.
            await asyncio.to_thread(raw_dev.wait_iouring, batch_id)
        except Exception as e:
            if write_error is None:
                write_error = e
            logger.error(f"Batched write failed for keys {valid_keys}: {e}")
        finally:
            with self._lock:
                for key in successfully_submitted:
                    self._inflight_io_count -= 1
                    self._last_io_ts = time.monotonic()

            for obj in objs_with_inc_ref:
                obj.ref_count_down()

            with self._put_lock:
                for key in valid_keys:
                    self._put_tasks.discard(key)

            if write_error is None:
                with self._lock:
                    for key, offset, header, obj in write_requests:
                        inflight = self._inflight.pop(key, None)
                        if inflight is not None and not inflight.canceled:
                            self._index[key] = _Entry(
                                offset=inflight.offset,
                                size=inflight.meta.size,
                                meta=inflight.meta,
                            )
                            self._touch(key)
                            self._meta_dirty_total += 1

                if on_complete_callback is not None:
                    for key in valid_keys:
                        try:
                            on_complete_callback(key)
                        except Exception as e:
                            logger.warning(
                                f"on_complete_callback failed for key {key}: {e}"
                            )
            else:
                with self._lock:
                    for key in valid_keys:
                        inflight = self._inflight.pop(key, None)
                        if inflight is not None:
                            self._append_free_slot_locked(
                                self._offset_to_slot(int(inflight.offset))
                            )
                            self._meta_dirty_total += 1
            if write_error is not None:
                raise write_error
        return None

    def _prepare_write_payload(self, memory_obj: MemoryObj) -> tuple[Any, int, int]:
        """Prepare payload view and aligned lengths for write path."""
        buf = memory_obj.byte_array
        if hasattr(buf, "cast"):
            buf = buf.cast("B")
        payload_len = len(memory_obj.byte_array)
        total_len = payload_len
        if self.use_odirect:
            total_len = _round_up(payload_len, self.block_align)
            if total_len > (self.slot_bytes - self.header_bytes):
                raise RuntimeError(f"O_DIRECT payload {total_len} > slot capacity")
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

    async def _submit_write(
        self,
        key: CacheEngineKey,
        offset: int,
        header: bytes,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        try:
            buf, payload_len, total_len = self._prepare_write_payload(memory_obj)

            def _do_write():
                with self._lock:
                    self._inflight_io_count += 1
                try:
                    raw_dev = self._rawdev()
                    hdr_total = (
                        _round_up(len(header), self.block_align)
                        if self.use_odirect
                        else len(header)
                    )
                    raw_dev.pwrite_from_buffer(offset, header, len(header), hdr_total)
                    raw_dev.pwrite_from_buffer(
                        offset + self.header_bytes, buf, payload_len, total_len
                    )
                except Exception as e:
                    logger.error(
                        f"Write failed for key {self._dbg_key_short(key)}: {e}"
                    )
                    raise
                finally:
                    with self._lock:
                        self._inflight_io_count -= 1
                        self._last_io_ts = time.monotonic()

            write_error: Optional[Exception] = None
            try:
                await asyncio.to_thread(_do_write)
            except Exception as e:
                write_error = e

            with self._lock:
                inflight = self._inflight.pop(key, None)
                if inflight is not None:
                    if inflight.canceled or write_error is not None:
                        self._append_free_slot_locked(
                            self._offset_to_slot(int(inflight.offset))
                        )
                        self._meta_dirty_total += 1
                    else:
                        self._index[key] = _Entry(
                            offset=inflight.offset,
                            size=inflight.meta.size,
                            meta=inflight.meta,
                        )
                        self._touch(key)
                        self._meta_dirty_total += 1

            if write_error is None:
                if on_complete_callback is not None:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            f"on_complete_callback failed for key {key}: {e}"
                        )
            else:
                raise write_error
        finally:
            memory_obj.ref_count_down()
            with self._put_lock:
                self._put_tasks.discard(key)

    def _encode_header(self, key: CacheEngineKey, payload_len: int) -> bytes:
        magic = b"LMCBLK01"
        chunk_hash = int(key.chunk_hash) & ((1 << 64) - 1)
        hdr = bytearray(self.header_bytes)
        hdr[0:8] = magic
        hdr[8:16] = chunk_hash.to_bytes(8, "little", signed=False)
        hdr[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
        return bytes(hdr)

    def _decode_slot_header(self, hdr: bytes) -> Optional[tuple[int, int]]:
        if len(hdr) < 24:
            return None
        if hdr[0:8] != b"LMCBLK01":
            return None
        chunk_hash = int.from_bytes(hdr[8:16], "little", signed=False)
        payload_len = int.from_bytes(hdr[16:24], "little", signed=False)
        return chunk_hash, payload_len

    def _read_slot_header(self, offset: int) -> Optional[tuple[int, int]]:
        raw = self._rawdev()
        buf = bytearray(self.header_bytes)
        try:
            with self._lock:
                self._inflight_io_count += 1
            if self.use_uring:
                raw.read_uring(offset, buf, self.header_bytes, self.header_bytes)
            else:
                raw.pread_into(offset, buf, self.header_bytes, self.header_bytes)
            return self._decode_slot_header(buf)
        except Exception:
            return None
        finally:
            with self._lock:
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()

    async def _batched_get_prefix_uring(
        self, keys: Sequence[CacheEngineKey]
    ) -> list[MemoryObj]:
        """Batched get using io_uring"""
        if not keys:
            return []

        if self._raw is None:
            raise RuntimeError("device is closed")

        items: list[tuple[CacheEngineKey, _Entry]] = []
        with self._lock:
            for key in keys:
                entry = self._index.get(key)
                if entry is None:
                    break
                items.append((key, entry))
            if not items:
                return []
            self._inflight_io_count += 1

        loaded: list[MemoryObj] = []
        touched: list[CacheEngineKey] = []
        offsets = []
        buffers = []
        total_lens = []
        valid_keys = []
        valid_objs = []
        try:
            raw_dev = self._rawdev()

            for key, entry in items:
                meta = entry.meta
                assert meta.shape is not None and meta.dtype is not None
                assert self.local_cpu_backend is not None
                memory_obj = self.local_cpu_backend.allocate(
                    meta.shape, meta.dtype, meta.fmt
                )
                if memory_obj is None:
                    logger.error(
                        "Failed to allocate memory for key %s",
                        self._dbg_key_short(key),
                    )

                    for obj in valid_objs:
                        obj.ref_count_down()
                    return []

                total_len = int(meta.size)
                assert (total_len % self.block_align) == 0
                if logger.isEnabledFor(10):
                    self._dbg_get_calls += 1
                    self._dbg_get_bytes += total_len
                    if self._dbg_should_log(self._dbg_get_calls):
                        logger.debug(
                            "RustRawBlockBackend GET (uring): %s offset=%d size=%d",
                            self._dbg_key_short(key),
                            int(entry.offset),
                            total_len,
                        )

                buf = memory_obj.byte_array
                try:
                    buf = buf.cast("B")
                except Exception:
                    pass

                direct_view = self._build_direct_odirect_view(
                    memory_obj=memory_obj,
                    payload_len=total_len,
                    total_len=total_len,
                    buffer_len=len(buf),
                    zero_tail=False,
                )

                if direct_view is not None:
                    offsets.append(entry.offset + self.header_bytes)
                    buffers.append(direct_view)
                    total_lens.append(total_len)
                else:
                    offsets.append(entry.offset + self.header_bytes)
                    buffers.append(buf)
                    total_lens.append(total_len)

                valid_keys.append(key)
                valid_objs.append(memory_obj)

            if not offsets:
                return []

            batch_id = raw_dev.batched_read(offsets, buffers, total_lens)
            # Wait for all reads to complete.
            # Pass batch_id to capture any error from this batch completions.
            # TODO(Ankit): Add a way to capture specific read failures.
            await asyncio.to_thread(raw_dev.wait_iouring, batch_id)

            # Update metadata for successfully loaded items
            for key, obj in zip(valid_keys, valid_objs, strict=False):
                entry = next((e for k, e in items if k == key), None)
                if entry is not None:
                    obj.metadata.cached_positions = entry.meta.cached_positions
                loaded.append(obj)
                touched.append(key)

        except Exception as e:
            for memory_obj in valid_objs:
                memory_obj.ref_count_down()
            loaded.clear()
            touched.clear()
            logger.error("Batched io_uring read failed: %s", e)
            raise
        finally:
            with self._lock:
                for key in touched:
                    self._touch(key)
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()
        return loaded

    def _batched_get_prefix(self, keys: Sequence[CacheEngineKey]) -> list[MemoryObj]:
        if not keys:
            return []

        items: list[tuple[CacheEngineKey, _Entry]] = []
        with self._lock:
            for key in keys:
                entry = self._index.get(key)
                if entry is None:
                    break
                items.append((key, entry))
            if not items:
                return []
            self._inflight_io_count += 1

        loaded: list[MemoryObj] = []
        touched: list[CacheEngineKey] = []
        try:
            raw_dev = self._rawdev()
            for key, entry in items:
                meta = entry.meta
                assert meta.shape is not None and meta.dtype is not None
                assert self.local_cpu_backend is not None
                memory_obj = self.local_cpu_backend.allocate(
                    meta.shape, meta.dtype, meta.fmt
                )
                if memory_obj is None:
                    logger.error(
                        "Failed to allocate memory for key %s",
                        self._dbg_key_short(key),
                    )
                    break

                try:
                    payload_len = int(meta.size)
                    total_len = (
                        _round_up(payload_len, self.block_align)
                        if self.use_odirect
                        else payload_len
                    )
                    if logger.isEnabledFor(10):
                        self._dbg_get_calls += 1
                        self._dbg_get_bytes += payload_len
                        if self._dbg_should_log(self._dbg_get_calls):
                            logger.debug(
                                "RustRawBlockBackend GET: %s offset=%d size=%d",
                                self._dbg_key_short(key),
                                int(entry.offset),
                                payload_len,
                            )

                    buf = memory_obj.byte_array
                    try:
                        buf = buf.cast("B")
                    except Exception:
                        pass
                    direct_view = self._build_direct_odirect_view(
                        memory_obj=memory_obj,
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

                    memory_obj.metadata.cached_positions = meta.cached_positions
                except Exception as e:
                    memory_obj.ref_count_down()
                    logger.error(
                        "Read failed for key %s: %s", self._dbg_key_short(key), e
                    )
                    raise

                loaded.append(memory_obj)
                touched.append(key)
        except Exception:
            for memory_obj in loaded:
                memory_obj.ref_count_down()
            loaded.clear()
            touched.clear()
            raise
        finally:
            with self._lock:
                for key in touched:
                    self._touch(key)
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()
        return loaded

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if self.use_uring:
            assert self.loop is not None
            loaded = asyncio.run_coroutine_threadsafe(
                self._batched_get_prefix_uring([key]),
                self.loop,
            ).result()
            return loaded[0] if loaded else None
        loaded = self._batched_get_prefix([key])
        return loaded[0] if loaded else None

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        Get a batch of cache entries until the first miss.

        :param List[CacheEngineKey] keys: Ordered keys to retrieve.

        :return: A list aligned to ``keys`` where the successful prefix contains
            loaded memory objects and the remaining suffix is ``None``.

        :raises Exception: Propagates raw-device initialization or read failures.
        """
        if not keys:
            return []

        if self.use_uring:
            assert self.loop is not None
            loaded = asyncio.run_coroutine_threadsafe(
                self._batched_get_prefix_uring(keys),
                self.loop,
            ).result()
            return [*loaded, *([None] * (len(keys) - len(loaded)))]

        loaded = self._batched_get_prefix(keys)
        return [*loaded, *([None] * (len(keys) - len(loaded)))]

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        del lookup_id
        hit = 0
        with self._lock:
            for k in keys:
                if k not in self._index:
                    break
                if pin:
                    self._pinned.add(k)
                hit += 1
        return hit

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        Asynchronously get a batch of cache entries until the first miss.

        :param str lookup_id: Lookup identifier used by the storage manager.
        :param list[CacheEngineKey] keys: Ordered keys to retrieve.
        :param Any transfer_spec: Unused transfer hint for API compatibility.

        :return: The successfully loaded prefix of ``keys`` in input order.

        :raises Exception: Propagates raw-device initialization or read failures.
        """
        del lookup_id, transfer_spec
        if self.use_uring:
            return await self._batched_get_prefix_uring(keys)
        return await asyncio.to_thread(self._batched_get_prefix, keys)

    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        assert self.local_cpu_backend is not None
        return self.local_cpu_backend

    def close(self) -> None:
        if logger.isEnabledFor(10):
            logger.debug(
                "RustRawBlockBackend stats: put=%d/%d/%d get=%d/%d",
                self._dbg_put_batches,
                self._dbg_put_keys,
                self._dbg_put_bytes,
                self._dbg_get_calls,
                self._dbg_get_bytes,
            )

        # Stop checkpoint background thread.
        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None

        # Wait briefly for inflight put tasks to drain.
        deadline = time.monotonic() + 10.0
        while True:
            with self._put_lock:
                pending = len(self._put_tasks)
            if pending == 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)

        # Force final metadata checkpoint.
        try:
            self._checkpoint_once(force=True)
        except Exception as e:
            logger.warning(f"Failed to write final on-device metadata checkpoint: {e}")

        if self._raw is not None:
            try:
                self._raw.close()
            except Exception as e:
                logger.warning(f"Failed to close raw block device: {e}")
            finally:
                self._raw = None

    # ------------------- On-device metadata checkpoint -------------------

    def _checkpoint_loop(self) -> None:
        interval = max(1, self.meta_checkpoint_interval_sec)
        while not self._meta_stop_evt.wait(interval):
            try:
                self._checkpoint_once(force=False)
            except Exception as e:
                logger.warning(f"Periodic metadata checkpoint failed: {e}")

    def _meta_payload_capacity(self) -> int:
        return self._meta_container_bytes - self.block_align

    def _meta_container_offsets(self) -> list[int]:
        return [
            idx * self._meta_container_bytes for idx in range(self._meta_copy_count)
        ]

    def _read_meta_header(self, container_offset: int) -> Optional[dict[str, int]]:
        raw = self._rawdev()
        buf = bytearray(self.block_align)
        try:
            if self.use_uring:
                raw.read_uring(
                    container_offset, buf, self.block_align, self.block_align
                )
            else:
                raw.pread_into(
                    container_offset, buf, self.block_align, self.block_align
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

    def _load_meta_payload(self, header: dict[str, int]) -> Optional[bytes]:
        raw = self._rawdev()
        payload_len = int(header["payload_len"])
        payload_off = int(header["container_offset"]) + self.block_align
        total_len = _round_up(payload_len, self.block_align)
        buf = bytearray(total_len)
        try:
            if self.use_uring:
                raw.read_uring(payload_off, buf, payload_len, total_len)
            else:
                raw.pread_into(payload_off, buf, payload_len, total_len)
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
                "lru_keys": [k.to_string() for k in self._lru.keys()],
                "entries": {
                    k.to_string(): {
                        "offset": e.offset,
                        "size": e.meta.size,
                        "shape": list(e.meta.shape)
                        if e.meta.shape is not None
                        else None,
                        "dtype": k._dtype_str,
                        "fmt": (
                            e.meta.fmt.name
                            if e.meta.fmt is not None and hasattr(e.meta.fmt, "name")
                            else str(e.meta.fmt)
                            if e.meta.fmt is not None
                            else None
                        ),
                        "cached_positions": (
                            e.meta.cached_positions.tolist()
                            if e.meta.cached_positions is not None
                            and hasattr(e.meta.cached_positions, "tolist")
                            else None
                        ),
                    }
                    for k, e in self._index.items()
                },
            }
        return snapshot, dirty_total

    def _write_checkpoint(self, payload: bytes, dirty_total_snapshot: int) -> bool:
        payload_cap = self._meta_payload_capacity()
        if len(payload) > payload_cap:
            logger.warning(
                "Metadata payload too large (%d > %d), skipping checkpoint",
                len(payload),
                payload_cap,
            )
            return False

        next_seq = self._meta_seq + 1
        target_idx = int((next_seq - 1) % self._meta_copy_count)
        target = self._meta_container_offsets()[target_idx]

        payload_len = len(payload)
        payload_total_len = _round_up(payload_len, self.block_align)
        payload_off = target + self.block_align
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        header_block = bytearray(self.block_align)
        header_block[: _META_HEADER_STRUCT.size] = _META_HEADER_STRUCT.pack(
            self.meta_magic,
            self.meta_version,
            int(next_seq),
            int(payload_len),
            int(crc),
        )

        raw = self._rawdev()

        raw.pwrite_from_buffer(payload_off, payload, payload_len, payload_total_len)
        raw.pwrite_from_buffer(target, header_block, self.block_align, self.block_align)

        with self._lock:
            self._meta_seq = int(next_seq)
            self._meta_persisted = max(self._meta_persisted, int(dirty_total_snapshot))

        return True

    def _checkpoint_once(self, force: bool) -> bool:
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

        ok = self._write_checkpoint(payload, dirty_total_snapshot)
        if ok and logger.isEnabledFor(10):
            logger.debug(
                "RustRawBlockBackend: checkpoint saved seq=%d entries=%d",
                self._meta_seq,
                len(snapshot.get("entries", {})),
            )
        return ok

    def _is_valid_checkpoint_entry(self, offset: int, size: int) -> bool:
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
        if not isinstance(data, dict):
            return False
        if int(data.get("version", 0)) != 1:
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
            self._lru.clear()

            entries = data.get("entries", {})
            if isinstance(entries, dict):
                for k_str, entry in entries.items():
                    if not isinstance(entry, dict):
                        logger.warning(
                            "Invalid entry in metadata for key '%s': not a dict.", k_str
                        )
                        continue
                    try:
                        key = CacheEngineKey.from_string(k_str)
                    except Exception as e:
                        logger.warning(
                            "Failed to parse key string '%s' from metadata: %s",
                            k_str,
                            e,
                        )
                        continue

                    offset = int(entry.get("offset", 0))
                    size = int(entry.get("size", 0))
                    shape_list = entry.get("shape")
                    fmt_name = entry.get("fmt")
                    cached_positions_list = entry.get("cached_positions")

                    if not self._is_valid_checkpoint_entry(offset, size):
                        logger.warning(
                            "Skipping invalid checkpoint entry for key '%s': "
                            "offset=%d size=%d",
                            k_str,
                            offset,
                            size,
                        )
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

                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=size,
                        shape=shape,
                        dtype=key.dtype,
                        cached_positions=cached_positions,
                        fmt=fmt,
                        pin_count=0,
                    )
                    self._index[key] = _Entry(offset=offset, size=size, meta=meta)

            if self.metadata is not None and self._index:
                first_loaded_key = next(iter(self._index))
                expected_worker_id = int(self.metadata.worker_id)
                loaded_worker_id = int(first_loaded_key.worker_id)
                if loaded_worker_id != expected_worker_id:
                    logger.warning(
                        "RustRawBlockBackend: loaded metadata may belong to another "
                        "worker (device=%s, current_worker_id=%d, "
                        "first_entry_worker_id=%d, first_entry_key=%s)",
                        self.device_path,
                        expected_worker_id,
                        loaded_worker_id,
                        first_loaded_key.to_string(),
                    )

            # Remove free-slot entries that overlap with loaded index slots.
            used_slots = {
                self._offset_to_slot(int(entry.offset))
                for entry in self._index.values()
            }
            self._free_slots = [
                slot for slot in self._free_slots if slot not in used_slots
            ]

            lru_keys = data.get("lru_keys", [])
            if isinstance(lru_keys, list) and lru_keys:
                for k_str in lru_keys:
                    try:
                        key = CacheEngineKey.from_string(k_str)
                    except Exception:
                        continue
                    if key in self._index:
                        self._lru[key] = None
            else:
                for key in self._index:
                    self._lru[key] = None

            # Loaded state should start as clean.
            self._meta_dirty_total = 0
            self._meta_persisted = 0

        if self.meta_verify_on_load:
            self._validate_loaded_entries()
        return True

    def _validate_loaded_entries(self) -> None:
        to_drop: list[CacheEngineKey] = []
        with self._lock:
            entries = list(self._index.items())

        for key, entry in entries:
            slot_hdr = self._read_slot_header(int(entry.offset))
            if slot_hdr is None:
                to_drop.append(key)
                continue
            chunk_hash, payload_len = slot_hdr
            if int(chunk_hash) != (int(key.chunk_hash) & ((1 << 64) - 1)):
                to_drop.append(key)
                continue
            if int(payload_len) != int(entry.size):
                to_drop.append(key)

        if not to_drop:
            return

        with self._lock:
            for key in to_drop:
                removed_entry: Optional[_Entry] = None
                if key in self._index:
                    removed_entry = self._index.pop(key)
                self._lru.pop(key, None)
                self._pinned.discard(key)
                if removed_entry is not None:
                    self._append_free_slot_locked(
                        self._offset_to_slot(int(removed_entry.offset))
                    )
            self._meta_dirty_total += 1

        logger.warning(
            "RustRawBlockBackend: dropped %d stale metadata entries after slot-header "
            "validation",
            len(to_drop),
        )

    def _load_checkpoint_from_device(self) -> None:
        header, payload = self._select_latest_checkpoint()
        if header is None:
            logger.info(
                "RustRawBlockBackend: no valid on-device metadata checkpoint found"
            )
            return
        assert payload is not None
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning("RustRawBlockBackend: failed to decode metadata payload")
            return
        applied = self._apply_loaded_state(data)
        if not applied:
            logger.warning("RustRawBlockBackend: metadata payload rejected by checks")
            return
        self._meta_seq = int(header["seq"])
        logger.info(
            "RustRawBlockBackend: loaded on-device metadata checkpoint "
            "(entries=%d, next_slot=%d, seq=%d)",
            len(self._index),
            self._next_slot,
            self._meta_seq,
        )
