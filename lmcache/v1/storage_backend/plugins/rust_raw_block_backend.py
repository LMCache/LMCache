# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence
import asyncio
import json
import os
import threading

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


def _round_up(x: int, align: int) -> int:
    """Round up to nearest multiple of alignment (required for O_DIRECT)."""
    return ((x + align - 1) // align) * align


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
    using a Rust extension for pread/pwrite.

    Features:
    - High-throughput I/O via direct block device access
    - O_DIRECT support to bypass page cache (requires aligned buffers)
    - Manifest persistence for recovery across restarts
    - Efficient buffer operations via Rust extension

    .. warning::
       **This backend currently only supports TP=1 (single GPU) deployments.**

       When using Tensor Parallelism (TP > 1), multiple vLLM workers would
       independently access the same raw block device without coordination,
       leading to metadata conflicts and data corruption.

       TP > 1 support will be added in a future release via LMCache
       Multi-Process (MP) mode integration.

       Current status:
       - TP=1: Fully supported ✓
       - TP > 1: Not yet supported (requires MP mode integration)
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

        # TP > 1 not supported: multiple workers would conflict on device access.
        if self.metadata is not None:
            is_single_worker = self.metadata.world_size == 1

            if not is_single_worker:
                raise ValueError(
                    "RustRawBlockBackend currently only supports TP=1 "
                    "(single GPU) deployments. "
                    f"Current world_size={self.metadata.world_size}. "
                    "TP > 1 support will be added in a future release.\n"
                    "For now, please use TP=1 or choose a different storage backend."
                )

        extra = self.config.extra_config or {}
        self.device_path: str = extra.get("rust_raw_block.device_path", "")
        if not self.device_path:
            raise ValueError("extra_config['rust_raw_block.device_path'] is required")

        self.manifest_path: Optional[str] = extra.get("rust_raw_block.manifest_path")
        self.capacity_bytes: int = int(
            extra.get("rust_raw_block.capacity_bytes", 0)
        )  # 0 = use full device
        self.block_align: int = int(extra.get("rust_raw_block.block_align", 4096))
        self.header_bytes: int = int(extra.get("rust_raw_block.header_bytes", 4096))
        self.use_odirect: bool = bool(extra.get("rust_raw_block.use_odirect", False))

        full_chunk_bytes = int(self.local_cpu_backend.get_full_chunk_size())
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

        self._lock = threading.Lock()
        self._index: dict[
            CacheEngineKey, _Entry
        ] = {}  # key -> entry (successfully written)
        self._pinned: set[CacheEngineKey] = set()  # keys that cannot be evicted
        self._inflight: dict[
            CacheEngineKey, _Inflight
        ] = {}  # keys currently being written
        self._lru: "OrderedDict[CacheEngineKey, None]" = (
            OrderedDict()
        )  # LRU order (oldest first)

        self._next_slot: int = 0  # next slot index to allocate
        self._free_slots: list[int] = []  # reusable slots from evicted chunks
        self._max_slots: int = 0  # computed lazily from device size

        # Manifest persistence: save index to disk periodically and on shutdown.
        # Default interval: every 100 writes. Set to 0 to disable periodic saves.
        self._manifest_write_interval: int = int(
            extra.get("rust_raw_block.manifest_write_interval", 100)
        )
        self._writes_since_manifest_save: int = 0

        # Default manifest path: /tmp/lmcache_raw_block_<device_name>.manifest.json
        if not self.manifest_path:
            device_name = os.path.basename(self.device_path.rstrip("/"))
            self.manifest_path = f"/tmp/lmcache_raw_block_{device_name}.manifest.json"
            logger.info(
                "RustRawBlockBackend: using default manifest_path=%s",
                self.manifest_path,
            )

        # Debug logging (rate-limited).
        # Only emits when LMCache log level is DEBUG.
        self._dbg_first_n: int = int(extra.get("rust_raw_block.debug_first_n", 4) or 0)
        self._dbg_every_n: int = int(
            extra.get("rust_raw_block.debug_every_n", 256) or 0
        )
        self._dbg_put_batches: int = 0
        self._dbg_put_keys: int = 0
        self._dbg_put_bytes: int = 0
        self._dbg_get_calls: int = 0
        self._dbg_get_bytes: int = 0

        # Lazy import so normal LMCache usage doesn't require Rust extension installed
        self._raw = None

        # Track ongoing put tasks to match exists_in_put_tasks semantics.
        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()

        logger.info(
            "RustRawBlockBackend init: device=%s cap=%s slot=%d align=%d header=%d",
            self.device_path,
            self.capacity_bytes,
            self.slot_bytes,
            self.block_align,
            self.header_bytes,
        )
        logger.warning(
            "RustRawBlockBackend: Currently only TP=1 is supported. "
            "TP > 1 support will be added in a future release."
        )

        # Best-effort restore from manifest (if configured).
        if self.manifest_path:
            self._load_manifest(self.manifest_path)

    def _dbg_should_log(self, n: int) -> bool:
        if not logger.isEnabledFor(10):  # logging.DEBUG
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
        """Lazy init: create single-FD device for synchronous read/write operations."""
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
            )
        return self._raw

    def _allocate_slot(self) -> int:
        """Allocate a slot on device.

        Reuses free slots first, then allocates new ones.
        """
        if self.capacity_bytes <= 0:
            self.capacity_bytes = int(self._rawdev().size_bytes())
        if self._max_slots <= 0:
            self._max_slots = self.capacity_bytes // self.slot_bytes
            if self._max_slots <= 0:
                raise RuntimeError("raw block capacity too small for slot size")

        if self._free_slots:
            return self._free_slots.pop() * self.slot_bytes

        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            return slot * self.slot_bytes

        raise RuntimeError("No free slots available; eviction required")

    def _touch(self, key: CacheEngineKey) -> None:
        """Update LRU: move key to most-recently-used position."""
        self._lru.pop(key, None)
        self._lru[key] = None

    def _evict_one(self) -> bool:
        """Evict least recently used chunk that is not pinned or in-flight."""
        for victim in list(self._lru.keys()):
            if victim in self._pinned or victim in self._inflight:
                continue
            entry = self._index.pop(victim, None)
            if entry is None:
                self._lru.pop(victim, None)
                continue
            self._lru.pop(victim, None)
            self._pinned.discard(victim)
            self._free_slots.append(int(entry.offset // self.slot_bytes))
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

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove key from index and reclaim slot for reuse.

        If the key is currently in-flight, defer slot reuse until the
        async write completes to avoid reusing the same offset while the
        write is still running.
        """
        with self._lock:
            existed = key in self._index or key in self._inflight
            entry = self._index.pop(key, None)
            inflight = self._inflight.get(key)
            self._pinned.discard(key)
            self._lru.pop(key, None)
            if entry is not None:
                self._free_slots.append(int(entry.offset // self.slot_bytes))
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
        """Submit batch of put tasks.

        Allocates slots, encodes headers, and submits async writes.
        """
        if logger.isEnabledFor(10):  # DEBUG
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

    async def _submit_write(
        self,
        key: CacheEngineKey,
        offset: int,
        header: bytes,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Execute write: synchronous blocking write wrapped in async thread."""
        try:
            buf = memory_obj.byte_array
            if hasattr(buf, "cast"):
                buf = buf.cast("B")
            payload_len = len(memory_obj.byte_array)
            total_len = payload_len
            if self.use_odirect:
                total_len = _round_up(payload_len, self.block_align)
                if total_len > (self.slot_bytes - self.header_bytes):
                    raise RuntimeError(f"O_DIRECT payload {total_len} > slot capacity")

            # Synchronous blocking write executed in thread pool
            def _do_write():
                try:
                    raw_dev = self._rawdev()
                    # Write header
                    hdr_total = (
                        _round_up(len(header), self.block_align)
                        if self.use_odirect
                        else len(header)
                    )
                    raw_dev.pwrite_from_buffer(offset, header, len(header), hdr_total)
                    # Write payload
                    raw_dev.pwrite_from_buffer(
                        offset + self.header_bytes, buf, payload_len, total_len
                    )
                except Exception as e:
                    logger.error(
                        f"Write failed for key {self._dbg_key_short(key)}: {e}"
                    )
                    raise

            await asyncio.to_thread(_do_write)

            with self._lock:
                inflight = self._inflight.pop(key, None)
                if inflight is not None:
                    if inflight.canceled:
                        self._free_slots.append(int(inflight.offset // self.slot_bytes))
                    else:
                        self._index[key] = _Entry(
                            offset=inflight.offset,
                            size=inflight.meta.size,
                            meta=inflight.meta,
                        )
                        self._touch(key)

            self._maybe_save_manifest()
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(f"on_complete_callback failed for key {key}: {e}")
        finally:
            memory_obj.ref_count_down()
            with self._put_lock:
                self._put_tasks.discard(key)

    def _encode_header(self, key: CacheEngineKey, payload_len: int) -> bytes:
        """Encode header: magic(8) + chunk_hash(8) + payload_len(8) + zero padding."""
        magic = b"LMCBLK01"
        chunk_hash = int(key.chunk_hash) & ((1 << 64) - 1)
        hdr = bytearray(self.header_bytes)
        hdr[0:8] = magic
        hdr[8:16] = chunk_hash.to_bytes(8, "little", signed=False)
        hdr[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
        return bytes(hdr)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking read: lookup key, allocate buffer, read from device."""
        if logger.isEnabledFor(10):
            self._dbg_get_calls += 1
        with self._lock:
            entry = self._index.get(key)
        if entry is None:
            return None

        meta = entry.meta
        assert meta.shape is not None and meta.dtype is not None
        payload_len = int(meta.size)
        total_len = (
            _round_up(payload_len, self.block_align)
            if self.use_odirect
            else payload_len
        )

        if logger.isEnabledFor(10):
            self._dbg_get_bytes += int(payload_len)
            if self._dbg_should_log(self._dbg_get_calls):
                logger.debug(
                    "RustRawBlockBackend GET: %s offset=%d size=%d",
                    self._dbg_key_short(key),
                    int(entry.offset),
                    int(payload_len),
                )

        assert self.local_cpu_backend is not None
        memory_obj = self.local_cpu_backend.allocate(meta.shape, meta.dtype, meta.fmt)
        assert memory_obj is not None
        buf = memory_obj.byte_array
        try:
            buf = buf.cast("B")
        except Exception:
            pass

        try:
            self._rawdev().pread_into(
                entry.offset + self.header_bytes, buf, payload_len, total_len
            )
        except Exception as e:
            logger.error(f"Read failed for key {self._dbg_key_short(key)}: {e}")
            raise
        memory_obj.metadata.cached_positions = meta.cached_positions
        with self._lock:
            self._touch(key)
        return memory_obj

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,  # noqa: ARG002
    ) -> int:
        # Prefix semantics: stop at first miss.
        hit = 0
        with self._lock:
            for k in keys:
                if k not in self._index:
                    break
                if pin:
                    self._pinned.add(k)
                hit += 1
        return hit

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

        # Best-effort persist manifest before closing I/O.
        if self.manifest_path:
            try:
                self._save_manifest(self.manifest_path)
                logger.info(
                    "RustRawBlockBackend: saved manifest to %s (entries=%d)",
                    self.manifest_path,
                    len(self._index),
                )
            except Exception as e:
                logger.warning(f"Failed to save rust_raw_block manifest: {e}")

        # Close rust device fd if opened
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception as e:
                logger.warning(f"Failed to close raw block device: {e}")
            finally:
                self._raw = None

    def _maybe_save_manifest(self) -> None:
        """Save manifest periodically (every N writes) for crash recovery."""
        if self._manifest_write_interval <= 0:
            return
        self._writes_since_manifest_save += 1
        if self._writes_since_manifest_save >= self._manifest_write_interval:
            self._writes_since_manifest_save = 0
            if self.manifest_path:
                try:
                    self._save_manifest(self.manifest_path)
                    logger.debug(
                        "RustRawBlockBackend: manifest saved, entries=%d",
                        len(self._index),
                    )
                except Exception as e:
                    logger.warning(f"Failed to save periodic manifest: {e}")

    def _save_manifest(self, path: str) -> None:
        """Save index to JSON file for crash recovery (atomic write via temp file)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            data = {
                "version": 1,
                "device_path": self.device_path,
                "capacity_bytes": self.capacity_bytes,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "slot_bytes": self.slot_bytes,
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
                            if e.meta.fmt and hasattr(e.meta.fmt, "name")
                            else str(e.meta.fmt)
                            if e.meta.fmt
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
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # Atomic rename

    def _load_manifest(self, path: str) -> None:
        """Load index from manifest file (validates compatibility before restoring)."""
        if not os.path.exists(path):
            logger.info("RustRawBlockBackend: no manifest found at %s", path)
            return
        logger.info("RustRawBlockBackend: loading manifest from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != 1:
            logger.warning("Ignoring incompatible rust_raw_block manifest")
            return
        if data.get("device_path") and data.get("device_path") != self.device_path:
            logger.warning("Manifest device_path mismatch; ignoring manifest")
            return
        if "slot_bytes" in data and int(data["slot_bytes"]) != int(self.slot_bytes):
            logger.warning("Manifest slot_bytes mismatch; ignoring manifest")
            return
        with self._lock:
            self.capacity_bytes = int(data.get("capacity_bytes", self.capacity_bytes))
            self.block_align = int(data.get("block_align", self.block_align))
            self.header_bytes = int(data.get("header_bytes", self.header_bytes))
            self.slot_bytes = int(data.get("slot_bytes", self.slot_bytes))
            self._next_slot = int(data.get("next_slot", 0))
            self._free_slots = [int(x) for x in data.get("free_slots", [])]

            # Restore entries
            self._index.clear()
            self._lru.clear()
            entries = data.get("entries", {})
            if isinstance(entries, dict):
                for k_str, e in entries.items():
                    try:
                        key = CacheEngineKey.from_string(k_str)
                    except Exception:
                        continue
                    if not isinstance(e, dict):
                        continue
                    offset = int(e.get("offset", 0))
                    size = int(e.get("size", 0))
                    shape_list = e.get("shape")
                    fmt_name = e.get("fmt")
                    shape = (
                        torch.Size(list(shape_list)) if shape_list is not None else None
                    )
                    fmt = (
                        MemoryFormat[fmt_name]
                        if isinstance(fmt_name, str)
                        and fmt_name in MemoryFormat.__members__
                        else MemoryFormat.UNDEFINED
                    )
                    # Restore cached_positions if present
                    cached_positions_list = e.get("cached_positions")
                    cached_positions = (
                        torch.tensor(cached_positions_list, dtype=torch.long)
                        if cached_positions_list is not None
                        else None
                    )
                    # Metadata recovery is best-effort.
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

            # Restore LRU order (fallback to insertion order if missing)
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
                for k in self._index.keys():
                    self._lru[k] = None

            logger.info(
                "RustRawBlockBackend: loaded manifest with %d entries, next_slot=%d",
                len(self._index),
                self._next_slot,
            )
