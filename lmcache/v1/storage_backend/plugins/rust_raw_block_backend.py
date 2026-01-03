# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence
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


class RustRawBlockBackend(StoragePluginInterface):
    """
    A storage plugin backend that stores KV chunks into a block device (raw)
    using a Rust extension for pread/pwrite.

    Features:
    - High-throughput I/O via direct block device access
    - O_DIRECT support to bypass page cache
    - Manifest persistence for recovery across restarts
    - Zero-copy operations via Rust extension

    Supported Configurations:
    - **TP=1 (Direct Mode)**: Single vLLM worker directly uses this backend
    - **TP > 1 (MP Mode)**: LMCache MP server uses this backend to serve
      multiple vLLM workers

    .. warning::
       **This backend does NOT support TP > 1 in direct connector mode.**

       When using Tensor Parallelism (TP > 1), multiple vLLM workers would
       independently access the same raw block device without coordination,
       leading to metadata conflicts and data corruption.

       For TP > 1 setups, use LMCache Multi-process mode where a single
       MP server manages the raw block device:

       .. code-block:: bash

           # Start MP server with Raw Block
           python3 -m lmcache.v1.multiprocess.server --raw-block-device /dev/nvme0n1

           # Start vLLM with MP connector
           vllm serve model --tensor-parallel-size 4 \\
               --kv-transfer-config '{"kv_connector": ...}'

       See the LMCache documentation on Multi-process Mode for details.
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

        # Check for Tensor Parallelism: Raw Block backend does NOT support TP > 1
        # because multiple workers would independently access the raw block device
        # without metadata synchronization, causing conflicts and data corruption.
        #
        # Allowed configurations:
        # 1. world_size == 1 (single worker, e.g., TP=1)
        # 2. role == "mp_server" (MP server manages storage for all workers)
        if self.metadata is not None:
            is_mp_server = getattr(self.metadata, "role", None) == "mp_server"
            is_single_worker = self.metadata.world_size == 1

            if not is_single_worker and not is_mp_server:
                raise ValueError(
                    "RustRawBlockBackend does not support "
                    f"Tensor Parallelism (TP > 1). "
                    f"Current world_size={self.metadata.world_size}. "
                    "For TP > 1, use LMCache Multi-process mode instead:\n"
                    "  1. Start LMCache server: "
                    "python3 -m lmcache.v1.multiprocess.server "
                    "--raw-block-device /dev/nvme0n1\n"
                    "  2. Configure vLLM with: --kv-transfer-config "
                    '\'{"kv_connector":"LMCacheMPConnector", '
                    '"kv_role":"kv_both"}\'\n'
                    "See: https://docs.lmcache.ai/kv_cache/multiprocess_mode.html"
                )

        extra = self.config.extra_config or {}
        self.device_path: str = extra.get("rust_raw_block.device_path", "")
        if not self.device_path:
            raise ValueError("extra_config['rust_raw_block.device_path'] is required")

        # Optional manifest for persistence across restart (best-effort).
        self.manifest_path: Optional[str] = extra.get("rust_raw_block.manifest_path")

        # Bytes available for this backend to use on the device.
        # If unset, use the whole device size (blockdev --getsize64 equivalent).
        self.capacity_bytes: int = int(extra.get("rust_raw_block.capacity_bytes", 0))

        # Slot sizing: header + payload, aligned up to a block boundary.
        # Default alignment 4096 (safe for NVMe/loop).
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
        self._index: dict[CacheEngineKey, _Entry] = {}
        self._pinned: set[CacheEngineKey] = set()
        self._inflight: dict[CacheEngineKey, _Inflight] = {}
        self._lru: "OrderedDict[CacheEngineKey, None]" = OrderedDict()

        self._next_slot: int = 0
        self._free_slots: list[int] = []
        self._max_slots: int = 0  # computed lazily once device size is known

        # Optional: issue I/O in a separate process (reduces GIL contention).
        self.use_mp_issuer: bool = bool(
            extra.get("rust_raw_block.use_mp_issuer", False)
        )
        self.mp_zero_copy: bool = bool(extra.get("rust_raw_block.mp_zero_copy", True))

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
        # NOTE: we always use the Rust scheduler when not using the mp issuer.
        # The legacy Python PQ executor path was removed to reduce config surface.

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
        self._dbg_prefetch_batches: int = 0
        self._dbg_prefetch_keys: int = 0
        self._dbg_prefetch_bytes: int = 0

        # Lazy import so normal LMCache usage doesn't require Rust extension installed
        self._raw = None
        self._scheduler = None
        self._mp = None

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
        if self._raw is None:
            try:
                # Third Party
                from lmcache_rust_raw_block_io import (  # type: ignore
                    RawBlockDevice,
                    RawBlockDevicePool,
                )
            except Exception as e:
                raise RuntimeError(
                    "Rust raw-block extension is not installed. "
                    "Install / build `rust_raw_block_io` and retry."
                ) from e
            # io_fds is optional; default to per-CPU fds (>=1).
            extra = self.config.extra_config or {}
            io_fds = int(extra.get("rust_raw_block.io_fds", 0) or 0) or (
                os.cpu_count() or 1
            )
            if io_fds <= 1:
                self._raw = RawBlockDevice(
                    self.device_path,
                    writable=True,
                    use_odirect=self.use_odirect,
                    alignment=self.block_align,
                )
            else:
                # Per-CPU submission: choose FD based on sched_getcpu() in Rust.
                self._raw = RawBlockDevicePool(
                    self.device_path,
                    writable=True,
                    num_fds=io_fds,
                    use_odirect=self.use_odirect,
                    alignment=self.block_align,
                )
        return self._raw

    def _get_scheduler(self):
        if self._scheduler is None:
            try:
                # Third Party
                from lmcache_rust_raw_block_io import RawBlockScheduler  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Rust raw-block scheduler extension is not installed."
                ) from e
            # io_fds/io_workers are optional; default to CPU count (>=1).
            extra = self.config.extra_config or {}
            io_fds = int(extra.get("rust_raw_block.io_fds", 0) or 0) or (
                os.cpu_count() or 1
            )
            workers = int(extra.get("rust_raw_block.io_workers", 0) or 0) or (
                os.cpu_count() or 1
            )
            self._scheduler = RawBlockScheduler(
                self.device_path,
                writable=True,
                num_workers=workers,
                num_fds=io_fds,
                use_odirect=self.use_odirect,
                alignment=self.block_align,
            )
        return self._scheduler

    def _get_mp(self):
        if self._mp is None:
            # First Party
            from lmcache.v1.storage_backend.mp_raw_block_io import (
                RawBlockMPClient,  # type: ignore
            )

            # Keep mp configuration minimal: default to spawn and scratch SHM
            # (no explicit SHM pool required).
            extra = self.config.extra_config or {}
            io_fds = int(extra.get("rust_raw_block.io_fds", 0) or 0) or (
                os.cpu_count() or 1
            )
            workers = int(extra.get("rust_raw_block.io_workers", 0) or 0) or (
                os.cpu_count() or 1
            )
            self._mp = RawBlockMPClient(
                self.device_path,
                num_workers=workers,
                num_fds=io_fds,
                use_odirect=self.use_odirect,
                alignment=self.block_align,
                # ctx/shm_pool_* are intentionally not configurable here anymore.
            )
        return self._mp

    def _allocate_slot(self) -> int:
        # Fixed-slot allocator with reuse via free list.
        if self.capacity_bytes <= 0:
            # Probe device size lazily via rust helper if capacity unset.
            self.capacity_bytes = int(self._rawdev().size_bytes())
        if self._max_slots <= 0:
            self._max_slots = self.capacity_bytes // self.slot_bytes
            if self._max_slots <= 0:
                raise RuntimeError("raw block capacity too small for slot size")

        if self._free_slots:
            slot = self._free_slots.pop()
            return slot * self.slot_bytes

        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            return slot * self.slot_bytes

        raise RuntimeError("No free slots available; eviction required")

    def _touch(self, key: CacheEngineKey) -> None:
        # Maintain simple LRU (least recent at beginning).
        self._lru.pop(key, None)
        self._lru[key] = None

    def _evict_one(self) -> bool:
        # Evict LRU victim that is not pinned and not inflight.
        for victim in list(self._lru.keys()):
            if victim in self._pinned:
                continue
            if victim in self._inflight:
                continue
            entry = self._index.pop(victim, None)
            if entry is None:
                self._lru.pop(victim, None)
                continue
            self._lru.pop(victim, None)
            self._pinned.discard(victim)
            slot = entry.offset // self.slot_bytes
            self._free_slots.append(int(slot))
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
        # WIP: does not reclaim space yet; just drops index entry.
        with self._lock:
            existed = key in self._index or key in self._inflight
            entry = self._index.pop(key, None)
            inflight = self._inflight.pop(key, None)
            self._pinned.discard(key)
            self._lru.pop(key, None)
            if entry is not None:
                self._free_slots.append(int(entry.offset // self.slot_bytes))
            if inflight is not None:
                self._free_slots.append(int(inflight.offset // self.slot_bytes))
            return existed

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ):
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

        # Fast path: when using mp issuer, batch the entire put into a single
        # pwritev request to avoid per-key IPC overhead.
        if self.use_mp_issuer:
            batch_keys: list[CacheEngineKey] = []
            batch_offsets: list[int] = []
            batch_headers: list[bytes] = []
            batch_objs: list[MemoryObj] = []

            for key, obj in zip(keys, objs, strict=False):
                # Skip if already storing
                with self._put_lock:
                    if key in self._put_tasks:
                        continue
                    self._put_tasks.add(key)

                with self._lock:
                    if key in self._index or key in self._inflight:
                        with self._put_lock:
                            self._put_tasks.discard(key)
                        continue
                    # Ensure capacity (evict if needed) before allocating a slot
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
                batch_keys.append(key)
                batch_offsets.append(offset)
                batch_headers.append(header)
                batch_objs.append(obj)

            if not batch_keys:
                return None

            if logger.isEnabledFor(10) and self._dbg_should_log(self._dbg_put_batches):
                # This is the actual raw-device write submission point.
                # (2 segments per key: header + payload)
                logger.debug(
                    "RustRawBlockBackend PUT(mp): segs=%d keys=%d",
                    2 * len(batch_keys),
                    len(batch_keys),
                )

            assert self.loop is not None
            fut = asyncio.run_coroutine_threadsafe(
                self._submit_write_batch(
                    keys=batch_keys,
                    offsets=batch_offsets,
                    headers=batch_headers,
                    memory_objs=batch_objs,
                ),
                self.loop,
            )
            return [fut]

        # Default path: per-key async writes.
        futures = []
        for key, obj in zip(keys, objs, strict=False):
            # Skip if already storing
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
                    key=key, offset=offset, header=header, memory_obj=obj
                ),
                self.loop,
            )
            futures.append(fut)
        return futures or None

    async def _submit_write_batch(
        self,
        *,
        keys: Sequence[CacheEngineKey],
        offsets: Sequence[int],
        headers: Sequence[bytes],
        memory_objs: Sequence[MemoryObj],
    ) -> None:
        """
        Batch write for mp issuer: one IPC request + one Rust future.

        IMPORTANT: ref_count_down happens on the event loop thread only.
        """
        mp = self._get_mp()
        segs = []
        try:
            for key, offset, header, memory_obj in zip(
                keys, offsets, headers, memory_objs, strict=False
            ):
                # Header segment
                hdr_total = len(header)
                if self.use_odirect:
                    hdr_total = _round_up(hdr_total, self.block_align)
                segs.append((offset, header, 0, len(header), hdr_total))

                # Payload segment
                buf = memory_obj.byte_array
                if hasattr(buf, "cast"):
                    buf = buf.cast("B")
                payload_len = len(memory_obj.byte_array)
                total_len = payload_len
                if self.use_odirect:
                    total_len = _round_up(payload_len, self.block_align)
                    max_payload = self.slot_bytes - self.header_bytes
                    if total_len > max_payload:
                        raise RuntimeError(
                            f"O_DIRECT payload {total_len} > slot {max_payload}"
                        )

                if (
                    self.mp_zero_copy
                    and hasattr(memory_obj, "shm_view")
                    and hasattr(memory_obj, "shm_off")
                ):
                    try:
                        shm_view = memory_obj.shm_view()  # type: ignore[attr-defined]
                        shm_off = int(memory_obj.shm_off)  # type: ignore[attr-defined]
                        segs.append(
                            (
                                offset + self.header_bytes,
                                shm_view,
                                shm_off,
                                payload_len,
                                total_len,
                            )
                        )
                    except Exception:
                        segs.append(
                            (offset + self.header_bytes, buf, 0, payload_len, total_len)
                        )
                else:
                    segs.append(
                        (offset + self.header_bytes, buf, 0, payload_len, total_len)
                    )

            f = mp.submit_pwritev(segs, priority=2)
            await asyncio.wrap_future(f)

            # Commit inflight -> index
            with self._lock:
                for key in keys:
                    inflight = self._inflight.pop(key, None)
                    if inflight is not None:
                        self._index[key] = _Entry(
                            offset=inflight.offset,
                            size=inflight.meta.size,
                            meta=inflight.meta,
                        )
                        self._touch(key)

            # Periodic manifest save
            self._maybe_save_manifest()
        except Exception:
            # On error: roll back inflight slots so allocator can reuse them.
            with self._lock:
                for key in keys:
                    inflight = self._inflight.pop(key, None)
                    if inflight is not None:
                        self._free_slots.append(int(inflight.offset // self.slot_bytes))
            raise
        finally:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                memory_obj.ref_count_down()
                with self._put_lock:
                    self._put_tasks.discard(key)

    async def _submit_write(
        self, key: CacheEngineKey, offset: int, header: bytes, memory_obj: MemoryObj
    ) -> None:
        try:
            buf = memory_obj.byte_array
            if hasattr(buf, "cast"):
                buf = buf.cast("B")
            payload_len = len(memory_obj.byte_array)
            total_len = payload_len
            if self.use_odirect:
                total_len = _round_up(payload_len, self.block_align)
                max_payload = self.slot_bytes - self.header_bytes
                if total_len > max_payload:
                    raise RuntimeError(
                        f"O_DIRECT payload {total_len} > slot {max_payload}"
                    )

            if self.use_mp_issuer:
                mp = self._get_mp()
                hdr_total = len(header)
                if self.use_odirect:
                    hdr_total = _round_up(len(header), self.block_align)

                # Header write (copy is fine; small)
                f0 = mp.submit_pwrite(offset, header, total_len=hdr_total, priority=2)

                # Payload write: try SHM zero-copy if MemoryObj is SHM-backed.
                f1 = None
                if (
                    self.mp_zero_copy
                    and hasattr(memory_obj, "shm_view")
                    and hasattr(memory_obj, "shm_off")
                ):
                    try:
                        shm_view = memory_obj.shm_view()  # type: ignore[attr-defined]
                        shm_off = int(memory_obj.shm_off)  # type: ignore[attr-defined]
                        f1 = mp.submit_pwrite_shm(
                            offset + self.header_bytes,
                            shm_view,
                            shm_off=shm_off,
                            payload_len=payload_len,
                            total_len=total_len,
                            priority=2,
                        )
                    except Exception:
                        f1 = None
                if f1 is None:
                    f1 = mp.submit_pwrite(
                        offset + self.header_bytes,
                        buf,
                        total_len=total_len,
                        priority=2,
                    )

                await asyncio.wrap_future(f0)
                await asyncio.wrap_future(f1)
            else:
                sched = self._get_scheduler()
                # priority 2 = put (lowest)
                hdr_total = len(header)
                if self.use_odirect:
                    hdr_total = _round_up(len(header), self.block_align)
                fb = sched.submit_pwritev_from_buffers(
                    [
                        (offset, header, len(header), hdr_total),
                        (offset + self.header_bytes, buf, payload_len, total_len),
                    ],
                    priority=2,
                )
                await asyncio.wrap_future(fb)

            # Commit inflight -> index on success
            with self._lock:
                inflight = self._inflight.pop(key, None)
                if inflight is not None:
                    self._index[key] = _Entry(
                        offset=inflight.offset,
                        size=inflight.meta.size,
                        meta=inflight.meta,
                    )
                    self._touch(key)

            # Periodic manifest save
            self._maybe_save_manifest()
        finally:
            # IMPORTANT: ref_count_down on the event loop thread only.
            memory_obj.ref_count_down()
            with self._put_lock:
                self._put_tasks.discard(key)

    def _encode_header(self, key: CacheEngineKey, payload_len: int) -> bytes:
        # Header is block-aligned fixed-size region.
        # Layout (little endian):
        # 0..8   magic
        # 8..16  chunk_hash (u64)
        # 16..24 payload_len (u64)
        # rest   zero
        magic = b"LMCBLK01"
        chunk_hash = int(key.chunk_hash) & ((1 << 64) - 1)
        hdr = bytearray(self.header_bytes)
        hdr[0:8] = magic
        hdr[8:16] = chunk_hash.to_bytes(8, "little", signed=False)
        hdr[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
        return bytes(hdr)

    def _decode_header(self, header: bytes) -> tuple[int, int]:
        # NOTE: header decoding is intentionally unused on the hot path.
        # We rely on the in-memory index/manifest for payload size.
        if len(header) < 24:
            raise RuntimeError("short header")
        if header[0:8] != b"LMCBLK01":
            raise RuntimeError("bad magic")
        chunk_hash = int.from_bytes(header[8:16], "little", signed=False)
        payload_len = int.from_bytes(header[16:24], "little", signed=False)
        return chunk_hash, payload_len

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if logger.isEnabledFor(10):
            self._dbg_get_calls += 1
        with self._lock:
            entry = self._index.get(key)
        if entry is None:
            return None

        meta = entry.meta
        assert meta.shape is not None
        assert meta.dtype is not None
        payload_len = int(meta.size)
        total_len = int(payload_len)
        if self.use_odirect:
            total_len = _round_up(payload_len, self.block_align)

        if logger.isEnabledFor(10):
            self._dbg_get_bytes += int(payload_len)
            if self._dbg_should_log(self._dbg_get_calls):
                logger.debug(
                    "RustRawBlockBackend GET: %s offset=%d size=%d",
                    self._dbg_key_short(key),
                    int(entry.offset),
                    int(payload_len),
                )

        # Default: allocate destination MemoryObj from LocalCPUBackend
        assert self.local_cpu_backend is not None
        memory_obj = self.local_cpu_backend.allocate(meta.shape, meta.dtype, meta.fmt)
        assert memory_obj is not None
        buf = memory_obj.byte_array
        try:
            buf = buf.cast("B")
        except Exception:
            pass
        if self.use_mp_issuer:
            mp = self._get_mp()
            # If the destination is SHM-backed, do process-to-process zero-copy.
            if (
                self.mp_zero_copy
                and hasattr(memory_obj, "shm_view")
                and hasattr(memory_obj, "shm_off")
            ):
                shm_view = memory_obj.shm_view()  # type: ignore[attr-defined]
                shm_off = int(memory_obj.shm_off)  # type: ignore[attr-defined]
                f = mp.submit_pread_shm(
                    entry.offset + self.header_bytes,
                    shm_view,
                    shm_off=shm_off,
                    payload_len=payload_len,
                    total_len=total_len,
                    priority=0,
                )
            else:
                f = mp.submit_pread_into(
                    entry.offset + self.header_bytes,
                    buf,
                    payload_len=payload_len,
                    total_len=total_len,
                    priority=0,
                )
            f.result(timeout=60)
        else:
            raw = self._rawdev()
            raw.pread_into(
                entry.offset + self.header_bytes, buf, payload_len, total_len
            )

        memory_obj.metadata.cached_positions = meta.cached_positions
        with self._lock:
            self._touch(key)
        return memory_obj

    async def batched_async_contains(
        self, lookup_id: str, keys: list[CacheEngineKey], pin: bool = False
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

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        # Allocate upfront (like LocalDiskBackend) then read in background.
        entries: list[_Entry] = []
        mem_objs: list[MemoryObj] = []
        with self._lock:
            for k in keys:
                e = self._index.get(k)
                if e is None:
                    break
                entries.append(e)
                meta = e.meta
                assert meta.shape is not None
                assert meta.dtype is not None
                assert self.local_cpu_backend is not None
                m = self.local_cpu_backend.allocate(meta.shape, meta.dtype, meta.fmt)
                assert m is not None
                mem_objs.append(m)

        if logger.isEnabledFor(10):
            self._dbg_prefetch_batches += 1
            self._dbg_prefetch_keys += int(len(entries))
            try:
                self._dbg_prefetch_bytes += int(sum(int(e.meta.size) for e in entries))
            except Exception:
                pass
            if self._dbg_should_log(self._dbg_prefetch_batches):
                logger.debug(
                    "RustRawBlockBackend PREFETCH: req=%d hit=%d",
                    len(keys),
                    len(entries),
                )

        async def _prefetch_async():
            if self.use_mp_issuer:
                mp = self._get_mp()
                reqs = []
                for k, e, m in zip(keys, entries, mem_objs, strict=False):
                    payload_len = int(e.meta.size)
                    total_len = payload_len
                    if self.use_odirect:
                        total_len = _round_up(payload_len, self.block_align)
                    # Prefer SHM zero-copy when destination is SHM-backed.
                    if hasattr(m, "shm_view") and hasattr(m, "shm_off"):
                        shm_view = m.shm_view()  # type: ignore[attr-defined]
                        shm_off = int(m.shm_off)  # type: ignore[attr-defined]
                        reqs.append(
                            (
                                e.offset + self.header_bytes,
                                shm_view,
                                shm_off,
                                payload_len,
                                total_len,
                            )
                        )
                    else:
                        buf = m.byte_array
                        try:
                            buf = buf.cast("B")
                        except Exception:
                            pass
                        reqs.append(
                            (
                                e.offset + self.header_bytes,
                                buf,
                                0,
                                payload_len,
                                total_len,
                            )
                        )
                    m.metadata.cached_positions = e.meta.cached_positions
                fb = mp.submit_preadv_into(reqs, priority=0)
                await asyncio.wrap_future(fb)
                return mem_objs

            sched = self._get_scheduler()
            reqs = []
            for _k, e, m in zip(keys, entries, mem_objs, strict=False):
                payload_len = int(e.meta.size)
                buf = m.byte_array
                try:
                    buf = buf.cast("B")
                except Exception:
                    pass
                total_len = payload_len
                if self.use_odirect:
                    total_len = _round_up(payload_len, self.block_align)
                reqs.append((e.offset + self.header_bytes, buf, payload_len, total_len))
                m.metadata.cached_positions = e.meta.cached_positions
            fb = sched.submit_preadv_into(reqs, priority=0)
            await asyncio.wrap_future(fb)
            return mem_objs

        # prefetch = highest priority (0)
        return await _prefetch_async()

    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        assert self.local_cpu_backend is not None
        return self.local_cpu_backend

    def close(self) -> None:
        if logger.isEnabledFor(10):
            logger.debug(
                "RustRawBlockBackend stats: put=%d/%d/%d get=%d/%d prefetch=%d/%d/%d",
                self._dbg_put_batches,
                self._dbg_put_keys,
                self._dbg_put_bytes,
                self._dbg_get_calls,
                self._dbg_get_bytes,
                self._dbg_prefetch_batches,
                self._dbg_prefetch_keys,
                self._dbg_prefetch_bytes,
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

        if self._scheduler is not None:
            try:
                self._scheduler.close()
            except Exception:
                pass
            self._scheduler = None

        if self._mp is not None:
            try:
                self._mp.close()
            except Exception:
                pass
            self._mp = None

        # Close rust device fd if opened
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception:
                pass
            self._raw = None

    def _maybe_save_manifest(self) -> None:
        """Periodically save manifest based on write interval."""
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
        os.replace(tmp, path)

    def _load_manifest(self, path: str) -> None:
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
        # Restore core allocator state
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
