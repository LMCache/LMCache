# SPDX-License-Identifier: Apache-2.0
"""MP storage-manager bridge backed by the C++ tiered cache."""

# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal
import threading
import time

# Third Party
from lmcache_mp_cpp.bindings import TieredCache
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey, PrefetchHandle
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

_KEY_SEP = "@"


def object_key_to_string(key: ObjectKey) -> str:
    base = (
        f"{key.model_name}{_KEY_SEP}{key.kv_rank:08x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}"
    return base


@dataclass
class _PrefetchedObject:
    obj: MemoryObj
    refs: int


class CxxTieredStorageManager:
    """Subset of the Python StorageManager API used by MPCacheEngine.

    The object bytes live in the C++ TieredCache. Python MemoryObj instances
    are transient staging buffers used for HBM<->DRAM copies.
    """

    def __init__(
        self,
        dram_capacity_bytes: int,
        disk_path: str | Path,
    ) -> None:
        self._cache = TieredCache(dram_capacity_bytes, disk_path)
        self._lock = threading.Lock()
        self._pending_writes: dict[ObjectKey, MemoryObj] = {}
        self._prefetched: dict[ObjectKey, _PrefetchedObject] = {}
        self._next_prefetch_id = 0

    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, MemoryObj]:
        del mode
        result: dict[ObjectKey, MemoryObj] = {}
        with self._lock:
            for key in keys:
                obj = _allocate_cpu_memory_obj(layout_desc)
                self._pending_writes[key] = obj
                result[key] = obj
        return result

    def finish_write(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                obj = self._pending_writes.pop(key, None)
                if obj is None:
                    continue
                self._cache.put(object_key_to_string(key), obj.byte_array)

    def submit_prefetch_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
        external_request_id: str = "",
    ) -> PrefetchHandle:
        found_count = 0
        with self._lock:
            prefetch_id = self._next_prefetch_id
            self._next_prefetch_id += 1
            refs = 1 + max(extra_count, 0)
            for key in keys:
                if not self._cache.exists(object_key_to_string(key)):
                    break
                obj = _allocate_cpu_memory_obj(layout_desc)
                loaded = self._cache.get_into(object_key_to_string(key), obj.byte_array)
                if not loaded:
                    break
                self._prefetched[key] = _PrefetchedObject(obj=obj, refs=refs)
                found_count += 1
        return PrefetchHandle(
            prefetch_request_id=prefetch_id,
            external_request_id=external_request_id,
            l1_prefix_hit_count=found_count,
            total_requested_keys=len(keys),
            submit_time=time.monotonic(),
        )

    def query_prefetch_lookup_hits(self, handle: PrefetchHandle) -> int | None:
        return handle.l1_prefix_hit_count

    def query_prefetch_status(self, handle: PrefetchHandle) -> int | None:
        return handle.l1_prefix_hit_count

    @contextmanager
    def read_prefetched_results(
        self,
        keys: list[ObjectKey],
    ) -> Iterator[list[MemoryObj] | None]:
        with self._lock:
            entries = [self._prefetched.get(key) for key in keys]
            if any(entry is None for entry in entries):
                objs = None
            else:
                objs = [entry.obj for entry in entries if entry is not None]
        yield objs

    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> None:
        release_count = 1 + max(extra_count, 0)
        with self._lock:
            for key in keys:
                entry = self._prefetched.get(key)
                if entry is None:
                    continue
                entry.refs -= release_count
                if entry.refs <= 0:
                    self._prefetched.pop(key, None)

    def touch_l1_keys(self, keys: list[ObjectKey]) -> None:
        del keys

    def clear(self, force: bool = False) -> None:
        with self._lock:
            self._pending_writes.clear()
            self._prefetched.clear()
            self._cache.clear(force=force)

    def memcheck(self) -> None:
        return None

    def close(self) -> None:
        self._cache.close()

    def report_status(self) -> dict:
        stats = self._cache.stats()
        return {
            "is_healthy": True,
            "storage_backend": "lmcache_mp_cpp",
            "l1": {
                "backend": "cxx_dram",
                "used_bytes": stats.dram_bytes,
                "entries": stats.dram_entries,
            },
            "l2": {
                "backend": "cxx_disk",
                "entries": stats.disk_entries,
            },
            "cxx_tiered_cache": {
                "dram_bytes": stats.dram_bytes,
                "dram_entries": stats.dram_entries,
                "disk_entries": stats.disk_entries,
                "total_entries": stats.total_entries,
            },
        }


def _allocate_cpu_memory_obj(layout_desc: MemoryLayoutDesc) -> TensorMemoryObj:
    size = _layout_size_bytes(layout_desc)
    raw_data = torch.empty(size, dtype=torch.uint8, device="cpu")
    return TensorMemoryObj(
        raw_data=raw_data,
        metadata=MemoryObjMetadata(
            shape=layout_desc.shapes[0],
            dtype=layout_desc.dtypes[0],
            address=0,
            phy_size=size,
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
            shapes=list(layout_desc.shapes),
            dtypes=list(layout_desc.dtypes),
        ),
        parent_allocator=None,
    )


def _layout_size_bytes(layout_desc: MemoryLayoutDesc) -> int:
    total = 0
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        total += shape.numel() * dtype.itemsize
    return total
