# SPDX-License-Identifier: Apache-2.0
"""DramL2Adapter: in-DRAM L2 adapter backed by a Python dict.

This adapter stores compressed (or serialized) KV data as raw bytes in
DRAM. It implements L2AdapterInterface so it can be used as the inner
adapter inside SerdeL2AdapterWrapper for in-DRAM compression workflows
(e.g., QAT compression to reduce KV cache memory footprint).

Physical storage is a dict[ObjectKey, bytes]. Operations complete
synchronously on the calling thread; event fds are signaled immediately
after each submit so the controller's poll loop processes completions
on its next iteration.

Typical wiring:
    SerdeL2AdapterWrapper(
        inner=DramL2Adapter(max_capacity_bytes=...),
        serde=accel_kv_compress_processor,
        l1_manager=l1_mgr,
    )
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from lmcache.v1.distributed.api import KeyEntry, KeyListPage, MemoryLayoutDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


# Config class


class DramL2AdapterConfig(L2AdapterConfigBase):
    """Config for DramL2Adapter.

    Fields:
        max_size_gb: Maximum DRAM budget for compressed data in GB.
    """

    def __init__(self, max_size_gb: float):
        self.max_size_gb = max_size_gb

    @classmethod
    def from_dict(cls, d: dict) -> "DramL2AdapterConfig":
        max_size_gb = d.get("max_size_gb")
        if not isinstance(max_size_gb, (int, float)) or max_size_gb <= 0:
            raise ValueError("max_size_gb must be a positive number")
        return cls(max_size_gb=max_size_gb)

    @classmethod
    def help(cls) -> str:
        return (
            "DramL2Adapter config fields:\n"
            "- max_size_gb (float): maximum DRAM budget for "
            "compressed data in GB (required, >0)"
        )


# Main class


class DramL2Adapter(L2AdapterInterface):
    """In-DRAM L2 adapter that stores serialized bytes in a dict.

    All operations complete synchronously and signal event fds immediately.
    Uses an OrderedDict for LRU-friendly iteration during eviction.

    Args:
        config: Adapter configuration.
    """

    def __init__(self, config: DramL2AdapterConfig):
        max_capacity_bytes = int(config.max_size_gb * (1024**3))
        super().__init__(max_capacity_bytes=max_capacity_bytes)

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # OrderedDict for LRU access order tracking
        self._data: OrderedDict[ObjectKey, bytes] = OrderedDict()
        self._current_size_bytes: int = 0
        self._lock = threading.Lock()

        # Task ID management
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._task_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Event fd interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        task_id = self._get_next_task_id()
        total_bytes = 0
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []

        with self._lock:
            for key, obj in zip(keys, objects, strict=True):
                obj_size = obj.get_size()

                # Skip if already stored
                if key in self._data:
                    # Move to end (most recently used)
                    self._data.move_to_end(key)
                    continue

                # Skip if over capacity
                if (
                    self._max_capacity_bytes > 0
                    and self._current_size_bytes + obj_size > self._max_capacity_bytes
                ):
                    logger.warning(
                        "DramL2Adapter: capacity exceeded, skipping key %s "
                        "(used=%d, needed=%d, max=%d)",
                        key,
                        self._current_size_bytes,
                        obj_size,
                        self._max_capacity_bytes,
                    )
                    continue

                # Copy bytes from MemoryObj into our dict.
                # byte_array gives a memoryview of exactly get_size() bytes.
                data = bytes(obj.byte_array)

                self._data[key] = data
                self._current_size_bytes += len(data)
                total_bytes += len(data)
                stored_keys.append(key)
                stored_sizes.append(len(data))

        with self._task_lock:
            self._completed_store_tasks[task_id] = L2StoreResult(True, total_bytes)

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)
        self._store_efd.notify()
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._task_lock:
            result = self._completed_store_tasks
            self._completed_store_tasks = {}
        return result

    # ------------------------------------------------------------------
    # Lookup and lock interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: "MemoryLayoutDesc"
    ) -> L2TaskId:
        task_id = self._get_next_task_id()
        bitmap = Bitmap(len(keys))

        with self._lock:
            for i, key in enumerate(keys):
                if key in self._data:
                    bitmap.set(i)
                    # Move to end (accessed)
                    self._data.move_to_end(key)

        with self._task_lock:
            self._completed_lookup_tasks[task_id] = bitmap

        # Notify accessed keys
        accessed = [k for k in keys if k in self._data]
        if accessed:
            self._notify_keys_accessed(accessed)
        self._lookup_efd.notify()
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> "Bitmap | None":
        with self._task_lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        # No locking mechanism needed for dict-based storage
        pass

    # ------------------------------------------------------------------
    # Load interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        task_id = self._get_next_task_id()
        bitmap = Bitmap(len(keys))
        accessed_keys: list[ObjectKey] = []

        with self._lock:
            for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
                if key not in self._data:
                    continue

                data = self._data[key]
                # Copy stored bytes into the provided MemoryObj buffer.
                # byte_array returns a ctypes memoryview with format '<B';
                # cast to 'B' so slice assignment from bytes works.
                dst = obj.byte_array.cast("B")  # type: ignore[attr-defined]
                dst[: len(data)] = data
                # Narrow the MemoryObj's logical size so downstream
                # consumers (e.g. deserializer) see only valid bytes.
                if hasattr(obj, "set_used_size"):
                    obj.set_used_size(len(data))

                bitmap.set(i)
                accessed_keys.append(key)
                # Move to end (accessed)
                self._data.move_to_end(key)

        with self._task_lock:
            self._completed_load_tasks[task_id] = bitmap

        if accessed_keys:
            self._notify_keys_accessed(accessed_keys)
        self._load_efd.notify()
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> "Bitmap | None":
        with self._task_lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Eviction interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []

        with self._lock:
            for key in keys:
                if key not in self._data:
                    continue
                data = self._data.pop(key)
                size = len(data)
                self._current_size_bytes -= size
                deleted_keys.append(key)
                deleted_sizes.append(size)

        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    def list_l2_keys(
        self,
        model_name: str | None = None,
        page_size: int = 500,
        cursor: str | None = None,
    ) -> "KeyListPage":
        # First Party
        from lmcache.v1.distributed.api import KeyListPage

        with self._lock:
            all_keys = list(self._data.keys())

        if model_name is not None:
            all_keys = [k for k in all_keys if k.model_name == model_name]

        # Simple cursor-based pagination using integer offset
        start = int(cursor) if cursor else 0
        page = all_keys[start : start + page_size]
        next_cursor = (
            str(start + page_size) if start + page_size < len(all_keys) else None
        )

        entries = tuple(
            KeyEntry(
                key=k.to_encoded_object_key(), size_bytes=len(self._data.get(k, b""))
            )
            for k in page
        )
        return KeyListPage(entries=entries, next_page_token=next_cursor)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._data.clear()
            self._current_size_bytes = 0
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    def report_status(self) -> dict:
        with self._lock:
            return {
                "is_healthy": True,
                "type": "DramL2Adapter",
                "stored_object_count": len(self._data),
                "current_size_bytes": self._current_size_bytes,
                "max_capacity_bytes": self._max_capacity_bytes,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_next_task_id(self) -> L2TaskId:
        with self._task_lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        return task_id


# Registration

register_l2_adapter_type("dram", DramL2AdapterConfig)


def _create_dram_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc=None,
) -> L2AdapterInterface:
    """Create a DramL2Adapter from config."""
    return DramL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("dram", _create_dram_l2_adapter)
