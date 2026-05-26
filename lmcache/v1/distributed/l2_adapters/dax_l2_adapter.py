# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional
import os
import threading

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.storage_backend.dax.core import DaxCore

logger = init_logger(__name__)


def _parse_positive_int(value, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


class DaxL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for the built-in MP Device-DAX L2 adapter."""

    def __init__(
        self,
        *,
        device_path: str,
        max_dax_size_gb: float,
        slot_bytes: int,
        num_store_workers: int = 1,
        num_lookup_workers: int = 1,
        num_load_workers: int = min(4, os.cpu_count() or 1),
    ) -> None:
        """Initialize a validated DAX L2 adapter config.

        Args:
            device_path: Path to the mmap-able DAX device or test file.
            max_dax_size_gb: Number of GiB to map from ``device_path``.
            slot_bytes: Fixed slot size for each stored object.
            num_store_workers: Number of worker threads for store tasks.
            num_lookup_workers: Number of worker threads for lookup tasks.
            num_load_workers: Number of worker threads for load tasks.
        """
        self.device_path = device_path
        self.max_dax_size_gb = max_dax_size_gb
        self.slot_bytes = slot_bytes
        self.num_store_workers = num_store_workers
        self.num_lookup_workers = num_lookup_workers
        self.num_load_workers = num_load_workers

    @classmethod
    def from_dict(cls, d: dict) -> "DaxL2AdapterConfig":
        """Build a DAX L2 adapter config from CLI JSON.

        Args:
            d: Parsed ``--l2-adapter`` JSON object.

        Returns:
            A validated ``DaxL2AdapterConfig`` instance.

        Raises:
            ValueError: If a required field is missing or any numeric field
                is not positive.
        """
        device_path = d.get("device_path")
        if not isinstance(device_path, str) or not device_path.strip():
            raise ValueError("device_path must be a non-empty string")

        max_dax_size_gb = d.get("max_dax_size_gb")
        if not isinstance(max_dax_size_gb, (int, float)) or max_dax_size_gb <= 0:
            raise ValueError("max_dax_size_gb must be a positive number")

        slot_bytes = d.get("slot_bytes")
        if not isinstance(slot_bytes, int) or slot_bytes <= 0:
            raise ValueError("slot_bytes must be a positive integer")

        num_store_workers = _parse_positive_int(
            d.get("num_store_workers", 1),
            "num_store_workers",
        )
        num_lookup_workers = _parse_positive_int(
            d.get("num_lookup_workers", 1),
            "num_lookup_workers",
        )
        num_load_workers = _parse_positive_int(
            d.get("num_load_workers", min(4, os.cpu_count() or 1)),
            "num_load_workers",
        )

        max_dax_size_bytes = int(float(max_dax_size_gb) * 1024**3)
        if max_dax_size_bytes // slot_bytes <= 0:
            raise ValueError("configured DAX arena does not fit even one slot")

        return cls(
            device_path=device_path.strip(),
            max_dax_size_gb=float(max_dax_size_gb),
            slot_bytes=slot_bytes,
            num_store_workers=num_store_workers,
            num_lookup_workers=num_lookup_workers,
            num_load_workers=num_load_workers,
        )

    @classmethod
    def help(cls) -> str:
        """Return CLI help text for the DAX L2 adapter config.

        Returns:
            Human-readable field descriptions used by adapter config parsing.
        """
        return (
            "DAX L2 adapter config fields:\n"
            "- device_path (str): mmap-able dax device or file path (required)\n"
            "- max_dax_size_gb (float): mapped size in GiB (required, >0)\n"
            "- slot_bytes (int): fixed slot size in bytes (required, >0)\n"
            "- num_store_workers (int): store worker threads (optional, default 1)\n"
            "- num_lookup_workers (int): lookup worker threads (optional, default 1)\n"
            "- num_load_workers (int): load worker threads "
            "(optional, default min(4, cpu_count))"
        )


class DaxL2Adapter(L2AdapterInterface):
    """MP L2 adapter that stores fixed-size objects in a DAX mmap arena."""

    def __init__(self, config: DaxL2AdapterConfig) -> None:
        """Initialize the DAX adapter and its worker pools.

        Args:
            config: Validated DAX adapter configuration.

        Raises:
            RuntimeError: If the DAX device cannot be opened or mapped.
            ValueError: If the mapped arena cannot fit at least one slot.
        """
        super().__init__(max_capacity_bytes=int(config.max_dax_size_gb * 1024**3))
        self._config = config
        self._max_dax_size_bytes = int(config.max_dax_size_gb * 1024**3)

        self._core = DaxCore[ObjectKey](
            device_path=config.device_path,
            max_dax_size_bytes=self._max_dax_size_bytes,
            slot_bytes=config.slot_bytes,
        )

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._store_executor: Optional[ThreadPoolExecutor] = None
        self._lookup_executor: Optional[ThreadPoolExecutor] = None
        self._load_executor: Optional[ThreadPoolExecutor] = None

        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        self._lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._inflight_store_tasks = 0
        self._inflight_lookup_tasks = 0
        self._inflight_load_tasks = 0

        try:
            self._store_executor = ThreadPoolExecutor(
                max_workers=config.num_store_workers,
                thread_name_prefix="dax-l2-store",
            )
            self._lookup_executor = ThreadPoolExecutor(
                max_workers=config.num_lookup_workers,
                thread_name_prefix="dax-l2-lookup",
            )
            self._load_executor = ThreadPoolExecutor(
                max_workers=config.num_load_workers,
                thread_name_prefix="dax-l2-load",
            )
        except Exception:
            self.close()
            raise

    def get_store_event_fd(self) -> int:
        """Return the pollable fd signaled when store tasks complete.

        Returns:
            File descriptor owned by this adapter's store notifier.
        """
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the pollable fd signaled when lookup tasks complete.

        Returns:
            File descriptor owned by this adapter's lookup notifier.
        """
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the pollable fd signaled when load tasks complete.

        Returns:
            File descriptor owned by this adapter's load notifier.
        """
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit an asynchronous L1-to-DAX store task.

        Args:
            keys: Object keys to store.
            objects: Caller-owned memory objects containing the payloads.

        Returns:
            Adapter-local task id for the submitted store task.

        Raises:
            ValueError: If ``keys`` and ``objects`` have different lengths.
            RuntimeError: If the adapter is closing or already closed.
        """
        if len(keys) != len(objects):
            raise ValueError(
                "keys and objects must have the same length, "
                f"got {len(keys)} and {len(objects)}"
            )

        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_store_tasks += 1

        assert self._store_executor is not None
        self._store_executor.submit(
            self._execute_store_task,
            task_id,
            list(keys),
            list(objects),
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Drain completed store task results.

        Returns:
            Mapping from task id to store result. Each task appears at
            most once; subsequent calls do not return already drained tasks.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Submit an asynchronous lookup-and-lock task.

        Found keys have their DAX external lock refcount incremented until
        ``submit_unlock`` is called.

        Args:
            keys: Object keys to look up.

        Returns:
            Adapter-local task id for the submitted lookup task.

        Raises:
            RuntimeError: If the adapter is closing or already closed.
        """
        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_lookup_tasks += 1

        assert self._lookup_executor is not None
        self._lookup_executor.submit(self._execute_lookup_task, task_id, list(keys))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed lookup result.

        Args:
            task_id: Adapter-local lookup task id.

        Returns:
            A bitmap with bits set for keys that were found and locked, or
            ``None`` if the task is still pending or was already queried.
        """
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release DAX external locks acquired by lookup tasks.

        Args:
            keys: Keys whose lock refcount should be decremented.
        """
        self._core.unlock_many(keys)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit an asynchronous DAX-to-L1 load task.

        Args:
            keys: Object keys to load.
            objects: Caller-owned destination buffers. Loaded bytes are
                written directly into these ``MemoryObj`` instances.

        Returns:
            Adapter-local task id for the submitted load task.

        Raises:
            ValueError: If ``keys`` and ``objects`` have different lengths.
            RuntimeError: If the adapter is closing or already closed.
        """
        if len(keys) != len(objects):
            raise ValueError(
                "keys and objects must have the same length, "
                f"got {len(keys)} and {len(objects)}"
            )

        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_load_tasks += 1

        assert self._load_executor is not None
        self._load_executor.submit(
            self._execute_load_task,
            task_id,
            list(keys),
            list(objects),
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed load result.

        Args:
            task_id: Adapter-local load task id.

        Returns:
            A bitmap with bits set for keys that were successfully loaded, or
            ``None`` if the task is still pending or was already queried.
        """
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete unlocked keys from the DAX index.

        Externally locked keys are skipped. Slots borrowed by active reads are
        reclaimed after the read finalizes.

        Args:
            keys: Object keys to delete.
        """
        if not keys or self._closed:
            return

        deleted = self._core.delete_many(keys, force=False)
        deleted_keys = [key for key, ok in zip(keys, deleted, strict=True) if ok]
        if deleted_keys:
            self._notify_keys_deleted(
                deleted_keys,
                [self._config.slot_bytes] * len(deleted_keys),
            )

    def get_usage(self) -> AdapterUsage:
        """Return slot-based capacity usage for this adapter.

        Returns:
            Adapter usage where ``total_bytes_used`` is derived from occupied
            DAX slots rather than payload bytes.
        """
        current_usage, _ = self._core.usage()
        base_usage = super().get_usage()
        total_capacity_bytes = self._config.slot_bytes * self._core.max_slots
        total_bytes_used = int(round(current_usage * total_capacity_bytes))
        return AdapterUsage(
            total_bytes_used=total_bytes_used,
            total_capacity_bytes=total_capacity_bytes,
            bytes_by_cache_salt=MappingProxyType(dict(base_usage.bytes_by_cache_salt)),
        )

    def close(self) -> None:
        """Stop worker pools and release DAX resources.

        The call waits for already submitted worker tasks to finish before
        closing the shared DAX core and event notifiers.
        """
        store_executor = None
        lookup_executor = None
        load_executor = None

        with self._lock:
            if self._closed:
                return
            self._closing = True
            store_executor = self._store_executor
            lookup_executor = self._lookup_executor
            load_executor = self._load_executor
            self._store_executor = None
            self._lookup_executor = None
            self._load_executor = None

        if store_executor is not None:
            store_executor.shutdown(wait=True)
        if lookup_executor is not None:
            lookup_executor.shutdown(wait=True)
        if load_executor is not None:
            load_executor.shutdown(wait=True)

        self._core.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

        with self._lock:
            self._closed = True

    def report_status(self) -> dict:
        """Return a health and capacity snapshot for this adapter.

        Returns:
            Dictionary containing health, DAX capacity, slot occupancy,
            lock/borrow counts, in-flight task counts, and restart-recovery
            capability.
        """
        core_status = self._core.report_status()
        with self._lock:
            inflight_store = self._inflight_store_tasks
            inflight_lookup = self._inflight_lookup_tasks
            inflight_load = self._inflight_load_tasks
            closing = self._closing or self._closed

        return {
            **core_status,
            "is_healthy": core_status["is_healthy"] and not self._closed,
            "type": "dax",
            "device_path": self._config.device_path,
            "max_dax_size_bytes": self._max_dax_size_bytes,
            "slot_bytes": self._config.slot_bytes,
            "inflight_store_tasks": inflight_store,
            "inflight_lookup_tasks": inflight_lookup,
            "inflight_load_tasks": inflight_load,
            "closing": closing or bool(core_status["closing"]),
            "supports_restart_recovery": False,
        }

    def _ensure_open_locked(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("DaxL2Adapter is closing")

    def _get_next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _signal_eventfd(self, notifier) -> None:
        try:
            notifier.notify()
        except OSError:
            logger.debug("Skipping eventfd write during adapter shutdown")

    def _execute_store_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        per_key_results: list[bool] = []
        stored_keys: list[ObjectKey] = []

        try:
            for key, obj in zip(keys, objects, strict=True):
                try:
                    ok = self._core.put_many([key], [obj])[0]
                except Exception:
                    logger.exception("DAX L2 store failed for key %s", key)
                    ok = False
                per_key_results.append(ok)
                if ok:
                    stored_keys.append(key)
        finally:
            bytes_transferred = self._config.slot_bytes * len(stored_keys)
            with self._lock:
                self._completed_store_tasks[task_id] = L2StoreResult(
                    all(per_key_results), bytes_transferred
                )
                self._inflight_store_tasks -= 1

            if stored_keys:
                self._notify_keys_stored(
                    stored_keys,
                    [self._config.slot_bytes] * len(stored_keys),
                )
            self._signal_eventfd(self._store_efd)

    def _execute_lookup_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
    ) -> None:
        bitmap = Bitmap(len(keys))
        try:
            results = self._core.exists_many(keys, lock=True)
            for i, found in enumerate(results):
                if found:
                    bitmap.set(i)
        except Exception:
            logger.exception("DAX L2 lookup failed")
        finally:
            with self._lock:
                self._completed_lookup_tasks[task_id] = bitmap
                self._inflight_lookup_tasks -= 1
            self._signal_eventfd(self._lookup_efd)

    def _execute_load_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        bitmap = Bitmap(len(keys))
        loaded_keys: list[ObjectKey] = []

        try:
            results = self._core.load_many_into(keys, objects)
            for i, loaded in enumerate(results):
                if loaded:
                    bitmap.set(i)
                    loaded_keys.append(keys[i])
        except Exception:
            logger.exception("DAX L2 load failed")
        finally:
            with self._lock:
                self._completed_load_tasks[task_id] = bitmap
                self._inflight_load_tasks -= 1

            if loaded_keys:
                self._notify_keys_accessed(loaded_keys)
            self._signal_eventfd(self._load_efd)


register_l2_adapter_type("dax", DaxL2AdapterConfig)


def _create_dax_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    del l1_memory_desc
    return DaxL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("dax", _create_dax_adapter)
