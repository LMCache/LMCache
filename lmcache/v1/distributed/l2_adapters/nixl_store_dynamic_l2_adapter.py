# SPDX-License-Identifier: Apache-2.0
"""
Dynamic-file-mode Nixl L2 adapter.

Unlike the static ``NixlStoreL2Adapter`` which pre-allocates all storage
files at init time, this adapter opens/registers files per operation.

Atomic publish:
- Stores DMA-write to a per-operation ``<final_path>.tmp.<uuid>`` and
  atomically ``rename()`` to the final deterministic path on completion.
  This guarantees that readers (including other processes sharing the
  same directory) never observe a partially-written file.

Persist (enabled by default via ``persist_enabled``, can be opted out):
- Keeps data files on disk at shutdown (no metadata dump).

Secondary lookup (always on):
- Lookup always checks secondary storage (disk) on miss and lazily
  populates the in-memory index when a file is found. File names are
  derived deterministically from ObjectKey.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
import asyncio
import threading

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.dynamic_nixl_store_agent import (  # noqa: E501
    DynamicNixlStorageAgent,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.file_dynamic_nixl_store_agent import (  # noqa: E501
    FILE_DYNAMIC_BACKENDS,
    FileDynamicNixlStorageAgent,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_l2_adapter import (
    NixlStoreObj,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


def _create_dynamic_nixl_storage_agent(
    device: str,
    backend: str,
    backend_params: dict[str, str],
    l1_memory_desc: L1MemoryDesc,
) -> DynamicNixlStorageAgent:
    """Create the dynamic storage agent registered for ``backend``.

    Args:
        device: Device that owns the registered L1 memory.
        backend: NIXL storage backend name.
        backend_params: Backend-specific NIXL parameters.
        l1_memory_desc: L1 memory region shared with the agent.

    Returns:
        The concrete storage agent for the requested backend.

    Raises:
        ValueError: If the dynamic adapter has no agent for ``backend``.
    """
    if backend in FILE_DYNAMIC_BACKENDS:
        return FileDynamicNixlStorageAgent(
            device, backend, backend_params, l1_memory_desc
        )
    raise ValueError(f"No dynamic NIXL storage agent for backend {backend!r}")


# ---------------------------------------------------------------
# Dynamic L2 adapter
# ---------------------------------------------------------------


class DynamicNixlStoreL2Adapter(L2AdapterInterface):
    """Nixl L2 adapter using dynamic per-operation file registration.

    Each store creates a new file on disk; each load re-opens the file.

    When ``persist_enabled`` is True (the default), data files are kept
    on disk at shutdown.  Lookup always checks secondary storage (disk)
    for keys not in the in-memory index and populates the index lazily.
    """

    def __init__(
        self,
        config: DynamicNixlStoreL2AdapterConfig,
        l1_memory_desc: L1MemoryDesc,
    ):
        max_capacity_gb = float(config.backend_params.get("max_capacity_gb", 0))
        if max_capacity_gb <= 0:
            raise ValueError("backend_params must include a positive 'max_capacity_gb'")
        super().__init__(max_capacity_bytes=int(max_capacity_gb * (1024**3)))

        # Initialize the storage agent before allocating event notifiers or
        # starting the event-loop thread. Backend validation, storage setup,
        # or NIXL registration may fail during construction; failing here
        # avoids leaking adapter-owned file descriptors or a background thread.
        self.nixl_agent = _create_dynamic_nixl_storage_agent(
            device="cpu",
            backend=config.backend,
            backend_params=config.backend_params,
            l1_memory_desc=l1_memory_desc,
        )

        self._config = config

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Cache data structures
        self._memory_objects: dict[ObjectKey, NixlStoreObj] = {}
        self._inflight_stores: set[ObjectKey] = set()
        self._total_bytes: int = 0

        # Task ID management
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        # Asyncio event loop running in a background thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        self._persist_enabled = config.persist_config.persist_enabled

    # --------------------
    # Event Fd Interface
    # --------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    #####################
    # Store Interface
    #####################

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_store_in_the_loop(keys, objects, task_id), self._loop
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    #####################
    # Lookup and Lock Interface
    #####################

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        self._loop.call_soon_threadsafe(self._execute_lookup_in_the_loop, keys, task_id)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        def _unlock_keys(keys: list[ObjectKey]) -> None:
            for key in keys:
                if (obj := self._memory_objects.get(key)) is not None:
                    obj.decrease_pin_count()

        self._loop.call_soon_threadsafe(_unlock_keys, keys)

    #####################
    # Load Interface
    #####################

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_load_in_loop(keys, objects, task_id), self._loop
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    #####################
    # Eviction Interface
    #####################

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete objects from storage, removing their files from disk."""
        to_delete: list[tuple[ObjectKey, int]] = []
        with self._lock:
            for key in keys:
                obj = self._memory_objects.get(key)
                if obj is None:
                    continue
                if obj.pin_count > 0:
                    logger.debug(
                        "Skipping eviction of pinned key %s (pin_count=%d)",
                        key,
                        obj.pin_count,
                    )
                    continue
                self._total_bytes -= obj.size
                del self._memory_objects[key]
                to_delete.append((key, obj.size))
        # Filesystem I/O outside the lock to avoid blocking concurrent
        # store/lookup/load operations.
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for key, size in to_delete:
            self.nixl_agent.dynamic_delete(key)
            deleted_keys.append(key)
            deleted_sizes.append(size)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage`` is inherited from L2AdapterInterface; byte accounting
    # is driven by ``_notify_keys_*`` through the base class now.

    #####################
    # Status Interface
    #####################

    def report_status(self) -> dict:
        with self._lock:
            stored_object_count = len(self._memory_objects)
            pinned_object_count = sum(
                1 for obj in self._memory_objects.values() if obj.pin_count > 0
            )
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "DynamicNixlStoreL2Adapter",
            "backend": self._config.backend,
            "stored_object_count": stored_object_count,
            "pinned_object_count": pinned_object_count,
            "event_loop_alive": self._loop_thread.is_alive(),
        }

    #####################
    # Cleanup Interface
    #####################

    def close(self):
        # Stop the event loop and wait for all in-flight tasks to finish
        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop)
            future.result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()
        self._loop.close()

        # If persist is enabled, keep data files on disk; otherwise clean up.
        if self._persist_enabled:
            logger.info("persist_enabled=True, keeping data files on disk")
        else:
            logger.info("persist_enabled=False, deleting all data files")
            with self._lock:
                for key in list(self._memory_objects.keys()):
                    self.nixl_agent.dynamic_delete(key)

        # Best-effort cleanup of orphaned temp files from crashed stores.
        self.nixl_agent.cleanup()

        self.nixl_agent.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    ##################
    # Helper functions
    ##################

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _signal_store_event(self) -> None:
        self._store_efd.notify()

    def _signal_lookup_event(self) -> None:
        self._lookup_efd.notify()

    def _signal_load_event(self) -> None:
        self._load_efd.notify()

    async def _execute_store_in_the_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Store each key-object pair to its own file via concurrent DMA writes.

        All eligible keys in the batch are reserved under the lock first, then
        their ``dynamic_store_file`` coroutines are launched concurrently via
        ``asyncio.gather`` so the underlying io_uring ring can pipeline multiple
        SQE submissions in a single pass -- the same pattern used by the load
        path (``_execute_load_in_loop``).

        Per-key failure isolation: ``return_exceptions=True`` means one file's
        failure does not abort the rest of the batch; each key is committed or
        un-reserved independently based on its own coroutine result.
        """
        success = True
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        try:
            # ── Phase 1: reserve capacity and build the coroutine list ────────
            # Each entry in ``prepared`` carries everything needed to either
            # commit or roll back after the gather completes.
            coros = []
            prepared: list[tuple[ObjectKey, int, MemoryObj]] = []

            for key, obj in zip(keys, objects, strict=False):
                mem_size = obj.meta.phy_size

                with self._lock:
                    if key in self._memory_objects or key in self._inflight_stores:
                        continue
                    if self._total_bytes + mem_size > self._max_capacity_bytes:
                        logger.warning(
                            "Storage capacity exceeded, skipping store for key %s",
                            key,
                        )
                        success = False
                        break
                    self._inflight_stores.add(key)
                    self._total_bytes += mem_size

                mem_indices = self.nixl_agent.get_memory_indices(
                    obj.meta.address, mem_size
                )
                coros.append(self.nixl_agent.dynamic_store(mem_indices, key))
                prepared.append((key, mem_size, obj))

            # ── Phase 2: run all DMA writes concurrently ──────────────────────
            if coros:
                # return_exceptions=True: one file's failure does not abort the
                # rest of the batch -- commit each key independently below.
                results = await asyncio.gather(*coros, return_exceptions=True)

                for (key, mem_size, obj), result in zip(prepared, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.error(
                            "Dynamic NIXL store failed for key %s: %r", key, result
                        )
                        # Un-reserve so capacity accounting stays correct.
                        with self._lock:
                            self._inflight_stores.discard(key)
                            self._total_bytes -= mem_size
                        success = False
                    else:
                        store_obj = NixlStoreObj(
                            page_indices=[],  # not used in dynamic mode
                            size=mem_size,
                            layout=MemoryLayoutDesc(
                                [obj.meta.shape],
                                [obj.meta.dtype],
                            ),
                            pin_count=1,
                        )
                        with self._lock:
                            self._inflight_stores.discard(key)
                            self._memory_objects[key] = store_obj
                            store_obj.decrease_pin_count()
                        stored_keys.append(key)
                        stored_sizes.append(mem_size)

        except Exception:
            logger.exception("Dynamic NIXL store task %d failed", task_id)
            success = False

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)

        bytes_transferred = sum(stored_sizes)
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )
        self._signal_store_event()

    def _execute_lookup_in_the_loop(
        self, keys: list[ObjectKey], task_id: L2TaskId
    ) -> None:
        """Look up keys and pin found objects.

        Also checks secondary storage (disk) for keys not in the
        in-memory index and lazily populates ``_memory_objects`` for any
        data files found on disk.
        """
        bitmap = Bitmap(len(keys))
        recovered_keys: list[ObjectKey] = []
        recovered_sizes: list[int] = []
        missing: list[tuple[int, ObjectKey]] = []

        with self._lock:
            for i, key in enumerate(keys):
                obj = self._memory_objects.get(key)
                if obj is not None:
                    bitmap.set(i)
                    obj.increase_pin_count()
                else:
                    missing.append((i, key))

        # Filesystem I/O outside the lock to avoid blocking concurrent
        # store/lookup/load operations.
        stats: dict[tuple[int, ObjectKey], int] = {}
        for i, key in missing:
            obj_size = self.nixl_agent.get_stored_size(key)
            if obj_size is not None:
                stats[(i, key)] = obj_size

        with self._lock:
            for (i, key), obj_size in stats.items():
                obj = self._memory_objects.get(key)
                if obj is not None:
                    bitmap.set(i)
                    obj.increase_pin_count()
                    continue
                if key in self._inflight_stores:
                    continue
                if self._total_bytes + obj_size > self._max_capacity_bytes:
                    logger.debug(
                        "Secondary lookup hit for %s but capacity exceeded, skipping",
                        key,
                    )
                    continue
                obj = NixlStoreObj(
                    page_indices=[],
                    size=obj_size,
                    layout=None,
                )
                self._memory_objects[key] = obj
                self._total_bytes += obj_size
                recovered_keys.append(key)
                recovered_sizes.append(obj_size)
                bitmap.set(i)
                obj.increase_pin_count()
            self._completed_lookup_tasks[task_id] = bitmap

        if recovered_keys:
            self._notify_keys_stored(recovered_keys, recovered_sizes)
        self._signal_lookup_event()

    async def _execute_load_in_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Execute a queued load task for ``task_id``.

        For each requested key present in this adapter, read its stored
        data into the caller-provided ``objects[i]`` and set bit ``i`` of
        the result bitmap; keys that are missing or fail to load are left
        unset. The bitmap is recorded under ``task_id`` (retrieve via
        ``query_load_result``) and the load event fd is signaled.
        """
        bitmap = Bitmap(len(keys))
        accessed_keys: list[ObjectKey] = []
        try:
            # Read every present key's file concurrently (asyncio.gather
            # below) so a multi-chunk request does not pay per-file NIXL
            # latency serially -- the dominant cost of a single-request load.
            #
            # TODO(perf): batch a request's files into one NIXL
            # ``register_memory`` (a single transfer + single deregister),
            # like the static adapter's pre-registered flat-index pool,
            # instead of the current per-file register/transfer/deregister.
            # Trade-off: one large per-request registration list vs. many
            # small ones.
            coros = []
            found_positions: list[int] = []
            for i, key in enumerate(keys):
                with self._lock:
                    storage_obj = self._memory_objects.get(key)
                if storage_obj is None:
                    continue

                mem_addr = objects[i].meta.address
                mem_size = objects[i].meta.phy_size
                mem_indices = self.nixl_agent.get_memory_indices(mem_addr, mem_size)
                coros.append(self.nixl_agent.dynamic_load(mem_indices, key))
                found_positions.append(i)

            if coros:
                # return_exceptions=True so one chunk's failure doesn't
                # discard the chunks that loaded successfully: mark each
                # key by its own result rather than all-or-nothing.
                results = await asyncio.gather(*coros, return_exceptions=True)
                for pos, result in zip(found_positions, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.error(
                            "Dynamic NIXL load failed for key %s: %r",
                            keys[pos],
                            result,
                        )
                        continue
                    bitmap.set(pos)
                    accessed_keys.append(keys[pos])

        except Exception:
            logger.exception("Dynamic NIXL load task %d failed", task_id)

        if accessed_keys:
            self._notify_keys_accessed(accessed_keys)
        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()


# ---------------------------------------------------------------------
# Config and self-registration
# ---------------------------------------------------------------------

# TODO(Jiayi): OBJ backend is not supported in the dynamic adapter yet.
# Only file-based backends are supported.
_VALID_DYNAMIC_BACKENDS = FILE_DYNAMIC_BACKENDS


class DynamicNixlStoreL2AdapterConfig(L2AdapterConfigBase):
    """Config for the dynamic-file Nixl L2 adapter.

    Fields:
    - backend: Nixl storage backend (GDS, GDS_MT, POSIX, HF3FS).
    - backend_params: Backend-specific parameters as a dict of string
      key-value pairs. Must include ``file_path`` and ``use_direct_io``.
    """

    def __init__(
        self,
        backend: str,
        backend_params: dict[str, str],
    ):
        if backend not in _VALID_DYNAMIC_BACKENDS:
            raise ValueError(
                "backend must be one of %s, got %r" % (_VALID_DYNAMIC_BACKENDS, backend)
            )
        self.backend = backend
        self.backend_params = backend_params

    @classmethod
    def from_dict(cls, d: dict) -> DynamicNixlStoreL2AdapterConfig:
        backend = d.get("backend")
        if backend not in _VALID_DYNAMIC_BACKENDS:
            raise ValueError(
                "backend must be one of %s, got %r" % (_VALID_DYNAMIC_BACKENDS, backend)
            )

        backend_params = d.get("backend_params", {})
        if not isinstance(backend_params, dict):
            raise ValueError("backend_params must be a dict of string key-value pairs")

        return cls(backend=backend, backend_params=backend_params)

    @classmethod
    def help(cls) -> str:
        return (
            "Dynamic Nixl store L2 adapter config fields:\n"
            "- backend (str): Nixl storage backend, "
            "one of %s (required)\n"
            "- backend_params (dict): backend-specific "
            "string key-value pairs. Must include "
            "'file_path' and 'use_direct_io'.\n"
            "- persist_enabled (bool): if True, keep data files on disk "
            "at shutdown (optional, default True)\n"
            "Lookup always checks secondary storage (disk) on miss."
            % (_VALID_DYNAMIC_BACKENDS,)
        )


# Self-register config type and adapter factory
register_l2_adapter_type("nixl_store_dynamic", DynamicNixlStoreL2AdapterConfig)


def _create_dynamic_nixl_store_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create a DynamicNixlStoreL2Adapter from config."""
    if l1_memory_desc is None:
        raise ValueError(
            "l1_memory_desc is required to create a DynamicNixlStoreL2Adapter."
        )
    return DynamicNixlStoreL2Adapter(config, l1_memory_desc)  # type: ignore[arg-type]


register_l2_adapter_factory("nixl_store_dynamic", _create_dynamic_nixl_store_adapter)
