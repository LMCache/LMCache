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

Index/disk consistency:
- ``delete()`` runs on the eviction thread and unlinks files outside the
  lock. Keys being deleted are tombstoned in ``_deleting`` until the unlink
  finishes, so a concurrent lookup cannot re-add them from disk and a
  concurrent store cannot publish a file that the delete then removes.
- Eviction works on whole chunks: all objects sharing
  ``(model_name, chunk_hash, cache_salt)`` (every kv_rank and object group)
  are deleted together, or kept together when any of them is pinned.
- A load that finds its file missing (ENOENT/ESTALE) drops the whole chunk
  from the index instead of leaving a stale entry that fails on every load.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
import asyncio
import errno
import threading

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, MemoryTags, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.nixl_native_l2_adapter import (
    nixl_backend_memory_tags,
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

ChunkId = tuple[str, bytes, str]


def _chunk_id(key: ObjectKey) -> ChunkId:
    """Identify the chunk an object belongs to (all kv_ranks and groups)."""
    return (key.model_name, key.chunk_hash, key.cache_salt)


def _missing_file_reason(exc: BaseException) -> Optional[str]:
    """Return ``enoent``/``estale`` when ``exc`` means the file is gone."""
    if isinstance(exc, FileNotFoundError):
        return "enoent"
    if isinstance(exc, OSError) and exc.errno == errno.ESTALE:
        return "estale"
    return None


_meter = metrics.get_meter("lmcache_mp.health")
_LOAD_MISSING_COUNTER = _meter.create_counter(
    "lmcache_mp.l2_load_missing_object",
    description=(
        "L2 objects that were in the index but whose file was missing at load "
        "time. Tagged by reason = enoent | estale."
    ),
    unit="objects",
)
_DROPPED_CHUNK_COUNTER = _meter.create_counter(
    "lmcache_mp.l2_dropped_missing_chunk",
    description="Chunks dropped from the L2 index after a missing-file load.",
    unit="chunks",
)
_TOMBSTONE_SKIP_COUNTER = _meter.create_counter(
    "lmcache_mp.l2_tombstone_skip",
    description=(
        "Lookups or stores that skipped a key because it was being deleted. "
        "Tagged by op = lookup | store."
    ),
    unit="objects",
)
_PINNED_CHUNK_SKIP_COUNTER = _meter.create_counter(
    "lmcache_mp.l2_eviction_pinned_chunk_skip",
    description="Eviction candidates kept because a sibling object was pinned.",
    unit="chunks",
)


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
        # Keys whose index entry is gone but whose file may still
        # be on disk (delete in progress), and the per-chunk member index.
        self._deleting: set[ObjectKey] = set()
        self._chunk_members: dict[ChunkId, set[ObjectKey]] = {}
        self._consistency_stats: dict[str, int] = {
            "load_missing_objects": 0,
            "dropped_missing_chunks": 0,
            "tombstone_skips": 0,
            "pinned_chunk_skips": 0,
        }

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
            """Unlock keys on the event loop thread under the adapter lock."""
            with self._lock:
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
        """Delete objects from storage, removing their files from disk.

        Each requested key expands to every object of its chunk. A chunk is
        skipped as a whole when any of its objects is pinned. Deleted keys
        stay tombstoned until their files are unlinked.
        """
        to_delete: list[tuple[ObjectKey, int]] = []
        pinned_chunks = 0
        with self._lock:
            seen: set[ChunkId] = set()
            for key in keys:
                if key not in self._memory_objects:
                    continue
                cid = _chunk_id(key)
                if cid in seen:
                    continue
                seen.add(cid)
                members = list(self._chunk_members.get(cid, ()))
                pinned = [k for k in members if self._memory_objects[k].pin_count > 0]
                if pinned:
                    logger.debug(
                        "Skipping eviction of chunk of %s: %d pinned object(s)",
                        key,
                        len(pinned),
                    )
                    pinned_chunks += 1
                    continue
                for member in members:
                    obj = self._index_remove_locked(member)
                    assert obj is not None
                    self._total_bytes -= obj.size
                    self._deleting.add(member)
                    to_delete.append((member, obj.size))
            self._consistency_stats["pinned_chunk_skips"] += pinned_chunks
        if pinned_chunks:
            _PINNED_CHUNK_SKIP_COUNTER.add(pinned_chunks)
        self._unlink_tombstoned(to_delete)

    def _index_add_locked(self, key: ObjectKey, obj: NixlStoreObj) -> None:
        """Insert ``key`` into the index. Caller holds ``self._lock``."""
        self._memory_objects[key] = obj
        self._chunk_members.setdefault(_chunk_id(key), set()).add(key)

    def _index_remove_locked(self, key: ObjectKey) -> Optional[NixlStoreObj]:
        """Remove ``key`` from the index. Caller holds ``self._lock``.

        Byte accounting is left to the caller.
        """
        obj = self._memory_objects.pop(key, None)
        if obj is None:
            return None
        cid = _chunk_id(key)
        members = self._chunk_members.get(cid)
        if members is not None:
            members.discard(key)
            if not members:
                del self._chunk_members[cid]
        return obj

    def _unlink_tombstoned(
        self,
        entries: list[tuple[ObjectKey, int]],
        skip_unlink: Optional[set[ObjectKey]] = None,
    ) -> None:
        """Unlink files of tombstoned keys, clear the tombstones, notify.

        Filesystem I/O runs outside the lock so concurrent store, lookup and
        load are not blocked; the tombstones keep them away from these keys.
        """
        if not entries:
            return
        try:
            for key, _ in entries:
                if skip_unlink is None or key not in skip_unlink:
                    self.nixl_agent.dynamic_delete(key)
        finally:
            with self._lock:
                for key, _ in entries:
                    self._deleting.discard(key)
        self._notify_keys_deleted([k for k, _ in entries], [s for _, s in entries])

    def _drop_missing_chunks(self, missing: list[tuple[ObjectKey, str]]) -> None:
        """Drop every chunk that has an object whose file is missing.

        The missing objects themselves are not unlinked again; their present
        siblings are, because a partial chunk cannot serve a hit.
        """
        to_drop: list[tuple[ObjectKey, int]] = []
        dropped_chunks: list[ObjectKey] = []
        with self._lock:
            seen: set[ChunkId] = set()
            for key, reason in missing:
                self._consistency_stats["load_missing_objects"] += 1
                _LOAD_MISSING_COUNTER.add(1, {"reason": reason})
                cid = _chunk_id(key)
                if cid in seen:
                    continue
                seen.add(cid)
                members = set(self._chunk_members.get(cid, ()))
                if not members:
                    continue  # already dropped by another load or delete
                for member in members:
                    obj = self._index_remove_locked(member)
                    assert obj is not None
                    self._total_bytes -= obj.size
                    self._deleting.add(member)
                    to_drop.append((member, obj.size))
                dropped_chunks.append(key)
            self._consistency_stats["dropped_missing_chunks"] += len(dropped_chunks)
        for key in dropped_chunks:
            logger.warning("L2 object missing on load, dropped its chunk: %s", key)
        if dropped_chunks:
            _DROPPED_CHUNK_COUNTER.add(len(dropped_chunks))
        self._unlink_tombstoned(to_drop, skip_unlink={k for k, _ in missing})

    def _count_tombstone_skip(self, op: str) -> None:
        """Record a lookup/store skipped by a tombstone. Caller holds the lock."""
        self._consistency_stats["tombstone_skips"] += 1
        _TOMBSTONE_SKIP_COUNTER.add(1, {"op": op})

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
            consistency = dict(self._consistency_stats)
            consistency["deleting_objects"] = len(self._deleting)
            consistency["stored_chunk_count"] = len(self._chunk_members)
        consistency["store_dir_recreated"] = getattr(
            self.nixl_agent, "dir_recreated_count", 0
        )
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "DynamicNixlStoreL2Adapter",
            "backend": self._config.backend,
            "stored_object_count": stored_object_count,
            "pinned_object_count": pinned_object_count,
            "event_loop_alive": self._loop_thread.is_alive(),
            "consistency": consistency,
        }

    #####################
    # Cleanup Interface
    #####################

    def get_memory_tags(self) -> MemoryTags:
        """Return the memory tags of this adapter's NIXL backend."""
        return nixl_backend_memory_tags(self._config.backend, self._config.shared)

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
        reserved_keys: list[tuple[ObjectKey, int]] = []
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
                    if key in self._deleting:
                        # The old file is about to be unlinked; publishing a
                        # new one now would lose it to that unlink.
                        self._count_tombstone_skip("store")
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
                    reserved_keys.append((key, mem_size))

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
                            self._index_add_locked(key, store_obj)
                            store_obj.decrease_pin_count()
                        stored_keys.append(key)
                        stored_sizes.append(mem_size)

        except Exception:
            logger.exception("Dynamic NIXL store task %d failed", task_id)
            success = False
        finally:
            if not success:
                # Un-reserve any reserved keys that were not successfully
                # stored and committed
                with self._lock:
                    for key, mem_size in reserved_keys:
                        if key in self._inflight_stores:
                            self._inflight_stores.discard(key)
                            self._total_bytes -= mem_size

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
                elif key in self._deleting:
                    self._count_tombstone_skip("lookup")
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
                if key in self._deleting:
                    # Deleted while we were stat-ing it: do not resurrect.
                    self._count_tombstone_skip("lookup")
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
                self._index_add_locked(key, obj)
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
        missing: list[tuple[ObjectKey, str]] = []
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
                        reason = _missing_file_reason(result)
                        if reason is not None:
                            missing.append((keys[pos], reason))
                            continue
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

        if missing:
            self._drop_missing_chunks(missing)
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
