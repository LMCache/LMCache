# SPDX-License-Identifier: Apache-2.0
"""
L2 adapter that wraps any pybind-wrapped C++ IStorageConnector (native client).

This bridge lets any native storage connector (Redis, RDMA, Mooncake, etc.)
serve as an MP-mode L2 adapter.  The same C++ connector implementation is
also usable in non-MP mode via ConnectorClientBase.

Architecture:
  - The native client has 1 eventfd + drain_completions() for all operations.
  - This adapter creates 3 Python eventfds (store, lookup, load) and runs a
    background demux thread that routes native completions to the right
    category based on a future_id → op_type mapping.
  - ObjectKey serialization and MemoryObj buffer extraction happen at the
    submit call boundary.
  - Locking is client-side (refcount dict) since remote backends don't have
    our eviction concept.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from functools import cache
from typing import Any
import ctypes
import select
import threading

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.memory_management import MemoryObj, TensorMemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


# Key separator — kept in sync with fs_l2_adapter.py and
# csrc/storage_backends/fs/connector.cpp. Both ``@`` in ``model_name``
# and ``@`` in ``cache_salt`` are rejected by ObjectKey.__post_init__
# so splitting on ``@`` is unambiguous.
_KEY_SEP = "@"


@cache
def _cached_ubyte_array_type(num_bytes: int) -> "type[ctypes.Array[ctypes.c_ubyte]]":
    """Return a cached ``ctypes.c_ubyte * num_bytes`` array type.

    Mirrors ``memory_management._get_cached_ubyte_array_type`` (re-implemented
    here because that helper is module-private): ctypes builds a fresh heap
    type for every ``(c_ubyte * N)`` expression and never reclaims it, leaking
    ~1-2 kB per distinct ``N`` per call. Caching the type per length avoids
    that leak; the ``from_address`` instances never own the underlying buffer,
    so sharing the type is safe. See
    https://github.com/LMCache/LMCache/issues/3767.

    Args:
        num_bytes: The length of the array type in bytes.

    Returns:
        The cached ``ctypes.Array`` subclass for the given length.
    """
    return ctypes.c_ubyte * num_bytes


def _object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey to the native-connector wire format.

    Unsalted::

        <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>

    Salted (trailing ``cache_salt``)::

        <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>
    """
    base = (
        f"{key.model_name}{_KEY_SEP}{key.kv_rank:08x}"
        f"{_KEY_SEP}{key.object_group_id:x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}"
    return base


def _obj_to_memoryview(
    obj: MemoryObj,
    pad_to_physical: bool = False,
) -> memoryview:  # type: ignore[type-arg]
    """Extract a byte-oriented memoryview from a MemoryObj.

    By default this returns ``obj.byte_array``: a ctypes-backed memoryview
    with itemsize=1 spanning the logical size (``get_size()``), so pybind's
    ``buffer_info.size == num_bytes``.

    When ``pad_to_physical`` is true and ``obj`` is a TensorMemoryObj whose
    physical allocation exceeds its logical size (alignment padding from the
    L1 allocator), the view instead spans the full physical slot:
    ``get_physical_size()`` bytes starting at ``data_ptr``. The view is a
    zero-copy pointer overlay (no payload is allocated or copied). The
    padding tail belongs to the object's own allocation and stays within the
    L1 arena, but its contents are undefined and are transferred as-is. This
    mode exists for O_DIRECT / DMA native connectors that require buffer
    lengths aligned to the L1 alignment (e.g. 4096 bytes).

    Args:
        obj: The memory object to expose.
        pad_to_physical: Whether to include allocator alignment padding when
            present.

    Returns:
        A byte-oriented memoryview over the object's bytes.
    """
    if (
        pad_to_physical
        and isinstance(obj, TensorMemoryObj)
        and obj.get_physical_size() > obj.get_size()
    ):
        num_bytes = obj.get_physical_size()
        arr_type = _cached_ubyte_array_type(num_bytes)
        ubyte_ptr = ctypes.cast(obj.data_ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = arr_type.from_address(ctypes.addressof(ubyte_ptr.contents))
        return memoryview(byte_array)
    return obj.byte_array  # type: ignore[return-value]


class NativeConnectorL2Adapter(L2AdapterInterface):
    """
    Wraps a pybind-wrapped C++ IStorageConnector to
    implement L2AdapterInterface.

    The native_client must expose:
      - event_fd() -> int
      - submit_batch_get(keys, memoryviews) -> int
      - submit_batch_set(keys, memoryviews) -> int
      - submit_batch_exists(keys) -> int
      - drain_completions()
          -> list[tuple[int, bool, str, list[bool]|None]]
      - close()
    """

    # Operation type tags for the pending-ops map
    _OP_STORE = "store"
    _OP_LOOKUP = "lookup"
    _OP_LOAD = "load"
    _OP_DELETE = "delete"

    def __init__(
        self,
        native_client: Any,
        max_capacity_gb: float = 0,
        type_name: str = "",
        extra_status: dict[str, Any] | None = None,
        pad_buffers_to_alignment: bool = False,
    ) -> None:
        """Initialize the adapter over a native connector client.

        Args:
            native_client: Pybind-wrapped C++ connector client exposing
                ``event_fd``, ``submit_batch_*``, ``drain_completions`` and
                ``close``.
            max_capacity_gb: Capacity in GiB used for L2 usage accounting.
                Zero disables the capacity limit.
            type_name: Stable adapter type label for status reporting.
            extra_status: Extra key-value pairs merged into
                ``report_status()``.
            pad_buffers_to_alignment: When true, store/load submit the full
                physical (alignment-padded) slot of each TensorMemoryObj
                instead of just its logical bytes, so transfer lengths are
                aligned to the L1 allocator's ``align_bytes``. Enable for
                native connectors whose I/O path requires aligned buffer
                lengths (e.g. O_DIRECT file storage). See
                ``_obj_to_memoryview``.
        """
        super().__init__(max_capacity_bytes=int(max_capacity_gb * (1024**3)))
        self._client = native_client
        self._client_fd: int = int(native_client.event_fd())
        self._type_name: str = type_name or type(native_client).__name__
        self._extra_status: dict[str, Any] = dict(extra_status or {})
        self._pad_buffers_to_alignment = pad_buffers_to_alignment

        # 3 distinct cross-platform notifiers for the L2 adapter
        # interface
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Pending ops: native future_id →
        #   (op_type, task_id, num_keys, keys_for_locking)
        # keys_for_locking is only set for lookup ops so
        # we can apply locks
        self._pending_ops: dict[
            int,
            tuple[str, L2TaskId, int, list[ObjectKey] | None],
        ] = {}

        # Completed results (same pattern as MockL2Adapter)
        self._completed_stores: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookups: dict[L2TaskId, Bitmap] = {}
        self._completed_loads: dict[L2TaskId, Bitmap] = {}

        # Client-side lock tracking (refcount per key)
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)

        # Delete capability detection
        self._has_delete = callable(getattr(native_client, "submit_batch_delete", None))

        # Pending delete events for synchronous delete() calls
        self._pending_delete_events: dict[L2TaskId, threading.Event] = {}

        # Per-key size tracking. ``_key_sizes`` lets us look up byte sizes
        # at delete time (the native completion only carries booleans, not
        # sizes) so we can pass them to ``_notify_keys_deleted``. Aggregate
        # and per-user totals live in the base class — see ``get_usage``.
        self._key_sizes: dict[ObjectKey, int] = {}
        # Pending store sizes: native future_id -> (keys, per_key_sizes).
        # Bridges the async store submit → demux completion gap so the
        # demux thread can fire ``_notify_keys_stored(keys, sizes)``.
        self._pending_store_sizes: dict[int, tuple[list[ObjectKey], list[int]]] = {}

        # Task ID counter
        self._next_task_id: L2TaskId = 0

        # Lock for all shared state above
        self._lock = threading.Lock()

        # Background demux thread
        self._stop = threading.Event()
        self._demux_thread = threading.Thread(
            target=self._demux_loop,
            daemon=True,
            name="l2-adapter-demux",
        )
        self._demux_thread.start()

    # ---------------------------------------------------------------
    # Event Fd Interface
    # ---------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ---------------------------------------------------------------
    # Store Interface
    # ---------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        key_strings = [_object_key_to_string(k) for k in keys]
        memviews = [
            _obj_to_memoryview(obj, self._pad_buffers_to_alignment) for obj in objects
        ]
        # Charge the byte count actually submitted to the backend: the
        # physical (alignment-padded) size when padding is enabled and
        # present, otherwise the logical size.
        per_key_sizes = [len(mv) for mv in memviews]

        # Register pending op BEFORE submit to avoid race
        # with demux thread. The native submit is
        # non-blocking so holding the lock is brief.
        with self._lock:
            task_id = self._get_next_task_id()
            future_id = int(self._client.submit_batch_set(key_strings, memviews))
            self._pending_ops[future_id] = (
                self._OP_STORE,
                task_id,
                len(keys),
                None,
            )
            self._pending_store_sizes[future_id] = (list(keys), per_key_sizes)

        return task_id

    def pop_completed_store_tasks(
        self,
    ) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            completed = self._completed_stores
            self._completed_stores = {}
        return completed

    # ---------------------------------------------------------------
    # Lookup and Lock Interface
    # ---------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> L2TaskId:
        key_strings = [_object_key_to_string(k) for k in keys]

        with self._lock:
            task_id = self._get_next_task_id()
            future_id = int(self._client.submit_batch_exists(key_strings))
            self._pending_ops[future_id] = (
                self._OP_LOOKUP,
                task_id,
                len(keys),
                list(keys),
            )

        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookups.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                if key not in self._locked_keys:
                    continue
                if self._locked_keys[key] <= 1:
                    del self._locked_keys[key]
                else:
                    self._locked_keys[key] -= 1

    # ---------------------------------------------------------------
    # Load Interface
    # ---------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        key_strings = [_object_key_to_string(k) for k in keys]
        memviews = [
            _obj_to_memoryview(obj, self._pad_buffers_to_alignment) for obj in objects
        ]

        with self._lock:
            task_id = self._get_next_task_id()
            future_id = int(self._client.submit_batch_get(key_strings, memviews))
            self._pending_ops[future_id] = (
                self._OP_LOAD,
                task_id,
                len(keys),
                list(keys),
            )

        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_loads.pop(task_id, None)

    # ---------------------------------------------------------------
    # Eviction Interface
    # ---------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete a batch of keys from the remote backend.

        Submits a batch delete to the native connector and blocks
        until the demux thread signals completion (up to 30s timeout).
        Fires ``_notify_keys_deleted`` on success so eviction policy
        tracking stays in sync.

        No-op if the connector does not expose ``submit_batch_delete``
        or if the key list is empty.
        """
        if not keys or not self._has_delete:
            return

        key_strings = [_object_key_to_string(k) for k in keys]
        done_event = threading.Event()

        with self._lock:
            task_id = self._get_next_task_id()
            future_id = int(self._client.submit_batch_delete(key_strings))
            self._pending_ops[future_id] = (
                self._OP_DELETE,
                task_id,
                len(keys),
                list(keys),
            )
            self._pending_delete_events[task_id] = done_event

        # Block until demux thread signals completion
        if not done_event.wait(timeout=30.0):
            with self._lock:
                self._pending_delete_events.pop(task_id, None)
                # Note: _pending_ops entry may already be consumed
                # by the demux thread; pop is safe either way.
                for fid, entry in list(self._pending_ops.items()):
                    if entry[1] == task_id:
                        self._pending_ops.pop(fid, None)
                        break
            logger.warning(
                "delete() timed out after 30s for %d keys",
                len(keys),
            )
            return

        # ``_notify_keys_deleted`` is fired by the demux thread (with
        # accurate per-key sizes drawn from ``_key_sizes``) when the
        # backend reports per-key deletion results, so we don't notify
        # again here.

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class tracks aggregate + per-user totals via ``_notify_keys_*``;
    # we feed it the byte sizes from each store/delete completion.

    # ---------------------------------------------------------------
    # Status Interface
    # ---------------------------------------------------------------

    def report_status(self) -> dict[str, Any]:
        """Return a status dict for this native-connector L2 adapter.

        Returns:
            A dict with at minimum:
              * ``is_healthy`` (bool): ``True`` while the background demux
                thread is alive and not stopping.
              * ``type`` (str): Stable adapter type label, supplied by the
                factory or derived from the native client class name.
            Plus any caller-supplied ``extra_status`` fields (e.g. backend
            configuration like ``base_path``, ``num_workers``).
        """
        status: dict[str, Any] = {
            "is_healthy": (self._demux_thread.is_alive() and not self._stop.is_set()),
            "type": self._type_name,
        }
        status.update(self._extra_status)
        return status

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._demux_thread.join(timeout=5)

        self._client.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _get_next_task_id(self) -> L2TaskId:
        """Increment and return the next task ID.
        Must be called under _lock."""
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _demux_loop(self) -> None:
        """Background thread that polls the native
        connector's eventfd, drains completions, and
        routes them to the correct L2 result category.
        """
        poller = select.poll()
        poller.register(self._client_fd, select.POLLIN)

        while not self._stop.is_set():
            events = poller.poll(500)
            if not events:
                continue

            try:
                completions = self._client.drain_completions()
            except Exception:
                logger.exception("drain_completions failed")
                continue

            if not completions:
                continue

            # Collect listener notifications to fire after
            # releasing the lock. Sizes are collected in parallel so
            # ``_notify_keys_*`` can update the base class's byte
            # accounting in one shot.
            keys_stored: list[ObjectKey] = []
            sizes_stored: list[int] = []
            keys_accessed: list[ObjectKey] = []
            keys_deleted: list[ObjectKey] = []
            sizes_deleted: list[int] = []
            # Events for synchronous ``delete()`` callers. We set these
            # AFTER firing ``_notify_keys_deleted`` below so that when
            # the caller unblocks and calls ``get_usage()``, the base
            # class's byte counters already reflect the deletion.
            delete_done_events: list[threading.Event] = []

            with self._lock:
                for (
                    future_id,
                    ok,
                    error,
                    result_bools,
                ) in completions:
                    fid = int(future_id)
                    entry = self._pending_ops.pop(fid, None)
                    if entry is None:
                        logger.warning(
                            "Received completion for unknown future_id=%d",
                            fid,
                        )
                        continue

                    (
                        op_type,
                        task_id,
                        num_keys,
                        lookup_keys,
                    ) = entry

                    if op_type == self._OP_STORE:
                        store_info = self._pending_store_sizes.pop(fid, None)
                        task_bytes = 0
                        if ok and store_info is not None:
                            store_keys, sizes = store_info
                            for key, size in zip(store_keys, sizes, strict=True):
                                # First-store wins for byte accounting:
                                # a re-store of an existing key adds 0
                                # bytes (the backend already holds it).
                                # We still notify the listener for every
                                # store so LRU policies can ``move_to_end``
                                # on re-store — passing size=0 in that
                                # case is a no-op for the base counters.
                                if key not in self._key_sizes:
                                    self._key_sizes[key] = size
                                    keys_stored.append(key)
                                    sizes_stored.append(size)
                                    task_bytes += size
                                else:
                                    keys_stored.append(key)
                                    sizes_stored.append(0)
                        self._completed_stores[task_id] = L2StoreResult(ok, task_bytes)
                        self._store_efd.notify()

                    elif op_type == self._OP_LOOKUP:
                        bitmap = Bitmap(num_keys)
                        if ok and result_bools is not None:
                            for i, found in enumerate(result_bools):
                                if found:
                                    bitmap.set(i)
                                    if lookup_keys is not None:
                                        self._locked_keys[lookup_keys[i]] += 1
                        self._completed_lookups[task_id] = bitmap
                        self._lookup_efd.notify()

                    elif op_type == self._OP_LOAD:
                        bitmap = Bitmap(num_keys)
                        loaded_keys: list[ObjectKey] = []
                        if result_bools is not None:
                            for i, loaded in enumerate(result_bools):
                                if loaded:
                                    bitmap.set(i)
                                    if lookup_keys is not None:
                                        loaded_keys.append(lookup_keys[i])
                        elif ok:
                            # Fallback for connectors that
                            # do not report per-key results
                            for i in range(num_keys):
                                bitmap.set(i)
                            if lookup_keys is not None:
                                loaded_keys.extend(lookup_keys)
                        keys_accessed.extend(loaded_keys)
                        self._completed_loads[task_id] = bitmap
                        self._load_efd.notify()

                    elif op_type == self._OP_DELETE:
                        if result_bools is not None and lookup_keys is not None:
                            for i, deleted in enumerate(result_bools):
                                if not deleted:
                                    continue
                                key = lookup_keys[i]
                                # Only notify (with size) for keys we've
                                # actually accounted for via a prior store.
                                if key in self._key_sizes:
                                    sizes_deleted.append(self._key_sizes.pop(key))
                                    keys_deleted.append(key)
                        evt = self._pending_delete_events.pop(task_id, None)
                        if evt is not None:
                            delete_done_events.append(evt)

            # Fire listener notifications outside the lock so a slow
            # listener cannot stall further demux iterations.
            if keys_stored:
                self._notify_keys_stored(keys_stored, sizes_stored)
            if keys_accessed:
                self._notify_keys_accessed(keys_accessed)
            if keys_deleted:
                self._notify_keys_deleted(keys_deleted, sizes_deleted)
            # Unblock any synchronous ``delete()`` callers only AFTER
            # ``_notify_keys_deleted`` has updated the base class byte
            # accounting, so ``get_usage()`` never briefly reports stale
            # (too-high) usage in the window between ``delete()``
            # returning and the notify running.
            for evt in delete_done_events:
                evt.set()
