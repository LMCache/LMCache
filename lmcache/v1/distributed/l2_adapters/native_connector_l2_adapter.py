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
from typing import TYPE_CHECKING, Any, Optional
import os
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import (
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

logger = init_logger(__name__)


# Key separator — kept in sync with fs_l2_adapter.py and
# csrc/storage_backends/fs/connector.cpp. Both ``@`` in ``model_name``
# and ``@`` in ``cache_salt`` are rejected by ObjectKey.__post_init__
# so splitting on ``@`` is unambiguous.
_KEY_SEP = "@"


def _object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey to the native-connector wire format.

    Unsalted::

        <model_name>@<kv_rank_hex>@<chunk_hash_hex>

    Salted (trailing ``cache_salt``)::

        <model_name>@<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>

    Keys with ``cache_salt=""`` produce the 3-field shape, which is
    bit-identical to the format used before ``cache_salt`` existed —
    so existing un-salted caches remain valid with no migration.
    """
    base = (
        f"{key.model_name}{_KEY_SEP}{key.kv_rank:08x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}"
    return base


def _obj_to_memoryview(
    obj: MemoryObj,
) -> memoryview:  # type: ignore[type-arg]
    """
    Extract a byte-oriented memoryview from a MemoryObj.

    Uses the MemoryObj's byte_array property which returns
    a ctypes-backed memoryview with itemsize=1, so pybind's
    buffer_info.size == num_bytes.
    """
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

    def __init__(self, native_client: Any, max_capacity_gb: float = 0) -> None:
        super().__init__(max_capacity_bytes=int(max_capacity_gb * (1024**3)))
        self._client = native_client
        self._client_fd: int = int(native_client.event_fd())

        # 3 distinct Python eventfds for the L2 adapter
        # interface
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # Pending ops: native future_id →
        #   (op_type, task_id, num_keys, keys_for_locking)
        # keys_for_locking is only set for lookup ops so
        # we can apply locks
        self._pending_ops: dict[
            int,
            tuple[str, L2TaskId, int, list[ObjectKey] | None],
        ] = {}

        # Completed results (same pattern as MockL2Adapter)
        self._completed_stores: dict[L2TaskId, bool] = {}
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
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        return self._load_efd

    # ---------------------------------------------------------------
    # Store Interface
    # ---------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        key_strings = [_object_key_to_string(k) for k in keys]
        memviews = [_obj_to_memoryview(obj) for obj in objects]
        per_key_sizes = [obj.get_size() for obj in objects]

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
    ) -> dict[L2TaskId, bool]:
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
        memviews = [_obj_to_memoryview(obj) for obj in objects]

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
    # Cleanup
    # ---------------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._demux_thread.join(timeout=5)

        self._client.close()

        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)

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
                        self._completed_stores[task_id] = ok
                        store_info = self._pending_store_sizes.pop(fid, None)
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
                                else:
                                    keys_stored.append(key)
                                    sizes_stored.append(0)
                        os.eventfd_write(self._store_efd, 1)

                    elif op_type == self._OP_LOOKUP:
                        bitmap = Bitmap(num_keys)
                        if ok and result_bools is not None:
                            for i, found in enumerate(result_bools):
                                if found:
                                    bitmap.set(i)
                                    if lookup_keys is not None:
                                        self._locked_keys[lookup_keys[i]] += 1
                        self._completed_lookups[task_id] = bitmap
                        os.eventfd_write(self._lookup_efd, 1)

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
                        os.eventfd_write(self._load_efd, 1)

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


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------


class RESPL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by a native RESP
    connector (Redis/Valkey).

    Fields:
    - host: server hostname or IP.
    - port: server port.
    - num_workers: C++ worker threads for I/O (default 8).
    - username: optional auth username.
    - password: optional auth password.
    """

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int = 8,
        username: str = "",
        password: str = "",
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.username = username
        self.password = password

    @classmethod
    def from_dict(cls, d: dict) -> "RESPL2AdapterConfig":
        host = d.get("host")
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")

        port = d.get("port")
        if not isinstance(port, int) or port <= 0:
            raise ValueError("port must be a positive integer")

        num_workers = d.get("num_workers", 8)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        username = d.get("username", "")
        password = d.get("password", "")

        return cls(
            host=host,
            port=port,
            num_workers=num_workers,
            username=str(username),
            password=str(password),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "RESP L2 adapter config fields:\n"
            "- host (str): Redis/Valkey server hostname "
            "or IP (required)\n"
            "- port (int): server port (required, >0)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 8, >0)\n"
            "- username (str): auth username "
            "(default empty)\n"
            "- password (str): auth password "
            "(default empty)"
        )


def _create_resp_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ Redis connector."""
    try:
        # First Party
        from lmcache.lmcache_redis import (
            LMCacheRedisClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "RESP L2 adapter requires the C++ Redis "
            "extension. Build with: pip install -e ."
        ) from e

    assert isinstance(config, RESPL2AdapterConfig)
    native_client = LMCacheRedisClient(
        config.host,
        config.port,
        config.num_workers,
        config.username,
        config.password,
    )
    logger.info(
        "Created RESP L2 adapter: %s:%d (workers=%d)",
        config.host,
        config.port,
        config.num_workers,
    )
    return NativeConnectorL2Adapter(native_client)


# Self-register config type and adapter factory
register_l2_adapter_type("resp", RESPL2AdapterConfig)
register_l2_adapter_factory("resp", _create_resp_l2_adapter)


# -------------------------------------------------------------------
# FS native L2 adapter config + factory
# -------------------------------------------------------------------


class FSNativeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    filesystem connector.

    Fields:
    - base_path: directory for storing KV cache files.
    - num_workers: C++ worker threads for I/O (default 4).
    - relative_tmp_dir: relative sub-dir for temp files.
    - use_odirect: bypass page cache via O_DIRECT.
    - read_ahead_size: trigger filesystem readahead by
      reading this many bytes first (optional).
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size

    @classmethod
    def from_dict(cls, d: dict) -> "FSNativeL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        relative_tmp_dir = d.get("relative_tmp_dir", "")
        if not isinstance(relative_tmp_dir, str):
            raise ValueError("relative_tmp_dir must be a string")

        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")

        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS native L2 adapter config fields:\n"
            "- base_path (str): directory for KV "
            "cache files (required)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- relative_tmp_dir (str): relative "
            "sub-dir for temp files (default empty)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (default false)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes "
            "first (optional)"
        )


def _create_fs_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ filesystem connector."""
    try:
        # First Party
        from lmcache.lmcache_fs import (
            LMCacheFSClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "FS native L2 adapter requires the C++ FS "
            "extension. Build with: pip install -e ."
        ) from e

    assert isinstance(config, FSNativeL2AdapterConfig)
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
    )
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, read_ahead=%s)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
    )
    return NativeConnectorL2Adapter(native_client)


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)


# -------------------------------------------------------------------
# Blkio (libblkio) native L2 adapter config + factory
# -------------------------------------------------------------------


class BlkioL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    libblkio connector (io_uring block device I/O).

    Fields:
    - device_path: path to the block device
      (e.g. ``/dev/nvme0n1``).
    - num_workers: C++ worker threads for I/O
      (default 4).  Each worker gets its own
      io_uring instance through libblkio.
    - direct_io: bypass page cache via O_DIRECT
      (default true).
    """

    def __init__(
        self,
        device_path: str,
        num_workers: int = 4,
        direct_io: bool = True,
    ) -> None:
        """Initialize a BlkioL2AdapterConfig.

        Args:
            device_path: Path to the block device
                (e.g. ``/dev/nvme0n1``).
            num_workers: Number of C++ worker threads for I/O.
                Each worker gets its own io_uring instance
                through libblkio.  Default 4.
            direct_io: If ``True``, bypass the page cache via
                ``O_DIRECT``.  Default ``True``.
        """
        super().__init__()
        self.device_path = device_path
        self.num_workers = num_workers
        self.direct_io = direct_io

    @classmethod
    def from_dict(cls, d: dict) -> "BlkioL2AdapterConfig":
        """Create a :class:`BlkioL2AdapterConfig` from a dictionary.

        Args:
            d: Dictionary with keys ``device_path`` (required),
                ``num_workers`` (optional, default 4), and
                ``direct_io`` (optional, default ``True``).

        Returns:
            A validated :class:`BlkioL2AdapterConfig` instance.

        Raises:
            ValueError: If any field has an invalid type or value.
        """
        device_path = d.get("device_path")
        if not isinstance(device_path, str) or not device_path:
            raise ValueError(
                "device_path must be a non-empty string"
            )

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError(
                "num_workers must be a positive integer"
            )

        direct_io = d.get("direct_io", True)
        if not isinstance(direct_io, bool):
            raise ValueError("direct_io must be a boolean")

        return cls(
            device_path=device_path,
            num_workers=num_workers,
            direct_io=direct_io,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Blkio L2 adapter config fields:\n"
            "- device_path (str): path to the block "
            "device (required, e.g. /dev/nvme0n1)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- direct_io (bool): bypass page cache "
            "via O_DIRECT (default true)"
        )


def _create_blkio_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ libblkio connector."""
    try:
        # First Party
        from lmcache.lmcache_blkio import (
            LMCacheBlkioClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "Blkio L2 adapter requires the C++ libblkio "
            "extension. Install libblkio-dev and rebuild "
            "with: pip install -e ."
        ) from e

    assert isinstance(config, BlkioL2AdapterConfig)
    native_client = LMCacheBlkioClient(
        config.device_path,
        config.num_workers,
        config.direct_io,
    )
    logger.info(
        "Created blkio L2 adapter: %s (workers=%d, direct_io=%s)",
        config.device_path,
        config.num_workers,
        config.direct_io,
    )
    return NativeConnectorL2Adapter(native_client)


register_l2_adapter_type("blkio", BlkioL2AdapterConfig)
register_l2_adapter_factory("blkio", _create_blkio_l2_adapter)
