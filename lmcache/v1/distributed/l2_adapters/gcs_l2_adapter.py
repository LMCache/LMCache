# SPDX-License-Identifier: Apache-2.0
"""
GCS L2 adapter using the Google Cloud Storage Python client.

Wraps the synchronous ``google.cloud.storage`` client in an asyncio
event loop (daemon thread) with a ``ThreadPoolExecutor`` for
non-blocking GCS I/O.  Exposes the poll-driven ``L2AdapterInterface``
contract (3 eventfds for store / lookup / load completions).

Pattern follows ``S3L2Adapter`` (asyncio loop on a daemon thread + 3
eventfds) with refcount-based locking and capacity tracking for eviction.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from typing import TYPE_CHECKING, Optional
import asyncio
import concurrent.futures
import ctypes
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.memory_management import MemoryObj

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
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
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey to a deterministic GCS blob name.

    Unsalted::

        <model_name>@<kv_rank_hex>@<chunk_hash_hex>

    Salted (trailing ``cache_salt``)::

        <model_name>@<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>

    Keys with ``cache_salt=""`` produce the 3-field shape, so existing
    un-salted caches remain valid with no migration.  ``@`` in
    ``model_name`` and ``cache_salt`` is rejected by
    ``ObjectKey.__post_init__``, so the format is unambiguous.
    """
    base = f"{key.model_name}@{key.kv_rank:08x}@{key.chunk_hash.hex()}"
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _is_connection_error(error_msg: str) -> bool:
    """Return True when *error_msg* looks like a transient network failure."""
    upper = error_msg.upper()
    return any(
        kw in upper
        for kw in (
            "CONNECTION",
            "TIMEOUT",
            "UNAVAILABLE",
            "DNS",
            "SOCKET",
            "DEADLINE",
        )
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GCSL2AdapterConfig(L2AdapterConfigBase):
    """Config for the GCS L2 adapter.

    Fields:
    - gcs_bucket (str, required): GCS bucket name.
    - gcs_credentials_file (str, optional): path to a service-account
      JSON key file.  When omitted, Application Default Credentials
      (ADC) are used (``GOOGLE_APPLICATION_CREDENTIALS`` env var,
      ``gcloud`` CLI auth, or Workload Identity metadata service).
    - gcs_project (str, optional): GCP project ID.  Required only when
      ADC cannot infer a project from the environment.
    - gcs_num_workers (int): thread-pool size for GCS I/O (default 64).
    - max_capacity_gb (float): aggregate capacity used by ``get_usage()``;
      ``0`` disables aggregate eviction (``usage_fraction == -1.0``).
    """

    def __init__(
        self,
        gcs_bucket: str,
        gcs_credentials_file: Optional[str] = None,
        gcs_project: Optional[str] = None,
        gcs_num_workers: int = 64,
        max_capacity_gb: float = 0.0,
    ):
        self.gcs_bucket = gcs_bucket
        self.gcs_credentials_file = gcs_credentials_file
        self.gcs_project = gcs_project
        self.gcs_num_workers = gcs_num_workers
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "GCSL2AdapterConfig":
        """Build config from a dict (e.g. parsed from JSON).

        Args:
            d: Adapter spec dict.

        Returns:
            A new ``GCSL2AdapterConfig`` instance.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        bucket = d.get("gcs_bucket")
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("gcs_bucket must be a non-empty string")

        def _opt_str(key: str) -> Optional[str]:
            v = d.get(key)
            if v is None:
                return None
            if not isinstance(v, str):
                raise ValueError(f"{key} must be a string")
            return v

        num_workers = d.get("gcs_num_workers", 64)
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers <= 0
        ):
            raise ValueError("gcs_num_workers must be a positive integer")

        max_cap = d.get("max_capacity_gb", 0.0)
        if not isinstance(max_cap, (int, float)) or isinstance(max_cap, bool):
            raise ValueError("max_capacity_gb must be a number")

        cfg = cls(
            gcs_bucket=bucket,
            gcs_credentials_file=_opt_str("gcs_credentials_file"),
            gcs_project=_opt_str("gcs_project"),
            gcs_num_workers=num_workers,
            max_capacity_gb=float(max_cap),
        )
        cfg.eviction_config = cls._parse_eviction_config(d)
        return cfg

    @classmethod
    def help(cls) -> str:
        """Return a help string describing the config fields.

        Returns:
            A human-readable description of all supported fields.
        """
        return (
            "GCS L2 adapter config fields:\n"
            "- gcs_bucket (str, required): GCS bucket name\n"
            "- gcs_credentials_file (str, optional): path to service-account "
            "JSON key file; defaults to Application Default Credentials\n"
            "- gcs_project (str, optional): GCP project ID; inferred from "
            "ADC when unset\n"
            "- gcs_num_workers (int): thread-pool size for GCS I/O (default 64)\n"
            "- max_capacity_gb (float): capacity for get_usage (0 = disabled)\n"
            "- eviction (dict): optional, see L2AdapterConfigBase"
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GCSL2Adapter(L2AdapterInterface):
    """GCS-backed L2 adapter.

    Concurrency model: one asyncio event loop on a dedicated daemon
    thread.  Each ``submit_*`` schedules a coroutine via
    ``run_coroutine_threadsafe``; that coroutine dispatches synchronous
    GCS SDK calls to a ``ThreadPoolExecutor`` via
    ``loop.run_in_executor``, awaits their futures with
    ``asyncio.gather``, then signals the corresponding eventfd.

    Locking: client-side refcount in ``_locked_keys``.  ``delete()``
    skips any key whose refcount is > 0 to prevent evicting an object a
    concurrent load is about to read.

    Circuit breaker: after ``max_connection_failures`` consecutive
    connection-class errors, ``_connection_disabled`` is set; all
    subsequent submits short-circuit and record failure without touching
    GCS.
    """

    max_connection_failures = 3

    def __init__(self, config: GCSL2AdapterConfig):
        super().__init__(max_capacity_bytes=int(config.max_capacity_gb * (1024**3)))
        self._config = config
        self._bucket_name = config.gcs_bucket

        try:
            # Third Party
            from google.cloud import storage as gcs_storage  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "GCSL2Adapter requires google-cloud-storage. "
                "Install it with: pip install google-cloud-storage"
            ) from e

        if config.gcs_credentials_file:
            try:
                # Third Party
                from google.oauth2 import service_account as gcs_sa  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(
                    "GCSL2Adapter requires google-auth when "
                    "gcs_credentials_file is set. "
                    "Install it with: pip install google-auth"
                ) from e
            logger.info(
                "GCSL2Adapter using service-account credentials: %s",
                config.gcs_credentials_file,
            )
            credentials = gcs_sa.Credentials.from_service_account_file(
                config.gcs_credentials_file
            )
            self._gcs_client = gcs_storage.Client(
                project=config.gcs_project,
                credentials=credentials,
            )
        else:
            logger.info("GCSL2Adapter using Application Default Credentials")
            self._gcs_client = gcs_storage.Client(project=config.gcs_project)

        self._bucket = self._gcs_client.bucket(self._bucket_name)

        self._executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=config.gcs_num_workers,
                thread_name_prefix="gcs-l2-adapter",
            )
        )

        # 3 distinct cross-platform notifiers for the L2 interface.
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        # Refcounted locks (like NativeConnectorL2Adapter).
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)

        # Per-key size map — retained so ``delete`` can recover each
        # key's stored size and pass it to ``_notify_keys_deleted``.
        self._key_sizes: dict[ObjectKey, int] = {}

        # Cached object sizes (keyed by blob name).
        self._object_size_cache: dict[str, int] = {}

        # Circuit breaker state.
        self._connection_failures = 0
        self._connection_disabled = False

        self._lock = threading.Lock()

        # Background asyncio event loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="gcs-l2-adapter-loop",
        )
        self._loop_thread.start()

        self._closed = False

        logger.info(
            "Initialized GCSL2Adapter (bucket=%s num_workers=%d max_capacity_gb=%.2f)",
            self._bucket_name,
            config.gcs_num_workers,
            config.max_capacity_gb,
        )

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Return the store-completion eventfd.

        Returns:
            int: file descriptor signalled on store task completion.
        """
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the lookup-completion eventfd.

        Returns:
            int: file descriptor signalled on lookup task completion.
        """
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the load-completion eventfd.

        Returns:
            int: file descriptor signalled on load task completion.
        """
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a store task to write a batch of memory objects to GCS.

        Args:
            keys: Keys identifying the objects to store.
            objects: Memory objects to write; parallel to ``keys``.

        Returns:
            L2TaskId: task identifier for polling via
            ``pop_completed_store_tasks``.
        """
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_store_tasks[task_id] = L2StoreResult(False, 0)
                disabled = True
            else:
                disabled = False

        if disabled:
            self._store_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_store(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Pop and return all completed store task results.

        Returns:
            dict mapping task id to ``L2StoreResult``.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: "MemoryLayoutDesc"
    ) -> L2TaskId:
        """Submit a lookup-and-lock task for a batch of keys.

        Args:
            keys: Keys to look up in GCS.
            layout_desc: Memory layout descriptor (unused by GCS adapter but
                required by L2AdapterInterface).

        Returns:
            L2TaskId: task identifier for polling via
            ``query_lookup_and_lock_result``.
        """
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_lookup_tasks[task_id] = Bitmap(len(keys))
                disabled = True
            else:
                disabled = False

        if disabled:
            self._lookup_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(list(keys), task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        """Non-blockingly retrieve and consume the result of a lookup task.

        Args:
            task_id: The task id returned by ``submit_lookup_and_lock_task``.

        Returns:
            Bitmap (1 = found, 0 = not found) or ``None`` if not yet complete.
        """
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Decrement the refcount lock for each key; allow eviction when zero.

        Args:
            keys: Keys to unlock.
        """
        with self._lock:
            for key in keys:
                if key not in self._locked_keys:
                    continue
                if self._locked_keys[key] <= 1:
                    del self._locked_keys[key]
                else:
                    self._locked_keys[key] -= 1

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a load task to read objects from GCS into memory buffers.

        Args:
            keys: Keys identifying the objects to load.
            objects: Pre-allocated memory buffers; parallel to ``keys``.

        Returns:
            L2TaskId: task identifier for polling via ``query_load_result``.
        """
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_load_tasks[task_id] = Bitmap(len(keys))
                disabled = True
            else:
                disabled = False

        if disabled:
            self._load_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_load(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        """Non-blockingly retrieve and consume the result of a load task.

        Args:
            task_id: The task id returned by ``submit_load_task``.

        Returns:
            Bitmap (1 = loaded, 0 = failed) or ``None`` if not yet complete.
        """
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete a batch of objects from GCS, skipping locked keys.

        Args:
            keys: Keys of objects to delete.
        """
        if not keys:
            return

        with self._lock:
            if self._connection_disabled:
                return
            deletable = [k for k in keys if self._locked_keys.get(k, 0) == 0]

        if not deletable:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._execute_delete(deletable),
            self._loop,
        )
        try:
            deleted_keys, deleted_sizes = fut.result(timeout=30.0)
        except Exception as e:
            logger.warning("GCSL2Adapter delete failed: %s", e)
            return

        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``.

    # ------------------------------------------------------------------
    # Status / Cleanup
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """Return a health and usage snapshot for this adapter.

        Returns:
            dict with ``is_healthy``, ``type``, ``bucket``,
            ``connection_failures``, ``connection_disabled``,
            ``current_size_bytes``, and ``max_capacity_bytes`` keys.
        """
        with self._lock:
            failures = self._connection_failures
            disabled = self._connection_disabled
        usage = self.get_usage()
        return {
            "is_healthy": self._loop_thread.is_alive() and not disabled,
            "type": "GCSL2Adapter",
            "bucket": self._bucket_name,
            "connection_failures": failures,
            "connection_disabled": disabled,
            "current_size_bytes": usage.total_bytes_used,
            "max_capacity_bytes": usage.total_capacity_bytes,
        }

    def close(self) -> None:
        """Shut down the adapter and release all resources."""
        if self._closed:
            return
        self._closed = True

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
            try:
                asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop).result(
                    timeout=5
                )
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join(timeout=5)
        try:
            self._loop.close()
        except Exception:
            pass

        self._executor.shutdown(wait=False)

        if self._gcs_client is not None:
            try:
                self._gcs_client.close()
            except Exception:
                pass
            self._gcs_client = None

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

        logger.info("GCSL2Adapter closed")

    # ------------------------------------------------------------------
    # Internal: event loop
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _record_connection_outcome(self, error_msg: Optional[str]) -> None:
        """Update the circuit breaker state under the lock.

        Args:
            error_msg: ``None`` on success; the error message string on
                failure.  Only connection-class errors advance the
                failure counter.
        """
        with self._lock:
            if error_msg is None:
                if self._connection_failures > 0:
                    logger.info("GCSL2Adapter connection recovered")
                self._connection_failures = 0
                return
            if not _is_connection_error(error_msg):
                return
            self._connection_failures += 1
            logger.error(
                "GCSL2Adapter connection error (%d/%d): %s",
                self._connection_failures,
                self.max_connection_failures,
                error_msg,
            )
            if self._connection_failures >= self.max_connection_failures:
                self._connection_disabled = True
                logger.error(
                    "GCSL2Adapter disabled after %d consecutive connection failures",
                    self.max_connection_failures,
                )

    # ------------------------------------------------------------------
    # Internal: synchronous GCS operations (run in thread pool)
    # ------------------------------------------------------------------

    def _sync_put(self, blob_name: str, data: bytes) -> int:
        """Upload *data* to GCS blob *blob_name* and return its byte length.

        Args:
            blob_name: GCS object name (within the configured bucket).
            data: Raw bytes to upload.

        Returns:
            Number of bytes uploaded.

        Raises:
            Exception: Propagated from the GCS client on failure.
        """
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data, content_type="application/octet-stream")
        return len(data)

    def _sync_head(self, blob_name: str) -> Optional[int]:
        """Return the byte size of a GCS blob, or ``None`` if not found.

        Args:
            blob_name: GCS object name to probe.

        Returns:
            Blob size in bytes, or ``None`` if the object does not exist.
        """
        blob = self._bucket.get_blob(blob_name)
        if blob is None:
            return None
        return blob.size

    def _sync_get(self, blob_name: str) -> bytes:
        """Download and return the contents of a GCS blob.

        Args:
            blob_name: GCS object name to download.

        Returns:
            Raw bytes of the blob.

        Raises:
            Exception: Propagated from the GCS client on failure.
        """
        return self._bucket.blob(blob_name).download_as_bytes()

    def _sync_delete(self, blob_name: str) -> None:
        """Delete a GCS blob; silently ignores not-found errors.

        Args:
            blob_name: GCS object name to delete.
        """
        try:
            self._bucket.blob(blob_name).delete()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal: coroutines
    # ------------------------------------------------------------------

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        futures = []
        indexed = []
        for key, obj in zip(keys, objects, strict=True):
            blob_name = _object_key_to_string(key)
            # Copy to bytes on the event-loop thread before handing off so
            # the MemoryObj buffer is not read concurrently from the
            # thread pool while the L2 controller may reclaim it.
            data = bytes(memoryview(obj.byte_array))
            size = obj.get_size()
            fut = self._loop.run_in_executor(
                self._executor, self._sync_put, blob_name, data
            )
            futures.append(fut)
            indexed.append((key, blob_name, size))

        results = await asyncio.gather(*futures, return_exceptions=True)

        success = True
        newly_stored_keys: list[ObjectKey] = []
        newly_stored_sizes: list[int] = []
        last_error: Optional[str] = None

        for (key, blob_name, size), result in zip(indexed, results, strict=True):
            if isinstance(result, Exception):
                success = False
                last_error = str(result)
                continue
            with self._lock:
                is_new = key not in self._key_sizes
                self._key_sizes[key] = size
                self._object_size_cache[blob_name] = size
            if is_new:
                newly_stored_keys.append(key)
                newly_stored_sizes.append(size)

        self._record_connection_outcome(last_error if not success else None)

        bytes_transferred = sum(newly_stored_sizes)
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )

        if newly_stored_keys:
            self._notify_keys_stored(newly_stored_keys, newly_stored_sizes)
        self._store_efd.notify()

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        futures: list[asyncio.Future[Optional[int]] | None] = []
        blob_names: list[str] = []
        cache_hits: list[Optional[int]] = []

        with self._lock:
            for key in keys:
                blob_name = _object_key_to_string(key)
                blob_names.append(blob_name)
                cache_hits.append(self._object_size_cache.get(blob_name))

        for blob_name, cached_size in zip(blob_names, cache_hits, strict=True):
            if cached_size is not None:
                futures.append(None)
            else:
                fut = self._loop.run_in_executor(
                    self._executor, self._sync_head, blob_name
                )
                futures.append(fut)

        real_futures = [f for f in futures if f is not None]
        real_results = await asyncio.gather(*real_futures, return_exceptions=True)
        real_iter = iter(real_results)
        resolved: list = []
        for f, cached_size in zip(futures, cache_hits, strict=True):
            if f is None:
                resolved.append(cached_size)
            else:
                resolved.append(next(real_iter))

        last_error: Optional[str] = None
        any_success = False

        for i, (key, blob_name, size_or_err) in enumerate(
            zip(keys, blob_names, resolved, strict=True)
        ):
            if isinstance(size_or_err, Exception):
                last_error = str(size_or_err)
                continue
            if size_or_err is None:
                continue
            size = int(size_or_err)
            if size > 0:
                bitmap.set(i)
                any_success = True
                with self._lock:
                    self._object_size_cache[blob_name] = size
                    self._locked_keys[key] += 1

        if any_success:
            self._record_connection_outcome(None)
        elif last_error is not None:
            self._record_connection_outcome(last_error)

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._lookup_efd.notify()

        accessed = [keys[i] for i in range(len(keys)) if bitmap.test(i)]
        if accessed:
            self._notify_keys_accessed(accessed)

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        futures = []
        launched: list[tuple[int, MemoryObj]] = []

        for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
            blob_name = _object_key_to_string(key)
            fut = self._loop.run_in_executor(self._executor, self._sync_get, blob_name)
            futures.append(fut)
            launched.append((i, obj))

        results = await asyncio.gather(*futures, return_exceptions=True)
        last_error: Optional[str] = None
        any_success = False

        for (idx, obj), result in zip(launched, results, strict=True):
            if isinstance(result, BaseException):
                last_error = str(result)
                continue
            data: bytes = result
            ctypes.memmove(obj.data_ptr, data, len(data))
            bitmap.set(idx)
            any_success = True

        if any_success:
            self._record_connection_outcome(None)
        elif last_error is not None:
            self._record_connection_outcome(last_error)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()

    async def _execute_delete(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[int]]:
        """Run DELETE for each key and remove its size-tracking entry.

        Args:
            keys: Keys to delete from GCS.

        Returns:
            Parallel lists of successfully deleted keys and their sizes.
        """
        futures = []
        indexed: list[tuple[ObjectKey, str]] = []
        for key in keys:
            blob_name = _object_key_to_string(key)
            fut = self._loop.run_in_executor(
                self._executor, self._sync_delete, blob_name
            )
            futures.append(fut)
            indexed.append((key, blob_name))

        results = await asyncio.gather(*futures, return_exceptions=True)
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []

        for (key, blob_name), result in zip(indexed, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "GCSL2Adapter DELETE failed for %s: %s", blob_name, result
                )
                continue
            with self._lock:
                sz = self._key_sizes.pop(key, None)
                self._object_size_cache.pop(blob_name, None)
            deleted_keys.append(key)
            deleted_sizes.append(sz if sz is not None else 0)

        return deleted_keys, deleted_sizes


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_l2_adapter_type("gcs", GCSL2AdapterConfig)


def _create_gcs_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    return GCSL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("gcs", _create_gcs_adapter)
