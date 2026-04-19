# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import json
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    CacheEngineKey,
    DiskCacheMetadata,
    TORCH_DTYPE_TO_STR_DTYPE,
    _lmcache_nvtx_annotate,
    parse_cache_key,
)
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.path_sharder import PathSharder

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def _build_disk_cache_metadata(
    path: str,
    size: int,
    shape: Optional[torch.Size],
    dtype: Optional[torch.dtype],
    fmt: Optional[MemoryFormat],
    cached_positions: Optional[torch.Tensor],
    shapes: Optional[list[torch.Size]],
    dtypes: Optional[list[torch.dtype]],
    *,
    created_ts: float,
    last_access_ts: float,
    hit_count: int,
) -> DiskCacheMetadata:
    """Construct persisted disk-cache metadata for an entry."""
    return DiskCacheMetadata(
        path=path,
        size=size,
        shape=shape,
        dtype=dtype,
        cached_positions=cached_positions,
        fmt=fmt,
        pin_count=0,
        shapes=shapes,
        dtypes=dtypes,
        created_ts=created_ts,
        last_access_ts=last_access_ts,
        hit_count=max(1, int(hit_count or 1)),
    )


def _clone_disk_cache_metadata(metadata: DiskCacheMetadata) -> DiskCacheMetadata:
    """Return a detached copy of persisted metadata for lock-free updates."""
    return replace(metadata)


def _has_same_persisted_state(
    current: DiskCacheMetadata, snapshot: DiskCacheMetadata
) -> bool:
    """Check whether the persisted fields still match a saved snapshot."""
    return (
        current.created_ts == snapshot.created_ts
        and current.last_access_ts == snapshot.last_access_ts
        and current.hit_count == snapshot.hit_count
    )


# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class LocalDiskWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        # TODO(Jiayi): make executor and its parameters configurable
        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=4)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
            # self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return await self.executor.submit_job(
            task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)


class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None
        sharder = PathSharder(
            raw_csv=config.local_disk,
            strategy=config.local_disk_path_sharding,
            dst_device=dst_device,
            create_dirs=True,
        )
        self.local_disk_path_sharding = sharder.strategy
        self.path: str = sharder.selected
        logger.info(
            "Local disk cache path: %s (device %s, %d path(s) configured)",
            self.path,
            dst_device,
            len(sharder.all_paths),
        )

        self.loop = loop

        self.use_local_cpu = config.local_cpu
        self.metadata = metadata

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        self.disk_worker = LocalDiskWorker(loop)

        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        # Batched message sender for controller communication
        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        # Initialize batched message sender
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

        self._recover_persisted_index()

    def __str__(self) -> str:
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def _meta_path_from_data_path(self, path: str) -> str:
        return path + ".meta.json"

    def _key_to_meta_path(self, key: CacheEngineKey) -> str:
        return self._meta_path_from_data_path(self._key_to_path(key))

    def _serialize_cached_positions(
        self, cached_positions: Optional[torch.Tensor]
    ) -> Optional[list[int]]:
        if cached_positions is None:
            return None
        return cached_positions.detach().cpu().tolist()

    def _serialize_dtype(self, dtype: Optional[torch.dtype]) -> Optional[str]:
        if dtype is None:
            return None
        return TORCH_DTYPE_TO_STR_DTYPE[dtype]

    def _deserialize_dtype(self, dtype_str: Optional[str]) -> Optional[torch.dtype]:
        if dtype_str is None:
            return None
        return STR_DTYPE_TO_TORCH_DTYPE[dtype_str]

    def _serialize_disk_metadata(
        self,
        key: CacheEngineKey,
        metadata: DiskCacheMetadata,
    ) -> dict:
        """Convert disk metadata into the JSON payload stored beside the tensor."""
        return {
            "key": key.to_string(),
            "size": metadata.size,
            "shape": list(metadata.shape) if metadata.shape is not None else None,
            "dtype": self._serialize_dtype(metadata.dtype),
            "fmt": metadata.fmt.value if metadata.fmt is not None else None,
            "cached_positions": self._serialize_cached_positions(
                metadata.cached_positions
            ),
            "shapes": [list(s) for s in metadata.shapes]
            if metadata.shapes is not None
            else None,
            "dtypes": [self._serialize_dtype(d) for d in metadata.dtypes]
            if metadata.dtypes is not None
            else None,
            "created_ts": metadata.created_ts,
            "last_access_ts": metadata.last_access_ts,
            "hit_count": metadata.hit_count,
        }

    def _persist_sidecar_for_metadata(
        self,
        key: CacheEngineKey,
        metadata: DiskCacheMetadata,
    ) -> None:
        """Atomically persist a metadata sidecar for the given cache entry."""
        meta_path = self._key_to_meta_path(key)
        payload = self._serialize_disk_metadata(key, metadata)
        tmp_meta_path = meta_path + ".tmp"
        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_meta_path, meta_path)

    def _record_hit_locked(
        self,
        key: CacheEngineKey,
    ) -> DiskCacheMetadata:
        """Update in-memory hit metadata and return a persistence snapshot.

        This method must be called while ``disk_lock`` is held.
        """
        metadata = self.dict[key]
        metadata.hit_count = max(1, int(metadata.hit_count or 1)) + 1
        metadata.last_access_ts = time.time()
        self.cache_policy.update_on_hit(key, self.dict)
        return _clone_disk_cache_metadata(metadata)

    def _recover_persisted_index(self) -> None:
        """Rebuild the in-memory disk index and cache policy state from sidecars."""
        recovered = 0
        total_bytes = 0
        evicted_recovered = 0
        evicted_bytes = 0
        recovered_entries: list[tuple[CacheEngineKey, DiskCacheMetadata]] = []

        for entry in os.scandir(self.path):
            if not entry.name.endswith(".meta.json"):
                continue

            meta_path = entry.path
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                key = parse_cache_key(payload["key"])
                path = self._key_to_path(key)
                if not os.path.exists(path):
                    logger.warning("Skipping stale disk metadata without data file: %s", meta_path)
                    continue

                size = os.path.getsize(path)
                shape = (
                    torch.Size(payload["shape"])
                    if payload.get("shape") is not None
                    else None
                )
                dtype = self._deserialize_dtype(payload.get("dtype"))
                fmt = (
                    MemoryFormat(payload["fmt"])
                    if payload.get("fmt") is not None
                    else None
                )
                cached_positions = payload.get("cached_positions")
                shapes = payload.get("shapes")
                dtypes = payload.get("dtypes")
                stored_created_ts = payload.get("created_ts")
                created_ts = (
                    float(stored_created_ts)
                    if stored_created_ts is not None
                    else os.path.getmtime(path)
                )
                stored_last_access_ts = payload.get("last_access_ts")
                last_access_ts = (
                    float(stored_last_access_ts)
                    if stored_last_access_ts is not None
                    else created_ts
                )
                hit_count = max(1, int(payload.get("hit_count", 1) or 1))

                recovered_entries.append(
                    (
                        key,
                        DiskCacheMetadata(
                            path=path,
                            size=size,
                            shape=shape,
                            dtype=dtype,
                            cached_positions=(
                                torch.tensor(cached_positions, dtype=torch.long)
                                if cached_positions is not None
                                else None
                            ),
                            fmt=fmt,
                            pin_count=0,
                            shapes=[torch.Size(s) for s in shapes]
                            if shapes is not None
                            else None,
                            dtypes=[self._deserialize_dtype(d) for d in dtypes]
                            if dtypes is not None
                            else None,
                            created_ts=created_ts,
                            last_access_ts=last_access_ts,
                            hit_count=hit_count,
                        ),
                    )
                )
                total_bytes += size
                recovered += 1
            except Exception as e:
                logger.warning(
                    "Failed to recover disk cache metadata %s: %s",
                    meta_path,
                    e,
                )

        if recovered_entries:
            recovered_entries.sort(
                key=lambda item: (
                    self.cache_policy.get_recovery_sort_key(item[1]),
                    item[0].to_string(),
                )
            )
            for key, metadata in recovered_entries:
                self.dict[key] = metadata
                self.cache_policy.restore_on_recover(key, self.dict, metadata)

        if recovered:
            recovered_after_eviction = recovered
            while total_bytes > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict,
                    num_candidates=1,
                )
                if not evict_keys:
                    logger.warning(
                        "Recovered disk cache exceeds configured capacity "
                        "(%d > %d bytes), but no eviction candidates were found.",
                        total_bytes,
                        self.max_cache_size,
                    )
                    break

                for evict_key in evict_keys:
                    metadata = self.dict.pop(evict_key, None)
                    if metadata is None:
                        continue

                    self.cache_policy.update_on_force_evict(evict_key)
                    total_bytes -= metadata.size
                    evicted_recovered += 1
                    evicted_bytes += metadata.size
                    recovered_after_eviction -= 1

                    if os.path.exists(metadata.path):
                        os.remove(metadata.path)

                    meta_path = self._meta_path_from_data_path(metadata.path)
                    if os.path.exists(meta_path):
                        os.remove(meta_path)

            self.current_cache_size = total_bytes
            self.usage = total_bytes
            self.stats_monitor.update_local_storage_usage(self.usage)
            logger.info(
                "Recovered %d persisted disk cache entries (%d bytes) from %s",
                recovered_after_eviction,
                total_bytes,
                self.path,
            )
            if evicted_recovered:
                logger.info(
                    "Evicted %d recovered disk cache entries (%d bytes) to fit "
                    "the configured disk cache size (%d bytes)",
                    evicted_recovered,
                    evicted_bytes,
                    self.max_cache_size,
                )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        metadata_snapshots: list[tuple[CacheEngineKey, DiskCacheMetadata]] = []
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                metadata_snapshots.append((key, self._record_hit_locked(key)))
            self.keys_in_request = []
        for key, metadata in metadata_snapshots:
            self._submit_metadata_persist(key, metadata)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.disk_worker.exists_in_put_tasks(key)

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        if force:
            self.disk_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.disk_lock.release()
            return False

        path = meta.path
        meta_path = self._meta_path_from_data_path(path)
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # NOTE: The following code will cause deadlock
        # res = asyncio.run_coroutine_threadsafe(
        #     self.disk_worker.submit_task("delete", os.remove, path),
        #     self.loop,
        # )
        # res.result()

        os.remove(path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        # Push kv evict msg with batching
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
        shapes: Optional[list[torch.Size]] = None,
        dtypes: Optional[list[torch.dtype]] = None,
        *,
        created_ts: Optional[float] = None,
        last_access_ts: Optional[float] = None,
        hit_count: int = 1,
        persist_sidecar: bool = True,
    ) -> None:
        path = self._key_to_path(key)
        created_at = float(created_ts if created_ts is not None else time.time())
        last_access_at = float(
            last_access_ts if last_access_ts is not None else created_at
        )

        has_stored = False
        metadata_to_persist: Optional[DiskCacheMetadata] = None
        with self.disk_lock:
            if key in self.dict:
                # Update cache recency
                metadata_to_persist = self._record_hit_locked(key)
                has_stored = True
            else:
                self.dict[key] = _build_disk_cache_metadata(
                    path=path,
                    size=size,
                    shape=shape,
                    dtype=dtype,
                    fmt=fmt,
                    cached_positions=cached_positions,
                    shapes=shapes,
                    dtypes=dtypes,
                    created_ts=created_at,
                    last_access_ts=last_access_at,
                    hit_count=hit_count,
                )
                if persist_sidecar:
                    metadata_to_persist = _clone_disk_cache_metadata(self.dict[key])

        if metadata_to_persist is not None and persist_sidecar:
            if has_stored:
                self._submit_metadata_persist(key, metadata_to_persist)
            else:
                self._persist_sidecar_for_metadata(key, metadata_to_persist)

        # Push kv admit msg with batching
        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.ADMIT,
                key=key.chunk_hash,
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        """
        Submit a single put task to store KV cache to disk asynchronously.

        :param key: The cache key for this KV chunk.
        :param memory_obj: The memory object containing the KV data.
        :param on_complete_callback: Optional callback invoked once per key
            after the disk write completes. Callback exceptions are caught
            and logged.
        """
        # skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.disk_worker.insert_put_task(key)

        # TODO(Jiayi): Fragmentation is not considered here.
        required_size = memory_obj.get_physical_size()
        all_evict_keys = []
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)

                all_evict_keys.extend(evict_keys)
            if evict_success:
                self.current_cache_size += required_size
                self.cache_policy.update_on_put(key)

        if not evict_success:
            return None

        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.disk_worker.submit_task(
                "put",
                self.async_save_bytes_to_disk,
                key=key,
                memory_obj=memory_obj,
                on_complete_callback=on_complete_callback,
            ),
            self.loop,
        )

    # TODO(Jiayi): enable real batching
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to disk asynchronously.

        :param keys: The cache keys for the KV chunks.
        :param memory_objs: The memory objects containing the KV data.
        :param transfer_spec: Optional transfer specification (unused).
        :param on_complete_callback: Optional callback invoked once per key
            after that key's disk write completes (not once per batch).
            Callback exceptions are caught and logged.
        """
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        # Update cache recency
        metadata_to_persist = self._record_hit_locked(key)

        disk_meta = self.dict[key]
        path = disk_meta.path
        dtype = disk_meta.dtype
        shape = disk_meta.shape
        dtypes = disk_meta.dtypes
        shapes = disk_meta.shapes
        fmt = disk_meta.fmt
        assert dtype is not None
        assert shape is not None

        self.disk_lock.release()
        self._submit_metadata_persist(key, metadata_to_persist)
        memory_obj = self.load_bytes_from_disk(
            key,
            path,
            dtype=dtype,
            shape=shape,
            fmt=fmt,
            dtypes=dtypes,
            shapes=shapes,
        )

        return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs: list[MemoryObj] = []
        paths: list[str] = []

        logger.debug(f"lookup_id: {lookup_id}; Prefetching {len(keys)} keys from disk.")
        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, f"Key {key} not found in disk cache after pinning"

            path = self.dict[key].path
            dtype = self.dict[key].dtype
            shape = self.dict[key].shape
            dtypes = self.dict[key].dtypes
            shapes = self.dict[key].shapes
            fmt = self.dict[key].fmt

            assert dtype is not None
            assert shape is not None

            # busy_loop=False prevents spinning on the event loop thread;
            # if staging memory is exhausted the caller will get a logged
            # error rather than a silent deadlock.
            memory_obj = self.local_cpu_backend.allocate(
                shapes if shapes is not None and dtypes is not None else shape,
                dtypes if shapes is not None and dtypes is not None else dtype,
                fmt,
                busy_loop=False,
            )

            if memory_obj is None:
                logger.error(
                    "Memory allocation failed during async disk load for key %s. "
                    "CPU staging pool may be exhausted (unpin() not called after "
                    "a previous retrieve). Returning partial results.",
                    key,
                )
                return mem_objs

            self.dict[key].pin()

            # NOTE(Jiayi): Currently, we consider prefetch as cache hit.
            # Update cache recency
            metadata_to_persist = self._record_hit_locked(key)

            self.disk_lock.release()
            self._submit_metadata_persist(key, metadata_to_persist)
            logger.debug(f"Prefetching {key} from disk.")
            memory_obj.pin()
            mem_objs.append(memory_obj)
            paths.append(path)

        return await self.disk_worker.submit_task(
            "prefetch",
            self.batched_async_load_bytes_from_disk,
            paths=paths,
            keys=keys,
            memory_objs=mem_objs,
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.

        :param on_complete_callback: Optional callback invoked after the disk
            write completes for this key. Callback exceptions are caught and
            logged.
        """
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # TODO(Jiayi): need to add ref count in disk memory object
        self.write_file(buffer, path)

        # ref count down here because there's a ref_count_up in
        # `submit_put_task` above.
        # Ref count down better be before `insert_key` for testing
        # purposes (e.g., testing mem_leak).
        # TODO(Jiayi): This could be problematic if the
        # freed memory object is immediately reused.
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        cached_positions = memory_obj.metadata.cached_positions
        shapes = memory_obj.metadata.shapes
        dtypes = memory_obj.metadata.dtypes
        memory_obj.ref_count_down()
        created_at = time.time()
        last_access_at = created_at
        initial_hit_count = 1

        self.insert_key(
            key,
            size,
            shape,
            dtype,
            fmt,
            cached_positions=cached_positions,
            shapes=shapes,
            dtypes=dtypes,
            created_ts=created_at,
            last_access_ts=last_access_at,
            hit_count=initial_hit_count,
            persist_sidecar=True,
        )

        self.disk_worker.remove_put_task(key)

        # Call the completion callback if provided
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"on_complete_callback failed for key {key}: {e}")

    def batched_async_load_bytes_from_disk(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        write_back: bool = False,
    ) -> list[MemoryObj]:
        """
        Async load bytearray from disk.
        """

        logger.debug("Executing `async_load_bytes` from disk.")
        # TODO (Jiayi): handle the case where loading fails.
        for path, key, mem_obj in zip(paths, keys, memory_objs, strict=False):
            buffer = mem_obj.byte_array
            self.read_file(key, buffer, path)

            # TODO(Jiayi): Please recover the metadata in a more
            # elegant way in the future.
            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions

            self.disk_lock.acquire()
            self.dict[key].unpin()
            self.disk_lock.release()

        return memory_objs

    def load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
        dtypes: Optional[list[torch.dtype]] = None,
        shapes: Optional[list[torch.Size]] = None,
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """

        memory_obj = self.local_cpu_backend.allocate(
            shapes if shapes is not None and dtypes is not None else shape,
            dtypes if shapes is not None and dtypes is not None else dtype,
            fmt,
            busy_loop=False,
        )
        if memory_obj is None:
            logger.warning(
                "Memory allocation failed during blocking disk load for key %s. "
                "CPU staging pool may be exhausted; treating this as a cache miss.",
                key,
            )
            return None

        buffer = memory_obj.byte_array
        self.read_file(key, buffer, path)

        # TODO(Jiayi): Please recover the metadata in a more
        # elegant way in the future.
        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions

        return memory_obj

    def write_file(self, buffer, path):
        start_time = time.time()
        size = len(buffer)
        if size % self.os_disk_bs != 0 or not self.use_odirect:
            with open(path, "wb") as f:
                f.write(buffer)
        else:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            os.write(fd, buffer)
            os.close(fd)
        disk_write_time = time.time() - start_time
        logger.debug(
            f"Disk write size: {size} bytes, "
            f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
        )

    def read_file(self, key, buffer, path):
        start_time = time.time()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to disk block size."
            )

        try:
            if not fblock_aligned or not self.use_odirect:
                with open(path, "rb") as f:
                    f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    fdo.readinto(buffer)
        except FileNotFoundError:
            logger.warning(f"File not found on disk: {path}")
            if self.dict.get(key, None):
                self.dict.pop(key)
            return

        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self.disk_worker.close()

    def _persist_metadata_snapshot_if_current(
        self,
        key: CacheEngineKey,
        metadata_snapshot: DiskCacheMetadata,
    ) -> None:
        """Persist metadata only if no newer in-memory state superseded it."""
        with self.disk_lock:
            current_metadata = self.dict.get(key)
            if current_metadata is None:
                return
            if not _has_same_persisted_state(current_metadata, metadata_snapshot):
                return
            latest_metadata = _clone_disk_cache_metadata(current_metadata)

        self._persist_sidecar_for_metadata(key, latest_metadata)

    def _submit_metadata_persist(
        self,
        key: CacheEngineKey,
        metadata_snapshot: DiskCacheMetadata,
    ) -> None:
        """Persist a metadata snapshot after the caller releases ``disk_lock``."""
        self._persist_metadata_snapshot_if_current(key, metadata_snapshot)
