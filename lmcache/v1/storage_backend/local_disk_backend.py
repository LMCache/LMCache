# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
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


def _get_available_space_bytes(path: str) -> int:
    """Return the currently usable filesystem space for ``path`` in bytes."""
    stat = os.statvfs(path)
    fragment_size = stat.f_frsize if stat.f_frsize > 0 else stat.f_bsize
    return stat.f_bavail * fragment_size


def _get_filesystem_id(path: str) -> int:
    """Return the device id for the filesystem backing ``path``."""
    return os.stat(path).st_dev


def _split_evenly(total_bytes: int, num_parts: int) -> list[int]:
    """Split ``total_bytes`` across ``num_parts`` nearly-equal integer quotas."""
    base_quota = total_bytes // num_parts
    remainder = total_bytes % num_parts
    return [base_quota + (1 if idx < remainder else 0) for idx in range(num_parts)]


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
        dst_device: str = torch_device_type,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        if torch_dev.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.dict: dict[CacheEngineKey, DiskCacheMetadata] = {}

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None

        # Pass local-worker metadata so that, when more disks than workers
        # are configured, this worker is assigned a contiguous subset of
        # paths instead of a single one. When metadata is absent the
        # sharder falls back to the legacy device-index single-path mapping.
        local_rank = metadata.local_worker_id if metadata is not None else None
        local_world_size = metadata.local_world_size if metadata is not None else None

        sharder = PathSharder(
            raw_csv=config.local_disk,
            strategy=config.local_disk_path_sharding,
            dst_device=dst_device,
            local_rank=local_rank,
            local_world_size=local_world_size,
            create_dirs=True,
        )
        self.path_sharder = sharder
        self.assigned_paths: list[str] = sharder.assigned_paths
        self.path: str = sharder.selected

        logger.info(
            "Local disk cache strategy=%s device=%s "
            "configured_paths=%d assigned_paths=%s",
            sharder.strategy,
            dst_device,
            len(sharder.all_paths),
            self.assigned_paths,
        )

        self.loop = loop

        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O), per assigned path so that
        # multi-path workers use the correct block size for each file.
        self.path_block_sizes = {
            path: os.statvfs(path).f_bsize for path in self.assigned_paths
        }
        self.os_disk_bs = self.path_block_sizes[self.path]
        self.path_max_cache_sizes = dict(
            zip(
                self.assigned_paths,
                _split_evenly(
                    int(config.max_local_disk_size * 1024**3),
                    len(self.assigned_paths),
                ),
                strict=True,
            )
        )
        self.path_current_cache_sizes = {path: 0 for path in self.assigned_paths}
        self.path_cache_policies = {
            path: get_cache_policy(config.cache_policy) for path in self.assigned_paths
        }
        self.path_dicts = {
            path: policy.init_mutable_mapping()
            for path, policy in self.path_cache_policies.items()
        }
        self.keys_in_request_by_path: dict[str, List[CacheEngineKey]] = {
            path: [] for path in self.assigned_paths
        }
        self._validate_assigned_path_capacity()
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        self.disk_worker = LocalDiskWorker(loop)

        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = sum(self.path_max_cache_sizes.values())
        self.current_cache_size = 0

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

    def __str__(self) -> str:
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        base_path = self.path_sharder.resolve_path_for_key(key)
        return os.path.join(base_path, key.to_string().replace("/", "-") + ".pt")

    def _get_disk_block_size(self, path: str) -> int:
        # path is always produced by _key_to_path(), which places files
        # directly under an assigned mount point with no subdirectories.
        root_path = os.path.dirname(path)
        return self.path_block_sizes.get(root_path, self.os_disk_bs)

    def _validate_assigned_path_capacity(self) -> None:
        """Ensure assigned filesystems have enough currently available space."""
        quotas_by_filesystem: dict[int, tuple[str, int, int]] = {}
        for path, quota_bytes in self.path_max_cache_sizes.items():
            filesystem_id = _get_filesystem_id(path)
            available_bytes = _get_available_space_bytes(path)
            sample_path, current_required, _ = quotas_by_filesystem.get(
                filesystem_id,
                (path, 0, available_bytes),
            )
            quotas_by_filesystem[filesystem_id] = (
                sample_path,
                current_required + quota_bytes,
                available_bytes,
            )

        for path, required_bytes, available_bytes in quotas_by_filesystem.values():
            if available_bytes < required_bytes:
                raise ValueError(
                    "Assigned local disk paths do not have enough available "
                    "filesystem space for their LMCache quota "
                    f"(path={path}, available_bytes={available_bytes}, "
                    f"required_bytes={required_bytes})"
                )

    def _get_path_for_key(self, key: CacheEngineKey) -> str:
        """Return the assigned base path for ``key`` on this worker."""
        return self.path_sharder.resolve_path_for_key(key)

    def _get_cache_path_for_key(self, key: CacheEngineKey) -> str:
        """Return the path-local cache selected for ``key``."""
        if meta := self.dict.get(key):
            return os.path.dirname(meta.path)
        return self._get_path_for_key(key)

    def _get_path_cache_policy(self, path: str):
        return self.path_cache_policies[path]

    def _get_path_cache_dict(self, path: str):
        return self.path_dicts[path]

    def _update_path_usage(self, path: str, size_delta: int) -> None:
        """Apply a signed usage delta to one assigned path and the global total."""
        self.path_current_cache_sizes[path] = (
            self.path_current_cache_sizes.get(path, 0) + size_delta
        )
        self.current_cache_size += size_delta

    def _get_evict_candidates_for_path(
        self,
        path: str,
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        """Return eviction candidates from one path-local cache."""
        return self._get_path_cache_policy(path).get_evict_candidates(
            self._get_path_cache_dict(path),
            num_candidates=num_candidates,
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                cache_path = self._get_cache_path_for_key(key)
                self.keys_in_request_by_path[cache_path].append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.disk_lock:
            for path, keys_in_request in self.keys_in_request_by_path.items():
                path_dict = self._get_path_cache_dict(path)
                path_policy = self._get_path_cache_policy(path)
                for key in reversed(keys_in_request):
                    if key in path_dict:
                        path_policy.update_on_hit(key, path_dict)
                self.keys_in_request_by_path[path] = []

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
        base_path = os.path.dirname(path)
        path_dict = self._get_path_cache_dict(base_path)
        path_dict.pop(key, None)
        size = meta.size
        self._update_path_usage(base_path, -size)
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # NOTE: The following code will cause deadlock
        # res = asyncio.run_coroutine_threadsafe(
        #     self.disk_worker.submit_task("delete", os.remove, path),
        #     self.loop,
        # )
        # res.result()

        if force:
            self._get_path_cache_policy(base_path).update_on_force_evict(key)
            self.disk_lock.release()

        try:
            os.remove(path)
        except FileNotFoundError:
            logger.warning(f"File already removed: {path}")
        except OSError as e:
            logger.error(f"Failed to remove file {path}: {e}")

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
    ) -> None:
        path = self._key_to_path(key)
        base_path = os.path.dirname(path)
        path_dict = self._get_path_cache_dict(base_path)
        path_policy = self._get_path_cache_policy(base_path)

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                # Update cache recency
                path_policy.update_on_hit(key, path_dict)
                has_stored = True
            else:
                meta = DiskCacheMetadata(
                    path, size, shape, dtype, cached_positions, fmt, 0
                )
                self.dict[key] = meta
                path_dict[key] = meta

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
        assert memory_obj.tensor is not None

        # skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.disk_worker.insert_put_task(key)

        # TODO(Jiayi): Fragmentation is not considered here.
        required_size = memory_obj.get_physical_size()
        target_path = self._get_path_for_key(key)
        path_quota = self.path_max_cache_sizes[target_path]
        evict_success = True
        with self.disk_lock:
            while (
                self.path_current_cache_sizes[target_path] + required_size > path_quota
            ):
                evict_keys = self._get_evict_candidates_for_path(
                    target_path,
                    num_candidates=1,
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found for local disk path under "
                        "pressure (path=%s, required_size=%s, quota=%s, current=%s).",
                        target_path,
                        required_size,
                        path_quota,
                        self.path_current_cache_sizes[target_path],
                    )
                    evict_success = False
                    break

                self.batched_remove(evict_keys, force=False)

            if evict_success:
                self._update_path_usage(target_path, required_size)
                self._get_path_cache_policy(target_path).update_on_put(key)

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
        Load a cached KV chunk from disk synchronously.

        The cache policy is updated only after a successful load so that a
        failed load (``load_bytes_from_disk`` returning ``None``) does not
        record a phantom cache hit and skew future eviction decisions.

        :param key: The cache key identifying the KV chunk.
        :returns: A ``MemoryObj`` containing the loaded KV data, or ``None``
            if the key is not present or the load fails.
        """
        with self.disk_lock:
            if key not in self.dict:
                return None

            disk_meta = self.dict[key]
            path = disk_meta.path
            dtype = disk_meta.dtype
            shape = disk_meta.shape
            fmt = disk_meta.fmt
            assert dtype is not None
            assert shape is not None
            assert fmt is not None

        # Load is performed outside the lock: it can block for a non-trivial
        # amount of time (CPU staging pool allocation + memcpy from disk) and
        # must not hold disk_lock while waiting, or concurrent insert/evict
        # operations would deadlock.
        memory_obj = self.load_bytes_from_disk(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

        if memory_obj is not None:
            # Re-acquire the lock to update the eviction policy.  The key
            # membership check guards against the entry being evicted between
            # the two lock regions — in that case the policy state is already
            # consistent and no update is needed.
            with self.disk_lock:
                if key in self.dict:
                    cache_path = self._get_cache_path_for_key(key)
                    path_dict = self._get_path_cache_dict(cache_path)
                    if key in path_dict:
                        self._get_path_cache_policy(cache_path).update_on_hit(
                            key,
                            path_dict,
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
            fmt = self.dict[key].fmt

            assert dtype is not None
            assert shape is not None

            # busy_loop=False prevents spinning on the event loop thread;
            # if staging memory is exhausted the caller will get a logged
            # error rather than a silent deadlock.
            memory_obj = self.local_cpu_backend.allocate(
                shape,
                dtype,
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
            cache_path = self._get_cache_path_for_key(key)
            self._get_path_cache_policy(cache_path).update_on_hit(
                key,
                self._get_path_cache_dict(cache_path),
            )

            self.disk_lock.release()
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
                    cache_path = self._get_cache_path_for_key(key)
                    self.keys_in_request_by_path[cache_path].append(key)
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
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
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
        memory_obj.ref_count_down()

        self.insert_key(key, size, shape, dtype, fmt, cached_positions=cached_positions)

        self.disk_worker.remove_put_task(key)

        # Call the completion callback if provided
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"on_complete_callback failed for key {key}: {e}")

    @_lmcache_nvtx_annotate
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
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during disk load."

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
        disk_block_size = self._get_disk_block_size(path)
        if size % disk_block_size != 0 or not self.use_odirect:
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

    @_lmcache_nvtx_annotate
    def read_file(self, key, buffer, path):
        start_time = time.time()
        size = len(buffer)
        disk_block_size = self._get_disk_block_size(path)
        fblock_aligned = size % disk_block_size == 0
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
            with self.disk_lock:
                if meta := self.dict.pop(key, None):
                    cache_path = os.path.dirname(meta.path)
                    self._get_path_cache_dict(cache_path).pop(key, None)
                    self._get_path_cache_policy(cache_path).update_on_force_evict(key)
                    self._update_path_usage(cache_path, -meta.size)
                    self.usage -= meta.size
                    self.stats_monitor.update_local_storage_usage(self.usage)
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
