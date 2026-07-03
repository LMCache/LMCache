# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
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

_DEFAULT_THREAD_COUNT = 4


@dataclass(frozen=True)
class _DiskPutItem:
    key: CacheEngineKey
    memory_obj: MemoryObj
    size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    cached_positions: Optional[torch.Tensor]


# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class LocalDiskWorker:
    def __init__(
        self, loop: asyncio.AbstractEventLoop, max_workers: int = _DEFAULT_THREAD_COUNT
    ) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=max_workers)
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
        self.remove_put_tasks([key])

    def remove_put_tasks(self, keys: Sequence[CacheEngineKey]) -> None:
        missing_keys = []
        with self.put_lock:
            for key in keys:
                if key in self.put_tasks:
                    self.put_tasks.remove(key)
                else:
                    missing_keys.append(key)

        for key in missing_keys:
            logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        self.insert_put_tasks([key])

    def insert_put_tasks(
        self, keys: Sequence[CacheEngineKey]
    ) -> list[CacheEngineKey]:
        inserted_keys = []
        with self.put_lock:
            for key in keys:
                if key in self.put_tasks:
                    continue
                self.put_tasks.append(key)
                inserted_keys.append(key)
        return inserted_keys

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
        self.path: str = sharder.selected

        logger.info(
            "Local disk cache path: %s (device %s, %d path(s) configured)",
            self.path,
            dst_device,
            len(sharder.all_paths),
        )

        self.loop = loop

        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        thread_count = config.get_extra_config_value(
            "disk_io_threads", _DEFAULT_THREAD_COUNT
        )
        self.disk_worker = LocalDiskWorker(loop, max_workers=thread_count)

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

    def __str__(self) -> str:
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

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
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

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
        lock_context = self.disk_lock if force else nullcontext()
        with lock_context:
            if not (meta := self.dict.pop(key, None)):
                return False

            path = meta.path
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

            if force:
                self.cache_policy.update_on_force_evict(key)

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

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                # Update cache recency
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path, size, shape, dtype, cached_positions, fmt, 0
                )

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

            if evict_success:
                self.current_cache_size += required_size
                self.cache_policy.update_on_put(key)

        if not evict_success:
            self.disk_worker.remove_put_task(key)
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

        Raises:
            ValueError: If ``keys`` and ``memory_objs`` do not have the same
                length.
        """
        put_items = self._prepare_batched_put_items(keys, memory_objs)
        if not put_items:
            return None

        for item in put_items:
            item.memory_obj.ref_count_up()

        coro = self.disk_worker.submit_task(
            "put",
            self.async_batched_save_bytes_to_disk,
            put_items=put_items,
            on_complete_callback=on_complete_callback,
        )
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        except Exception:
            coro.close()
            self._rollback_batched_put_reservation(put_items)
            raise

        return None

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
                    self.cache_policy.update_on_hit(key, self.dict)

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
            self.cache_policy.update_on_hit(key, self.dict)

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
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        put_item = self._build_disk_put_item(key, memory_obj)
        self._save_disk_put_item(
            put_item,
            on_complete_callback=on_complete_callback,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_batched_save_bytes_to_disk(
        self,
        put_items: list[_DiskPutItem],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Convert a batch of KV chunks to bytes and store them in one worker job.

        Args:
            put_items: Prepared disk put items whose capacity and in-flight
                task slots have already been reserved.
            on_complete_callback: Optional callback invoked once for each key
                after that key's disk write completes.
        """
        errors: list[Exception] = []
        for put_item in put_items:
            try:
                self._save_disk_put_item(
                    put_item,
                    on_complete_callback=on_complete_callback,
                )
            except Exception as e:
                errors.append(e)

        if errors:
            raise RuntimeError(
                f"Failed to write {len(errors)} local disk cache item(s)"
            ) from errors[0]

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

            with self.disk_lock:
                self.dict[key].unpin()

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

    @_lmcache_nvtx_annotate
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

    def _prepare_batched_put_items(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> list[_DiskPutItem]:
        if len(keys) != len(memory_objs):
            raise ValueError("keys and memory_objs must have the same length")

        indexed_items = []
        skipped_keys = []
        seen_keys = set()
        for key, memory_obj in zip(keys, memory_objs, strict=True):
            if key in seen_keys:
                skipped_keys.append(key)
                continue
            seen_keys.add(key)
            indexed_items.append((key, self._build_disk_put_item(key, memory_obj)))

        inserted_keys = self.disk_worker.insert_put_tasks(
            [key for key, _ in indexed_items]
        )
        inserted_key_set = set(inserted_keys)
        put_items = [
            put_item for key, put_item in indexed_items if key in inserted_key_set
        ]
        skipped_keys.extend(
            [key for key, _ in indexed_items if key not in inserted_key_set]
        )

        for key in skipped_keys:
            logger.debug(f"Put task for {key} is already in progress.")

        reserved_items = []
        failed_keys = []
        with self.disk_lock:
            for put_item in put_items:
                evict_success = True
                while self.current_cache_size + put_item.size > self.max_cache_size:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.dict, num_candidates=1
                    )
                    if not evict_keys:
                        logger.warning(
                            "No eviction candidates found. "
                            "Disk space under pressure."
                        )
                        evict_success = False
                        break

                    for evict_key in evict_keys:
                        self.current_cache_size -= self.dict[evict_key].size

                    self.batched_remove(evict_keys, force=False)

                if evict_success:
                    self.current_cache_size += put_item.size
                    self.cache_policy.update_on_put(put_item.key)
                    reserved_items.append(put_item)
                else:
                    failed_keys.append(put_item.key)

        if failed_keys:
            self.disk_worker.remove_put_tasks(failed_keys)

        return reserved_items

    def _rollback_batched_put_reservation(
        self,
        put_items: list[_DiskPutItem],
    ) -> None:
        with self.disk_lock:
            for put_item in put_items:
                self.current_cache_size -= put_item.size
            if self.current_cache_size < 0:
                self.current_cache_size = 0

        for put_item in put_items:
            put_item.memory_obj.ref_count_down()

        self.disk_worker.remove_put_tasks([put_item.key for put_item in put_items])

    def _build_disk_put_item(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> _DiskPutItem:
        assert memory_obj.tensor is not None
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        cached_positions = memory_obj.metadata.cached_positions
        assert shape is not None
        assert dtype is not None
        return _DiskPutItem(
            key=key,
            memory_obj=memory_obj,
            size=size,
            shape=shape,
            dtype=dtype,
            fmt=fmt,
            cached_positions=cached_positions,
        )

    def _save_disk_put_item(
        self,
        put_item: _DiskPutItem,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        key = put_item.key
        memory_obj = put_item.memory_obj
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)
        inserted = False

        try:
            # TODO(Jiayi): need to add ref count in disk memory object
            self.write_file(buffer, path)

            with self.disk_lock:
                self.usage += len(buffer)
                self.stats_monitor.update_local_storage_usage(self.usage)

            # Ref count down before insert_key for mem-leak regression tests.
            # TODO(Jiayi): This could be problematic if the freed memory object
            # is immediately reused.
            memory_obj.ref_count_down()
            inserted = True

            self.insert_key(
                key,
                put_item.size,
                put_item.shape,
                put_item.dtype,
                put_item.fmt,
                cached_positions=put_item.cached_positions,
            )

            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(
                        "on_complete_callback failed for key %s: %s",
                        key,
                        e,
                    )
        except Exception:
            if not inserted:
                memory_obj.ref_count_down()
            with self.disk_lock:
                self.current_cache_size -= put_item.size
                if self.current_cache_size < 0:
                    self.current_cache_size = 0
            logger.exception("Failed to write key %s to local disk", key)
            raise
        finally:
            self.disk_worker.remove_put_task(key)
