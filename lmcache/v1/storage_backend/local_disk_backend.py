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
from lmcache.utils import (
    CacheEngineKey,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
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

        self.metadata = metadata if metadata is not None else local_cpu_backend.metadata
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
        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending

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

    def __str__(self) -> str:
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def _default_memory_format(self) -> MemoryFormat:
        if not self.layerwise:
            return MemoryFormat.KV_2LTD
        if self.enable_blending:
            return MemoryFormat.KV_2TD
        return MemoryFormat.KV_T2D

    def _shape_size_bytes(self, shape: torch.Size, dtype: torch.dtype) -> int:
        return shape.numel() * dtype.itemsize

    def _layerwise_shape_from_metadata(self, fmt: MemoryFormat) -> Optional[torch.Size]:
        if self.metadata is None:
            return None

        kv_size = self.metadata.kv_shape[1]
        chunk_size = self.metadata.chunk_size
        hidden_dim = self.metadata.kv_shape[3] * self.metadata.kv_shape[4]

        if fmt == MemoryFormat.KV_2TD:
            return torch.Size([kv_size, chunk_size, hidden_dim])
        if fmt == MemoryFormat.KV_T2D:
            return torch.Size([chunk_size, kv_size, hidden_dim])
        return None

    def _infer_disk_entry_metadata(
        self,
        key: CacheEngineKey,
        size: int,
    ) -> tuple[Optional[torch.Size], torch.dtype, MemoryFormat]:
        dtype = key.dtype
        fmt = self._default_memory_format()
        if self.metadata is None:
            return None, dtype, fmt

        if self.layerwise:
            shape = self._layerwise_shape_from_metadata(fmt)
            if shape is not None and self._shape_size_bytes(shape, dtype) == size:
                return shape, dtype, fmt
            return None, dtype, fmt

        shapes = self.metadata.get_shapes()
        dtypes = self.metadata.get_dtypes()
        for shape, candidate_dtype in zip(shapes, dtypes, strict=False):
            if (
                candidate_dtype == dtype
                and self._shape_size_bytes(shape, candidate_dtype) == size
            ):
                return shape, candidate_dtype, fmt

        return None, dtype, fmt

    def _drop_disk_entry_locked(
        self,
        key: CacheEngineKey,
        *,
        update_policy: bool = True,
        update_cache_size: bool = True,
    ) -> Optional[DiskCacheMetadata]:
        meta = self.dict.pop(key, None)
        if meta is None:
            return None

        if update_cache_size:
            self.current_cache_size = max(self.current_cache_size - meta.size, 0)
        self.usage = max(self.usage - meta.size, 0)
        self.keys_in_request = [
            request_key for request_key in self.keys_in_request if request_key != key
        ]
        if update_policy:
            self.cache_policy.update_on_force_evict(key)
        return meta

    def _ensure_disk_entry_from_file_locked(
        self,
        key: CacheEngineKey,
        *,
        pin: bool = False,
    ) -> bool:
        if key in self.dict:
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

        path = self._key_to_path(key)
        if not os.path.isfile(path):
            return False

        try:
            size = os.path.getsize(path)
        except OSError:
            return False

        shape, dtype, fmt = self._infer_disk_entry_metadata(key, size)
        self.dict[key] = DiskCacheMetadata(
            path=path,
            size=size,
            shape=shape,
            dtype=dtype,
            cached_positions=None,
            fmt=fmt,
            pin_count=1 if pin else 0,
        )
        self.current_cache_size += size
        self.usage += size
        self.cache_policy.update_on_put(key)
        if pin:
            self.keys_in_request.append(key)
        self.stats_monitor.update_local_storage_usage(self.usage)
        return True

    def _get_load_metadata_locked(
        self,
        key: CacheEngineKey,
    ) -> Optional[
        tuple[str, torch.dtype, torch.Size, MemoryFormat, Optional[torch.Tensor]]
    ]:
        meta = self.dict.get(key)
        if meta is None:
            return None

        if meta.shape is None or meta.dtype is None or meta.fmt is None:
            shape, dtype, fmt = self._infer_disk_entry_metadata(key, meta.size)
            meta.shape = shape
            meta.dtype = dtype
            meta.fmt = fmt

        if meta.shape is None or meta.dtype is None or meta.fmt is None:
            logger.warning("Cannot infer disk cache metadata for key %s", key)
            self._drop_disk_entry_locked(key)
            self.stats_monitor.update_local_storage_usage(self.usage)
            return None

        return meta.path, meta.dtype, meta.shape, meta.fmt, meta.cached_positions

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            return self._ensure_disk_entry_from_file_locked(key, pin=pin)

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
        if force:
            self.disk_lock.acquire()

        try:
            meta = self._drop_disk_entry_locked(
                key,
                update_policy=force,
                update_cache_size=force,
            )
            if meta is None:
                path = self._key_to_path(key)
                if not os.path.exists(path):
                    return False
                size = 0
            else:
                path = meta.path
                size = meta.size
                self.stats_monitor.update_local_storage_usage(self.usage)

            # NOTE: The following code will cause deadlock
            # res = asyncio.run_coroutine_threadsafe(
            #     self.disk_worker.submit_task("delete", os.remove, path),
            #     self.loop,
            # )
            # res.result()
            try:
                os.remove(path)
            except FileNotFoundError:
                logger.warning("File already missing on disk: %s", path)

            if meta is None and force:
                self.current_cache_size = max(self.current_cache_size - size, 0)
        finally:
            if force:
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
            if not self._ensure_disk_entry_from_file_locked(key):
                return None

            load_meta = self._get_load_metadata_locked(key)
            if load_meta is None:
                return None
            path, dtype, shape, fmt, cached_positions = load_meta

        # Load is performed outside the lock: it can block for a non-trivial
        # amount of time (CPU staging pool allocation + memcpy from disk) and
        # must not hold disk_lock while waiting, or concurrent insert/evict
        # operations would deadlock.
        memory_obj = self.load_bytes_from_disk(
            key,
            path,
            dtype=dtype,
            shape=shape,
            fmt=fmt,
            cached_positions=cached_positions,
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
        cached_positions_list: list[Optional[torch.Tensor]] = []

        logger.debug(f"lookup_id: {lookup_id}; Prefetching {len(keys)} keys from disk.")
        for key in keys:
            with self.disk_lock:
                assert self._ensure_disk_entry_from_file_locked(key), (
                    f"Key {key} not found in disk cache after pinning"
                )

                load_meta = self._get_load_metadata_locked(key)
                if load_meta is None:
                    return mem_objs
                path, dtype, shape, fmt, cached_positions = load_meta

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

            logger.debug(f"Prefetching {key} from disk.")
            memory_obj.pin()
            mem_objs.append(memory_obj)
            paths.append(path)
            cached_positions_list.append(cached_positions)

        return await self.disk_worker.submit_task(
            "prefetch",
            self.batched_async_load_bytes_from_disk,
            paths=paths,
            keys=keys,
            memory_objs=mem_objs,
            cached_positions_list=cached_positions_list,
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
                if not self._ensure_disk_entry_from_file_locked(key, pin=pin):
                    return num_hit_counts
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

    def batched_async_load_bytes_from_disk(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        cached_positions_list: list[Optional[torch.Tensor]],
        write_back: bool = False,
    ) -> list[MemoryObj]:
        """
        Async load bytearray from disk.
        """

        logger.debug("Executing `async_load_bytes` from disk.")
        loaded_mem_objs: list[MemoryObj] = []
        for idx, (path, key, mem_obj, cached_positions) in enumerate(
            zip(paths, keys, memory_objs, cached_positions_list, strict=False)
        ):
            buffer = mem_obj.byte_array
            if not self.read_file(key, buffer, path):
                self._release_prefetch_resources([key], [mem_obj])
                self._release_prefetch_resources(
                    keys[idx + 1 :], memory_objs[idx + 1 :]
                )
                break

            mem_obj.metadata.cached_positions = cached_positions

            with self.disk_lock:
                if key in self.dict:
                    self.dict[key].unpin()
                    self.cache_policy.update_on_hit(key, self.dict)
            loaded_mem_objs.append(mem_obj)

        return loaded_mem_objs

    def load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor],
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during disk load."

        buffer = memory_obj.byte_array
        if not self.read_file(key, buffer, path):
            memory_obj.ref_count_down()
            return None

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

    def read_file(self, key, buffer, path) -> bool:
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
                    bytes_read = f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    bytes_read = fdo.readinto(buffer)
        except FileNotFoundError:
            logger.warning(f"File not found on disk: {path}")
            with self.disk_lock:
                self._drop_disk_entry_locked(key)
            self.stats_monitor.update_local_storage_usage(self.usage)
            return False

        if bytes_read != size:
            logger.warning(
                "Incomplete disk cache read for %s: expected %d bytes, read %d",
                path,
                size,
                bytes_read,
            )
            with self.disk_lock:
                self._drop_disk_entry_locked(key)
            self.stats_monitor.update_local_storage_usage(self.usage)
            return False

        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )
        return True

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def _release_prefetch_resources(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: Sequence[MemoryObj],
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            with self.disk_lock:
                if key in self.dict:
                    self.dict[key].unpin()
            memory_obj.unpin()
            memory_obj.ref_count_down()

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self.disk_worker.close()
