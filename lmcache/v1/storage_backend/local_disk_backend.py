# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import itertools
import os
import queue
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalDiskWorker:
    def __init__(self) -> None:
        self.pq: queue.PriorityQueue[tuple[int, int, str, Callable, dict[str, Any]]] = (
            queue.PriorityQueue()
        )

        # TODO(Jiayi): remove this hard code.
        num_workers = 1
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        self.put_lock = threading.Lock()
        self.prefetch_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        # Optional means the pretch task in queue but not
        # started yet.
        self.prefetch_tasks: dict[CacheEngineKey, Optional[Future]] = {}

        self.counter = itertools.count()

        self.thread = threading.Thread(target=self.process_task, daemon=True)
        self.thread.start()

    def submit_task(
        self,
        task_type: str,
        task: Callable,
        **kwargs,
    ):
        if task_type == "prefetch":
            priority = 0
            self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
            self.insert_put_task(kwargs["key"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.pq.put((priority, next(self.counter), task_type, task, kwargs))

    def process_task(self):
        while True:
            _, _, task_type, task, kwargs = self.pq.get(block=True)

            future = self.executor.submit(task, **kwargs)
            if task_type == "prefetch":
                # Remove the prefetch task from the queue
                self.insert_prefetch_task(kwargs["key"], future)

            self.pq.task_done()

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

    def remove_prefetch_task(self, key: CacheEngineKey):
        with self.prefetch_lock:
            if key in self.prefetch_tasks:
                self.prefetch_tasks.pop(key)
            else:
                logger.warning(f"Key {key} not found in prefetch tasks.")

    def insert_prefetch_task(
        self,
        key: CacheEngineKey,
        future_or_none: Optional[Future] = None,
    ):
        with self.prefetch_lock:
            self.prefetch_tasks[key] = future_or_none

    def exists_in_prefetch_tasks(self, key: CacheEngineKey) -> bool:
        with self.prefetch_lock:
            return key in self.prefetch_tasks

    def wait_prefetch_task(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Wait for the prefetch task to complete and return the MemoryObj.
        If the key is not in the prefetch tasks, return None.
        """

        while True:
            self.prefetch_lock.acquire()
            if key not in self.prefetch_tasks:
                self.prefetch_lock.release()
                return None

            logger.debug(f"Waiting for prefetch task for key {key} to complete.")
            future = self.prefetch_tasks[key]
            if future is None:
                self.prefetch_lock.release()
                time.sleep(0.01)
                continue

            self.prefetch_lock.release()

            memory_obj = future.result()
            return memory_obj

    def close(self):
        self.executor.shutdown(wait=True)
        self.thread.join()


# FIXME(Jiayi): need batched prefetch


class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        self.lookup_server = lookup_server

        self.loop = loop

        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        self.disk_worker = LocalDiskWorker()

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

    def __str__(self):
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
        if force:
            self.disk_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.disk_lock.release()
            return False

        path = meta.path
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)
        self.disk_worker.submit_task("delete", os.remove, path=path)

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

        return True

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt

        has_stored = False
        with self.disk_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                self.dict.pop(key)
                has_stored = True

            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype, fmt, False)

        # push kv admit msg
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(
                KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        assert memory_obj.tensor is not None

        # skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        # TODO(Jiayi): Fragmentation is not considered here.
        required_size = memory_obj.get_physical_size()
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found.", "Disk space under pressure."
                    )
                    return None

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)

                if self.lookup_server is not None:
                    self.lookup_server.batched_remove(evict_keys)
            self.current_cache_size += required_size

        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        self.disk_worker.submit_task(
            "put",
            self.async_save_bytes_to_disk,
            key=key,
            memory_obj=memory_obj,
        )

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> bool:
        # TODO(Jiayi): prefetch and local_cpu must be enabled together
        # Need to consider gpu direct cases.
        assert self.use_local_cpu, "prefetch and local_cpu must be enabled together"

        logger.debug("Submitting prefetch task")

        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return False

        # NOTE(Jiayi): Currently, we consider prefetch as cache hit.
        self.cache_policy.update_on_hit(key, self.dict)

        if self.disk_worker.exists_in_prefetch_tasks(key):
            logger.debug(f"Prefetch task for {key} is already in progress.")
            self.disk_lock.release()
            return False

        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        fmt = self.dict[key].fmt

        assert dtype is not None
        assert shape is not None

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            self.disk_lock.release()
            logger.debug("Memory allocation failed during async disk load.")
            return False

        self.dict[key].pin()

        # Update cache recency
        self.cache_policy.update_on_hit(key, self.dict)

        self.disk_lock.release()
        logger.debug(f"Prefetching {key} from disk.")

        self.disk_worker.submit_task(
            "prefetch",
            self.async_load_bytes_from_disk,
            path=path,
            key=key,
            memory_obj=memory_obj,
        )

        return True

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

        self.cache_policy.update_on_hit(key, self.dict)

        self.disk_lock.release()

        if memory_obj := self.disk_worker.wait_prefetch_task(key):
            # NOTE(Jiayi): We don't directly use pin here as
            # the memory_obj could be evicted from cpu backend
            # before pin.
            # TODO(Jiayi): Cache recency is not strictly
            # handled in prefetching.
            if self.local_cpu_backend.contains(key, pin=True):
                return memory_obj

        self.disk_lock.acquire()
        # Update cache recency
        self.cache_policy.update_on_hit(key, self.dict)

        disk_meta = self.dict[key]
        path = disk_meta.path
        dtype = disk_meta.dtype
        shape = disk_meta.shape
        fmt = disk_meta.fmt
        assert dtype is not None
        assert shape is not None

        memory_obj = self.load_bytes_from_disk(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )
        self.disk_lock.release()

        return memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Non-blocking get function.
        Using a dummy wrapper around prefetch for now.
        """
        # TODO(Jiayi): Need to align prefetch and get_non_blocking
        raise NotImplementedError(
            "Non-blocking get is not implemented for LocalDiskBackend. "
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # FIXME(Jiayi): need to add ref count in disk memory object
        self.write_file(buffer, path)

        self.insert_key(key, memory_obj)

        # ref count down here because there's a ref_count_up in
        # `submit_put_task` above
        memory_obj.ref_count_down()

        self.disk_worker.remove_put_task(key)

    # TODO(Jiayi): use `bytes_read = await f.readinto(buffer)`
    # for better performance (i.e., fewer copy)
    def async_load_bytes_from_disk(
        self,
        path: str,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        """
        Async load bytearray from disk.
        """

        logger.debug("Executing `async_load_bytes` from disk.")
        # FIXME (Jiayi): handle the case where loading fails.
        buffer = memory_obj.byte_array
        self.read_file(key, buffer, path)

        self.disk_lock.acquire()
        self.dict[key].unpin()
        self.disk_lock.release()

        # Write back to cpu
        self.local_cpu_backend.submit_put_task(key, memory_obj)

        self.disk_worker.remove_prefetch_task(key)

        return memory_obj

    # TODO(Jiayi): use memory allocator to redeuce cpu buffer allocation
    # TODO(Jiayi): the pinned cpu memory_obj should directly be passed into
    # gpu connector; this gpu buffer could be avoided
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

        # TODO(Jiayi): Consider adding write-back here.
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during disk load."

        buffer = memory_obj.byte_array
        self.read_file(key, buffer, path)
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
            if self.dict.get(key, None):
                self.dict.pop(key)
            return

        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

    def close(self) -> None:
        if self.lookup_server is not None:
            self.disk_lock.acquire()
            self.lookup_server.batched_remove(list(self.dict.keys()))
            self.disk_lock.release()
