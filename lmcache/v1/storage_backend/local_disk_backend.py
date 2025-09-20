# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
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
    CacheEngineKey,
    DiskCacheMetadata,
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

_METADATA_SUFFIX = ".meta"
_METADATA_VERSION = 1


def _dtype_to_string(dtype: torch.dtype) -> str:
    if dtype in TORCH_DTYPE_TO_STR_DTYPE:
        return TORCH_DTYPE_TO_STR_DTYPE[dtype]
    dtype_str = str(dtype)
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[len("torch.") :]
    return dtype_str


def _string_to_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    if dtype_str in STR_DTYPE_TO_TORCH_DTYPE:
        return STR_DTYPE_TO_TORCH_DTYPE[dtype_str]
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[len("torch.") :]
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        logger.warning("Unsupported torch dtype string in metadata: %s", dtype_str)
        return None


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
    ):
        super().__init__(dst_device)
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

        self.loop = loop

        self.use_local_cpu = config.local_cpu

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

        # Disk persistence: repopulate in-memory index from disk if enabled
        self.local_disk_persistence = bool(
            getattr(config, "local_disk_persistence", False)
        )
        self.populate_disk_cache_to_cpu_on_start = bool(
            getattr(config, "populate_disk_cache_to_cpu_on_start", True)
        )
        if self.local_disk_persistence:
            disk_keys = self._restore_persistent_cache()
            if (
                self.populate_disk_cache_to_cpu_on_start
                and self.use_local_cpu
                and disk_keys
            ):
                self._prefetch_persisted_cache(disk_keys)

    def __str__(self):
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def _metadata_path(self, data_path: str) -> str:
        return data_path + _METADATA_SUFFIX

    def _write_metadata_file(self, metadata: DiskCacheMetadata) -> None:
        if not self.local_disk_persistence:
            return        
        if metadata.shape is None or metadata.dtype is None:
            raise ValueError("Metadata must contain shape and dtype to persist to disk")

        fmt_name = metadata.fmt.name if metadata.fmt is not None else None
        metadata_dict = {
            "version": _METADATA_VERSION,
            "size": int(metadata.size),
            "shape": [int(dim) for dim in metadata.shape],
            "dtype": _dtype_to_string(metadata.dtype),
            "fmt": fmt_name,
        }
        meta_path = self._metadata_path(metadata.path)
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f)
        os.replace(tmp_path, meta_path)

    def _read_metadata_from_disk(
        self, data_path: str
    ) -> Optional[tuple[int, torch.Size, torch.dtype, Optional[MemoryFormat]]]:
        meta_path = self._metadata_path(data_path)
        if not os.path.exists(meta_path):
            logger.warning("Missing metadata file for persisted cache %s", data_path)
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read metadata file %s: %s", meta_path, exc)
            return None

        version = metadata.get("version")
        if version != _METADATA_VERSION:
            logger.warning(
                "Unsupported metadata version %s for %s", version, meta_path
            )
            return None

        shape_value = metadata.get("shape")
        if not isinstance(shape_value, list):
            logger.warning("Invalid shape metadata for %s: %s", data_path, shape_value)
            return None
        try:
            shape = torch.Size([int(dim) for dim in shape_value])
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to parse shape metadata for %s: %s", data_path, exc)
            return None

        dtype = _string_to_dtype(metadata.get("dtype"))
        if dtype is None:
            logger.warning("Invalid dtype metadata for %s", data_path)
            return None

        fmt_str = metadata.get("fmt")
        fmt: Optional[MemoryFormat] = None
        if fmt_str is not None:
            try:
                fmt = MemoryFormat[fmt_str]
            except KeyError:
                logger.warning(
                    "Invalid memory format metadata for %s: %s", data_path, fmt_str
                )
                fmt = None

        size = metadata.get("size")
        if size is None:
            size_int = os.path.getsize(data_path)
        else:
            try:
                size_int = int(size)
            except (TypeError, ValueError):
                logger.warning("Invalid size metadata for %s: %s", data_path, size)
                size_int = os.path.getsize(data_path)

        return size_int, shape, dtype, fmt

    def _remove_metadata_file(self, data_path: str) -> None:
        if not self.local_disk_persistence:
            return
        meta_path = self._metadata_path(data_path)
        try:
            os.remove(meta_path)
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning("Failed to remove metadata file %s: %s", meta_path, e)

    def _restore_persistent_cache(self) -> list[CacheEngineKey]:
        logger.info(
            "Local disk persistence enabled. Scanning for existing cache files..."
        )
        disk_keys: list[CacheEngineKey] = []
        total_size = 0
        for entry in sorted(os.listdir(self.path)):
            if not entry.endswith(".pt"):
                continue
            data_path = os.path.join(self.path, entry)
            if not os.path.isfile(data_path):
                continue
            key_str = entry[:-3].replace("-", "/")
            try:
                key = CacheEngineKey.from_string(key_str)
            except ValueError as exc:
                logger.warning(
                    "Failed to parse cache key from file %s: %s", entry, exc
                )
                continue

            metadata = self._read_metadata_from_disk(data_path)
            if metadata is None:
                continue
            size, shape, dtype, fmt = metadata
            disk_meta = DiskCacheMetadata(data_path, size, shape, dtype, fmt, 0)
            self.dict[key] = disk_meta
            self.cache_policy.update_on_put(key)
            disk_keys.append(key)
            total_size += size

        self.current_cache_size = float(total_size)
        self.usage = total_size
        self.stats_monitor.update_local_storage_usage(self.usage)
        if total_size:
            logger.info(
                "Restored %d disk cache entries (%.2f MB) from %s.",
                len(disk_keys),
                total_size / 1e6,
                self.path,
            )
        else:
            logger.info("No existing disk cache entries found in %s.", self.path)

        return disk_keys

    def _prefetch_persisted_cache(self, keys: list[CacheEngineKey]) -> None:
        logger.info(
            "Prefetching %d disk cache entries to CPU memory...",
            len(keys),
        )
        for key in keys:
            memory_obj: Optional[MemoryObj] = None
            try:
                memory_obj = self.get_blocking(key)
            except (IOError, FileNotFoundError) as exc:
                logger.warning("Failed to prefetch cache for key %s: %s", key, exc)
                continue

            if memory_obj is None:
                logger.warning("Persisted cache for key %s missing on disk", key)
                continue

            try:
                self.local_cpu_backend.submit_put_task(key, memory_obj)
            except (RuntimeError, MemoryError) as exc:
                logger.warning(
                    "Failed to populate CPU cache for persisted key %s: %s", key, exc
                )
            finally:
                memory_obj.ref_count_down()

        logger.info("Disk cache prefetch to CPU complete.")

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

        # NOTE: The following code will cause deadlock
        # res = asyncio.run_coroutine_threadsafe(
        #     self.disk_worker.submit_task("delete", os.remove, path),
        #     self.loop,
        # )
        # res.result()

        os.remove(path)
        self._remove_metadata_file(path)

        if force:
            self.current_cache_size = max(0.0, self.current_cache_size - size)
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> None:
        path = self._key_to_path(key)

        has_stored = False
        metadata_entry = DiskCacheMetadata(path, size, shape, dtype, fmt, False)
        with self.disk_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                self.dict.pop(key)
                has_stored = True

            self.dict[key] = metadata_entry

        try:
            self._write_metadata_file(metadata_entry)
        except (ValueError, OSError) as exc:
            logger.warning("Failed to persist metadata for key %s: %s", key, exc)

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

        if all_evict_keys:
            self._on_evict(all_evict_keys)

        if not evict_success:
            return None

        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.disk_worker.submit_task(
                "put",
                self.async_save_bytes_to_disk,
                key=key,
                memory_obj=memory_obj,
            ),
            self.loop,
        )

    # TODO(Jiayi): enable real batching
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

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

        self.disk_lock.release()
        memory_obj = self.load_bytes_from_disk(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

        return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
    ) -> list[MemoryObj]:
        mem_objs: list[MemoryObj] = []
        paths: list[str] = []

        logger.info(f"lookup_id: {lookup_id}; Prefetching {len(keys)} keys from disk.")
        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, f"Key {key} not found in disk cache after pinning"

            # NOTE(Jiayi): Currently, we consider prefetch as cache hit.
            self.cache_policy.update_on_hit(key, self.dict)

            path = self.dict[key].path
            dtype = self.dict[key].dtype
            shape = self.dict[key].shape
            fmt = self.dict[key].fmt

            assert dtype is not None
            assert shape is not None

            memory_obj = self.local_cpu_backend.allocate(
                shape,
                dtype,
                fmt,
            )

            assert memory_obj is not None, (
                "Memory allocation failed during async disk load."
            )

            self.dict[key].pin()

            # Update cache recency
            self.cache_policy.update_on_hit(key, self.dict)

            self.disk_lock.release()
            logger.debug(f"Prefetching {key} from disk.")

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

        # TODO(Jiayi): need to add ref count in disk memory object
        self.write_file(buffer, path)

        # ref count down here because there's a ref_count_up in
        # `submit_put_task` above.
        # Ref count down better be before `insert_key` for testing
        # purposes (e.g., testing mem_leak).
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        memory_obj.ref_count_down()

        self.insert_key(key, size, shape, dtype, fmt)

        self.disk_worker.remove_put_task(key)

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

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        self.disk_worker.close()
        with self.disk_lock:
            keys = list(self.dict.keys())
        if keys:
            super()._on_evict(keys)
        # Close worker executor
        self.disk_worker.close()
