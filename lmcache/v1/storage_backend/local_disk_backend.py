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
    MANIFEST_VERSION = 1

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

        self.config = config
        self.metadata = metadata if metadata is not None else local_cpu_backend.metadata
        self.cache_policy_name = config.cache_policy.upper()
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
        self.manifest_path = self._get_manifest_path()

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

        self._restore_disk_cache_state()

    def __str__(self):
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def _get_manifest_path(self) -> str:
        if self.metadata is None:
            namespace = self.instance_id or "default"
        else:
            namespace = (
                f"{self.metadata.model_name}@{self.metadata.world_size}"
                f"@{self.metadata.worker_id}"
            )

        return os.path.join(
            self.path,
            f".lmcache-local-disk-manifest-{namespace.replace('/', '-')}.json",
        )

    def _get_legacy_namespace_prefix(self) -> Optional[str]:
        if self.metadata is None:
            return None

        return (
            f"{self.metadata.model_name}@{self.metadata.world_size}"
            f"@{self.metadata.worker_id}@"
        ).replace("/", "-")

    def _serialize_cached_positions(
        self, cached_positions: Optional[torch.Tensor]
    ) -> Optional[list[int]]:
        if cached_positions is None:
            return None
        return cached_positions.detach().cpu().tolist()

    def _deserialize_cached_positions(
        self, cached_positions: Any
    ) -> Optional[torch.Tensor]:
        if cached_positions is None:
            return None
        if not isinstance(cached_positions, list):
            return None
        try:
            return torch.tensor(cached_positions, dtype=torch.long)
        except (TypeError, ValueError):
            return None

    def _serialize_policy_state_locked(self) -> dict[str, Any]:
        ordered_keys = [key.to_string() for key in self.dict.keys()]
        policy_state: dict[str, Any] = {"ordered_keys": ordered_keys}

        if self.cache_policy_name == "LFU":
            key_freqs = {
                key.to_string(): int(freq)
                for key, freq in self.cache_policy.key_to_freq.items()
            }

            ordered_keys = []
            for freq, keys in self.cache_policy.freq_to_keys.items():
                for key in keys.keys():
                    key_str = key.to_string()
                    ordered_keys.append(key_str)
                    key_freqs.setdefault(key_str, int(freq))

            policy_state["ordered_keys"] = ordered_keys
            policy_state["key_freqs"] = key_freqs

        return policy_state

    def _build_manifest_data_locked(self) -> dict[str, Any]:
        entries = {}
        for key, meta in self.dict.items():
            entries[key.to_string()] = {
                "path": os.path.relpath(meta.path, self.path),
                "size": meta.size,
                "shape": list(meta.shape) if meta.shape is not None else None,
                "dtype": key._dtype_str,
                "fmt": (
                    meta.fmt.name
                    if meta.fmt is not None and hasattr(meta.fmt, "name")
                    else None
                ),
                "cached_positions": self._serialize_cached_positions(
                    meta.cached_positions
                ),
            }

        data: dict[str, Any] = {
            "version": self.MANIFEST_VERSION,
            "cache_policy": self.cache_policy_name,
            "entries": entries,
            "policy_state": self._serialize_policy_state_locked(),
        }
        if self.metadata is not None:
            data["namespace"] = {
                "model_name": self.metadata.model_name,
                "world_size": self.metadata.world_size,
                "worker_id": self.metadata.worker_id,
            }

        return data

    def _write_manifest_data(self, data: dict[str, Any]) -> None:
        tmp_path = self.manifest_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.manifest_path)
        except Exception as e:
            logger.warning("Failed to persist local disk manifest: %s", e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _save_manifest(self) -> None:
        with self.disk_lock:
            data = self._build_manifest_data_locked()
        self._write_manifest_data(data)

    def _reset_policy_state_locked(self) -> None:
        if self.cache_policy_name == "LFU":
            self.cache_policy.key_to_freq.clear()
            self.cache_policy.freq_to_keys.clear()

        if hasattr(self.cache_policy, "chunk_hash_to_init_timestamp"):
            self.cache_policy.chunk_hash_to_init_timestamp.clear()

    def _recompute_usage_locked(self) -> None:
        self.current_cache_size = sum(meta.size for meta in self.dict.values())
        self.usage = 0
        for meta in self.dict.values():
            if os.path.exists(meta.path):
                self.usage += os.path.getsize(meta.path)

    def _get_restored_key_order(
        self,
        key_by_string: dict[str, CacheEngineKey],
        policy_state: dict[str, Any],
    ) -> list[CacheEngineKey]:
        ordered_keys: list[CacheEngineKey] = []
        seen_keys: set[CacheEngineKey] = set()

        for key_str in policy_state.get("ordered_keys", []):
            key = key_by_string.get(key_str)
            if key is None or key in seen_keys:
                continue
            ordered_keys.append(key)
            seen_keys.add(key)

        for key in key_by_string.values():
            if key in seen_keys:
                continue
            ordered_keys.append(key)
            seen_keys.add(key)

        return ordered_keys

    def _restore_policy_state_locked(
        self,
        saved_policy_name: str,
        policy_state: dict[str, Any],
        ordered_keys: list[CacheEngineKey],
    ) -> None:
        self._reset_policy_state_locked()

        if saved_policy_name != self.cache_policy_name:
            for key in ordered_keys:
                self.cache_policy.update_on_put(key)
            return

        if self.cache_policy_name != "LFU":
            return

        key_freqs = policy_state.get("key_freqs", {})
        if not isinstance(key_freqs, dict):
            key_freqs = {}

        for key in ordered_keys:
            freq = int(key_freqs.get(key.to_string(), 1))
            self.cache_policy.key_to_freq[key] = freq
            if freq not in self.cache_policy.freq_to_keys:
                self.cache_policy.freq_to_keys[freq] = {}
            self.cache_policy.freq_to_keys[freq][key] = None

    def _load_manifest(self) -> bool:
        if not os.path.exists(self.manifest_path):
            return False

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(
                "Failed to load local disk manifest %s: %s", self.manifest_path, e
            )
            return False

        if not isinstance(data, dict) or data.get("version") != self.MANIFEST_VERSION:
            logger.warning(
                "Ignoring incompatible local disk manifest: %s", self.manifest_path
            )
            return False

        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            logger.warning(
                "Ignoring malformed local disk manifest: %s", self.manifest_path
            )
            return False

        policy_state = data.get("policy_state", {})
        if not isinstance(policy_state, dict):
            policy_state = {}

        saved_policy_name = str(
            data.get("cache_policy", self.cache_policy_name)
        ).upper()
        restored_entries: dict[str, DiskCacheMetadata] = {}
        key_by_string: dict[str, CacheEngineKey] = {}

        for key_str, entry in entries.items():
            if not isinstance(entry, dict):
                continue

            try:
                key = parse_cache_key(key_str)
            except ValueError:
                continue

            rel_path = entry.get("path")
            if not isinstance(rel_path, str):
                continue

            path = os.path.join(self.path, rel_path)
            if not os.path.isfile(path):
                continue

            shape_list = entry.get("shape")
            if not isinstance(shape_list, list):
                continue

            try:
                shape = torch.Size([int(v) for v in shape_list])
            except (TypeError, ValueError):
                continue

            fmt_name = entry.get("fmt")
            if (
                not isinstance(fmt_name, str)
                or fmt_name not in MemoryFormat.__members__
            ):
                continue

            try:
                size = int(entry.get("size", os.path.getsize(path)))
            except (OSError, TypeError, ValueError):
                continue

            canonical_key = key.to_string()
            key_by_string[canonical_key] = key
            restored_entries[canonical_key] = DiskCacheMetadata(
                path=path,
                size=size,
                shape=shape,
                dtype=key.dtype,
                cached_positions=self._deserialize_cached_positions(
                    entry.get("cached_positions")
                ),
                fmt=MemoryFormat[fmt_name],
                pin_count=0,
            )

        ordered_keys = self._get_restored_key_order(key_by_string, policy_state)

        with self.disk_lock:
            self.dict = self.cache_policy.init_mutable_mapping()
            for key in ordered_keys:
                self.dict[key] = restored_entries[key.to_string()]

            self._restore_policy_state_locked(
                saved_policy_name,
                policy_state,
                ordered_keys,
            )
            self._recompute_usage_locked()

        self.stats_monitor.update_local_storage_usage(self.usage)
        logger.info(
            "Loaded %d local disk entries from manifest %s",
            len(ordered_keys),
            self.manifest_path,
        )
        return True

    def _get_legacy_restore_defaults(
        self,
    ) -> Optional[tuple[torch.Size, torch.dtype, MemoryFormat, int]]:
        if self.metadata is None:
            return None

        shapes = self.metadata.get_shapes()
        dtypes = self.metadata.get_dtypes()
        if not shapes or not dtypes:
            return None

        if self.metadata.use_mla:
            fmt = MemoryFormat.KV_MLA_FMT
        elif self.local_cpu_backend.layerwise:
            fmt = (
                MemoryFormat.KV_2TD
                if self.local_cpu_backend.enable_blending
                else MemoryFormat.KV_T2D
            )
        else:
            fmt = MemoryFormat.KV_2LTD

        return (
            shapes[0],
            dtypes[0],
            fmt,
            self.local_cpu_backend.get_full_chunk_size_bytes(),
        )

    def _scan_legacy_disk_cache_locked(self) -> int:
        prefix = self._get_legacy_namespace_prefix()
        restore_defaults = self._get_legacy_restore_defaults()
        if prefix is None or restore_defaults is None:
            return 0

        shape, dtype, fmt, expected_size = restore_defaults
        migrated_keys: list[CacheEngineKey] = []
        skipped_count = 0

        self.dict = self.cache_policy.init_mutable_mapping()
        self._reset_policy_state_locked()

        for entry in os.scandir(self.path):
            if not entry.is_file() or not entry.name.endswith(".pt"):
                continue

            if not entry.name.startswith(prefix):
                continue

            try:
                file_size = entry.stat().st_size
            except OSError:
                skipped_count += 1
                continue

            if file_size != expected_size:
                skipped_count += 1
                continue

            tail = entry.name[len(prefix) : -3]
            if not tail:
                skipped_count += 1
                continue

            try:
                key = parse_cache_key(
                    f"{self.metadata.model_name}@{self.metadata.world_size}"
                    f"@{self.metadata.worker_id}@{tail}"
                )
            except (AttributeError, ValueError):
                skipped_count += 1
                continue

            self.dict[key] = DiskCacheMetadata(
                path=entry.path,
                size=file_size,
                shape=shape,
                dtype=dtype,
                cached_positions=None,
                fmt=fmt,
                pin_count=0,
            )
            self.cache_policy.update_on_put(key)
            migrated_keys.append(key)

        self._recompute_usage_locked()
        self.stats_monitor.update_local_storage_usage(self.usage)

        if migrated_keys:
            logger.info(
                "Migrated %d legacy local disk entries from %s (skipped=%d)",
                len(migrated_keys),
                self.path,
                skipped_count,
            )

        return len(migrated_keys)

    def _restore_disk_cache_state(self) -> None:
        if self._load_manifest():
            self._save_manifest()
            return

        with self.disk_lock:
            migrated_entries = self._scan_legacy_disk_cache_locked()

        if migrated_entries > 0:
            self._save_manifest()

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

        manifest_data = None
        try:
            if not (meta := self.dict.pop(key, None)):
                return False

            path = meta.path
            size = meta.size
            if force:
                self.current_cache_size = max(self.current_cache_size - size, 0)
            self.usage = max(self.usage - size, 0)
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

            if force:
                self.cache_policy.update_on_force_evict(key)

            manifest_data = self._build_manifest_data_locked()
        finally:
            if force:
                self.disk_lock.release()

        if manifest_data is not None:
            self._write_manifest_data(manifest_data)

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

            memory_obj = self.local_cpu_backend.allocate(
                shape,
                dtype,
                fmt,
            )

            assert memory_obj is not None, (
                "Memory allocation failed during async disk load."
            )

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
        self._save_manifest()

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
            manifest_data = None
            with self.disk_lock:
                if disk_meta := self.dict.pop(key, None):
                    self.current_cache_size = max(
                        self.current_cache_size - disk_meta.size, 0
                    )
                    self.usage = max(self.usage - disk_meta.size, 0)
                    self.cache_policy.update_on_force_evict(key)
                    manifest_data = self._build_manifest_data_locked()
            self.stats_monitor.update_local_storage_usage(self.usage)
            if manifest_data is not None:
                self._write_manifest_data(manifest_data)
            return

        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self.disk_worker.close()
        self._save_manifest()
