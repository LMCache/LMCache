# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, List, Optional, Sequence, Tuple, Union
import asyncio
import ctypes
import json
import mmap
import os
import random
import string
import struct
import threading
import time

# Third Party
import aiofile
import numpy as np
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    GPUMemoryAllocator,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_FULL_SUFFIX_LENGTH = len(_DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX)  # 29 characters
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata.
# TODO: It is possible to read this 4KB block without triggering read-ahead by
# various means.


class UnsupportedMetadataVersion(Exception):
    pass


torch_dtypes = {
    torch.half: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.uint8: "U8",
    torch.uint16: "U16",
    torch.uint32: "U32",
    torch.uint64: "U64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float8_e4m3fn: "F8E4M3FN",
    torch.float8_e5m2: "F8E5M2",
}


torch_dtypes_inverse = dict([(v, k) for k, v in torch_dtypes.items()])


def get_fstype(path):
    with open("/proc/mounts", "r") as f:
        lines = f.readlines()

    # Find the best matching mount point
    best_match = ""
    best_fstype = ""
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            _, mount_point, fstype = parts[0], parts[1], parts[2]
            if path.startswith(mount_point) and len(mount_point) > len(best_match):
                best_match = mount_point
                best_fstype = fstype

    if not best_fstype:
        raise RuntimeError(f"Unable to detect fstype for {path}")

    return best_fstype


def pack_metadata(tensor, fmt: MemoryFormat, **extra_metadata) -> bytes:
    if tensor.dtype not in torch_dtypes:
        raise RuntimeError(f"unhandled dtype {tensor.dtype}")

    # Metadata
    data_size = tensor.numel() * tensor.element_size()
    tensor_meta = {
        "dtype": torch_dtypes[tensor.dtype],
        "shape": list(tensor.size()),
        "data_offsets": [0, data_size],
        "fmt": fmt.value,
        "__metadata__": extra_metadata,
    }
    meta = {"kvcache": tensor_meta}
    str_meta = json.dumps(meta).encode("utf-8")
    meta_len = len(str_meta)
    assert meta_len <= _METADATA_MAX_SIZE - 8

    # Align to _METADATA_MAX_SIZE - 8
    str_meta += b" " * (_METADATA_MAX_SIZE - 8 - meta_len)

    # Pack it all up so it is sized _METADATA_MAX_SIZE exactly.
    return struct.pack("<Q", len(str_meta)) + str_meta


def unpack_metadata(buffer: bytes):
    meta_len = struct.unpack("<Q", buffer[:8])[0]

    str_meta = buffer[8 : 8 + meta_len]
    json_meta = str_meta.rstrip(b" ")

    meta = json.loads(json_meta.decode("utf-8"))
    tensor_meta = meta["kvcache"]

    shape = tensor_meta["shape"]
    dtype_str = tensor_meta["dtype"]
    data_offsets = tensor_meta["data_offsets"]
    fmt = MemoryFormat(tensor_meta["fmt"])

    nbytes = data_offsets[1] - data_offsets[0]
    dtype = torch_dtypes_inverse[dtype_str]

    return torch.Size(shape), dtype, nbytes, fmt, tensor_meta["__metadata__"]


def rand_suffix(rand, n: int):
    return "".join(
        rand.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )


async def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path + tmp
    async with aiofile.async_open(tmp_path, "wb") as f:
        await f.write(metadata)
    os.rename(tmp_path, path)


def get_extra_config_bool(key, config: LMCacheEngineConfig) -> bool | None:
    value = config.extra_config.get(key, None)
    if value is None:
        return None

    if isinstance(value, str):
        bool_value = value.lower() == "true"
    elif value in [False, True]:
        bool_value = value
    else:
        raise RuntimeError(f"Invalid value `{value}` for `{key}` in extra_config")

    logger.info(f"Getting {key} = {bool_value} from extra_config")
    return bool_value


class GdsBackend(AllocatorBackendInterface):
    """
    Originally based on the open sourced WekaGdsBackend, this is a backend that
    leverages NVIDIA's cuFile API to issue GDS requests directly to the
    GDS-supported remote filesystem.  In order to use it, users need to specify
    `gds_path` and `cufile_buffer_size` in their LMCache config.

    Cache Directory Structure created by this Backend:
    /{gds_path}/{first_level}/{second_level}/{data & metadata} This structure
    is semi-arbitrary. We create two levels in the directory hierarchy to
    parallelize loading the data during initialization in the Python code.

    NOTE: If GPUDirect is not supported on that other filesystem, then CuFile will
    fall back to POSIX I/O.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
    ):
        assert dst_device.startswith("cuda")
        super().__init__(dst_device=dst_device)

        self.config = config
        self.loop = loop
        self.dst_device = dst_device

        assert config.gds_path is not None, "Need to specify gds_path for GdsBackend"
        self.gds_path = config.gds_path
        self.fstype = get_fstype(config.gds_path)

        # Log the fstype - this is useful in reports and varying optimizations
        # based on the kind of fstype used.
        logger.info(
            f"GDS backend using fstype '{self.fstype}' on path '{self.gds_path}'"
        )

        # Determine use_cufile BEFORE initializing allocator
        self.use_cufile = True
        use_cufile_from_config = False

        if config.extra_config is not None:
            use_cufile = get_extra_config_bool("use_cufile", config)
            if use_cufile is not None:
                self.use_cufile = use_cufile
                use_cufile_from_config = True

        if self.fstype in ["tmpfs", "overlayfs"]:
            # TODO: we can replace the auto-detection of unsupported cufile
            # file systems by doing a small cufile API test on them. If as
            # read/write test fails, we can fallback to not using cufile APIs.
            if use_cufile_from_config:
                logger.warning("No automatic disabling of cufile usage due to fstype")
            else:
                logger.info("Automatic disabling of cufile usage due to fstype")
                self.use_cufile = False

        if self.use_cufile:
            logger.info("Using cufile")
            # HACK(Jiayi): cufile import is buggy on some hardware
            # (e.g., without GPUDirect), so it's temporarily put here.
            # Third Party
            import cufile

            self.cudart = None
            self.cufile = cufile
            self._cufile_driver = self.cufile.CuFileDriver()
        else:
            logger.info("Not using cufile")
            self.cufile = None
            self.cudart = ctypes.CDLL("libcudart.so")

        # Initialize allocator AFTER determining use_cufile
        self.memory_allocator = self.initialize_allocator(config, metadata)

        self.use_direct_io = False

        if config.extra_config is not None:
            use_direct_io = get_extra_config_bool("use_direct_io", config)
            if use_direct_io is not None:
                self.use_direct_io = use_direct_io

        if not os.path.exists(self.gds_path):
            os.makedirs(self.gds_path, exist_ok=True)

        self.stats = None  # TODO: plug into LMCache Statistics

        # Cache policy and size tracking for GDS eviction
        # Use max_cache_size as the GDS cache size limit (unified naming)
        self.cache_policy = get_cache_policy(config.cache_policy)
        max_gds_size = getattr(config, "max_gds_size", None)
        if max_gds_size is None or max_gds_size == 0:
            env_val = os.environ.get("LMCACHE_MAX_GDS_SIZE")
            if env_val is not None:
                try:
                    max_gds_size = float(env_val)
                except ValueError as e:
                    logger.warning(f"[GDS CACHE] Failed to parse LMCACHE_MAX_GDS_SIZE: {env_val}, error: {e}")
            else:
                max_gds_size = 0
        self.max_cache_size = int(max_gds_size * 1024**3)
        self.current_cache_size = 0

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.metadata_dirs: set[str] = set()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        self.rand = random.Random(self.dst_device)

        if hasattr(self.memory_allocator, "base_pointer"):
            logger.debug(f"Using base pointer {self.memory_allocator.base_pointer}")
            self.cufile_base_pointer = self.memory_allocator.base_pointer
        else:
            logger.info("No base pointer found, cufile will use bounce buffers")
            self.cufile_base_pointer = None
        asyncio.run_coroutine_threadsafe(self._scan_metadata(), self.loop)
        self.save_metadata_tasks: set[asyncio.Task] = set()

    async def _scan_metadata(self):
        # TODO: even though we only run it once on startup, this is still
        # not super scalable - test whether Rust code will be faster here, or
        # whether we can serialize meta-data in groups for faster loading.
        tasks = []
        start = time.perf_counter()
        with os.scandir(self.gds_path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l1_dir = os.path.basename(entry.name)
                if len(l1_dir) != 2:
                    continue
                tasks.append(
                    asyncio.to_thread(
                        self._scan_metadata_subdir,
                        os.path.join(self.gds_path, l1_dir),
                        l1_dir,
                    )
                )
        # TODO: If Python 3.11+, can we use TaskGroup instead?
        await asyncio.gather(*tasks)
        end = time.perf_counter()
        logger.info(
            f"Read {len(self.hot_cache)} cache entries from persistent "
            f"storage in {end - start:.2f} seconds"
        )

    def _scan_metadata_subdir(self, path, l1_dir):
        target_suffix = _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l2_dir = os.path.basename(entry.name)
                if len(l2_dir) != 2:
                    continue
                with os.scandir(os.path.join(path, l2_dir)) as it2:
                    for fentry in it2:
                        if not fentry.is_file():
                            continue
                        if not fentry.name.endswith(target_suffix):
                            continue
                        filename = os.path.basename(fentry.name)
                        key_str = filename[:-_FULL_SUFFIX_LENGTH].replace("_", "/")
                        try:
                            key = CacheEngineKey.from_string(key_str)
                        except ValueError as e:
                            logger.error(
                                f"Filename {filename} can't be converted "
                                f"back into cache key: {e}"
                            )
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for "
                                f"{fentry.path}, ignoring"
                            )

    def _read_metadata(self, key, filename, subdir_key):
        with open(filename, "rb") as f:
            buf = f.read(_METADATA_MAX_SIZE)

        shape, dtype, size, fmt, extra_metadata = unpack_metadata(buf)
        if extra_metadata["lmcache_version"] != str(_METADATA_VERSION):
            raise RuntimeError("unhandled lmcache metadata")
        logger.debug(
            f"Read metadata for {key} from {filename}: "
            f"shape={shape}, dtype={dtype}, size={size}, fmt={fmt}, "
            f"extra_metadata={extra_metadata}"
        )
        # TODO(extra_metadata)
        # TODO(Jiayi): need to support `cached_positions`.
        # Currently we just fill it as None.
        metadata = DiskCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX),
            size,
            shape,
            dtype,
            None,
            fmt,
        )
        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
            self.current_cache_size += size
        return metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether the key exists in the GDS cache.

        Compared to LocalDiskBackend, GDS supports a "lazy metadata loading" path:
        if the key is not in the in-memory hot_cache, we try to load its metadata
        directly from the filesystem (.metadata file). If found, the entry will be
        inserted into hot_cache on-demand.

        Args:
            key: CacheEngineKey to check.
            pin: If True, mark the entry as pinned so that eviction policy
                will not evict it.

        Returns:
            True if the key exists either in hot_cache or on disk; False otherwise.
        """

        # --- Fast path: found in in-memory cache ---
        with self.hot_lock:
            meta = self.hot_cache.get(key)
            if meta is not None:
                # Cache hit — optional recency update can also be placed here.
                if pin:
                    meta.pin()
                return True

        # --- Slow path: try to load metadata from disk lazily ---
        meta = self._try_to_read_metadata(key)
        if meta is None:
            return False

        # _try_to_read_metadata() already inserted the metadata into hot_cache.
        # If pin=True, we pin it after insertion.
        if pin:
            with self.hot_lock:
                cached_meta = self.hot_cache.get(key)
                if cached_meta is not None:
                    cached_meta.pin()
                else:
                    # Race condition: entry was evicted after being read from disk
                    # but before it could be pinned. We cannot honor pin=True.
                    logger.warning(f"Key {key} was evicted before it could be pinned.")
                    return False

        return True

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[DiskCacheMetadata]:
        path, subdir_key, _, _ = self._key_to_path(key)
        path += _METADATA_FILE_SUFFIX
        if os.path.exists(path):
            try:
                return self._read_metadata(key, path, subdir_key)
            except UnsupportedMetadataVersion:
                logger.error(f"Unsupported metadata version for {path}, ignoring")
        return None

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> Tuple[str, str, str, str]:
        hash = str(key.chunk_hash)
        l1_dir = hash[:2]
        l2_dir = hash[2:4]
        key_str = key.to_string()
        assert "_" not in key_str, "key string should not contain `_`"
        return (
            os.path.join(
                self.gds_path,
                l1_dir,
                l2_dir,
                key_str.replace("/", "_") + _DATA_FILE_SUFFIX,
            ),
            l1_dir + l2_dir,
            l1_dir,
            l2_dir,
        )

    def _maybe_evict_for(self, required_size: int) -> bool:
        """
        Evict cache entries if needed to make room for a new entry.
        
        Args:
            required_size: Size in bytes needed for the new entry
            
        Returns:
            True if eviction was successful (or not needed), False otherwise
        """
        if self.max_cache_size <= 0:
            # No size limit configured
            return True

        logger.debug(
            f"[GDS EVICT CHECK] current_cache_size={self.current_cache_size / 1024 / 1024:.2f} MB, "
            f"required_size={required_size / 1024 / 1024:.2f} MB, "
            f"max_cache_size={self.max_cache_size / 1024 / 1024:.2f} MB, "
            f"current_cache_size+required_size={ (self.current_cache_size + required_size) / 1024 / 1024:.2f} MB, "
            f"hot_cache_entries={len(self.hot_cache)}"
        )

        if self.current_cache_size + required_size <= self.max_cache_size:
            logger.debug(
                f"[GDS EVICT] Not triggered: current_cache_size + required_size <= max_cache_size ({self.current_cache_size + required_size:.0f} <= {self.max_cache_size:.0f} bytes)"
            )
            return True

        with self.hot_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                # Get eviction candidates from cache policy
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.hot_cache, num_candidates=1
                )
                if not evict_keys:
                    logger.debug(
                        "[GDS EVICTION] No eviction candidates found. "
                        f"current_cache_size={self.current_cache_size / 1024 / 1024:.2f} MB, "
                        f"max_cache_size={self.max_cache_size / 1024 / 1024:.2f} MB, "
                        f"required_size={required_size / 1024 / 1024:.2f} MB, "
                        f"hot_cache_entries={len(self.hot_cache)}"
                    )
                    return False

                for evict_key in evict_keys:
                    metadata = self.hot_cache.pop(evict_key, None)
                    if metadata is None:
                        continue

                    evict_size = metadata.size
                    evict_path = metadata.path

                    # Remove files from disk
                    try:
                        os.remove(evict_path)
                        os.remove(evict_path + _METADATA_FILE_SUFFIX)
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        logger.error(f"[GDS EVICTION] Error removing files: {e}")

                    self.current_cache_size -= evict_size
                    self.cache_policy.update_on_force_evict(evict_key)

                    logger.debug(
                        f"[GDS EVICTION] Evicted key={evict_key}, "
                        f"size={evict_size / 1024 / 1024:.2f} MB. "
                        f"current_cache_size={self.current_cache_size / 1024 / 1024:.2f} MB, "
                        f"hot_cache_entries={len(self.hot_cache)}"
                    )

        return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Future:
        assert memory_obj.tensor is not None

        # Skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        with self.put_lock:
            self.put_tasks.add(key)

        required_size = memory_obj.get_physical_size()
        
        # Perform eviction if needed before saving
        evict_success = self._maybe_evict_for(required_size)
        if not evict_success:
            with self.put_lock:
                self.put_tasks.discard(key)
            logger.warning(
                f"[GDS CACHE] Cannot store key={key}, eviction failed. "
                f"required_size={required_size / 1024 / 1024:.2f} MB"
            )
            return None

        # Update current size after successful eviction check
        with self.hot_lock:
            self.current_cache_size += required_size
        
        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        future = asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj), self.loop
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> Union[List[Future], None]:
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            future = self.submit_put_task(key, memory_obj)
            futures.append(future)
        return futures

    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        path, subdir_key, l1_dir, l2_dir = self._key_to_path(key)
        required_size = memory_obj.get_physical_size()
        
        try:
            # TODO: maybe remove `metadata_dirs` and insert mkdir calls
            # only for the case where creating the CuFile fails on ENOENT. It
            # also makes the code more resilient to out-of-band deletions
            if subdir_key not in self.metadata_dirs:
                os.makedirs(os.path.join(self.gds_path, l1_dir, l2_dir), exist_ok=True)
                self.metadata_dirs.add(subdir_key)
            tmp = ".tmp" + rand_suffix(self.rand, 8)
            fmt = memory_obj.metadata.fmt
            metadata = await asyncio.to_thread(
                self._save_gds,
                path,
                tmp,
                kv_chunk,
                fmt,
                self.cufile_base_pointer,
                memory_obj.metadata.address,
            )

            logger.debug(
                f"Saved {kv_chunk.numel()} elements of {kv_chunk.dtype} "
                f"to {path} with metadata {metadata}"
            )
            self.insert_key(key, memory_obj)

            task = asyncio.create_task(
                save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)
            )
            self.save_metadata_tasks.add(task)
            task.add_done_callback(self.save_metadata_tasks.discard)
        except Exception as e:
            # Rollback cache size on failure
            with self.hot_lock:
                self.current_cache_size -= required_size
            logger.error(f"[GDS CACHE] Failed to save key={key}: {e}")
            raise
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path, _, _, _ = self._key_to_path(key)
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        with self.hot_lock:
            # TODO(Jiayi): need to support `cached_positions`.
            self.hot_cache[key] = DiskCacheMetadata(path, size, shape, dtype, None, fmt)
            logger.info(
                f"[GDS CACHE] Inserted key={key}, size={size / 1024 / 1024:.2f} MB. "
                f"Total hot_cache entries: {len(self.hot_cache)}, "
                f"current_cache_size={self.current_cache_size / 1024 / 1024:.2f} MB"
            )

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> bool:
        # with self.hot_lock:
        #     entry = self.hot_cache.get(key)
        # if entry is None:
        #     return None

        # path = entry.path
        # dtype = entry.dtype
        # shape = entry.shape
        # fmt = entry.fmt
        # assert dtype is not None
        # assert shape is not None
        # assert fmt is not None
        # return asyncio.run_coroutine_threadsafe(
        #     self._async_load_bytes_from_disk(key, path, dtype, shape，fmt), self.loop
        # )

        # TODO(Jiayi): Need to modify this when prefetch interface is determined.

        # TODO(Jiayi): add `test_gds_backend_sanity` back after implementing this
        return False

    async def _async_load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_disk(key, path, dtype, shape, fmt=fmt)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.hot_lock:
            entry = self.hot_cache.get(key)
            if entry is None:
                return None
            # Update cache policy on hit (for LRU/LFU etc.)
            self.cache_policy.update_on_hit(key, self.hot_cache)

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        fmt = entry.fmt
        logger.debug(f"[GDS CACHE] Cache hit: key={key}")
        assert dtype is not None
        assert shape is not None
        assert fmt is not None
        return self._load_bytes_from_disk(key, path, dtype=dtype, shape=shape, fmt=fmt)

    def _load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk.
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync disk load.")
            return None
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.is_cuda
        assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)

        offset = _METADATA_MAX_SIZE
        if self.cufile_base_pointer is None:
            addr = ctypes.c_void_p(memory_obj.tensor.data_ptr())
            dev_offset = 0
        else:
            addr = ctypes.c_void_p(self.cufile_base_pointer)
            dev_offset = memory_obj.metadata.address
        ret = self._load_gds(path, offset, addr, memory_obj.get_size(), dev_offset)
        if ret != memory_obj.get_size():
            if ret < 0:
                logger.error(
                    f"Error loading {path}: ret: {ret} removing entry from cache"
                )
                with self.hot_lock:
                    self.hot_cache.pop(key)
            else:
                # TODO: we should probably count errors and
                # remove the entry if it's a persistent problem.
                logger.error(
                    f"Error loading {path}: got only {ret} bytes "
                    f"out of {memory_obj.get_size()}, ignoring"
                )
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        # TODO: Using a dummy wrapper around prefetch for now.
        if not self.submit_prefetch_task(key):
            return None
        return Future()

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_gds(
        self,
        path: str,
        tmp: str,
        kv_chunk: torch.Tensor,
        fmt: MemoryFormat,
        base_pointer: int,
        device_offset: int,
    ):
        if base_pointer is None:
            addr = ctypes.c_void_p(kv_chunk.data_ptr())
            dev_offset = 0
        else:
            addr = ctypes.c_void_p(base_pointer)
            dev_offset = device_offset
        tmp_path = path + tmp
        offset = _METADATA_MAX_SIZE
        # TODO: We can add the chunk's metadata here, e.g. Tensor parallelism shard
        # and pipeline parallelism index.
        metadata = pack_metadata(
            kv_chunk, fmt=fmt, lmcache_version=str(_METADATA_VERSION)
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata)
            if self.cufile:
                with self.cufile.CuFile(
                    tmp_path, "r+", use_direct_io=self.use_direct_io
                ) as f:
                    f.write(
                        addr, kv_chunk.nbytes, file_offset=offset, dev_offset=dev_offset
                    )
            elif self.cudart:
                # mmap the file
                fd = os.open(tmp_path, os.O_RDWR)
                nbytes = kv_chunk.nbytes
                os.ftruncate(fd, nbytes + offset)
                mm = mmap.mmap(
                    fd, nbytes + offset, prot=mmap.PROT_WRITE, flags=mmap.MAP_SHARED
                )
                os.close(fd)

                # get mapped file address
                arr = np.frombuffer(mm, dtype=np.uint8)
                buf_addr = arr.__array_interface__["data"][0]

                assert addr.value is not None
                res = self.cudart.cudaMemcpy(
                    ctypes.c_void_p(buf_addr + offset),
                    ctypes.c_void_p(int(addr.value) + device_offset),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(2),
                )
                if res:
                    raise RuntimeError(f"cudaMemcpy failed {res}")
                del arr
                mm.close()

        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            raise e
        os.rename(tmp_path, path)
        return metadata

    def _load_gds(
        self,
        gds_path: str,
        file_offset: int,
        gpu_pointer: ctypes.c_void_p,
        size_in_bytes: int,
        dev_offset: int,
    ) -> int:
        # Read data from disk into a GPU buffer
        if self.cufile:
            with self.cufile.CuFile(
                gds_path, "r", use_direct_io=self.use_direct_io
            ) as f:
                return f.read(
                    gpu_pointer,
                    size_in_bytes,
                    file_offset=file_offset,
                    dev_offset=dev_offset,
                )
        elif self.cudart:
            fd = os.open(gds_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd,
                file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,  # type: ignore [attr-defined]
            )
            os.close(fd)

            arr = np.frombuffer(mm, dtype=np.uint8)
            addr = arr.__array_interface__["data"][0]

            assert gpu_pointer.value is not None
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(int(gpu_pointer.value) + dev_offset),
                ctypes.c_void_p(addr + file_offset),
                ctypes.c_size_t(size_in_bytes),
                ctypes.c_int(1),
            )

            if res != 0:
                raise RuntimeError(f"cudaMemcpy failed with code {res}")
            del arr
            mm.close()
            return size_in_bytes
        else:
            raise RuntimeError(
                "Both cufile and cudart are None, this should not happen"
            )

    def pin(self, key: CacheEngineKey) -> bool:
        """
        Mark a cache entry as pinned so that the eviction policy
        should not evict it while it is in use.
        """
        with self.hot_lock:
            meta = self.hot_cache.get(key)
            if meta is None:
                return False
            meta.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """
        Unmark a cache entry as pinned, allowing the eviction policy
        to evict it if needed.
        """
        with self.hot_lock:
            meta = self.hot_cache.get(key)
            if meta is None:
                return False
            meta.unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True):
        raise NotImplementedError("Remote backend does not support remove now.")

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> MemoryAllocatorInterface:
        assert config.cufile_buffer_size is not None
        size = config.cufile_buffer_size * 1024**2
        if self.use_cufile:
            return CuFileMemoryAllocator(size)
        else:
            logger.debug("Using GPUMemoryAllocator instead of CuFileMemoryAllocator")
            return GPUMemoryAllocator(size, device=self.dst_device)

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if busy_loop:
            logger.warning("GDS Backend does not support allocation with busy loop")

        result = self.memory_allocator.allocate(shape, dtype, fmt)
        
        if result is None:
            # Calculate required size for logging
            element_size = torch.tensor([], dtype=dtype).element_size()
            required_size = shape.numel() * element_size
            logger.warning(
                f"[GDS GPU BUFFER] GPU staging allocator failed! "
                f"This is unrelated to GDS disk eviction. "
                f"Requested shape={shape}, dtype={dtype}, "
                f"required_size={required_size / 1024 / 1024:.2f} MB. "
                f"Consider increasing cufile_buffer_size or GPU memory."
            )
        
        return result

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("GDS Backend does not support allocation with busy loop")

        result = self.memory_allocator.batched_allocate(shape, dtype, batch_size, fmt)
        
        if result is None:
            # Calculate required size for logging
            element_size = torch.tensor([], dtype=dtype).element_size()
            required_size = shape.numel() * element_size * batch_size
            logger.warning(
                f"[GDS GPU BUFFER] Batched GPU staging allocator failed! "
                f"This is unrelated to GDS disk eviction. "
                f"Requested shape={shape}, dtype={dtype}, batch_size={batch_size}, "
                f"required_size={required_size / 1024 / 1024:.2f} MB. "
                f"Consider increasing cufile_buffer_size or GPU memory."
            )
        
        return result

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def close(self) -> None:
        self.memory_allocator.close()
        logger.info("GDS backend closed.")
