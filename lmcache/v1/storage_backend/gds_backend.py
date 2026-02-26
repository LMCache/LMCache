# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
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
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.utils.run_with_timeout import OperationManager, OperationTimeoutError

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_WEKA_DATA_FILE_SUFFIX = ".weka1"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata.
# TODO: It is possible to read this 4KB block without triggering read-ahead by
# various means.
_DEFAULT_THREAD_COUNT = 4


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


def get_timeout_value(
    key: str, config: LMCacheEngineConfig, default: float
) -> float:
    """
    Get timeout value from environment variable or config, with environment
    taking priority.
    
    Args:
        key: The config key name (e.g., "timeout_contains")
        config: The LMCache engine config
        default: Default value if not found in env or config
        
    Returns:
        The timeout value in seconds as a float
    """
    # Check environment variable first (priority)
    env_name = f"LMCACHE_{key.upper()}"
    env_value = os.getenv(env_name)
    if env_value is not None:
        try:
            timeout = float(env_value)
            logger.info(f"Using {key} = {timeout} from environment variable {env_name}")
            return timeout
        except ValueError:
            logger.warning(
                f"Invalid value '{env_value}' for {env_name}, falling back to config/default"
            )
    
    # Fall back to config
    if config.extra_config is not None:
        timeout = config.extra_config.get(key, default)
        if timeout != default:
            logger.info(f"Using {key} = {timeout} from config")
            return float(timeout)
    
    logger.info(f"Using {key} = {default} (default)")
    return default


class GdsBackend(AllocatorBackendInterface):
    """
    Originally based on the open sourced WekaGdsBackend, this is a backend that
    leverages NVIDIA's cuFile API to issue GDS requests directly to the
    GDS-supported remote filesystem.  In order to use it, users need to specify
    `gds_path` and `cufile_buffer_size` in their LMCache config.

    NOTE: If GPUDirect is not supported on that other filesystem, then CuFile will
    fall back to POSIX I/O.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
    ):
        assert dst_device.startswith("cuda")
        super().__init__(dst_device=dst_device)

        self.config = config
        self.layerwise = config.use_layerwise
        self.loop = loop
        self.memory_allocator = self.initialize_allocator(config, metadata)
        self.dst_device = dst_device

        assert config.gds_path is not None, "Need to specify gds_path for GdsBackend"
        self.gds_path = config.gds_path
        self.fstype = get_fstype(config.gds_path)

        # Log the fstype - this is useful in reports and varying optimizations
        # based on the kind of fstype used.
        logger.info(
            f"GDS backend using fstype '{self.fstype}' on path '{self.gds_path}'"
        )

        self.use_cufile = True
        use_cufile_from_config = False

        if config.extra_config is not None:
            use_cufile = get_extra_config_bool("use_cufile", config)
            if use_cufile is not None:
                self.use_cufile = use_cufile
                use_cufile_from_config = True

        self.data_suffix = _DATA_FILE_SUFFIX
        self.use_thread_pool = False
        self._thread_pool = None

        if self.fstype in ["tmpfs", "overlayfs"]:
            # TODO: we can replace the auto-detection of unsupported cufile
            # file systems by doing a small cufile API test on them. If as
            # read/write test fails, we can fallback to not using cufile APIs.
            if use_cufile_from_config:
                logger.warning("No automatic disabling of cufile usage due to fstype")
            else:
                logger.info("Automatic disabling of cufile usage due to fstype")
                self.use_cufile = False
        elif self.fstype == "wekafs":
            logger.info("Weka filesystem detected, cufile usage is enforced")
            assert self.use_cufile
            self.data_suffix = _WEKA_DATA_FILE_SUFFIX
            self.use_thread_pool = True

        if self.use_thread_pool:
            thread_count = _DEFAULT_THREAD_COUNT
            if config.extra_config is not None:
                thread_count = config.extra_config.get(
                    "gds_io_threads", _DEFAULT_THREAD_COUNT
                )
            self._thread_pool = ThreadPoolExecutor(
                max_workers=thread_count, thread_name_prefix="weka-gds-io"
            )

        # TODO allow control from env
        self.op_manager = OperationManager(
            config.extra_config.get("operation_manager_threads", 4),
        )
        self.timeout_contains = get_timeout_value("timeout_contains", config, 10.0)
        self.timeout_get_blocking = get_timeout_value("timeout_get_blocking", config, 10.0)
        self.timeout_batched_get_blocking = get_timeout_value(
            "timeout_batched_get_blocking", config, 10.0
        )

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

        self.use_direct_io = False

        # TODO allow control from env
        self.max_alloc_attempts = config.extra_config.get("max_alloc_attempts", 10)
        self.alloc_attempt_delay_secs = config.extra_config.get(
            "allocation_attempt_delay_secs", 0.1
        )
        self.enable_blending = config.extra_config.get("enable_blending", False)

        if config.extra_config is not None:
            use_direct_io = get_extra_config_bool("use_direct_io", config)
            if use_direct_io is not None:
                self.use_direct_io = use_direct_io

        if self.fstype == "wekafs":
            # Construct a descriptive directory name based on metadata
            # Format:
            # {model_name}-{world_size}-{kv_dtype}-{kv_shape}-{worker_id}[-layerwise]
            dtype_str = str(metadata.kv_dtype).replace("torch.", "")
            shape_str = "x".join(map(str, metadata.kv_shape))
            dir_components = [
                # Replace / in model names like "meta/Llama-2-7b"
                metadata.model_name.replace("/", "_"),
                str(metadata.world_size),
                dtype_str,
                shape_str,
                str(metadata.worker_id),
            ]
            if self.layerwise:
                dir_components.append("layerwise")
            metadata_dir = "-".join(dir_components)
            self.gds_path = os.path.join(config.gds_path, metadata_dir)
        os.makedirs(self.gds_path, exist_ok=True)

        self.stats = None  # TODO: plug into LMCache Statistics

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
        self.save_metadata_tasks: set[asyncio.Task] = set()

    def _read_metadata_info(self, filename: str):
        # Use O_NOATIME to prevent updating access time and improve performance
        # Instead of using Python's open() and read(), we use the OS's open() and
        # read() because it is faster - the metadata file is small and we don't
        # need any buffering.
        #
        # Additionally, we use O_NOATIME for two reasons:
        # 1. Improve performance
        # 2. To prevent updating the access time and preserve our LRU ordering
        #    when we get rid of the metadata file separation.
        fd = os.open(filename, os.O_RDONLY | os.O_NOATIME)
        try:
            buf = os.read(fd, _METADATA_MAX_SIZE)
        finally:
            os.close(fd)
        return unpack_metadata(buf)

    def _import_key_with_metadata(
        self,
        key: CacheEngineKey,
        filename: str,
        subdir_key: str,
    ):
        shape, dtype, size, fmt, extra_metadata = self._read_metadata_info(filename)
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

        # Set the appropriate memory format for layerwise operations
        if self.layerwise:
            fmt = MemoryFormat.KV_T2D

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
        return metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO: implement pin() semantics
        with self.hot_lock:
            res = key in self.hot_cache
        if res:
            return True
        return self._contains_slow_path(key)

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[DiskCacheMetadata]:
        path, subdir_key, _, _ = self._key_to_path(key)
        path = (path + _METADATA_FILE_SUFFIX).strip()

        try:
            flags = os.O_RDONLY | os.O_NONBLOCK
            fd = os.open(path, flags)
            with os.fdopen(fd, "rb"):
                return self._import_key_with_metadata(key, path, subdir_key)
        except FileNotFoundError:
            return None
        except PermissionError:
            print(f"DEBUG [GDS]: Permission Denied for PID {os.getpid()} on {path}")
            return None
        except UnsupportedMetadataVersion:
            logger.error(f"Unsupported metadata version for {path}, ignoring")
        except (OSError, IOError) as e:
            logger.error(
                f"Failed to read metadata file {path}: {type(e).__name__}: {e}. "
                f"File may be corrupted or inaccessible. "
                f"Ignoring cache entry for key {key}."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error reading metadata file {path}: "
                f"{type(e).__name__}: {e}. Ignoring cache entry for key {key}."
            )
        return None

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> Tuple[str, str, str, str]:
        # FIX: Handle bytes correctly to get actual hex characters
        if isinstance(key.chunk_hash, bytes):
            hash_str = key.chunk_hash.hex()
        else:
            hash_str = str(key.chunk_hash)

        l1_dir = hash_str[:2]
        l2_dir = hash_str[2:4]

        # Ensure key_str is also a clean string
        key_str = key.to_string()
        if key_str.startswith("b'") or key_str.startswith('b"'):
            # This is a fallback in case CacheEngineKey.to_string()
            # also has the byte-stringification bug
            # Standard
            import re

            key_str = re.sub(r"^b['\"]|['\"]$", "", key_str)

        assert "_" not in key_str, f"key string '{key_str}' should not contain `_`"

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

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to GDS asynchronously.

        :param on_complete_callback: Optional callback invoked after the GDS
            write completes. Callback exceptions are caught and logged.
        """
        assert memory_obj.tensor is not None
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj, on_complete_callback),
            self.loop,
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        if not keys or not memory_objs:
            return None

        # Batch setup (Synchronous & Thread-safe)
        for memory_obj in memory_objs:
            memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.update(keys)

        # Offload the entire batch as one operation
        # This returns a single Future for the background coroutine
        master_future = asyncio.run_coroutine_threadsafe(
            self._async_batched_submit(keys, memory_objs), self.loop
        )

        # Return a list of the same future repeated N times.
        return [master_future] * len(keys)

    async def _async_batched_submit(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        """
        Asynchronously submit multiple put tasks in batch.
        The loop happens in the async context so it doesn't block the caller.
        """
        # Create all async tasks
        tasks = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            task = self._async_save_bytes_to_disk(key, memory_obj)
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        try:
            """
            Convert KV to bytes and async store bytes to disk.

            :param on_complete_callback: Optional callback invoked after the GDS
                write completes for this key. Callback exceptions are caught.
            """
            kv_chunk = memory_obj.tensor
            assert kv_chunk is not None
            path, subdir_key, l1_dir, l2_dir = self._key_to_path(key)
            # TODO: maybe remove `metadata_dirs` and insert mkdir calls
            # only for the case where creating the CuFile fails on ENOENT. It
            # also makes the code more resilient to out-of-band deletions
            if subdir_key not in self.metadata_dirs:
                os.makedirs(os.path.join(self.gds_path, l1_dir, l2_dir), exist_ok=True)
                self.metadata_dirs.add(subdir_key)
            tmp = ".tmp" + rand_suffix(self.rand, 8)
            fmt = memory_obj.metadata.fmt
            try:
                metadata = await asyncio.to_thread(
                    self._save_gds,
                    path,
                    tmp,
                    kv_chunk,
                    fmt,
                    self.cufile_base_pointer,
                    memory_obj.metadata.address,
                )
            except Exception as e:
                logger.error(
                    f"GDS/cuFile write operation failed for key {key} at path {path}: "
                    f"tensor_shape={kv_chunk.shape}, tensor_dtype={kv_chunk.dtype}, "
                    f"tensor_size_bytes={kv_chunk.nbytes}, error={e}",
                    exc_info=True,
                )
                with self.put_lock:
                    self.put_tasks.discard(key)
                return

            # Register key in cache
            self.insert_key(key, memory_obj)

            try:
                task = asyncio.create_task(
                    save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)
                )
                self.save_metadata_tasks.add(task)
                task.add_done_callback(self.save_metadata_tasks.discard)
            except Exception as e:
                logger.error(
                    f"POSIX metadata write operation failed for key {key} at path "
                    f"{path + _METADATA_FILE_SUFFIX}: "
                    f"metadata_size_bytes={len(metadata)}, "
                    f"tmp_suffix={tmp}, error={e}",
                    exc_info=True,
                )
                with self.hot_lock:
                    self.hot_cache.pop(key, None)
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)

        # Call the completion callback if provided
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"on_complete_callback failed for key {key}: {e}")

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path, _, _, _ = self._key_to_path(key)
        # size = memory_obj.get_physical_size()
        size = memory_obj.get_size()  # Use logical size to match what's stored in file
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        with self.hot_lock:
            # TODO(Jiayi): need to support `cached_positions`.
            self.hot_cache[key] = DiskCacheMetadata(path, size, shape, dtype, None, fmt)

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
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_disk_with_allocation(key, path, dtype, shape)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        fmt = entry.fmt
        logger.warning(entry)
        assert dtype is not None
        assert shape is not None
        assert fmt is not None

        try:
            return self.op_manager.run_with_timeout(
                lambda: self._load_bytes_from_disk_with_allocation(
                    key, path, dtype=dtype, shape=shape
                ),
                self.timeout_get_blocking,
                "get_blocking",
                key,
            )
        except OperationTimeoutError:
            logger.error(
                f"Get blocking timed out after {self.timeout_get_blocking} seconds",
                exc_info=True,
            )
            return None

    def _load_bytes_from_disk_with_allocation(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk by first allocating memory, then loading.

        Args:
            key: Cache key for error handling
            path: File path to load from
            dtype: Data type for memory allocation
            shape: Shape for memory allocation

        Returns:
            A new memory object with loaded data, or None if allocation or
            loading failed
        """
        if self.layerwise:
            fmt = MemoryFormat.KV_T2D
        else:
            fmt = MemoryFormat.KV_2LTD
        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
        if memory_obj is None:
            logger.error("Memory allocation failed during sync disk load.")
            return None

        return self._load_bytes_from_disk_with_memory(key, path, memory_obj)

    def _load_bytes_from_disk_with_memory(
        self,
        key: CacheEngineKey,
        path: str,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk into a pre-allocated memory object.

        Args:
            key: Cache key for error handling
            path: File path to load from
            memory_obj: Pre-allocated memory object to load data into

        Returns:
            The memory object with loaded data, or None if loading failed
        """
        if memory_obj is None or memory_obj.tensor is None:
            return None

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

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        if self.use_thread_pool:
            logger.info("Using batched_get_blocking with thread pool implementation")
            try:
                return self.op_manager.run_with_timeout(
                    lambda: self._batched_get_blocking_by_thread_pool_impl(keys),
                    self.timeout_batched_get_blocking,
                    "batched_get_blocking",
                    len(keys),
                )
            except OperationTimeoutError:
                logger.error(
                    f"Batched get blocking timed out after "
                    f"{self.timeout_batched_get_blocking} seconds",
                    exc_info=True,
                )
                return [None] * len(keys)
        else:
            return super().batched_get_blocking(keys)

    def _batched_get_blocking_by_thread_pool_impl(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        paths: list[str | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    logger.error(f"Lookup failed during get_blocking for {key}")
                    paths.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    continue
                paths.append(entry.path)
                dtypes.append(entry.dtype)
                shapes.append(entry.shape)
        fmt = None
        if self.layerwise:
            fmt = MemoryFormat.KV_T2D
        else:
            fmt = MemoryFormat.KV_2LTD

        memory_objs: list[MemoryObj | None] = []
        gds_reads, gds_read_bytes = 0, 0
        for dtype, shape, path in zip(dtypes, shapes, paths, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
            if memory_obj is None:
                logger.error(f"Memory allocation failed during get_blocking for {path}")
            else:
                gds_reads += 1
                gds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
        assert self._thread_pool is not None
        results = list(
            self._thread_pool.map(
                self._load_bytes_from_disk_with_memory, keys, paths, memory_objs
            )
        )
        total_time = time.perf_counter() - start_time
        logger.info(
            f"Time taken for batched_get_blocking: {total_time:.3f}s |"
            f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        )
        return results

    async def _async_batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        """
        Asynchronously run the batched get operation in a thread pool.
        This allows the event loop to handle other operations while I/O is happening.
        """
        return await asyncio.to_thread(self.batched_get_blocking, keys)

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
        try:
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
        except Exception as e:
            logger.error(f"CuFile read failed for {gds_path}: {e}", exc_info=True)
            return -1

    def pin(self, key: CacheEngineKey) -> bool:
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def remove(self, key: CacheEngineKey, force: bool = True):
        raise NotImplementedError("Remote backend does not support remove now.")

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> CuFileMemoryAllocator:
        assert config.cufile_buffer_size is not None
        return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary.
        """
        logger.debug(
            f"Allocating memory with busy loop: {busy_loop} with eviction: {eviction}"
        )
        memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if memory_obj is not None:
            return memory_obj
        if not busy_loop:
            logger.error(
                "GDS allocation failed and busy loop is disabled. Returning None."
            )
            return None

        num_attempts = 0
        logger.warning(
            "GDS allocation failed and busy loop is enabled. "
            f"Waiting for {self.alloc_attempt_delay_secs} seconds before retrying."
        )
        while True:
            time.sleep(self.alloc_attempt_delay_secs)

            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:
                break
            num_attempts += 1
            logger.warning(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of GDS backend allocate()"
            )
            if num_attempts >= self.max_alloc_attempts:
                logger.error(
                    "GDS allocation failed after "
                    f"{self.max_alloc_attempts} attempts. Returning None."
                )
                if not self.memory_allocator.memcheck():
                    logger.error(
                        "GDS allocation failed and memory allocator "
                        "is inconsistent. This is a bug in the memory allocator."
                    )
                return None
        return memory_obj

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("GDS Backend does not support allocation with busy loop")

        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary.
        """
        logger.debug(
            f"Batched allocating memory in GDS backend"
            f" with busy loop: {busy_loop} with eviction: {eviction}"
        )

        memory_objs = self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt
        )

        if memory_objs is not None:
            return memory_objs
        if not busy_loop:
            logger.error(
                "GDS batched allocation failed and "
                "busy loop is disabled. Returning None."
            )
            return None

        num_attempts = 0
        logger.warning(
            "GDS batched allocation failed and busy loop is enabled. "
            f"Waiting for {self.alloc_attempt_delay_secs} seconds before retrying."
        )
        while True:
            time.sleep(self.alloc_attempt_delay_secs)

            memory_objs = self.memory_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
            if memory_objs:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of GDS backend batched_allocate()"
            )
            if num_attempts >= self.max_alloc_attempts:
                logger.error(
                    "GDS batched allocation failed after "
                    f"{self.max_alloc_attempts} attempts. Returning None."
                )
                if not self.memory_allocator.memcheck():
                    logger.error(
                        "GDS batched allocation failed and memory allocator "
                        "is inconsistent. This is a bug in the memory allocator."
                    )
                return None
        return memory_objs

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    async def _wait_for_metadata_tasks(self) -> None:
        """Wait for all pending metadata save tasks to complete."""
        if self.save_metadata_tasks:
            await asyncio.gather(*self.save_metadata_tasks, return_exceptions=True)

    def wait_for_metadata_tasks(self) -> None:
        """Synchronously wait for all pending metadata save tasks to complete."""
        if self.save_metadata_tasks:
            future = asyncio.run_coroutine_threadsafe(
                self._wait_for_metadata_tasks(), self.loop
            )
            future.result()

    def _contains_slow_path(self, key: CacheEngineKey) -> bool:
        try:
            read_from_disk = self.op_manager.run_with_timeout(
                lambda: self._try_to_read_metadata(key),
                self.timeout_contains,
                "contains",
                key,
            )
            if read_from_disk:
                return True
        except OperationTimeoutError:
            logger.error(
                f"Contains timed out after {self.timeout_contains} seconds",
                exc_info=True,
            )
        return False

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check whether keys are in the storage backend.

        :param lookup_id: Identifier for the lookup operation
        :param keys: The keys to check
        :param pin: Whether to pin the keys if they exist
        :return: Number of keys that exist in the storage backend
        """
        num_hit_chunks = 0
        while num_hit_chunks < len(keys):
            # Keep the lock as long as we keep getting hits
            # in the hot cache.
            with self.hot_lock:
                while (
                    num_hit_chunks < len(keys)
                    and keys[num_hit_chunks] in self.hot_cache
                ):
                    if pin:
                        # TODO(Serapheim): implement pin() semantics
                        pass
                    num_hit_chunks += 1

            # If we've processed all keys, return the count
            if num_hit_chunks == len(keys):
                return num_hit_chunks

            # Check the current key that's not in hot cache using slow path
            current_key = keys[num_hit_chunks]
            if self._contains_slow_path(current_key):
                num_hit_chunks += 1
            else:
                return num_hit_chunks

        return num_hit_chunks

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        Non-blocking function to get memory objects from storage.

        :param lookup_id: Identifier for the lookup operation
        :param keys: The keys to retrieve
        :return: List of MemoryObj instances
        """
        mem_objs: list[MemoryObj] = []
        entries: list[DiskCacheMetadata] = []

        # TODO(Serapheim): Do this properly

        # First, collect metadata for all keys
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                assert entry is not None, f"Key {key} not found in hot cache"
                entries.append(entry)

        # Load memory objects for each key
        for key, entry in zip(keys, entries, strict=True):
            assert entry is not None, f"Key {key} not found in hot cache"
            try:
                memory_obj = await self._async_load_bytes_from_disk(
                    key,
                    entry.path,
                    entry.dtype,
                    entry.shape,
                )
                if memory_obj is not None:
                    memory_obj.ref_count_up()
                    mem_objs.append(memory_obj)
            except Exception as e:
                logger.error(
                    f"Failed to load memory object for key {key}: {e}",
                    exc_info=True,
                )

        return mem_objs

    def close(self) -> None:
        self.memory_allocator.close()
        self.op_manager.shutdown()
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
        logger.info("GDS backend closed.")
