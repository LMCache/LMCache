# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple
import asyncio
import ctypes
import os
import random
import string
import struct
import threading
import time

# Third Party
import aiofile
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".weka1"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata


class UnsupportedMetadataVersion(Exception):
    pass


torch_dtypes = [
    torch.half,
    torch.float16,
    torch.bfloat16,
    torch.float,
    torch.float32,
    torch.float64,
    torch.double,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
dtype_to_idx = {dtype: idx for idx, dtype in enumerate(torch_dtypes)}


def pack_metadata(shape, dtype, size) -> bytes:
    metadata_desc = "<QQQQ" + len(shape) * "Q"
    if struct.calcsize(metadata_desc) > _METADATA_MAX_SIZE:
        # TODO(Serapheim/Ilya): support variable offset for data
        raise ValueError(
            f"Metadata size {struct.calcsize(metadata_desc)} "
            f"exceeds max size {_METADATA_MAX_SIZE}"
        )
    return struct.pack(
        metadata_desc, _METADATA_VERSION, dtype_to_idx[dtype], size, len(shape), *shape
    )


def unpack_metadata(buffer):
    version, dt_idx, size, ndim = struct.unpack_from("<QQQQ", buffer)
    shape_offset = struct.calcsize("<QQQQ")
    if version != _METADATA_VERSION:
        # TODO(Serapheim): When we bump the _METADATA_VERSION for
        # the first time, we need to ensure that we can still
        # read older versions.
        raise UnsupportedMetadataVersion(f"Unsupported metadata version: {version}")
    shape = struct.unpack_from("<" + ndim * "Q", buffer, offset=shape_offset)
    return torch.Size(shape), torch_dtypes[dt_idx], size


def rand_suffix(rand, n: int):
    return "".join(
        rand.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )


async def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path + tmp
    async with aiofile.async_open(tmp_path, "wb") as f:
        await f.write(metadata)
    os.rename(tmp_path, path)


class WekaGdsBackend(StorageBackendInterface):
    """
    This is a backend that leverages NVIDIA's cuFile API to issue GDS requests
    directly to the Weka Filesystem.  In order to use it, users need to specify
    `weka_path` and `cufile_buffer_size` in their LMCache config.

    Cache Directory Structure created by this Backend:
    /{weka_path}/{first_level}/{second_level}/{data & metadata}
    This structure is semi-arbitrary. WekaFS can handle/scale many small files
    into a single directory so we could just put all the data/metadata directly
    under the weka_path, but we create two levels in the directory hierarchy to
    parallelize loading the data during initialization in the Python code.

    NOTE: The `weka_path` does not strictly need to be a WekaFS mount so if you
    want to test the backend without Weka you are free to do so for testing
    purposes. For production though it wouldn't scale as this backend is
    tailored to the performance characteristics of WekaFS. More specifically if
    used with non-Weka filesystems performance will suffer potentially for two
    reasons:
    (1) If GPUDirect is not supported on that other filesystem, then CuFile will
        fall back to POSIX I/O.
    (2) Our cache directory structure creates a lot of small files within a
        single directory and uses 4K block/buffer sizes. These align very well
        with Weka but not other filesystems.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        # HACK(Jiayi): cufile import is buggy on some hardware
        # (e.g., without GPUDirect), so it's temporarily put here.
        # Third Party
        import cufile

        self.cufile = cufile

        assert dst_device.startswith("cuda")
        super().__init__(dst_device)

        self.config = config
        self.loop = loop
        self.memory_allocator = memory_allocator
        self.dst_device = dst_device

        assert config.weka_path is not None, (
            "Need to specify weka_path for WekaGdsBackend"
        )
        self.weka_path = config.weka_path
        if not os.path.exists(self.weka_path):
            os.makedirs(self.weka_path, exist_ok=True)

        self.stats = None  # TODO(Serapheim): plug into LMCache Statistics

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.metadata_dirs: set[str] = set()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        self.rand = random.Random(self.dst_device)
        thread_count = config.extra_config.get("gds_io_threads", 4)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_count, thread_name_prefix="weka-gds-io"
        )

        self._cufile_driver = self.cufile.CuFileDriver()
        assert hasattr(self.memory_allocator, "base_pointer")
        self.cufile_base_pointer = self.memory_allocator.base_pointer
        asyncio.run_coroutine_threadsafe(self._scan_metadata(), self.loop)
        self.save_metadata_tasks: set[asyncio.Task] = set()

    async def _scan_metadata(self):
        # TODO(Serapheim): even though we only run it once on startup,
        # this is still not super scalable maybe we need to add metadata
        # snapshotting later.
        tasks = []
        start = time.perf_counter()
        with os.scandir(self.weka_path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l1_dir = os.path.basename(entry.name)
                if len(l1_dir) != 2:
                    continue
                tasks.append(
                    asyncio.to_thread(
                        self._scan_metadata_subdir,
                        os.path.join(self.weka_path, l1_dir),
                        l1_dir,
                    )
                )
        # TODO(Serapheim): If Python 3.11+, can we use TaskGroup instead?
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
                        key_str = filename[: -len(target_suffix)].replace("_", "/")
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
        shape, dtype, size = unpack_metadata(buf)
        metadata = DiskCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX), size, shape, dtype
        )
        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
        return metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO(Serapheim): implement pin() semantics
        with self.hot_lock:
            res = key in self.hot_cache
        if res:
            return True
        if self._try_to_read_metadata(key):
            return True
        return False

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
                self.weka_path,
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

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Future:
        assert memory_obj.tensor is not None
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj), self.loop
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

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
        if subdir_key not in self.metadata_dirs:
            os.makedirs(os.path.join(self.weka_path, l1_dir, l2_dir), exist_ok=True)
            self.metadata_dirs.add(subdir_key)
        tmp = ".tmp" + rand_suffix(self.rand, 8)
        metadata = await asyncio.to_thread(
            self._save_gds_cufile,
            path,
            tmp,
            kv_chunk,
            self.cufile_base_pointer,
            memory_obj.metadata.address,
        )

        self.insert_key(key, memory_obj)
        memory_obj.ref_count_down()

        task = asyncio.create_task(
            save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)
        )
        self.save_metadata_tasks.add(task)
        task.add_done_callback(self.save_metadata_tasks.discard)
        with self.put_lock:
            self.put_tasks.discard(key)

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path, _, _, _ = self._key_to_path(key)
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        with self.hot_lock:
            self.hot_cache[key] = DiskCacheMetadata(path, size, shape, dtype)

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
        # assert dtype is not None
        # assert shape is not None
        # return asyncio.run_coroutine_threadsafe(
        #     self._async_load_bytes_from_disk(key, path, dtype, shape), self.loop
        # )

        # TODO(Jiayi): Need to modify this when prefetch interface is determined.
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
        assert dtype is not None
        assert shape is not None
        return self._load_bytes_from_disk_with_allocation(key, path, dtype, shape)

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
        assert memory_obj.tensor.is_cuda
        assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)
        # TODO(Jiayi): We can optimize a bit by reading size instead of physical size.
        ret = self._load_gds_cufile(
            path,
            _METADATA_MAX_SIZE,
            ctypes.c_void_p(self.cufile_base_pointer),
            memory_obj.get_physical_size(),
            memory_obj.metadata.address,
        )
        if ret != memory_obj.get_physical_size():
            if ret < 0:
                logger.error(
                    f"Error loading {path}: ret: {ret} removing entry from cache"
                )
                with self.hot_lock:
                    self.hot_cache.pop(key)
            else:
                # TODO(Serapheim): we should probably count errors and
                # remove the entry if it's a persistent problem.
                logger.error(
                    f"Error loading {path}: got only {ret} bytes "
                    f"out of {memory_obj.get_physical_size()}, ignoring"
                )
            memory_obj.ref_count_down()
            return None
        return memory_obj

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
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync disk load.")
            return None
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.is_cuda
        assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)

        return self._load_bytes_from_disk_with_memory(key, path, memory_obj)

    def batched_get_blocking(
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

        memory_objs: list[MemoryObj | None] = []
        gds_reads, gds_read_bytes = 0, 0
        for dtype, shape, path in zip(dtypes, shapes, paths, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            memory_obj = self.memory_allocator.allocate(shape, dtype)
            if memory_obj is None:
                logger.error(f"Memory allocation failed during get_blocking for {path}")
            else:
                gds_reads += 1
                gds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
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

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        # TODO(Serapheim): Using a dummy wrapper around prefetch for now.
        self.submit_prefetch_task(key)
        f: Future = Future()
        f.set_result(None)
        return f

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_gds_cufile(
        self,
        path: str,
        tmp: str,
        kv_chunk: torch.Tensor,
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
        metadata = pack_metadata(kv_chunk.shape, kv_chunk.dtype, kv_chunk.nbytes)
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata)
            with self.cufile.CuFile(tmp_path, "r+") as f:
                f.write(
                    addr, kv_chunk.nbytes, file_offset=offset, dev_offset=dev_offset
                )
        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            raise e
        os.rename(tmp_path, path)
        return metadata

    def _load_gds_cufile(
        self,
        file_path: str,
        file_offset: int,
        gpu_pointer: ctypes.c_void_p,
        size_in_bytes: int,
        dev_offset: int,
    ) -> int:
        # Read data from disk into a GPU buffer
        with self.cufile.CuFile(file_path, "r") as f:
            return f.read(
                gpu_pointer,
                size_in_bytes,
                file_offset=file_offset,
                dev_offset=dev_offset,
            )

    def pin(self, key: CacheEngineKey) -> bool:
        # TODO(Serapheim): Implement this
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        # TODO(Serapheim): Implement this
        return False

    def remove(self, key, force=True):
        raise NotImplementedError("Remote backend does not support remove now.")

    def close(self) -> None:
        self._thread_pool.shutdown(wait=True)
        logger.info("Weka backend closed.")
