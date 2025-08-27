# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import List, Optional, Sequence, Tuple
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
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
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


def pack_metadata(tensor, **extra_metadata) -> bytes:
    if tensor.dtype not in torch_dtypes:
        raise RuntimeError(f"unhandled dtype {tensor.dtype}")

    # Metadata
    data_size = tensor.numel() * tensor.element_size()
    tensor_meta = {
        "dtype": torch_dtypes[tensor.dtype],
        "shape": list(tensor.size()),
        "data_offsets": [0, data_size],
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

    nbytes = data_offsets[1] - data_offsets[0]
    dtype = torch_dtypes_inverse[dtype_str]

    return torch.Size(shape), dtype, nbytes, tensor_meta["__metadata__"]


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


class GdsBackend(StorageBackendInterface):
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
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        assert dst_device.startswith("cuda")
        super().__init__(dst_device)

        self.config = config
        self.loop = loop
        self.memory_allocator = memory_allocator
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

        self.use_direct_io = False

        if config.extra_config is not None:
            use_direct_io = get_extra_config_bool("use_direct_io", config)
            if use_direct_io is not None:
                self.use_direct_io = use_direct_io

        if not os.path.exists(self.gds_path):
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
                        key_str = filename[:-14].replace("_", "/")
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

        shape, dtype, size, extra_metadata = unpack_metadata(buf)
        if extra_metadata["lmcache_version"] != str(_METADATA_VERSION):
            raise RuntimeError("unhandled lmcache metadata")

        # TODO(extra_metadata)
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
        # TODO: implement pin() semantics
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
        # TODO: maybe remove `metadata_dirs` and insert mkdir calls
        # only for the case where creating the CuFile fails on ENOENT. It
        # also makes the code more resilient to out-of-band deletions
        if subdir_key not in self.metadata_dirs:
            os.makedirs(os.path.join(self.gds_path, l1_dir, l2_dir), exist_ok=True)
            self.metadata_dirs.add(subdir_key)
        tmp = ".tmp" + rand_suffix(self.rand, 8)
        metadata = await asyncio.to_thread(
            self._save_gds,
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

        # TODO(Jiayi): add `test_gds_backend_sanity` back after implementing this
        return False

    async def _async_load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_disk(key, path, dtype, shape)

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
        return self._load_bytes_from_disk(key, path, dtype=dtype, shape=shape)

    def _load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk.
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype)
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
        ret = self._load_gds(
            path, offset, addr, memory_obj.get_physical_size(), dev_offset
        )
        if ret != memory_obj.get_physical_size():
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
                    f"out of {memory_obj.get_physical_size()}, ignoring"
                )
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
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
        metadata = pack_metadata(kv_chunk, lmcache_version=str(_METADATA_VERSION))
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
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
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
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def remove(self, key: CacheEngineKey, force: bool = True):
        raise NotImplementedError("Remote backend does not support remove now.")

    def close(self) -> None:
        logger.info("GDS backend closed.")
