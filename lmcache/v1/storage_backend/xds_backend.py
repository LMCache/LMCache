# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from functools import partial
import csv
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
import file_p2p
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

if TYPE_CHECKING:
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

_METADATA_VERSION = 1
_FILE_SUFFIX = ".pt"
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata.

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

class XdsBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        # why must start with cuda?
        assert dst_device.startswith("cuda")
        super().__init__(dst_device)

        self.block_size = config.block_size

        self.config = config
        self.loop = loop
        self.memory_allocator = memory_allocator
        self.dst_device = dst_device

        assert config.xds_path is not None, "Need to specify xds_path for XdsBackend"
        self.xds_path = config.xds_path
        self.bdev = config.xds_bdev

        if not os.path.exists(self.xds_path):
            os.makedirs(self.xds_path, exist_ok=True)

        self.stats = None  # TODO: plug into LMCache Statistics
        self._executor = ThreadPoolExecutor(max_workers=4)

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.metadata_dirs: set[str] = set()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        asyncio.run_coroutine_threadsafe(self._scan_metadata(), self.loop)
        self.save_metadata_tasks: set[asyncio.Task] = set()
    
    async def _scan_metadata(self):
        tasks = []
        with os.scandir(self.xds_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if not entry.name.lower().endswith(_FILE_SUFFIX):
                    continue
                key = entry.name[:-len(_FILE_SUFFIX)]
                tasks.append(asyncio.to_thread(self._read_metadata, key, entry.path))
        await asyncio.gather(*tasks)
    
    def _read_metadata(self, key: CacheEngineKey, filename: str):
        with open(filename, "rb") as f:
            buffer = f.read(_METADATA_MAX_SIZE)
        shape, dtype, size, extra_metadata = unpack_metadata(buffer)
        if extra_metadata["version"] != str(_METADATA_VERSION):
            raise UnsupportedMetadataVersion(
                f"Unsupported metadata version {extra_metadata['version']}"
            )
        metadata = DiskCacheMetadata(
            filename, size, shape, dtype
        )
        real_key = CacheEngineKey.from_string(key)
        real_key.model_name = real_key.model_name.replace("_NDS_", "/")
        with self.hot_lock:
            self.hot_cache[real_key] = metadata
        return metadata

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
        path = self._key_to_path(key)
        if os.path.exists(path):
            try:
                return self._read_metadata(key, path)
            except UnsupportedMetadataVersion:
                logger.error(f"Unsupported metadata version for {path}, ignoring")
        return None
    
    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        hash = str(key.chunk_hash)
        key_str = key.to_string()
        assert "_XDS_" not in key_str, "key string should not contain `_XDS_`"
        return os.path.join(
                self.xds_path,
                key_str.replace("/", "_XDS_") + _FILE_SUFFIX,
            )

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks
        
    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj, path: str) -> None:
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        with self.hot_cache:
            self.hot_cache[key] = DiskCacheMetadata(
                path, size, shape, dtype
            )
    
    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        assert memory_obj.tensor is not None

        with self.put_lock:
            self.put_tasks.add(key)
        
        asyncio.run(self._async_save_bytes_to_disk(key, memory_obj))

    def batched_submit_put_task(self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj], transfer_spec=None) -> Optional[List[Future]]:
        return [
            self.submit_put_task(key, memory_obj)
            for key, memory_obj in zip(keys, memory_objs, strict=False)
        ]

    async def _async_save_bytes_to_disk(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None

        byte_array = memory_obj.byte_array
        path = self._key_to_path(key)
        size = len(byte_array)
        metadata = pack_metadata(memory_obj.tensor, lmcache_version=str(_METADATA_VERSION))

        with open(path, "wb+") as f:
            f.write(metadata)
            f.seek(_METADATA_MAX_SIZE)
            f.write(byte_array)
            f.flush()
            os.fsync(f.fileno())
        
        self.insert_key(key, memory_obj, path)

        memory_obj.ref_count_down()
        with self.put_lock:
            self.put_tasks.discard(key)
        
    def get_xds_non_blocking(
        self,
        keys: List[CacheEngineKey],
        kv_pointers: List[torch.Tensor],
        slot_mappings: torch.Tensor,
        starts: List[int],
        ends: List[int]
    ):
        entrys = []
        with self.hot_lock:
            for key in keys:
                entrys.append(self.hot_cache.get(key))
        
        asyncio.run(self._load_bytes_from_disk(keys, kv_pointers, slot_mappings, starts, ends, entrys))

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        raise NotImplementedError("get_blocking is not implemented for XdsBackend")
    
    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        raise NotImplementedError("get_non_blocking is not implemented for XdsBackend")
    
    def submit_get_task(self, key: CacheEngineKey) -> None:
        raise NotImplementedError("submit_get_task is not implemented for XdsBackend")
    
    async def _async_load_bytes_from_disk(
        self,
        keys: List[CacheEngineKey],
        kv_pointers: List[torch.Tensor],
        slot_mappings: torch.Tensor,
        starts: List[int], 
        ends: List[int],
        entrys: List[DiskCacheMetadata]
    ) -> Optional[MemoryObj]:
        
        page_buffer_size = kv_pointers[0].shape[1] * kv_pointers[0].shape[2]
        kv_size = kv_pointers[0].shape[0]
        element_size = kv_pointers[0].element_size()
        dimm_size = kv_pointers[0].shape[3] * kv_pointers[0].shape[4]
        layer_size = len(kv_pointers)

        kv_ptr_host = []
        for layer in range(0, layer_size):
            ptr = int(kv_pointers[layer].data_ptr())
            if torch.is_tensor(ptr):
                kv_ptr_host.append(int(ptr.cpu().item()))
            else:
                kv_ptr_host.append(int(ptr))
        
        key_tasks = []
        loop = asyncio.get_running_loop
        for key, start, end, entry in zip(keys, starts, ends, entrys, strict=False):
            key_tasks.append(
                loop.run_in_executor(
                    self.executor,
                    self._process_one_key,
                    key,
                    start,
                    end,
                    entry,
                    kv_ptr_host,
                    kv_size,
                    layer_size,
                    page_buffer_size,
                    dimm_size,
                    element_size,
                    slot_mappings
                )
            )
        
        await asyncio.gather(*key_tasks, return_exceptions=False)
        file_p2p.drain_read()
    
    def _process_one_key(
        self,
        key: CacheEngineKey,
        start: int,
        end: int,
        entry: DiskCacheMetadata,
        kv_ptr: List[int],
        kv_size: int,
        layer_size: int,
        page_buffer_size: int,
        dimm_size: int,
        element_size: int,
        slot_mappings: torch.Tensor,
    ):
        # 1. 提取常量，转化数据类型
        dtype = np.int64
        slot_mapping = slot_mappings[start:end].cpu().numpy().astype(dtype)
        token_length = end - start
        sm_shape = slot_mapping.shape[0]
        block_size = self.block_size
        base_copy = dimm_size * element_size
        
        # 2. 转换torch为numpy
        kv_ptr_np = np.array(kv_ptr, dtype=dtype)                           # List[int] -> np.array

        # 3. 生成三个维度基础索引数组
        kv_indices = np.arange(kv_size, dtype=dtype)                        # [0,1,...,kv_size-1]
        layer_indices = np.arange(layer_size, dtype=dtype)                  # [0,1,...,layer_size-1]
        slot_indices = np.arange(0, token_length, block_size, dtype=dtype)  # [0,block_size,...,token_length-1]，步长为block_size

        # 4. 生成三个维度的笛卡尔积
        kv_mesh, layer_mesh, slot_mesh = np.meshgrid(kv_indices, layer_indices, slot_indices, indexing='ij')
        cart_prod = np.stack([kv_mesh.ravel(), layer_mesh.ravel(), slot_mesh.ravel()], axis=1)

        # 5. 拆分笛卡尔积为单维度数组
        kv_all = cart_prod[:, 0]
        layer_all = cart_prod[:, 1]
        slot_all = cart_prod[:, 2]

        #6. 向量化计算copy_size
        mask = (slot_all + block_size) < token_length
        copy_size = np.where(mask, base_copy * block_size, base_copy * (token_length - slot_all)).astype(dtype)

        # 7. 向量化计算HBM偏移量
        slot_map_vals = slot_mapping[slot_all]
        hbm_off = slot_map_vals * dimm_size + kv_all * page_buffer_size * dimm_size

        # 8. 向量化计算disk偏移量
        disk_mid = kv_all * layer_size * sm_shape + layer_all * sm_shape + slot_all
        disk_off = disk_mid * dimm_size * element_size + _METADATA_MAX_SIZE
        disk_off = disk_off.astype(dtype)

        # 9. 向量化计算HBM地址
        kv_ptr_vals = kv_ptr_np[layer_all]
        hbm_addr = kv_ptr_vals + hbm_off * element_size
        hbm_addr = hbm_addr.astype(dtype)

        # 10. 拼接结果
        zeros = np.zeros_like(copy_size, dtype=dtype)
        entries_np = np.column_stack([disk_off, hbm_addr, copy_size, zeros, zeros])
        entries = entries_np.tolist()
        
        self._load_xds_batched(entry.path, entries)

    def _load_xds_batched(self, path: str, entries: List[int]):
        ret = file_p2p.read_file_batch(path, self.bdev, entries)
        if ret != 0:
            logger.error(f"read_file_batch failed, ret: {ret}")

    def pin(self, key: CacheEngineKey):
        return
    def unpin(self, key: CacheEngineKey) -> bool:
        return
    def remove(self, key: CacheEngineKey, free_obj: bool = True) -> bool:
        raise NotImplementedError("remove not implemented in XDSBackend")
    def close(self):
        logger.info("close XDSBackend")
