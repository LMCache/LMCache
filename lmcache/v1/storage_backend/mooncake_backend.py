# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional
import asyncio
import json
import operator
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# --- Helper Functions (integrated from utils) ---


def _dtype_element_size(dtype: torch.dtype) -> int:
    """Get the size of a dtype in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


def _infer_actual_shape(
    meta_shape: torch.Size,
    actual_elements: int,
    fmt: MemoryFormat,
) -> torch.Size:
    """
    Given:
        1. meta_shape       – the shape recorded in the checkpoint
                              (it may include padded / maximum lengths),
        2. actual_elements  – the number of elements physically present
                              in the file buffer,
        3. fmt              – the memory layout format (KV_2LTD, KV_MLA_FMT),
    infer the true runtime shape of the tensor.

    We currently support two formats:
      • KV_2LTD:   [ 2, num_layers, seq_len, hidden_dim ]
      • KV_MLA_FMT:[ 1, num_layers, num_tokens, aligned_head_size ]

    In both cases, all dimensions except the 3rd one are considered
    “static”.  We solve for the 3rd dimension so that
        prod(static_dims) * dynamic_dim == actual_elements
    """
    if fmt is MemoryFormat.KV_2LTD:
        # shape = [ 2, L, *, H ]
        shape = list(meta_shape)
        static_prod = shape[0] * shape[1] * shape[3]  # 2 * L * H
        if actual_elements % static_prod != 0:
            raise ValueError(
                f"actual_elements ({actual_elements}) is not divisible by "
                f"product of static dims ({static_prod})"
            )
        shape[2] = actual_elements // static_prod  # solve for seq_len
        return torch.Size(shape)

    if fmt is MemoryFormat.KV_MLA_FMT:
        # shape = [ 1, L, *, A ]
        shape = list(meta_shape)
        static_prod = shape[0] * shape[1] * shape[3]  # 1 * L * A
        if actual_elements % static_prod != 0:
            raise ValueError(
                f"actual_elements ({actual_elements}) is not divisible by "
                f"product of static dims ({static_prod})"
            )
        shape[2] = actual_elements // static_prod  # solve for num_tokens
        return torch.Size(shape)

    raise ValueError(f"Partial-chunk reshape is not implemented for {fmt}")


def reshape_partial_chunk(
    memory_obj: "MemoryObj",
    bytes_read: int,
    expected_shape: torch.Size,
    expected_dtype: torch.dtype,
    expected_fmt: "MemoryFormat",
) -> "MemoryObj":
    """
    Adjust the metadata of `memory_obj` when the file holds only a
    partial chunk of the original tensor.

    Steps
    -----
    1. Validate that `bytes_read` forms an integral number of elements.
    2. Compare the number of elements present vs. the expected total.
       a. If equal  → nothing to do, return as is.
       b. If larger → error (buffer over-run).
       c. If smaller→ compute the true shape and slice the raw bytes.
    """
    # 1. Element-size alignment check
    elem_size = _dtype_element_size(expected_dtype)
    if bytes_read % elem_size != 0:
        raise ValueError(
            f"bytes_read ({bytes_read}) is not aligned with element size ({elem_size})"
        )

    # 2. How many elements do we have vs. expect?
    actual_elements = bytes_read // elem_size
    expected_elements = reduce(operator.mul, expected_shape)

    if actual_elements == expected_elements:
        # Exact match – no reshaping required
        return memory_obj

    if actual_elements > expected_elements:
        # File contained more data than requested slice
        raise ValueError(
            f"Buffer over-run: actual_elements {actual_elements} "
            f"> expected {expected_elements}"
        )

    # 3. We have a truncated chunk – infer the runtime shape
    actual_shape = _infer_actual_shape(expected_shape, actual_elements, expected_fmt)

    # 4. Update the MemoryObj:
    actual_bytes = actual_elements * elem_size
    memory_obj.raw_data = memory_obj.raw_data[:actual_bytes]
    memory_obj.meta.shape = actual_shape

    return memory_obj


@dataclass
class MooncakeStoreConfig:
    """Configuration for the Mooncake store."""

    DEFAULT_GLOBAL_SEGMENT_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    DEFAULT_LOCAL_BUFFER_SIZE = 128 * 1024 * 1024  # 128MB
    DEFAULT_PROTOCOL = "tcp"
    DEFAULT_DEVICE_NAME = ""
    DEFAULT_TRANSFER_TIMEOUT = 1
    DEFAULT_METADATA_SERVER = "http://127.0.0.1:8080/metadata"
    DEFAULT_MASTER_SERVER_ADDRESS = "localhost:50051"
    DEFAULT_PREFER_LOCAL_ALLOC = False

    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    transfer_timeout: int
    prefer_local_alloc: bool

    @staticmethod
    def from_lmcache_config(config: "LMCacheEngineConfig") -> "MooncakeStoreConfig":
        """Load config from LMCacheEngineConfig."""
        extra_config = config.extra_config
        if extra_config is None:
            raise ValueError("The extra_config in LMCacheEngineConfig is not set.")

        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path:
            with open(config_file_path) as fin:
                file_config = json.load(fin)
        else:
            file_config = {}

        return MooncakeStoreConfig(
            local_hostname=file_config.get(
                "local_hostname", extra_config.get("local_hostname")
            ),
            metadata_server=file_config.get(
                "metadata_server",
                extra_config.get(
                    "metadata_server", MooncakeStoreConfig.DEFAULT_METADATA_SERVER
                ),
            ),
            global_segment_size=file_config.get(
                "global_segment_size",
                extra_config.get(
                    "global_segment_size",
                    MooncakeStoreConfig.DEFAULT_GLOBAL_SEGMENT_SIZE,
                ),
            ),
            local_buffer_size=file_config.get(
                "local_buffer_size",
                extra_config.get(
                    "local_buffer_size", MooncakeStoreConfig.DEFAULT_LOCAL_BUFFER_SIZE
                ),
            ),
            protocol=file_config.get(
                "protocol",
                extra_config.get("protocol", MooncakeStoreConfig.DEFAULT_PROTOCOL),
            ),
            device_name=file_config.get(
                "device_name",
                extra_config.get(
                    "device_name", MooncakeStoreConfig.DEFAULT_DEVICE_NAME
                ),
            ),
            master_server_address=file_config.get(
                "master_server_address",
                extra_config.get(
                    "master_server_address",
                    MooncakeStoreConfig.DEFAULT_MASTER_SERVER_ADDRESS,
                ),
            ),
            transfer_timeout=file_config.get(
                "transfer_timeout",
                extra_config.get(
                    "transfer_timeout", MooncakeStoreConfig.DEFAULT_TRANSFER_TIMEOUT
                ),
            ),
            prefer_local_alloc=file_config.get(
                "mooncake_prefer_local_alloc",
                extra_config.get(
                    "mooncake_prefer_local_alloc",
                    MooncakeStoreConfig.DEFAULT_PREFER_LOCAL_ALLOC,
                ),
            ),
        )


class MooncakeBackend(StorageBackendInterface):
    """
    A self-contained storage backend that uses Mooncake for distributed KV storage.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device)

        # Validate configuration to prevent conflicting Mooncake setups
        self._validate_mooncake_config(config)

        if metadata.fmt != "vllm":
            raise ValueError(
                "MooncakeBackend only supports vllm format. "
                f"Got {metadata.fmt} instead."
            )
        self.put_tasks = {}
        self.lock = threading.Lock()
        self.loop = loop
        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend
        self.blocking_timeout_secs = config.blocking_timeout_secs
        self.registered_buffer_ptr = None

        try:
            # Third Party
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError("Please install mooncake to use MooncakeBackend.") from e

        self.store_config = MooncakeStoreConfig.from_lmcache_config(config)
        logger.info("Mooncake Configuration loaded: %s", self.store_config)

        self.store = MooncakeDistributedStore()
        self.store.setup(
            self.store_config.local_hostname,
            self.store_config.metadata_server,
            self.store_config.global_segment_size,
            self.store_config.local_buffer_size,
            self.store_config.protocol,
            self.store_config.device_name,
            self.store_config.master_server_address,
        )
        self._register_cpu_buffer()
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1

        # Set preferred_segment based on configuration
        if self.store_config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.store.get_hostname()

        # Setup metadata context for reshaping
        chunk_size = self.metadata.kv_shape[2]
        num_kv_head = self.metadata.kv_shape[3]
        head_size = self.metadata.kv_shape[4]
        hidden_dim = num_kv_head * head_size

        # Use MLA flag from metadata
        self.use_mla = self.metadata.use_mla

        if self.use_mla:
            # KV_MLA_FMT format: [1, num_layers, num_tokens, aligned_head_size]
            self.chunk_shape = torch.Size(
                [1, self.metadata.kv_shape[0], chunk_size, hidden_dim]
            )
            self.chunk_fmt = MemoryFormat.KV_MLA_FMT
        else:
            # KV_2LTD format: [2, num_layers, num_tokens, hidden_dim]
            self.chunk_shape = torch.Size(
                [2, self.metadata.kv_shape[0], chunk_size, hidden_dim]
            )
            self.chunk_fmt = MemoryFormat.KV_2LTD

        self.chunk_dtype = self.metadata.kv_dtype

        # Precompute MLA worker id as 0 mode status (same as RemoteBackend)
        self._mla_worker_id_as0_mode = (
            config.extra_config is not None
            and config.extra_config.get("remote_enable_mla_worker_id_as0", False)
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )

        logger.info(
            f"MooncakeBackend initialized successfully. "
            f"Format: {'KV_MLA_FMT' if self.use_mla else 'KV_2LTD'}, "
            f"Shape: {self.chunk_shape}, "
            f"Dtype: {self.chunk_dtype}, "
            f"MLA worker_id_as_0 mode: {self._mla_worker_id_as0_mode}"
        )

    def _validate_mooncake_config(self, config: LMCacheEngineConfig) -> None:
        """
        Validate that Mooncake backend and remote connector are not enabled
        simultaneously.

        Args:
            config: LMCache engine configuration

        Raises:
            ValueError: If both Mooncake backend and remote connector are enabled
        """
        # Check if remote_url is set (indicating remote connector mode)
        if config.remote_url is not None:
            raise ValueError(
                "Cannot enable both Mooncake backend and remote connector "
                "simultaneously. Please use either enable_mooncake: True OR "
                "remote_url, but not both."
            )

        # Validate that enable_mooncake is True when MooncakeBackend is being used
        if not config.enable_mooncake:
            raise ValueError(
                "MooncakeBackend requires enable_mooncake=True in the configuration. "
                "Please set enable_mooncake: True in your LMCache configuration."
            )

    def _register_cpu_buffer(self):
        """Register CPU buffer for zero-copy operations."""
        try:
            allocator = self.local_cpu_backend.memory_allocator
            if hasattr(allocator, "pin_allocator") and hasattr(
                allocator.pin_allocator, "buffer"
            ):
                buffer = allocator.pin_allocator.buffer
                self.registered_buffer_ptr = buffer.data_ptr()
                result = self.store.register_buffer(buffer.data_ptr(), buffer.numel())
                if result == 0:
                    logger.info(
                        f"Registered: {hex(buffer.data_ptr())}, {buffer.numel()} bytes"
                    )
                else:
                    logger.warning(f"Buffer registration failed: error={result}")
                    self.registered_buffer_ptr = None
            else:
                self.registered_buffer_ptr = None
        except Exception as e:
            logger.error(f"Buffer registration error: {e}")
            self.registered_buffer_ptr = None

    def _unregister_cpu_buffer(self):
        """Unregister CPU buffer."""
        if self.registered_buffer_ptr is not None:
            result = self.store.unregister_buffer(self.registered_buffer_ptr)
            if result == 0:
                logger.info(f"Unregistered buffer: {hex(self.registered_buffer_ptr)}")
            else:
                logger.warning(f"Buffer unregistration failed: error={result}")
            self.registered_buffer_ptr = None

    def _normalize_mla_key(self, key: CacheEngineKey) -> CacheEngineKey:
        """
        Apply the `worker_id-as-0` rule when MLA mode is on.

        This keeps the logic in one place instead of spreading it across
        multiple call sites.
        """
        if self._mla_worker_id_as0_mode:
            return CacheEngineKey(
                key.fmt, key.model_name, key.world_size, 0, key.chunk_hash
            )
        return key

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        key = self._normalize_mla_key(key)
        result = self.store.is_exist(key.to_string())
        return result == 1

    def batched_contains(
        self, keys: List[CacheEngineKey], pin: bool = False
    ) -> List[bool]:
        keys = [self._normalize_mla_key(key) for key in keys]
        results = self.store.batch_is_exist([key.to_string() for key in keys])
        return [result == 1 for result in results]

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def batched_submit_put_task(
        self, keys: List[CacheEngineKey], objs: List[MemoryObj], transfer_spec=None
    ) -> Optional[List[Future]]:
        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return None

        # Filter out keys that are already being processed
        valid_pairs = []
        for key, memory_obj in zip(keys, objs, strict=False):
            # Validate format compatibility
            if self.use_mla:
                if memory_obj.meta.fmt != MemoryFormat.KV_MLA_FMT:
                    raise ValueError(
                        "MooncakeBackend configured for MLA format. "
                        f"Got {memory_obj.meta.fmt} instead."
                    )
            else:
                if memory_obj.meta.fmt != MemoryFormat.KV_2LTD:
                    raise ValueError(
                        "MooncakeBackend configured for KV_2LTD format. "
                        f"Got {memory_obj.meta.fmt} instead."
                    )
            if not self.exists_in_put_tasks(key):
                valid_pairs.append((key, memory_obj))

        if not valid_pairs:
            return []

        # Mark all keys as being processed
        with self.lock:
            for key, _ in valid_pairs:
                self.put_tasks[key] = time.time()
        return self._batch_put_optimized(valid_pairs)

    def _batch_put_optimized(self, valid_pairs: List[tuple]) -> List[Future]:
        """Optimized batch put using Mooncake's batch_put_from."""
        keys, memory_objs = zip(*valid_pairs, strict=False)
        key_strs = [key.to_string() for key in keys]
        buffer_ptrs = [obj.tensor.data_ptr() for obj in memory_objs]
        buffer_sizes = [
            obj.tensor.numel() * obj.tensor.element_size() for obj in memory_objs
        ]

        future = asyncio.run_coroutine_threadsafe(
            asyncio.to_thread(
                self.store.batch_put_from,
                key_strs,
                buffer_ptrs,
                buffer_sizes,
                self.replica_config,
            ),
            self.loop,
        )

        # Add callback to clean up all keys
        def batch_callback(f):
            with self.lock:
                for key in keys:
                    if key in self.put_tasks:
                        del self.put_tasks[key]
            try:
                results = f.result()
                for i, result in enumerate(results):
                    if result != 0:
                        logger.error(
                            f"Failed to put key {keys[i]}, error code: {result}"
                        )
            except Exception as e:
                logger.error(f"Batch put failed: {e}")

        future.add_done_callback(batch_callback)
        return [future]

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key = self._normalize_mla_key(key)
        future = self.get_non_blocking(key)
        if future is None:
            return None
        try:
            return future.result(timeout=self.blocking_timeout_secs)
        except TimeoutError:
            logger.warning(f"Timeout getting key {key} from Mooncake.")
            future.cancel()
            return None
        except Exception as e:
            logger.error(f"Error getting key {key} from Mooncake: {e}")
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """Submit a non-blocking get operation."""
        key = self._normalize_mla_key(key)
        return asyncio.run_coroutine_threadsafe(self._async_get(key), self.loop)

    async def _async_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """The async part of the get operation."""
        memory_obj = self.local_cpu_backend.allocate(
            self.chunk_shape, self.chunk_dtype, self.chunk_fmt
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory for Mooncake get")
            return None

        key_str = key.to_string()
        buffer_ptr = memory_obj.tensor.data_ptr()
        buffer_size = memory_obj.tensor.numel() * memory_obj.tensor.element_size()

        try:
            bytes_read = await asyncio.to_thread(
                self.store.get_into, key_str, buffer_ptr, buffer_size
            )
            if bytes_read <= 0:
                logger.warning(
                    f"Failed to read data for key {key_str}, code: {bytes_read}"
                )
                memory_obj.ref_count_down()
                return None

            return reshape_partial_chunk(
                memory_obj,
                bytes_read,
                self.chunk_shape,
                self.chunk_dtype,
                self.chunk_fmt,
            )
        except Exception as e:
            logger.error(f"Failed to get key {key_str} from Mooncake. {e}")
            memory_obj.ref_count_down()
            return None

    def batched_get_blocking(self, keys: List[CacheEngineKey]) -> List[MemoryObj]:
        """Optimized batch get operation."""
        if not keys:
            return []

        keys = [self._normalize_mla_key(key) for key in keys]
        return self._batch_get_optimized(keys)

    def _batch_get_optimized(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Internal helper that leverages Mooncake’s `batch_get_into` API to
        retrieve multiple chunks with a single round-trip.
        """
        # Reserve a buffer for every requested chunk.  We have to allocate the
        # maximum possible size because the store cannot tell us the size
        # beforehand.
        memory_objs: list[Optional[MemoryObj]] = []
        valid_idx: list[int] = []

        for i, _ in enumerate(keys):
            buf = self.local_cpu_backend.allocate(
                self.chunk_shape, self.chunk_dtype, self.chunk_fmt
            )
            memory_objs.append(buf)
            if buf is not None:
                valid_idx.append(i)

        if not valid_idx:
            logger.warning("Batch-get aborted: unable to allocate any buffers.")
            return [None] * len(keys)

        # Build the argument lists for the C++ call.
        key_strs = [keys[i].to_string() for i in valid_idx]
        buffer_ptrs = [memory_objs[i].tensor.data_ptr() for i in valid_idx]
        buffer_sizes = [
            memory_objs[i].tensor.numel() * memory_objs[i].tensor.element_size()
            for i in valid_idx
        ]

        try:
            # One RPC, many chunks.
            bytes_read_list = self.store.batch_get_into(
                key_strs, buffer_ptrs, buffer_sizes
            )

            # Assemble the final result list, defaulting to None.
            results: list[Optional[MemoryObj]] = [None] * len(keys)

            for i, n_read in zip(valid_idx, bytes_read_list, strict=False):
                if n_read <= 0:
                    logger.warning(
                        f"batch_get_into failed for key {keys[i]} (code={n_read})"
                    )
                    memory_objs[i].ref_count_down()
                    continue

                try:
                    results[i] = reshape_partial_chunk(
                        memory_objs[i],
                        n_read,
                        self.chunk_shape,
                        self.chunk_dtype,
                        self.chunk_fmt,
                    )
                except Exception as exc:
                    logger.error(f"Reshape failed for key {keys[i]}: {exc}")
                    memory_objs[i].ref_count_down()

            return results

        except Exception as exc:
            logger.error(f"Batch-get threw an exception: {exc}")
            # Release any buffers we successfully allocated.
            for i in valid_idx:
                memory_objs[i].ref_count_down()
            return [None] * len(keys)

    def submit_prefetch_task(self, key: CacheEngineKey) -> bool:
        """
        Submit a prefetch task for the given key.

        :param key: The key of the MemoryObj to prefetch.
        :return: True if the prefetch task was successfully submitted, False otherwise.
        """
        future = self.get_non_blocking(key)
        return future is not None

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug("Pin not supported by MooncakeBackend.")
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug("Unpin not supported by MooncakeBackend.")
        return True

    def close(self):
        logger.info("Closing MooncakeBackend.")

        # Unregister buffer before closing the store
        self._unregister_cpu_buffer()

        future = asyncio.run_coroutine_threadsafe(
            asyncio.to_thread(self.store.close), self.loop
        )
        try:
            future.result(timeout=5)
            logger.info("Mooncake store closed successfully.")
        except Exception as e:
            logger.error(f"Error closing Mooncake store: {e}")
