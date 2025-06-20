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
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, no_type_check
import asyncio
import json
import operator
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    transfer_timeout: int

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
            transfer_timeout=config.get("transfer_timeout", 1),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_file_path)

    @staticmethod
    def load_from_lmcache_config(
        config: "LMCacheEngineConfig",
    ) -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        extra_config = config.extra_config
        if extra_config is None:
            raise ValueError("The extra config is not set.")
        return MooncakeStoreConfig(
            local_hostname=extra_config["local_hostname"],
            metadata_server=extra_config["metadata_server"],
            global_segment_size=extra_config.get("global_segment_size", 3355443200),
            local_buffer_size=extra_config.get("local_buffer_size", 1073741824),
            protocol=extra_config.get("protocol", "tcp"),
            device_name=extra_config.get("device_name", ""),
            master_server_address=extra_config["master_server_address"],
            transfer_timeout=extra_config.get("transfer_timeout", 1),
        )


class MooncakestoreConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        dev_name,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_config: Optional[LMCacheEngineConfig],
    ):
        super().__init__()
        try:
            # Third Party
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()
            config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
            if config_file_path is not None:
                self.config = MooncakeStoreConfig.from_file(config_file_path)
            elif lmcache_config is not None:
                self.config = MooncakeStoreConfig.load_from_lmcache_config(
                    lmcache_config
                )
            else:
                raise ValueError("MOONCAKE_CONFIG_PATH/lmcache_config must be provided")

            if host != "" and port != 0:
                self.config.master_server_address = host + ":" + str(port)
            if dev_name != "":
                self.config.device_name = dev_name
            logger.info("Mooncake Configuration loaded. config: %s", self.config)

            self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.lmcache_config = lmcache_config

        # Register CPU buffer with Mooncake for zero-copy operations
        self._register_cpu_buffer()

    def _register_cpu_buffer(self):
        """Register CPU buffer for zero-copy operations."""
        try:
            allocator = self.local_cpu_backend.memory_allocator
            if not hasattr(allocator, "pin_allocator") or not hasattr(
                allocator.pin_allocator, "buffer"
            ):
                logger.warning("Cannot register buffer: incompatible allocator")
                return

            buffer = allocator.pin_allocator.buffer
            result = self.store.register_buffer(buffer.data_ptr(), buffer.numel())

            if result == 0:
                logger.info(
                    f"Registered CPU buffer: {hex(buffer.data_ptr())}, "
                    f"{buffer.numel()} bytes"
                )
            else:
                logger.warning(f"Buffer registration failed: error={result}")
        except Exception as e:
            logger.error(f"Buffer registration error: {e}")

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string())

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if self.metadata_context is None:
            logger.error(
                "Metadata context not set. Cannot retrieve data without metadata."
            )
            return None

        metadata_shape = self.metadata_context.shape
        metadata_dtype = self.metadata_context.dtype
        metadata_fmt = self.metadata_context.fmt

        # Pre-allocate memory object
        memory_obj = self.local_cpu_backend.allocate(
            metadata_shape,
            metadata_dtype,
            metadata_fmt,
        )

        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        key_str = key.to_string()

        try:
            # Use get_into for zero-copy reading directly into the allocated buffer
            buffer_ptr = memory_obj.tensor.data_ptr()
            buffer_size = memory_obj.tensor.numel() * memory_obj.tensor.element_size()

            bytes_read = await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.get_into, key_str, buffer_ptr, buffer_size
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when getting key {key_str} from mooncake store."
                "The output may be incorrect."
            )
            return None
        except Exception as e:
            logger.error(f"Failed to get key {key_str}. {e}")
            return None

        if bytes_read < 0:
            logger.error(
                f"Failed to read data for key {key_str}, error code: {bytes_read}"
            )
            return None

        if bytes_read == 0:
            logger.warning(f"No data read for key {key_str}")
            return None

        if memory_obj.tensor is not None and bytes_read > 0:
            assert metadata_dtype is not None
            expected_data_size = (
                reduce(operator.mul, metadata_shape) * metadata_dtype.itemsize
            )
            actual_data_size = bytes_read

            if expected_data_size == actual_data_size:
                # Data size matches exactly, no need to copy since data is already
                # in place
                return memory_obj
            else:
                # Chunk is not full, need to reshape
                if actual_data_size % metadata_dtype.itemsize != 0:
                    logger.error(
                        f"Buffer size {actual_data_size} not aligned to dtype size "
                        f"{metadata_dtype.itemsize}"
                    )
                    return None

                actual_elements = actual_data_size // metadata_dtype.itemsize
                expected_elements = reduce(operator.mul, metadata_shape)

                if actual_elements > expected_elements:
                    logger.error(
                        f"Buffer has more elements ({actual_elements}) than expected "
                        f"({expected_elements})"
                    )
                    return None

                # Calculate actual shape for KV_2LTD format:
                # [2, num_layers, seq_len, hidden_dim]
                # Only sequence dimension (index 2) changes for unfull chunks
                actual_shape = list(metadata_shape)
                other_dims_product = actual_shape[0] * actual_shape[1] * actual_shape[3]
                actual_seq_len = actual_elements // other_dims_product
                actual_shape[2] = actual_seq_len

                actual_shape = torch.Size(actual_shape)

                # Validate the calculated shape
                if reduce(operator.mul, actual_shape) != actual_elements:
                    logger.error(
                        f"Cannot reshape {actual_elements} elements to shape "
                        f"{actual_shape}"
                    )
                    return None

                # Update the raw_data buffer to only include the actual elements
                # This is necessary because the tensor property uses
                # raw_data.view(dtype).view(shape)
                actual_bytes = actual_elements * metadata_dtype.itemsize
                memory_obj.raw_data = memory_obj.raw_data[:actual_bytes]

                # Update metadata shape to match actual data
                memory_obj.meta.shape = actual_shape

                # Data is already in the correct location due to get_into
                return memory_obj
        else:
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        try:
            # Check if we can use zero-copy put_from
            if hasattr(memory_obj, "tensor") and memory_obj.tensor is not None:
                # Use put_from for zero-copy writing directly from the tensor buffer
                buffer_ptr = memory_obj.tensor.data_ptr()
                buffer_size = (
                    memory_obj.tensor.numel() * memory_obj.tensor.element_size()
                )

                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.store.put_from, key_str, buffer_ptr, buffer_size
                    ),
                    timeout=self.config.transfer_timeout,
                )

                if result != 0:
                    logger.error(
                        f"Failed to put key {key_str} using put_from, "
                        f"error code: {result}"
                    )
            else:
                # Fallback to regular put method
                kv_bytes = memory_obj.byte_array
                await asyncio.wait_for(
                    asyncio.to_thread(self.store.put, key_str, kv_bytes),
                    timeout=self.config.transfer_timeout,
                )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when putting key {key_str} from mooncake store."
                "Decode instance may redo prefill."
            )
        except Exception as e:
            logger.error(f"Failed to put key {key_str},data: {type(memory_obj)}: {e}")

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.store.close()
        logger.info("Closed the mooncake store connection")
