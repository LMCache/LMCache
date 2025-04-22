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

import asyncio
import inspect
import json
import os
from dataclasses import dataclass
from typing import List, Optional, no_type_check

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
# reuse
from lmcache.experimental.protocol import RedisMetadata
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

METADATA_BYTES_LEN = 28


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeStoreConfig':
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
        )

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)


class MooncakestoreConnector(RemoteConnector):

    def __init__(self, host: str, port: int, dev_name,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            if host != "" and port != 0:
                self.config.master_server_address = host + ":" + str(port)
            if dev_name != "":
                self.config.device_name = dev_name
            logger.info("Mooncake Configuration loaded. config: %s",
                        self.config)

            self.store.setup(self.config.local_hostname,
                             self.config.metadata_server,
                             self.config.global_segment_size,
                             self.config.local_buffer_size,
                             self.config.protocol, self.config.device_name,
                             self.config.master_server_address)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        self.memory_allocator = memory_allocator
        self.loop = loop

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string() + "metadata")

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        metadata_bytes = self.store.get(key_str + "metadata")
        if metadata_bytes is None or len(metadata_bytes) != METADATA_BYTES_LEN:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RedisMetadata.deserialize(memoryview(metadata_bytes))

        memory_obj = self.memory_allocator.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        kv_bytes = self.store.get(key_str + "kv_bytes")
        assert not inspect.isawaitable(kv_bytes)
        assert kv_bytes is not None

        view = memoryview(memory_obj.byte_array)
        view[:metadata.length] = kv_bytes

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape, kv_dtype,
                                       memory_format).serialize()

        key_str = key.to_string()
        self.store.put(key_str + "metadata", metadata_bytes)
        key_byte_str = key_str + "kv_bytes"
        try:
            self.store.put(key_byte_str, kv_bytes)
        except Exception as e:
            logger.error(f"Failed to put key {key_byte_str},"
                         f"meta type: {type(metadata_bytes)},"
                         f"data: {type(kv_bytes)}: {e}")

        self.memory_allocator.ref_count_down(memory_obj)

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.store.close()
        logger.info("Closed the mooncake store connection")
