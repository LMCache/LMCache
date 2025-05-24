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
from pathlib import Path
from typing import List, Optional, no_type_check

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

METADATA_BYTES_LEN = 28


class FSConnector(RemoteConnector):
    """File system based connector that stores data in local files.

    Data is stored in the following format:
    - Each key is stored as a separate file
    - File content: metadata (METADATA_BYTES_LEN bytes) + serialized data
    """

    def __init__(self, base_path: str, loop: asyncio.AbstractEventLoop,
                 local_cpu_backend: LocalCPUBackend):
        """
        Args:
            base_path: Root directory to store all cache files
            loop: Asyncio event loop
            memory_allocator: Memory allocator interface
        """
        self.base_path = Path(base_path)
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        logger.info(f"Initialized FSConnector with base path {base_path}")
        # Create base directory if not exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: CacheEngineKey) -> Path:
        """Get file path for the given key"""
        key_path = key.to_string().replace("/", "-") + ".data"
        # Use key's string representation as filename
        return self.base_path / key_path

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system"""
        file_path = self._get_file_path(key)
        return file_path.exists()

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from file system"""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None

        try:
            # Read file content
            with open(file_path, 'rb') as f:
                data = f.read()

            # Split metadata and actual data
            metadata = RemoteMetadata.deserialize(
                memoryview(data[:METADATA_BYTES_LEN]))
            kv_bytes = data[METADATA_BYTES_LEN:METADATA_BYTES_LEN +
                            metadata.length]

            # Allocate memory and copy data
            memory_obj = self.local_cpu_backend.allocate(
                metadata.shape,
                metadata.dtype,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.warning("Failed to allocate memory during file read")
                return None

            if isinstance(memory_obj.byte_array, memoryview):
                view = memory_obj.byte_array
                if view.format == "<B":
                    view = view.cast("B")
            else:
                view = memoryview(memory_obj.byte_array)
            view[:metadata.length] = kv_bytes

            return memory_obj

        except Exception as e:
            logger.error(f"Failed to read from file {file_path}: {str(e)}")
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data to file system"""
        final_path = self._get_file_path(key)
        temp_path = final_path.with_suffix('.tmp')

        try:
            # Prepare metadata
            metadata = RemoteMetadata(len(memory_obj.byte_array),
                                      memory_obj.get_shape(),
                                      memory_obj.get_dtype(),
                                      memory_obj.get_memory_format())

            # Write to file (metadata + data)
            with open(temp_path, 'wb') as f:
                f.write(metadata.serialize())
                f.write(memory_obj.byte_array)

            # Atomically rename temp file to final destination
            temp_path.replace(final_path)

        except Exception as e:
            logger.error(f"Failed to write file {final_path}: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()  # Remove corrupted file
            raise
        finally:
            memory_obj.ref_count_down()

    @no_type_check
    async def list(self) -> List[str]:
        """List all keys in file system"""
        return [f.stem for f in self.base_path.glob("*.data")]

    async def close(self):
        """Clean up resources"""
        logger.info("Closed the file system connector")
