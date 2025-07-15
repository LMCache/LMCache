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
from pathlib import Path
from typing import List, Optional, no_type_check
import asyncio

# Third Party
import aiofiles
import aiofiles.os

# First Party
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

    def __init__(
        self,
        base_path: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
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
        return await aiofiles.os.path.exists(file_path)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from file system"""
        file_path = self._get_file_path(key)

        try:
            async with aiofiles.open(file_path, "rb") as f:
                # Read metadata buffer first to get shape, dtype, fmt
                # to be able to allocate memory object for the data and read into it
                md_buffer = bytearray(METADATA_BYTES_LEN)
                num_read = await f.readinto(md_buffer)
                if num_read != len(md_buffer):
                    raise RuntimeError(
                        f"Partial read meta {len(md_buffer)} got {num_read}"
                    )

                # Deserialize metadata and allocate memory
                metadata = RemoteMetadata.deserialize(md_buffer)
                memory_obj = self.local_cpu_backend.allocate(
                    metadata.shape, metadata.dtype, metadata.fmt
                )
                if memory_obj is None:
                    logger.debug("Memory allocation failed during async disk load.")
                    return None

                # Read the actual data into allocated memory
                buffer = memory_obj.byte_array
                num_read = await f.readinto(buffer)
                if num_read != len(buffer):
                    raise RuntimeError(
                        f"Partial read data {len(buffer)} got {num_read}"
                    )

            return memory_obj

        except FileNotFoundError:
            # Key does not exist is normal case
            return None
        except Exception as e:
            logger.error(f"Failed to read from file {file_path}: {str(e)}")
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data to file system"""
        final_path = self._get_file_path(key)
        temp_path = final_path.with_suffix(".tmp")

        try:
            # Prepare metadata
            buffer = memory_obj.byte_array
            metadata = RemoteMetadata(
                len(buffer),
                memory_obj.get_shape(),
                memory_obj.get_dtype(),
                memory_obj.get_memory_format(),
            )

            # Write to file (metadata + data)
            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(metadata.serialize())
                await f.write(buffer)

            # Atomically rename temp file to final destination
            await aiofiles.os.replace(temp_path, final_path)

        except Exception as e:
            logger.error(f"Failed to write file {final_path}: {str(e)}")
            if await aiofiles.os.path.exists(temp_path):
                await aiofiles.os.unlink(temp_path)  # Remove corrupted file
            raise

    @no_type_check
    async def list(self) -> List[str]:
        """List all keys in file system"""
        return [f.stem for f in self.base_path.glob("*.data")]

    async def close(self):
        """Clean up resources"""
        logger.info("Closed the file system connector")
