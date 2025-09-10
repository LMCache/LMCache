# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import List, Optional, no_type_check
import asyncio
import os

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
        base_paths_str: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        """
        Args:
            base_paths_str: Comma separated storage paths
            loop: Asyncio event loop
            local_cpu_backend: Memory allocator interface
        """
        # Parse comma separated paths
        self.base_paths = (
            [Path(p.strip()) for p in base_paths_str.split(",")]
            if "," in base_paths_str
            else [Path(base_paths_str)]
        )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        logger.info(f"Initialized FSConnector with base paths {self.base_paths}")
        # Create directories for all paths
        for path in self.base_paths:
            path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: CacheEngineKey) -> Path:
        """Get file path for the given key"""
        # If there's only one path, use it directly
        if len(self.base_paths) == 1:
            base_path = self.base_paths[0]
        else:
            # Calculate hash value and modulo to select path
            hash_val = abs(key.chunk_hash)
            idx = hash_val % len(self.base_paths)
            base_path = self.base_paths[idx]

        key_path = key.to_string().replace("/", "-") + ".data"
        return base_path / key_path

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system"""
        file_path = self._get_file_path(key)
        return await aiofiles.os.path.exists(file_path)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system synchronized"""
        file_path = self._get_file_path(key)
        return os.path.exists(file_path)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from file system"""
        file_path = self._get_file_path(key)

        try:
            async with aiofiles.open(file_path, "rb") as f:
                if self.save_chunk_meta:
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
                else:
                    memory_obj = self.local_cpu_backend.allocate(
                        self.meta_shape, self.meta_dtype, self.meta_fmt
                    )
                if memory_obj is None:
                    logger.debug("Memory allocation failed during async disk load.")
                    return None

                # Read the actual data into allocated memory
                buffer = memory_obj.byte_array
                num_read = await f.readinto(buffer)
                if self.save_chunk_meta:
                    if num_read != len(buffer):
                        raise RuntimeError(
                            f"Partial read data {len(buffer)} got {num_read}"
                        )
                else:
                    # reshape and check
                    assert num_read is not None
                    memory_obj = self.reshape_partial_chunk(memory_obj, num_read)

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
            metadata = (
                RemoteMetadata(
                    len(buffer),
                    memory_obj.get_shape(),
                    memory_obj.get_dtype(),
                    memory_obj.get_memory_format(),
                )
                if self.save_chunk_meta
                else None
            )

            # Write to file (metadata + data)
            async with aiofiles.open(temp_path, "wb") as f:
                if metadata is not None:
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
        keys = []
        for base_path in self.base_paths:
            keys.extend([f.stem for f in base_path.glob("*.data")])
        return keys

    async def close(self):
        """Clean up resources"""
        logger.info("Closed the file system connector")
