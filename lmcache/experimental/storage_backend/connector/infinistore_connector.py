import asyncio
import ctypes
from typing import List, Optional, Union, no_type_check

import infinistore

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


def _get_ptr(mv: Union[bytearray, memoryview]) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class InfinistoreConnector(RemoteConnector):

    def __init__(self, host: str, port: int, dev_name,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        config = infinistore.ClientConfig(
            host_addr=host,
            service_port=port,
            log_level="info",
            connection_type=infinistore.TYPE_RDMA,
            ib_port=1,
            link_type=infinistore.LINK_ETHERNET,
            dev_name=dev_name,
        )

        self.rdma_conn = infinistore.InfinityConnection(config)

        self.memory_allocator = memory_allocator
        self.loop = loop
        self.rdma_conn.connect()

        # allocate 4KB buffer for RDMA read
        self.buffer_size = 4 << 10
        self.buffer = bytearray(self.buffer_size)
        self.rdma_conn.register_mr(_get_ptr(self.buffer), self.buffer_size)

    async def exists(self, key: CacheEngineKey) -> bool:

        def blocking_io():
            return self.rdma_conn.check_exist(key.to_string() + "metadata")

        return await self.loop.run_in_executor(None, blocking_io)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        try:
            await self.rdma_conn.read_cache_single_async(
                key_str + "metadata", _get_ptr(self.buffer), len(self.buffer))
        except infinistore.lib.InfiniStoreKeyNotFound:
            return None

        metadata = RedisMetadata.deserialize(self.buffer[:METADATA_BYTES_LEN])

        memory_obj = self.memory_allocator.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # TODO: we could have memory allocator which pre-allocate
        # and register RDMA memory.
        # register memory is a heavy operation, so we should avoid it.

        kv_bytes = bytes(memory_obj.get_size())
        pointer = ctypes.cast(ctypes.c_char_p(kv_bytes),
                              ctypes.POINTER(ctypes.c_char))
        ptr = ctypes.addressof(pointer.contents)
        size = memory_obj.get_size()

        await self.loop.run_in_executor(None, self.rdma_conn.register_mr, ptr,
                                        size)

        try:
            await self.rdma_conn.read_cache_single_async(
                key_str + "kv_bytes", ptr, size)
        except infinistore.lib.InfiniStoreKeyNotFound:
            return None

        view = memoryview(memory_obj.byte_array)
        view[:metadata.length] = kv_bytes

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape, kv_dtype,
                                       memory_format).serialize()

        # not likely to happen
        assert len(metadata_bytes
                   ) <= self.buffer_size, "metadata size exceeds buffer size"

        # copy metadata to self.buffer
        self.buffer[:len(metadata_bytes)] = metadata_bytes

        await self.rdma_conn.rdma_write_cache_single_async(
            key.to_string() + "metadata", _get_ptr(self.buffer),
            len(self.buffer))

        pointer = ctypes.cast(ctypes.c_char_p(memory_obj.byte_array),
                              ctypes.POINTER(ctypes.c_char))
        ptr = ctypes.addressof(pointer.contents)
        size = memory_obj.get_size()
        await self.loop.run_in_executor(None, self.rdma_conn.register_mr, ptr,
                                        size)
        await self.rdma_conn.rdma_write_cache_single_async(
            key.to_string() + "kv_bytes", ptr, size)

        self.memory_allocator.ref_count_down(memory_obj)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.rdma_conn.close()
        logger.info("Closed the infinistore connection")
