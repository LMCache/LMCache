# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from typing import List, Optional, Tuple, no_type_check
import asyncio
import inspect

# Third Party
from glide import (
    Batch,
    ClusterBatch,
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClient,
    GlideClusterClientConfiguration,
    NodeAddress,
    ServerCredentials,
)

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class ValkeyConnector(RemoteConnector):
    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        username: str,
        password: str,
        database_id: Optional[int] = None,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        if ":" in url:
            host, port_str = url.split(":", 1)
            port = int(port_str)
        else:
            host = url
            port = 6379  # Default Valkey port

        self.host = host
        self.port = port
        self.database_id = database_id
        self.username = username
        self.password = password
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.executor = AsyncPQExecutor(loop)

        # Create connection properly using async create
        self.connection = self._init_connection()

    def _init_connection(self):
        """Initialize GlideClient connection with credentials and database"""

        async def create_connection():
            try:
                # Setup credentials if provided
                credentials = None
                if self.username or self.password:
                    credentials = ServerCredentials(self.username, self.password)

                # Build config with optional database_id
                config_kwargs = {
                    "addresses": [NodeAddress(self.host, self.port)],
                    "credentials": credentials,
                }

                if self.database_id is not None:
                    config_kwargs["database_id"] = self.database_id

                config = GlideClientConfiguration(**config_kwargs)
                return await GlideClient.create(config)
            except Exception as e:
                raise RuntimeError(f"Fail to init valkey connection {e}") from e

        future = asyncio.run_coroutine_threadsafe(create_connection(), self.loop)
        connection = future.result(timeout=1.0)
        return connection

    def _get_keys(self, key: CacheEngineKey) -> Tuple[str, str]:
        """Generate metadata and kv_bytes keys"""
        key_str = key.to_string()
        metadata_key = f"{key_str}:metadata"
        kv_key = f"{key_str}:kv_bytes"
        return metadata_key, kv_key

    async def _exists(self, key: CacheEngineKey) -> bool:
        metadata_key, _ = self._get_keys(key)
        return bool(await self.connection.exists([metadata_key]))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(
            self.executor.submit_job(self._exists, key=key, priority=Priorities.PEEK),
            self.loop,
        )
        return future.result()

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        metadata_key, kv_key = self._get_keys(key)

        results = await self.connection.mget([metadata_key, kv_key])

        if len(results) != 2:
            return None

        metadata_bytes, kv_bytes = results[0], results[1]

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shapes,
            metadata.dtypes,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        if kv_bytes is None:
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by valkey."
            )
            memory_obj.ref_count_down()
            return None

        assert not inspect.isawaitable(kv_bytes)

        try:
            if isinstance(memory_obj.byte_array, memoryview):
                view = memory_obj.byte_array
                if view.format == "<B":
                    view = view.cast("B")
            else:
                view = memoryview(memory_obj.byte_array)

            if isinstance(kv_bytes, (bytes, bytearray)):
                view[: metadata.length] = kv_bytes
            elif isinstance(kv_bytes, str):
                converted = kv_bytes.encode("utf-8")
                view[: metadata.length] = converted
            else:
                converted = bytes(kv_bytes)
                view[: metadata.length] = converted
            return memory_obj

        except Exception as exc:
            logger.error(f"Fail to converting : {exc}")
            return None

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        try:
            kv_bytes = bytes(memory_obj.byte_array)
            kv_shapes = memory_obj.get_shapes()
            kv_dtypes = memory_obj.get_dtypes()
            memory_format = memory_obj.get_memory_format()

            metadata_bytes = RemoteMetadata(
                len(kv_bytes), kv_shapes, kv_dtypes, memory_format
            ).serialize()

            metadata_key, kv_key = self._get_keys(key)

            # Use batch to set both keys in one operation
            # kv bytes needs to be set first to avoid race condition
            batch = Batch(False)
            batch.set(kv_key, kv_bytes)
            batch.set(metadata_key, metadata_bytes)

            await self.connection.exec(batch, raise_on_error=False)
        except Exception as exc:
            logger.error(f"Fail to put data: {exc}")

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.executor.shutdown(wait=True)
        await self.connection.close()
        logger.info("Closed the Valkey connection")


class ValkeyClusterConnector(RemoteConnector):
    """
    Uses GlideClusterClient to connect to a Valkey cluster.
    Supports both URL-based and hosts_and_ports-based initialization.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        username: str,
        password: str,
        hosts_and_ports: Optional[List[Tuple[str, int]]],
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.executor = AsyncPQExecutor(loop)
        self.username = username
        self.password = password
        self.hosts_and_ports = hosts_and_ports

        # Create connection
        self.connection = self._init_connection()

    def _init_connection(self):
        """Initialize GlideClusterClient connection with credentials"""

        async def create_connection():
            try:
                # Setup credentials if provided
                credentials = None
                if self.username or self.password:
                    credentials = ServerCredentials(self.username, self.password)

                addresses = [
                    NodeAddress(host, port) for host, port in self.hosts_and_ports
                ]
                config = GlideClusterClientConfiguration(
                    addresses=addresses, credentials=credentials
                )
                return await GlideClusterClient.create(config)
            except Exception as e:
                raise RuntimeError(f"Fail to init valkey connection {e}") from e

        future = asyncio.run_coroutine_threadsafe(create_connection(), self.loop)
        connection = future.result(timeout=1.0)
        return connection

    def _get_keys_with_hash_tag(self, key: CacheEngineKey) -> Tuple[str, str]:
        """Generate metadata and kv_bytes keys with hash tag for same slot placement"""
        key_str = key.to_string()
        # Use hash tag to ensure both keys go to same slot
        metadata_key = f"{{{key_str}}}:metadata"
        kv_key = f"{{{key_str}}}:kv_bytes"
        return metadata_key, kv_key

    async def _exists(self, key: CacheEngineKey) -> bool:
        metadata_key, _ = self._get_keys_with_hash_tag(key)
        return bool(await self.connection.exists([metadata_key]))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(
            self.executor.submit_job(self._exists, key=key, priority=Priorities.PEEK),
            self.loop,
        )
        return future.result()

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        metadata_key, kv_key = self._get_keys_with_hash_tag(key)

        results = await self.connection.mget([metadata_key, kv_key])

        if len(results) != 2:
            return None

        metadata_bytes, kv_bytes = results[0], results[1]

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shapes,
            metadata.dtypes,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        if kv_bytes is None:
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by valkey."
            )
            memory_obj.ref_count_down()
            return None

        assert not inspect.isawaitable(kv_bytes)

        try:
            if isinstance(memory_obj.byte_array, memoryview):
                view = memory_obj.byte_array
                if view.format == "<B":
                    view = view.cast("B")
            else:
                view = memoryview(memory_obj.byte_array)

            if isinstance(kv_bytes, (bytes, bytearray)):
                view[: metadata.length] = kv_bytes
            elif isinstance(kv_bytes, str):
                converted = kv_bytes.encode("utf-8")
                view[: metadata.length] = converted
            else:
                converted = bytes(kv_bytes)
                view[: metadata.length] = converted
            return memory_obj
        except Exception as exc:
            logger.error(f"Fail to converting : {exc}")
            return None

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        try:
            kv_bytes = bytes(memory_obj.byte_array)
            kv_shapes = memory_obj.get_shapes()
            kv_dtypes = memory_obj.get_dtypes()
            memory_format = memory_obj.get_memory_format()

            metadata_bytes = RemoteMetadata(
                len(kv_bytes), kv_shapes, kv_dtypes, memory_format
            ).serialize()

            metadata_key, kv_key = self._get_keys_with_hash_tag(key)

            # Use cluster batch to set both keys in one operation
            # kv bytes needs to be set first to avoid race condition
            batch = ClusterBatch(False)
            batch.set(kv_key, kv_bytes)
            batch.set(metadata_key, metadata_bytes)

            await self.connection.exec(batch, raise_on_error=False)
        except Exception as exc:
            logger.error(f"Fail to put data: {exc}")

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.executor.shutdown(wait=True)
        await self.connection.close()
        logger.info("Closed the Valkey connection")
