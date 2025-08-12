# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import asyncio

try:
    # Third Party
    from couchbase.auth import PasswordAuthenticator
    from couchbase.cluster import Cluster
    from couchbase.exceptions import DocumentNotFoundException
    from couchbase.options import ClusterOptions
except ImportError:
    PasswordAuthenticator = None
    Cluster = None
    DocumentNotFoundException = None
    ClusterOptions = None

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

METADATA_KEY_SUFFIX = "::metadata"
DATA_KEY_SUFFIX = "::data"


class CouchbaseConnector(RemoteConnector):
    """Couchbase-based connector for remote storage."""

    def __init__(
        self,
        host: str,
        port: int,
        username: Optional[str],
        password: Optional[str],
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        """
        Initialize Couchbase connector.

        Args:
            host: Couchbase server hostname
            port: Couchbase server port
            username: Authentication username
            password: Authentication password
            bucket_name: Couchbase bucket name
            scope_name: Couchbase scope name
            collection_name: Couchbase collection name
            loop: Asyncio event loop
            local_cpu_backend: Memory allocator interface
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.bucket_name = bucket_name
        self.scope_name = scope_name
        self.collection_name = collection_name
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # Initialize Couchbase connection
        self._initialize_connection()

        logger.info(
            f"Initialized CouchbaseConnector: {host}:{port}/{bucket_name}"
            f".{scope_name}.{collection_name}"
        )

    def _initialize_connection(self):
        """Initialize connection to Couchbase cluster."""
        if Cluster is None:
            raise ImportError(
                "Couchbase SDK is not installed. Please install it with: "
                "pip install couchbase"
            )

        connection_string = f"couchbase://{self.host}:{self.port}"

        if self.username and self.password:
            auth = PasswordAuthenticator(self.username, self.password)
            options = ClusterOptions(auth)
            self.cluster = Cluster(connection_string, options)
        else:
            self.cluster = Cluster(connection_string)

        self.bucket = self.cluster.bucket(self.bucket_name)
        self.scope = self.bucket.scope(self.scope_name)
        self.collection = self.scope.collection(self.collection_name)

    def _get_key_string(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to string format."""
        return key.to_string()

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in Couchbase."""
        key_str = self._get_key_string(key)
        metadata_key = key_str + METADATA_KEY_SUFFIX

        try:
            # Run in thread pool since Couchbase SDK is synchronous
            result = await self.loop.run_in_executor(
                None, lambda: self.collection.exists(metadata_key)
            )
            return result.exists
        except Exception as e:
            logger.error(f"Error checking existence for key {key_str}: {str(e)}")
            return False

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if key exists in Couchbase (synchronous)."""
        key_str = self._get_key_string(key)
        metadata_key = key_str + METADATA_KEY_SUFFIX

        try:
            result = self.collection.exists(metadata_key)
            return result.exists
        except Exception as e:
            logger.error(f"Error checking existence for key {key_str}: {str(e)}")
            return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve data from Couchbase."""
        key_str = self._get_key_string(key)
        metadata_key = key_str + METADATA_KEY_SUFFIX
        data_key = key_str + DATA_KEY_SUFFIX

        try:
            if self.save_chunk_meta:
                # Get metadata first
                metadata_result = await self.loop.run_in_executor(
                    None, lambda: self.collection.get(metadata_key)
                )
                metadata_bytes = metadata_result.content_as[bytes]
                metadata = RemoteMetadata.deserialize(metadata_bytes)

                # Allocate memory based on metadata
                memory_obj = self.local_cpu_backend.allocate(
                    metadata.shape, metadata.dtype, metadata.fmt
                )
            else:
                # Use pre-configured metadata
                memory_obj = self.local_cpu_backend.allocate(
                    self.meta_shape, self.meta_dtype, self.meta_fmt
                )

            if memory_obj is None:
                logger.debug("Memory allocation failed during Couchbase load.")
                return None

            # Get actual data
            data_result = await self.loop.run_in_executor(
                None, lambda: self.collection.get(data_key)
            )
            data_bytes = data_result.content_as[bytes]

            if self.save_chunk_meta:
                # Copy data into allocated memory
                if len(data_bytes) != len(memory_obj.byte_array):
                    raise RuntimeError(
                        f"Data size mismatch: expected {len(memory_obj.byte_array)}, "
                        f"got {len(data_bytes)}"
                    )

                # Handle memory view casting similar to Redis connector
                if isinstance(memory_obj.byte_array, memoryview):
                    view = memory_obj.byte_array
                    if view.format == "<B":
                        view = view.cast("B")
                else:
                    view = memoryview(memory_obj.byte_array)

                view[: len(data_bytes)] = data_bytes
            else:
                # Handle partial chunks
                if isinstance(memory_obj.byte_array, memoryview):
                    view = memory_obj.byte_array
                    if view.format == "<B":
                        view = view.cast("B")
                else:
                    view = memoryview(memory_obj.byte_array)

                view[: len(data_bytes)] = data_bytes
                memory_obj = self.reshape_partial_chunk(memory_obj, len(data_bytes))

            return memory_obj

        except Exception as e:
            if DocumentNotFoundException and isinstance(e, DocumentNotFoundException):
                # Key doesn't exist - this is normal
                return None
            else:
                logger.error(f"Error retrieving key {key_str}: {str(e)}")
                return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data in Couchbase."""
        key_str = self._get_key_string(key)
        metadata_key = key_str + METADATA_KEY_SUFFIX
        data_key = key_str + DATA_KEY_SUFFIX

        try:
            # Prepare data
            kv_bytes = memory_obj.byte_array

            if self.save_chunk_meta:
                # Store metadata
                kv_shape = memory_obj.get_shape()
                kv_dtype = memory_obj.get_dtype()
                memory_format = memory_obj.get_memory_format()

                metadata_bytes = RemoteMetadata(
                    len(kv_bytes), kv_shape, kv_dtype, memory_format
                ).serialize()

                await self.loop.run_in_executor(
                    None, lambda: self.collection.upsert(metadata_key, metadata_bytes)
                )

            # Store actual data
            await self.loop.run_in_executor(
                None, lambda: self.collection.upsert(data_key, kv_bytes)
            )

        except Exception as e:
            logger.error(f"Error storing key {key_str}: {str(e)}")
            raise
        finally:
            # Always decrease reference count
            memory_obj.ref_count_down()

    async def list(self) -> List[str]:
        """List all keys in Couchbase collection."""
        try:
            # Use N1QL query to get all document keys
            query = (
                f"SELECT META().id FROM "
                f"`{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
            )

            result = await self.loop.run_in_executor(
                None, lambda: self.cluster.query(query)
            )

            keys = []
            for row in result:
                doc_id = row["id"]
                # Filter out metadata keys and extract base keys
                if doc_id.endswith(DATA_KEY_SUFFIX):
                    base_key = doc_id[: -len(DATA_KEY_SUFFIX)]
                    keys.append(base_key)

            return keys

        except Exception as e:
            logger.error(f"Error listing keys: {str(e)}")
            return []

    async def close(self):
        """Clean up Couchbase connection."""
        try:
            await self.loop.run_in_executor(None, lambda: self.cluster.close())
            logger.info("Closed Couchbase connection")
        except Exception as e:
            logger.error(f"Error closing Couchbase connection: {str(e)}")

    def support_ping(self) -> bool:
        """Couchbase supports health checks."""
        return True

    async def ping(self) -> int:
        """Perform health check on Couchbase connection."""
        try:
            # Use bucket ping for health check
            result = await self.loop.run_in_executor(None, lambda: self.bucket.ping())
            # Return 0 for success, non-zero for failure
            return 0 if result else 1
        except Exception:
            return 1

    def support_batched_get(self) -> bool:
        """Couchbase supports batch operations."""
        return True

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Retrieve multiple keys in batch from Couchbase."""
        # For simplicity, implement as sequential gets
        # Could be optimized with Couchbase's multi-get operations
        results = []
        for key in keys:
            result = await self.get(key)
            results.append(result)
        return results
