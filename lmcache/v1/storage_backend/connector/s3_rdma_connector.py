# Hewlett Packard Enterprise Confidential
"""RDMA-enabled S3 connector for LMCache."""

# Standard
from typing import Dict, List, Optional

# Standard library imports
import asyncio
import ctypes
import ctypes.util
import os

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# CRITICAL: Must load HPE's cuFile library BEFORE any code imports hpe_object
# This must happen at module import time, not later
_HPE_CUFILE_LOADED = False
_HPE_OBJECT_AVAILABLE = False
_HPE_OBJECT_IMPORT_ERROR = None

def _preload_hpe_cufile():
    """Pre-load HPE's cuFile library to ensure correct version is used."""
    global _HPE_CUFILE_LOADED

    if _HPE_CUFILE_LOADED:
        return True

    hpe_cufile_paths = [
        "/opt/hpe/s3/lib64/libcufile.so.1.13.0",
        "/opt/hpe/s3/lib64/libcufile.so",
    ]

    for path in hpe_cufile_paths:
        if not os.path.exists(path):
            continue

        try:
            # Load with RTLD_GLOBAL to make symbols available globally
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

            # Verify the required symbol exists
            try:
                _ = lib._ZN10cuFileInfo19cuFileGetMemoryTypeEPKv
                _HPE_CUFILE_LOADED = True
                return True
            except AttributeError:
                # Wrong version, try next
                continue

        except Exception:
            continue

    return False

# Pre-load HPE cuFile library FIRST
_cufile_loaded = _preload_hpe_cufile()

# Now try to import hpe_object
try:
    # Third Party
    from hpe_object import (
        BufferGetObject,
        BufferPutObject,
        ClientConfig,
        S3RdmaClient,
    )
    _HPE_OBJECT_AVAILABLE = True
except ImportError as e:
    _HPE_OBJECT_IMPORT_ERROR = str(e)

    if "cuFileGetMemoryType" in str(e):
        _HPE_OBJECT_IMPORT_ERROR = (
            f"\n\nOriginal error: {e}"
        )

# # boto3 import
# try:
#     # Third Party
#     import boto3
#     _BOTO3_AVAILABLE = True
# except ImportError:
#     _BOTO3_AVAILABLE = False

# First Party
# First Party imports
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.s3_rdma_adapter import S3RdmaConnectorSettings
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# Unique prefix for easy log filtering
LOG_PREFIX = "[S3-RDMA]"


class S3RdmaConnector(RemoteConnector):
    """
    S3 RDMA connector using HPE's RDMA-enabled S3 client.

    Requirements:
    - HPE cuFile library (v1.13.0) at /opt/hpe/s3/lib64/
    - LD_LIBRARY_PATH must prioritize /opt/hpe/s3/lib64 over CUDA paths

    This ensures proper library load order.
    """

    def __init__(
        self,
        settings: S3RdmaConnectorSettings,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        logger.debug("%s __init__ ENTER", LOG_PREFIX)
        if not _HPE_OBJECT_AVAILABLE:
            error_msg = (
                f"hpe_object package is required for S3 RDMA connector.\n"
                f"{_HPE_OBJECT_IMPORT_ERROR}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        # if not _BOTO3_AVAILABLE:
        #     raise ImportError(
        #         "boto3 is required for S3 RDMA connector. "
        #         "Install with: pip install boto3"
        #     )

        if not _cufile_loaded:
            logger.warning(
                "Could not pre-load HPE cuFile library. "
                "Ensure you're using the fixed launcher script with correct LD_LIBRARY_PATH"
            )

        self.settings = settings
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self._client: Optional[S3RdmaClient] = None
        # self._client_lock = threading.Lock()
        self._boto_client = None
        self._object_size_cache: Dict[str, int] = {}
        self._inflight_sema: Optional[asyncio.Semaphore] = None
        self._io_executor: Optional[AsyncPQThreadPoolExecutor] = None

        self._prefixed_bucket_path = settings.prefix
        self._effective_parallelism = max(1, settings.max_parallel_requests)
        logger.debug("%s __init__ EXIT", LOG_PREFIX)

    def post_init(self) -> None:
        """Initialize clients after event loop is set up."""
        logger.debug("%s post_init ENTER", LOG_PREFIX)
        super().post_init()

        logger.info(
            "Initializing S3 RDMA client (endpoint=%s, bucket=%s, prefix=%s, parallelism=%d)",
            self.settings.endpoint,
            self.settings.bucket,
            self._prefixed_bucket_path or "(none)",
            self._effective_parallelism,
        )

        client_config = ClientConfig(
            endpoint=self.settings.endpoint,
            max_parallel_requests=self._effective_parallelism,
        )

        if self.settings.max_segment_size is not None:
            logger.info("Using max segment size: %s bytes", self.settings.max_segment_size)
            client_config.max_segment_size = self.settings.max_segment_size

        self._client = S3RdmaClient(client_config)
        logger.info("S3 RDMA client initialized")

        # boto_kwargs = {"endpoint_url": self.settings.endpoint}
        # if self.settings.region:
        #     boto_kwargs["region_name"] = self.settings.region

        # if self.settings.boto_profile:
        #     session = boto3.Session(profile_name=self.settings.boto_profile)
        #     self._boto_client = session.client("s3", **boto_kwargs)
        # else:
        #     self._boto_client = boto3.client("s3", **boto_kwargs)

        # logger.info("boto3 client initialized")

        self._inflight_sema = asyncio.Semaphore(self._effective_parallelism)
        self._io_executor = AsyncPQThreadPoolExecutor(
            self.loop, max_workers=self._effective_parallelism
        )
        logger.info("S3 RDMA connector initialization complete")
        logger.debug("%s post_init EXIT", LOG_PREFIX)

    def _make_s3_key(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to S3 object key with optional prefix."""
        # logger.debug("%s _make_s3_key ENTER: key=%s", LOG_PREFIX, key)
        key_str = key.to_string()
        if self._prefixed_bucket_path:
            result = f"{self._prefixed_bucket_path}/{key_str}"
        else:
            result = key_str
        # logger.debug("%s _make_s3_key EXIT: result=%s", LOG_PREFIX, result)
        return result

    def _get_object_size_sync(self, s3_key: str) -> Optional[int]:
        """Get object size using S3 HEAD request (synchronous)."""
        # logger.debug("%s _get_object_size_sync ENTER: s3_key=%s", LOG_PREFIX, s3_key)
        if s3_key in self._object_size_cache:
            cached_size = self._object_size_cache[s3_key]
            # logger.debug("%s _get_object_size_sync EXIT (cached): size=%s", LOG_PREFIX, cached_size)
            return cached_size

        try:
            size = self._client.get_object_size(
                bucket=self.settings.bucket,
                key=s3_key
            )
            self._object_size_cache[s3_key] = size
            # logger.debug("%s _get_object_size_sync EXIT: size=%s", LOG_PREFIX, size)
            return size
        except Exception as e:
            logger.debug("Failed to get size for %s: %s", s3_key, e)
            # logger.debug("%s _get_object_size_sync EXIT: error", LOG_PREFIX)
            return None

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in S3."""
        # logger.debug("%s exists ENTER: key=%s", LOG_PREFIX, key)
        s3_key = self._make_s3_key(key)
        size = await self.loop.run_in_executor(None, self._get_object_size_sync, s3_key)
        result = size is not None
        # logger.debug("%s exists EXIT: result=%s", LOG_PREFIX, result)
        return result

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronous version of exists."""
        # logger.debug("%s exists_sync ENTER: key=%s", LOG_PREFIX, key)
        s3_key = self._make_s3_key(key)
        result = self._get_object_size_sync(s3_key) is not None
        # logger.debug("%s exists_sync EXIT: result=%s", LOG_PREFIX, result)
        return result

    def _get_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> bool:
        """Synchronous RDMA GET operation."""
        # logger.debug("%s _get_object_sync ENTER: s3_key=%s", LOG_PREFIX, s3_key)

        try:
            # logger.debug("%s Begin RDMA GET to GPU memory: %s", LOG_PREFIX, s3_key)
            # Get the underlying storage
            storage = memory_obj.tensor.untyped_storage()

            # Create a ctypes pointer to the storage's data
            # This allows the HPE RDMA client to write directly to GPU memory
            storage_ptr = storage.data_ptr()
            storage_size = storage.nbytes()

            # Create buffer from the storage using ctypes
            buffer = (ctypes.c_ubyte * storage_size).from_address(storage_ptr)

            # logger.debug("RDMA GET: bucket=%s, key=%s, size=%s", self.settings.bucket, s3_key, storage_size)

            # with self._client_lock:
            #     self._client.get_object_buffers(
            #         BufferGetObject(
            #             bucket=self.settings.bucket,
            #             key=s3_key,
            #             buffer=memoryview(buffer)
            #             )
            #     )
            self._client.get_object_buffers(
                BufferGetObject(
                    bucket=self.settings.bucket,
                    key=s3_key,
                    buffer=memoryview(buffer)
                    )
            )

            # logger.debug("%s End RDMA GET to GPU memory: %s", LOG_PREFIX, s3_key)
            # logger.debug("%s RDMA transfer complete - data is in GPU memory", LOG_PREFIX)

            # The data is now in the GPU tensor's storage
            # The RemoteBackend will handle deserialization based on the format
            # logger.debug(
            #     "%s MemoryObj ready for deserialization: fmt=%s, size=%s",
            #     LOG_PREFIX, memory_obj.metadata.fmt, memory_obj.metadata.phy_size
            # )

            # logger.debug("%s _get_object_sync EXIT: success=True", LOG_PREFIX)
            return True

        except RuntimeError as e:
            # hpe_object raises RuntimeError for various errors
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str or "nosuchkey" in error_str:
                logger.debug("Object not found: %s", s3_key)
                logger.debug("%s _get_object_sync EXIT: success=False (not found)", LOG_PREFIX)
                return False
            else:
                logger.error("RDMA GET error for %s: %s", s3_key, e)
                logger.debug("%s _get_object_sync EXIT: error (RuntimeError)", LOG_PREFIX)
                raise

        except Exception as e:
            logger.error("Unexpected error during RDMA GET %s: %s", s3_key, e)
            logger.debug("%s _get_object_sync EXIT: error (Exception)", LOG_PREFIX)
            raise

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get object from S3 using RDMA."""
        s3_key = self._make_s3_key(key)
        # logger.debug("%s get ENTER: key=%s", LOG_PREFIX, s3_key)

        try:
            size = await self.loop.run_in_executor(None, self._get_object_size_sync, s3_key)
            if size is None:
                # logger.debug("%s get EXIT: result=None (not found): %s", LOG_PREFIX, s3_key)
                return None

            # logger.debug("%s get: Object size=%s bytes for key=%s", LOG_PREFIX, size, s3_key)

            # Allocate GPU memory directly for RDMA transfer
            try:
                gpu_device = self.local_cpu_backend.dst_device
                gpu_tensor = torch.empty(size, dtype=torch.uint8, device=gpu_device)
                # logger.debug(
                #     "%s get: Allocated GPU tensor on device %s with size %s bytes for key=%s",
                #     LOG_PREFIX, gpu_device, size, s3_key
                # )

                # Create metadata for the MemoryObj
                metadata = MemoryObjMetadata(
                    shape=self.meta_shape,
                    dtype=self.meta_dtype,
                    address=gpu_tensor.data_ptr(),
                    phy_size=size,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.KV_2LTD
                )

                # Create TensorMemoryObj wrapping the GPU tensor
                # parent_allocator=None since we're managing this memory directly
                memory_obj = TensorMemoryObj(
                    raw_data=gpu_tensor,
                    metadata=metadata,
                    parent_allocator=None
                )

                # logger.debug(
                #     "%s get: Created TensorMemoryObj (device=%s, size=%s)",
                #     LOG_PREFIX, gpu_tensor.device, size
                # )

            except Exception as e:
                logger.error("Failed to allocate GPU memory for %s: %s", s3_key, e)
                logger.debug("%s get EXIT: result=None (allocation failed)", LOG_PREFIX)
                return None

            # Verify memory location before RDMA transfer
            if not (hasattr(memory_obj, 'tensor') and memory_obj.tensor is not None):
                logger.error("%s memory_obj has no tensor attribute!", LOG_PREFIX)
                logger.debug("%s get EXIT: result=None (invalid memory_obj)", LOG_PREFIX)
                return None

            if not memory_obj.tensor.is_cuda:
                logger.error(
                    "%s Allocated memory is NOT on GPU! Device: %s. Cannot proceed with RDMA transfer.",
                    LOG_PREFIX, memory_obj.tensor.device
                )
                logger.debug("%s get EXIT: result=None (not GPU memory)", LOG_PREFIX)
                return None

            # logger.debug(
            #     "%s get: Verified GPU memory allocation (device=%s)",
            #     LOG_PREFIX, memory_obj.tensor.device
            # )

            # Log memory_obj attributes before RDMA call
            # logger.debug("%s get: memory_obj.tensor type=%s", LOG_PREFIX, type(memory_obj.tensor))
            # logger.debug("%s get: memory_obj.shape=%s", LOG_PREFIX, memory_obj.tensor.shape)
            # logger.debug("%s get: memory_obj.dtype=%s", LOG_PREFIX, memory_obj.tensor.dtype)
            # logger.debug("%s get: memory_obj.device=%s", LOG_PREFIX, memory_obj.tensor.device)
            # logger.debug("%s get: memory_obj.is_cuda=%s", LOG_PREFIX, memory_obj.tensor.is_cuda)

            try:
                async with self._inflight_sema:
                    success = await self.loop.run_in_executor(
                        None, self._get_object_sync, s3_key, memory_obj
                    )

                # logger.debug("%s get: RDMA completed, success=%s", LOG_PREFIX, success)
                # logger.debug("%s get: Retrieved MemoryObj: fmt=%s, shape=%s, dtype=%s", LOG_PREFIX, memory_obj.metadata.fmt, memory_obj.metadata.shape, memory_obj.metadata.dtype)
                if not success:
                    # Clean up on failure
                    # Since parent_allocator is None, we just delete the object
                    memory_obj.invalidate()
                    del memory_obj
                    del gpu_tensor
                    # logger.debug("%s get EXIT: result=None (get failed)", LOG_PREFIX)
                    return None

                # Log memory_obj attributes after RDMA call
                # logger.debug("%s get: After RDMA - tensor.shape=%s", LOG_PREFIX, memory_obj.tensor.shape)
                # logger.debug("%s get: After RDMA - tensor.device=%s", LOG_PREFIX, memory_obj.tensor.device)
                # logger.debug("%s get: After RDMA - tensor.is_cuda=%s", LOG_PREFIX, memory_obj.tensor.is_cuda)

                # Add diagnostic info about what will be deserialized
                # logger.debug(
                #     "%s get: Returning %s format MemoryObj to RemoteBackend for deserialization (size=%s bytes)",
                #     LOG_PREFIX, memory_obj.metadata.fmt, memory_obj.metadata.phy_size
                # )

                # logger.debug("Retrieved %s bytes for %s to GPU memory", size, s3_key)
                # logger.debug("%s get EXIT: result=MemoryObj (success)", LOG_PREFIX)
                return memory_obj

            except Exception as e:
                # Clean up on error
                memory_obj.invalidate()
                del memory_obj
                del gpu_tensor
                logger.error("Failed to get %s: %s", s3_key, e, exc_info=True)
                logger.debug("%s get EXIT: error raised", LOG_PREFIX)
                raise

        except Exception as e:
            # Catch any other exceptions from the outer try block
            logger.error("Failed to get %s (outer error): %s", s3_key, e, exc_info=True)
            logger.debug("%s get EXIT: outer exception raised", LOG_PREFIX)
            raise

    def _put_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> None:
        """Synchronous RDMA PUT operation."""
        # logger.debug("%s _put_object_sync ENTER: s3_key=%s", LOG_PREFIX, s3_key)
        try:
            buffer_view = memory_obj.byte_array
            # logger.debug("RDMA PUT: bucket=%s, key=%s, size=%s", self.settings.bucket, s3_key, len(buffer_view))

            # with self._client_lock:
            #     self._client.put_object_buffers(
            #         BufferPutObject(
            #             bucket=self.settings.bucket,
            #             key=s3_key,
            #             buffer=buffer_view
            #         )
            #     )
            self._client.put_object_buffers(
                BufferPutObject(
                    bucket=self.settings.bucket,
                    key=s3_key,
                    buffer=buffer_view
                )
            )

            # Cache the size
            self._object_size_cache[s3_key] = len(buffer_view)
            # logger.debug("RDMA PUT completed: %s", s3_key)
            # logger.debug("%s _put_object_sync EXIT: success", LOG_PREFIX)

        except RuntimeError as e:
            logger.error("RDMA PUT error for %s: %s", s3_key, e)
            logger.debug("%s _put_object_sync EXIT: error (RuntimeError)", LOG_PREFIX)
            raise
        except Exception as e:
            logger.error("Unexpected error during RDMA PUT %s: %s", s3_key, e)
            logger.debug("%s _put_object_sync EXIT: error (Exception)", LOG_PREFIX)
            raise

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Put object to S3 using RDMA."""
        # logger.debug("%s put ENTER: key=%s", LOG_PREFIX, key)
        # logger.debug("%s put memory_obj = %s", LOG_PREFIX, vars(memory_obj))
        s3_key = self._make_s3_key(key)

        async with self._inflight_sema:
            await self.loop.run_in_executor(None, self._put_object_sync, s3_key, memory_obj)

        # logger.debug("Stored %s bytes for %s", len(memory_obj.byte_array), key)
        # logger.debug("%s put EXIT: success", LOG_PREFIX)

    # def _list_objects_sync(self) -> List[str]:
    #     """Synchronous list operation using boto3."""
    #     logger.debug("%s _list_objects_sync ENTER", LOG_PREFIX)
    #     try:
    #         result = []
    #         continuation_token = None

    #         while True:
    #             kwargs = {'Bucket': self.settings.bucket}
    #             if self._prefixed_bucket_path:
    #                 kwargs['Prefix'] = self._prefixed_bucket_path + "/"
    #             if continuation_token:
    #                 kwargs['ContinuationToken'] = continuation_token

    #             response = self._boto_client.list_objects_v2(**kwargs)

    #             if 'Contents' in response:
    #                 for obj in response['Contents']:
    #                     key = obj['Key']
    #                     if self._prefixed_bucket_path and key.startswith(self._prefixed_bucket_path + "/"):
    #                         key = key[len(self._prefixed_bucket_path) + 1:]
    #                     result.append(key)

    #             if not response.get('IsTruncated', False):
    #                 break
    #             continuation_token = response.get('NextContinuationToken')

    #         logger.debug("Listed %s objects from S3", len(result))
    #         logger.debug("%s _list_objects_sync EXIT: count=%s", LOG_PREFIX, len(result))
    #         return result

    #     except Exception as e:
    #         logger.error("Failed to list objects: %s", e)
    #         logger.debug("%s _list_objects_sync EXIT: error", LOG_PREFIX)
    #         raise

    async def list(self) -> List[str]:
        # """List all objects."""
        # logger.debug("%s list ENTER", LOG_PREFIX)
        # result = await self.loop.run_in_executor(None, self._list_objects_sync)
        # logger.debug("%s list EXIT: count=%d", LOG_PREFIX, len(result))
        # return result
        raise NotImplementedError

    async def close(self) -> None:
        """Clean up resources."""
        # logger.debug("%s close ENTER", LOG_PREFIX)
        # logger.debug("Closing S3 RDMA connector")
        if self._io_executor:
            try:
                self._io_executor.shutdown(wait=True)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)
        self._object_size_cache.clear()
        # logger.debug("S3 RDMA connector closed")
        # logger.debug("%s close EXIT", LOG_PREFIX)
