# Hewlett Packard Enterprise Confidential
"""RDMA-enabled S3 connector for LMCache."""

# Standard library imports
import asyncio
import ctypes
import ctypes.util
import os
import sys
import torch
import threading
from typing import Dict, List, Optional

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
    from hpe_object import (
        ClientConfig,
        S3RdmaClient,
        BufferGetObject,
        BufferPutObject,
        AlignedBuffer,
    )
    _HPE_OBJECT_AVAILABLE = True
except ImportError as e:
    _HPE_OBJECT_IMPORT_ERROR = str(e)

    if "cuFileGetMemoryType" in str(e):
        _HPE_OBJECT_IMPORT_ERROR = (
            f"\n\nOriginal error: {e}"
        )

# boto3 import
try:
    import boto3
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False

# First Party imports
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.s3_rdma_adapter import S3RdmaConnectorSettings
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQThreadPoolExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# Unique prefix for easy log filtering
LOG_PREFIX = "[S3-RDMA-CONN]"


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
        logger.info(f"{LOG_PREFIX} __init__ ENTER")
        if not _HPE_OBJECT_AVAILABLE:
            error_msg = (
                f"hpe_object package is required for S3 RDMA connector.\n"
                f"{_HPE_OBJECT_IMPORT_ERROR}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        if not _BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 RDMA connector. "
                "Install with: pip install boto3"
            )

        if not _cufile_loaded:
            logger.warning(
                "Could not pre-load HPE cuFile library. "
                "Ensure you're using the fixed launcher script with correct LD_LIBRARY_PATH"
            )

        self.settings = settings
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self._client: Optional[S3RdmaClient] = None
        self._client_lock = threading.Lock()
        self._boto_client = None
        self._object_size_cache: Dict[str, int] = {}
        self._inflight_sema: Optional[asyncio.Semaphore] = None
        self._io_executor: Optional[AsyncPQThreadPoolExecutor] = None

        self._prefixed_bucket_path = settings.prefix
        self._effective_parallelism = max(1, settings.max_parallel_requests)
        logger.info(f"{LOG_PREFIX} __init__ EXIT")

    def post_init(self) -> None:
        """Initialize clients after event loop is set up."""
        logger.info(f"{LOG_PREFIX} post_init ENTER")
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
            logger.info(f"Using max segment size: {self.settings.max_segment_size} bytes")
            client_config.max_segment_size = self.settings.max_segment_size

        self._client = S3RdmaClient(client_config)
        logger.info("S3 RDMA client initialized")

        boto_kwargs = {"endpoint_url": self.settings.endpoint}
        if self.settings.region:
            boto_kwargs["region_name"] = self.settings.region

        if self.settings.boto_profile:
            session = boto3.Session(profile_name=self.settings.boto_profile)
            self._boto_client = session.client("s3", **boto_kwargs)
        else:
            self._boto_client = boto3.client("s3", **boto_kwargs)

        logger.info("boto3 client initialized")

        self._inflight_sema = asyncio.Semaphore(self._effective_parallelism)
        self._io_executor = AsyncPQThreadPoolExecutor(
            self.loop, max_workers=self._effective_parallelism
        )
        logger.info("S3 RDMA connector initialization complete")
        logger.info(f"{LOG_PREFIX} post_init EXIT")

    def _make_s3_key(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to S3 object key with optional prefix."""
        logger.debug(f"{LOG_PREFIX} _make_s3_key ENTER: key={key}")
        key_str = key.to_string()
        if self._prefixed_bucket_path:
            result = f"{self._prefixed_bucket_path}/{key_str}"
        else:
            result = key_str
        logger.debug(f"{LOG_PREFIX} _make_s3_key EXIT: result={result}")
        return result

    def _get_object_size_sync(self, s3_key: str) -> Optional[int]:
        """Get object size using boto3 HEAD request (synchronous)."""
        logger.debug(f"{LOG_PREFIX} _get_object_size_sync ENTER: s3_key={s3_key}")
        if s3_key in self._object_size_cache:
            cached_size = self._object_size_cache[s3_key]
            logger.debug(f"{LOG_PREFIX} _get_object_size_sync EXIT (cached): size={cached_size}")
            return cached_size

        try:
            response = self._boto_client.head_object(
                Bucket=self.settings.bucket,
                Key=s3_key
            )
            size = response['ContentLength']
            self._object_size_cache[s3_key] = size
            logger.debug(f"{LOG_PREFIX} _get_object_size_sync EXIT: size={size}")
            return size
        except self._boto_client.exceptions.NoSuchKey:
            logger.debug(f"{LOG_PREFIX} _get_object_size_sync EXIT: not found")
            return None
        except Exception as e:
            logger.warning(f"Failed to get size for {s3_key}: {e}")
            logger.debug(f"{LOG_PREFIX} _get_object_size_sync EXIT: error")
            return None

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in S3."""
        logger.info(f"{LOG_PREFIX} exists ENTER: key={key}")
        s3_key = self._make_s3_key(key)
        size = await self.loop.run_in_executor(None, self._get_object_size_sync, s3_key)
        result = size is not None
        logger.info(f"{LOG_PREFIX} exists EXIT: result={result}")
        return result

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronous version of exists."""
        logger.info(f"{LOG_PREFIX} exists_sync ENTER: key={key}")
        s3_key = self._make_s3_key(key)
        result = self._get_object_size_sync(s3_key) is not None
        logger.info(f"{LOG_PREFIX} exists_sync EXIT: result={result}")
        return result

    def _get_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> bool:
        """Synchronous RDMA GET operation."""
        logger.info(f"{LOG_PREFIX} _get_object_sync ENTER: s3_key={s3_key}")

        try:
            logger.info(f"{LOG_PREFIX} Begin RDMA GET to GPU memory: {s3_key}")
            buffer = memory_obj.tensor.untyped_storage()
            logger.info(f"RDMA GET: bucket={self.settings.bucket}, key={s3_key}")

            with self._client_lock:
                self._client.get_object_buffers(
                    BufferGetObject(
                        bucket=self.settings.bucket,
                        key=s3_key,
                        buffer=buffer  # Works with both memoryview and torch.UntypedStorage
                    )
                )

            logger.info(f"{LOG_PREFIX} End RDMA GET to GPU memory: {s3_key}")

            logger.info(f"{LOG_PREFIX} _get_object_sync EXIT: success=True")

            return True

        except RuntimeError as e:
            # hpe_object raises RuntimeError for various errors
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str or "nosuchkey" in error_str:
                logger.debug(f"Object not found: {s3_key}")
                logger.debug(f"{LOG_PREFIX} _get_object_sync EXIT: success=False (not found)")
                return False
            else:
                logger.error(f"RDMA GET error for {s3_key}: {e}")
                logger.debug(f"{LOG_PREFIX} _get_object_sync EXIT: error (RuntimeError)")
                raise

        except Exception as e:
            logger.error(f"Unexpected error during RDMA GET {s3_key}: {e}")
            logger.debug(f"{LOG_PREFIX} _get_object_sync EXIT: error (Exception)")
            raise

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get object from S3 using RDMA."""
        ####### DEBUG #######
        import traceback
        # Log the complete call stack
        logger.info(f"{LOG_PREFIX} CALL STACK TRACE:")
        for line in traceback.format_stack():
            logger.info(line.strip())
        
        logger.info(f"{LOG_PREFIX} get called by:")
        logger.info(f"{LOG_PREFIX} Caller info: {traceback.extract_stack()[-2]}")
        ####################
        s3_key = self._make_s3_key(key)
        logger.info(f"{LOG_PREFIX} get ENTER: key={s3_key}")

        try:
            size = await self.loop.run_in_executor(None, self._get_object_size_sync, s3_key)
            if size is None:
                logger.info(f"{LOG_PREFIX} get EXIT: result=None (not found): {s3_key}")
                return None

            logger.info(f"{LOG_PREFIX} get: Object size={size} bytes for key={s3_key}")

            # Allocate GPU memory directly for RDMA transfer
            try:
                gpu_device = self.local_cpu_backend.dst_device
                gpu_tensor = torch.empty(size, dtype=torch.uint8, device=gpu_device)
                logger.info(
                    f"{LOG_PREFIX} get: Allocated GPU tensor on device {gpu_device} "
                    f"with size {size} bytes for key={s3_key}"
                )
                
                # Import the MemoryObj classes
                from lmcache.v1.memory_management import (
                    TensorMemoryObj,
                    MemoryObjMetadata,
                    MemoryFormat
                )
                
                # Create metadata for the MemoryObj
                metadata = MemoryObjMetadata(
                    shape=torch.Size([size]),
                    dtype=torch.uint8,
                    address=gpu_tensor.data_ptr(),
                    phy_size=size,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.BINARY  # Use BINARY format for raw bytes
                )
                
                # Create TensorMemoryObj wrapping the GPU tensor
                # parent_allocator=None since we're managing this memory directly
                memory_obj = TensorMemoryObj(
                    raw_data=gpu_tensor,
                    metadata=metadata,
                    parent_allocator=None
                )
                
                logger.info(
                    f"{LOG_PREFIX} get: Created TensorMemoryObj "
                    f"(device={gpu_tensor.device}, size={size})"
                )
                
            except Exception as e:
                logger.error(f"Failed to allocate GPU memory for {s3_key}: {e}")
                logger.info(f"{LOG_PREFIX} get EXIT: result=None (allocation failed)")
                return None

            # Verify memory location before RDMA transfer
            if not (hasattr(memory_obj, 'tensor') and memory_obj.tensor is not None):
                logger.error(f"{LOG_PREFIX} memory_obj has no tensor attribute!")
                logger.info(f"{LOG_PREFIX} get EXIT: result=None (invalid memory_obj)")
                return None

            if not memory_obj.tensor.is_cuda:
                logger.error(
                    f"{LOG_PREFIX} Allocated memory is NOT on GPU! "
                    f"Device: {memory_obj.tensor.device}. "
                    f"Cannot proceed with RDMA transfer."
                )
                logger.info(f"{LOG_PREFIX} get EXIT: result=None (not GPU memory)")
                return None
            
            logger.info(
                f"{LOG_PREFIX} get: Verified GPU memory allocation "
                f"(device={memory_obj.tensor.device})"
            )

            # Log memory_obj attributes before RDMA call
            logger.info(f"{LOG_PREFIX} get: memory_obj.tensor type={type(memory_obj.tensor)}")
            logger.info(f"{LOG_PREFIX} get: memory_obj.shape={memory_obj.tensor.shape}")
            logger.info(f"{LOG_PREFIX} get: memory_obj.dtype={memory_obj.tensor.dtype}")
            logger.info(f"{LOG_PREFIX} get: memory_obj.device={memory_obj.tensor.device}")
            logger.info(f"{LOG_PREFIX} get: memory_obj.is_cuda={memory_obj.tensor.is_cuda}")

            try:
                async with self._inflight_sema:
                    success = await self.loop.run_in_executor(
                        None, self._get_object_sync, s3_key, memory_obj
                    )

                logger.info(f"{LOG_PREFIX} get: RDMA completed, success={success}")
                logger.info(f"{LOG_PREFIX} get: Retrieved MemoryObj: fmt={memory_obj.metadata.fmt}, shape={memory_obj.metadata.shape}, dtype={memory_obj.metadata.dtype}")
                if not success:
                    # Clean up on failure
                    # Since parent_allocator is None, we just delete the object
                    memory_obj.invalidate()
                    del memory_obj
                    del gpu_tensor
                    logger.info(f"{LOG_PREFIX} get EXIT: result=None (get failed)")
                    return None

                # Log memory_obj attributes after RDMA call
                logger.info(f"{LOG_PREFIX} get: After RDMA - tensor.shape={memory_obj.tensor.shape}")
                logger.info(f"{LOG_PREFIX} get: After RDMA - tensor.device={memory_obj.tensor.device}")
                logger.info(f"{LOG_PREFIX} get: After RDMA - tensor.is_cuda={memory_obj.tensor.is_cuda}")

                # Add diagnostic info about what will be deserialized
                logger.info(
                    f"{LOG_PREFIX} get: Returning {memory_obj.metadata.fmt} format MemoryObj to RemoteBackend "
                    f"for deserialization (size={memory_obj.metadata.phy_size} bytes)"
                )
                
                logger.info(f"Retrieved {size} bytes for {key} to GPU memory")
                logger.info(f"{LOG_PREFIX} get EXIT: result=MemoryObj (success)")
                return memory_obj

            except Exception as e:
                # Clean up on error
                memory_obj.invalidate()
                del memory_obj
                del gpu_tensor
                logger.error(f"Failed to get {key}: {e}", exc_info=True)
                logger.info(f"{LOG_PREFIX} get EXIT: error raised")
                raise

        except Exception as e:
            # Catch any other exceptions from the outer try block
            logger.error(f"Failed to get {key} (outer error): {e}", exc_info=True)
            logger.info(f"{LOG_PREFIX} get EXIT: outer exception raised")
            raise

    def _put_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> None:
        """Synchronous RDMA PUT operation."""
        logger.debug(f"{LOG_PREFIX} _put_object_sync ENTER: s3_key={s3_key}")
        try:
            buffer_view = memory_obj.byte_array
            logger.debug(f"RDMA PUT: bucket={self.settings.bucket}, key={s3_key}, size={len(buffer_view)}")

            with self._client_lock:
                self._client.put_object_buffers(
                    BufferPutObject(
                        bucket=self.settings.bucket,
                        key=s3_key,
                        buffer=buffer_view
                    )
                )

            # Cache the size
            self._object_size_cache[s3_key] = len(buffer_view)
            logger.debug(f"RDMA PUT completed: {s3_key}")
            logger.debug(f"{LOG_PREFIX} _put_object_sync EXIT: success")

        except RuntimeError as e:
            logger.error(f"RDMA PUT error for {s3_key}: {e}")
            logger.debug(f"{LOG_PREFIX} _put_object_sync EXIT: error (RuntimeError)")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during RDMA PUT {s3_key}: {e}")
            logger.debug(f"{LOG_PREFIX} _put_object_sync EXIT: error (Exception)")
            raise

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Put object to S3 using RDMA."""
        logger.info(f"{LOG_PREFIX} put ENTER: key={key}")
        s3_key = self._make_s3_key(key)

        async with self._inflight_sema:
            await self.loop.run_in_executor(None, self._put_object_sync, s3_key, memory_obj)

        logger.info(f"Stored {len(memory_obj.byte_array)} bytes for {key}")
        logger.info(f"{LOG_PREFIX} put EXIT: success")

    def _list_objects_sync(self) -> List[str]:
        """Synchronous list operation using boto3."""
        logger.debug(f"{LOG_PREFIX} _list_objects_sync ENTER")
        try:
            result = []
            continuation_token = None

            while True:
                kwargs = {'Bucket': self.settings.bucket}
                if self._prefixed_bucket_path:
                    kwargs['Prefix'] = self._prefixed_bucket_path + "/"
                if continuation_token:
                    kwargs['ContinuationToken'] = continuation_token

                response = self._boto_client.list_objects_v2(**kwargs)

                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if self._prefixed_bucket_path and key.startswith(self._prefixed_bucket_path + "/"):
                            key = key[len(self._prefixed_bucket_path) + 1:]
                        result.append(key)

                if not response.get('IsTruncated', False):
                    break
                continuation_token = response.get('NextContinuationToken')

            logger.debug(f"Listed {len(result)} objects from S3")
            logger.debug(f"{LOG_PREFIX} _list_objects_sync EXIT: count={len(result)}")
            return result

        except Exception as e:
            logger.error(f"Failed to list objects: {e}")
            logger.debug(f"{LOG_PREFIX} _list_objects_sync EXIT: error")
            raise

    async def list(self) -> List[str]:
        """List all objects."""
        logger.info(f"{LOG_PREFIX} list ENTER")
        result = await self.loop.run_in_executor(None, self._list_objects_sync)
        logger.info(f"{LOG_PREFIX} list EXIT: count={len(result)}")
        return result

    async def close(self) -> None:
        """Clean up resources."""
        logger.info(f"{LOG_PREFIX} close ENTER")
        logger.info("Closing S3 RDMA connector")
        if self._io_executor:
            try:
                self._io_executor.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")
        self._object_size_cache.clear()
        logger.info("S3 RDMA connector closed")
        logger.info(f"{LOG_PREFIX} close EXIT")
