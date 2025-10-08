# SPDX-License-Identifier: Apache-2.0
"""RDMA-enabled S3 connector for LMCache."""

from __future__ import annotations

# Standard
from typing import Dict, List, Optional
import asyncio
import functools

# Third Party
try:
    from hpe_object import (
        BufferGetObject,
        BufferPutObject,
        ClientConfig,
        S3Error,
        S3NotFoundError,
        S3RdmaClient,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "hpe_object package is required for the S3 RDMA connector. "
        "Install the s3-rdma-wrapper-aws-sdk-python package."
    ) from exc

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.s3_rdma_adapter import (
    S3RdmaConnectorSettings,
)
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class S3RdmaConnector(RemoteConnector):
    """Remote connector that talks to an S3-compatible endpoint using RDMA."""

    def __init__(
        self,
        settings: S3RdmaConnectorSettings,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: LMCacheEngineConfig,
        metadata,
    ) -> None:
        self.settings = settings
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.config = config
        self.metadata = metadata

        self._client: Optional[S3RdmaClient] = None
        self._object_size_cache: Dict[str, int] = {}
        self._inflight_sema: Optional[asyncio.Semaphore] = None
        self._io_executor: Optional[AsyncPQThreadPoolExecutor] = None

        self._prefixed_bucket_path = settings.prefix
        self._effective_parallelism = max(1, settings.max_parallel_requests)

    # ---------------------------------------------------------------------
    # lifecycle hooks
    # ---------------------------------------------------------------------
    def post_init(self) -> None:
        super().post_init()

        if self.full_chunk_size is None:
            if self.settings.max_segment_size is None:
                raise ValueError(
                    "S3 RDMA connector requires chunk metadata to determine transfer size."
                )
            chunk_size = self.settings.max_segment_size
        else:
            chunk_size = self.full_chunk_size

        if self.settings.max_segment_size is not None and (
            chunk_size > self.settings.max_segment_size
        ):
            raise ValueError(
                "Configured max_segment_size is smaller than the LMCache chunk size"
            )

        logger.info(
            "Initializing S3 RDMA client (parallelism=%d, segment_size=%s)",
            self._effective_parallelism,
            chunk_size,
        )

        client_config = ClientConfig(
            endpoint=self.settings.endpoint,
            max_parallel_requests=self._effective_parallelism,
            max_segment_size=chunk_size,
        )
        self._client = S3RdmaClient(client_config)

        self._inflight_sema = asyncio.Semaphore(self._effective_parallelism)
        self._io_executor = AsyncPQThreadPoolExecutor(
            self.loop, max_workers=self._effective_parallelism
        )

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _ensure_client(self) -> S3RdmaClient:
        if self._client is None:
            raise RuntimeError("S3 RDMA client is not initialized")
        return self._client

    def _ensure_semaphore(self) -> asyncio.Semaphore:
        if self._inflight_sema is None:
            raise RuntimeError("S3 RDMA semaphore not initialized")
        return self._inflight_sema

    @staticmethod
    def _flatten_key(raw_key: str) -> str:
        return raw_key.replace("/", "_")

    def _object_key(self, key: CacheEngineKey) -> str:
        base = self._flatten_key(key.to_string())
        if self._prefixed_bucket_path:
            return f"{self._prefixed_bucket_path}/{base}"
        return base

    def _record_object_size(self, object_key: str, size: int) -> None:
        self._object_size_cache[object_key] = size

    def _lookup_cached_size(self, object_key: str) -> Optional[int]:
        return self._object_size_cache.get(object_key)

    def _head_object(self, object_key: str) -> int:
        client = self._ensure_client()
        try:
            size = client.get_object_size(self.settings.bucket, object_key)
        except S3NotFoundError:
            return 0
        except S3Error as exc:
            logger.warning("S3 head_object failed for %s: %s", object_key, exc)
            raise
        self._record_object_size(object_key, size)
        return size

    async def _ensure_object_size(self, object_key: str) -> int:
        cached = self._lookup_cached_size(object_key)
        if cached is not None:
            return cached
        size = await asyncio.to_thread(self._head_object, object_key)
        return size

    def _bytes_view(self, memory_obj: MemoryObj):
        buffer = memory_obj.byte_array
        try:
            return buffer.cast("B")
        except TypeError:
            return buffer

    # ------------------------------------------------------------------
    # RemoteConnector API
    # ------------------------------------------------------------------
    async def exists(self, key: CacheEngineKey) -> bool:
        return await asyncio.to_thread(self.exists_sync, key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        object_key = self._object_key(key)
        cached = self._lookup_cached_size(object_key)
        if cached is not None:
            return cached > 0
        size = self._head_object(object_key)
        return size > 0

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        object_key = self._object_key(key)
        size = await self._ensure_object_size(object_key)
        if size <= 0:
            return None

        semaphore = self._ensure_semaphore()
        await semaphore.acquire()
        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shape,
            self.meta_dtype,
            self.meta_fmt,
        )
        if memory_obj is None:
            semaphore.release()
            raise RuntimeError("Failed to allocate CPU memory object for RDMA download")

        try:
            await self._schedule_get(memory_obj, object_key, size)
            if self.full_chunk_size is not None and size != self.full_chunk_size:
                memory_obj = self.reshape_partial_chunk(memory_obj, size)
            return memory_obj
        except Exception:
            memory_obj.invalidate()
            memory_obj.ref_count_down()
            raise
        finally:
            semaphore.release()

    async def _schedule_get(
        self,
        memory_obj: MemoryObj,
        object_key: str,
        expected_size: int,
    ) -> None:
        if self._io_executor is None:
            raise RuntimeError("RDMA executor not initialized")

        await self._io_executor.submit_job(
            functools.partial(
                self._get_sync,
                memory_obj=memory_obj,
                object_key=object_key,
                expected_size=expected_size,
            )
        )

    def _get_sync(
        self,
        memory_obj: MemoryObj,
        object_key: str,
        expected_size: int,
    ) -> None:
        client = self._ensure_client()
        buffer_view = self._bytes_view(memory_obj)
        try:
            request = BufferGetObject(
                bucket=self.settings.bucket,
                key=object_key,
                buffer=buffer_view,
                transfer_size=expected_size,
            )
            client.get_object_buffers(request)
        finally:
            try:
                buffer_view.release()  # type: ignore[attr-defined]
            except AttributeError:
                pass

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        object_key = self._object_key(key)
        semaphore = self._ensure_semaphore()
        await semaphore.acquire()
        try:
            await self._schedule_put(memory_obj, object_key)
            size = memory_obj.get_physical_size()
            self._record_object_size(object_key, size)
        finally:
            semaphore.release()

    async def _schedule_put(self, memory_obj: MemoryObj, object_key: str) -> None:
        if self._io_executor is None:
            raise RuntimeError("RDMA executor not initialized")
        await self._io_executor.submit_job(
            functools.partial(self._put_sync, memory_obj=memory_obj, object_key=object_key)
        )

    def _put_sync(self, memory_obj: MemoryObj, object_key: str) -> None:
        client = self._ensure_client()
        buffer_view = self._bytes_view(memory_obj)
        try:
            if self.full_chunk_size is not None and memory_obj.get_size() != self.full_chunk_size:
                raise ValueError("Partial chunk writes are not supported yet in RDMA S3 connector")
            request = BufferPutObject(
                bucket=self.settings.bucket,
                key=object_key,
                buffer=buffer_view,
                transfer_size=memory_obj.get_size(),
            )
            client.put_object_buffers(request)
        finally:
            try:
                buffer_view.release()  # type: ignore[attr-defined]
            except AttributeError:
                pass

    async def list(self) -> List[str]:
        raise NotImplementedError("Listing objects is not implemented for RDMA connector")

    async def close(self):
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=True)
            self._io_executor = None
        self._client = None

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        try:
            await asyncio.to_thread(self._ping_sync)
            return 0
        except Exception as exc:  # pragma: no cover - monitoring path
            logger.warning("S3 RDMA ping failed: %s", exc)
            return 1

    def _ping_sync(self) -> None:
        client = self._ensure_client()
        boto_client = client._boto_client  # type: ignore[attr-defined]
        boto_client.head_bucket(Bucket=self.settings.bucket)

    def support_batched_get(self) -> bool:
        return True

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        results = await asyncio.gather(*[self.get(key) for key in keys], return_exceptions=True)
        fixed: List[Optional[MemoryObj]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("S3 RDMA batched_get encountered error: %s", result)
                fixed.append(None)
            else:
                fixed.append(result)
        return fixed

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await asyncio.gather(
            *[
                self.put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=False)
            ],
            return_exceptions=False,
        )
