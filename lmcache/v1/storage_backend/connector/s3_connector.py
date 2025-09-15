# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from typing import List, Optional
from urllib.parse import quote as url_quote
import asyncio
import ctypes
import mmap
import os
import tempfile
import threading
import uuid

# Third Party
from awscrt import auth, io, s3
from awscrt.http import HttpHeaders, HttpRequest
from awscrt.io import ClientTlsContext, TlsConnectionOptions, TlsContextOptions

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


# TODO(Jiayi): Some pending problems.
# (1) We might need a filesystem-like allocator.
# This could be useful for local disk `LocalDiskBackend` and
# `/dev/shm` in `S3Connector`
# (2) Need to hack amazon python s3 crt library to enable `offset`
# to achieve zero-copy.
# (3) Need a job manager so that we can do sth like
# write priority, read priority, etc.
# (4) Potentially can drop the semaphore to reduce the complexity.
# Let crt handle the scheduling.


class AdhocSharedMemoryManager:
    """
    A shared memory manager that allocates shared memory buffers
    on demand.
    """

    def __init__(
        self,
        shm_buffers: list[int],
        shm_names: list[str],
        mmaps: list[mmap.mmap],
    ):
        self.shm_buffers = shm_buffers
        self.shm_names = shm_names
        self.mmaps = mmaps

    def allocate(self) -> tuple[str, int]:
        """
        Allocate a shared memory buffer and return its name and a bytearray
        that can be used to access the buffer.
        """
        if not self.shm_buffers:
            raise RuntimeError("No more shared memory buffers available")

        shm = self.shm_buffers.pop()
        shm_name = self.shm_names.pop()
        return shm_name, shm

    def free(
        self,
        shm_name: str,
        shm: int,
    ) -> None:
        """
        Free a shared memory buffer.
        """

        self.shm_buffers.append(shm)
        self.shm_names.append(shm_name)


class S3Connector(RemoteConnector):
    """
    S3 remote connector
    """

    def __init__(
        self,
        s3_endpoint: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        s3_part_size: Optional[int],
        s3_file_prefix: Optional[str],
        s3_max_io_concurrency: int,
        s3_max_inflight_reqs: int,
        s3_prefer_http2: bool,
        s3_region: str,
        s3_enable_s3express: bool,
    ):
        self.batched_async_contains_counter = 0
        self.batched_get_non_blocking_counter = 0
        if not s3_endpoint.startswith("s3://"):
            raise ValueError("S3 url must start with 's3://'")

        self.s3_endpoint = s3_endpoint.removeprefix("s3://")
        self.s3_prefix = s3_file_prefix
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.s3_part_size = s3_part_size

        # TODO(Jiayi): Now we only assume S3 part size = chunk size
        assert self.s3_part_size == self.full_chunk_size, (
            "S3 part size must be equal to chunk size in S3Connector"
        )

        self.s3_max_io_concurrency = s3_max_io_concurrency
        self.s3_max_inflight_reqs = s3_max_inflight_reqs
        self.s3_prefer_http2 = s3_prefer_http2
        self.s3_region = s3_region
        self.s3_enable_s3express = s3_enable_s3express

        event_loop_group = io.EventLoopGroup(s3_max_io_concurrency)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        self.credentials_provider = auth.AwsCredentialsProvider.new_default_chain(
            client_bootstrap
        )

        tls_opts = None
        if self.s3_prefer_http2:
            # Use HTTP/2 multiplexing if possible.
            tls_ctx = ClientTlsContext(TlsContextOptions())
            tls_opts = TlsConnectionOptions(tls_ctx)
            try:
                tls_opts.set_alpn_list(["h2", "http/1.1"])
            except Exception:
                tls_opts = None

        logger.info("Initializing S3 client")
        self.s3_client = s3.S3Client(
            bootstrap=client_bootstrap,
            region=s3_region,
            credential_provider=self.credentials_provider,
            enable_s3express=False,
            tls_connection_options=tls_opts,
            tls_mode=s3.S3RequestTlsMode.DISABLED,
        )

        # TODO(Jiayi): We need to handle cache consistency issues in a systematic way
        # across all connectors.
        # We assume S3 cache is never evicted and read-only for now.
        self.object_size_cache: dict[str, int] = {}
        # Non-greedy lock for cache access - if busy, skip cache and go to S3
        self.cache_lock = threading.Lock()

        self.inflight_sema = asyncio.Semaphore(s3_max_inflight_reqs)

        # # for async loading codepaths
        # self.pq_executor = AsyncPQExecutor(loop)

    def post_init(self):
        logger.info("Post-initializing S3 connector")

        if self.s3_part_size is None:
            # Default to chunk size
            self.s3_part_size = self.full_chunk_size
        assert self.s3_part_size == self.full_chunk_size, (
            "S3 part size must be equal to chunk size in S3Connector"
        )

        shm_name_prefix = "my_shm"
        shms = []
        shm_names = []
        mmaps = []
        for i in range(self.s3_max_inflight_reqs):
            shm_name = f"{shm_name_prefix}_{i}"

            shm = tempfile.NamedTemporaryFile(
                prefix=shm_name, suffix=".part", dir="/dev/shm", delete=False
            )

            os.ftruncate(shm.fileno(), self.full_chunk_size)

            with open(shm.name, "r+b") as f:
                mm = mmap.mmap(f.fileno(), self.full_chunk_size)
                # create a char buffer view over the mmap
                buf = ctypes.c_char.from_buffer(mm)
                addr = ctypes.addressof(buf)

            shms.append(addr)
            shm_names.append(shm.name)
            mmaps.append(mm)

        self.adhoc_shm_manager = AdhocSharedMemoryManager(
            shm_buffers=shms,
            shm_names=shm_names,
            mmaps=mmaps,
        )

    def _format_safe_path(self, key_str: str) -> str:
        """
        Generate a safe HTTP path for the S3 key.
        This is necessary because S3 keys can contain special characters
        that need to be URL-encoded.
        """
        flat_key_str = key_str.replace("/", "_")
        if self.s3_prefix:
            path = f"/{self.s3_prefix}/{flat_key_str}"
        else:
            path = f"/{flat_key_str}"
        # Keep slashes as they are path separators in S3.
        return url_quote(path, safe="/")

    def _try_get_cached_size(self, key_str: str) -> Optional[int]:
        """
        Non-greedy cache lookup. If lock is available, check cache.
        If lock is busy, return None (skip cache, go to S3).

        Returns:
            - cached size if found in cache
            - 0 if confirmed not to exist
            - None if cache unavailable (lock busy) or cache miss
        """
        if self.cache_lock.acquire(blocking=False):
            try:
                return self.object_size_cache.get(key_str, None)
            finally:
                self.cache_lock.release()
        else:
            # Lock is busy, skip cache
            return None

    def _update_cache_size(self, key_str: str, size: int) -> None:
        """
        Non-greedy cache update. If lock is available, update cache.
        If lock is busy, skip update.
        """
        if self.cache_lock.acquire(blocking=False):
            try:
                self.object_size_cache[key_str] = size
            finally:
                self.cache_lock.release()
        # If lock is busy, just skip the cache update

    # TODO(Jiayi): optimize this with async
    def _get_object_size(self, key_str: str) -> int:
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        req = HttpRequest("HEAD", self._format_safe_path(key_str), headers)

        got = {"len": None, "status": None, "err": None}

        def on_headers(status_code, headers, **kwargs):
            got["status"] = status_code
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        got["len"] = int(value)
                    except Exception:
                        pass

        def on_done(error=None, **kwargs):
            got["err"] = error

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="HeadObject",
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
        )

        try:
            s3_req.finished_future.result()
        except Exception as e:
            logger.debug(f"Exception in `_get_object_size`: {e}")
            return 0
        if got["err"] or got["status"] != 200:
            logger.warning("Encountering error in S3 HEAD request")
            return 0
        return got["len"] if got["len"] is not None else 0

    async def _get_object_size_async(self, key_str: str) -> int:
        """
        Async version of _get_object_size that doesn't block the event loop.
        """
        logger.info(f"_get_object_size_async called for {key_str}")
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        req = HttpRequest("HEAD", self._format_safe_path(key_str), headers)

        got = {"len": None, "status": None, "err": None}

        def on_headers(status_code, headers, **kwargs):
            got["status"] = status_code
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        got["len"] = int(value)
                    except Exception:
                        pass

        def on_done(error=None, **kwargs):
            got["err"] = error

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="HeadObject",
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
        )

        # Use the CRT library's built-in async support with timeout
        try:
            # Convert the Future to an asyncio-compatible awaitable with timeout
            await asyncio.wait_for(
                asyncio.wrap_future(s3_req.finished_future), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"S3 HEAD request timeout for {key_str} after 10s")
            return 0
        except Exception as e:
            logger.debug(f"S3 HEAD request failed for {key_str}: {e}")
            return 0

        if got["err"] or got["status"] != 200:
            logger.info(
                f"S3 HEAD request failed for {key_str}: ",
                "{got['err']} status={got['status']}",
            )
            return 0
        result = got["len"] if got["len"] is not None else 0
        logger.info(f"_get_object_size_async returning {result} for {key_str}")
        return result

    # TODO(Jiayi): implement real async
    async def exists(self, key: CacheEngineKey) -> bool:
        return self.exists_sync(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        key_str = key.to_string()

        # Try non-greedy cache lookup
        cached_size = self._try_get_cached_size(key_str)
        if cached_size is not None:
            return cached_size > 0

        # Cache miss or unavailable, check S3
        actual_size = self._get_object_size(key_str)

        # Try to update cache (non-blocking)
        self._update_cache_size(key_str, actual_size)

        return actual_size > 0

    def _s3_download(
        self,
        key_str: str,
        recv_path: str,
        done_event: threading.Event,
    ):
        """
        Download a file from S3.
        """
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)

        # TODO(Jiayi): Enable more finegrained data partition
        # range_header = f"bytes={start_byte}-{end_byte}"
        # headers.add("Range", range_header)

        req = HttpRequest("GET", self._format_safe_path(key_str), headers)

        # NOTE(Jiayi): Run in crt threads (not this thread) with GIL
        # See https://github.com/awslabs/aws-crt-python/blob/4250709624119de1af3ca86816e1a154fcac7cc8/source/common.c#L51
        def on_done(error=None, status_code=None, **kwargs):
            ok = (status_code in (200, 206)) or (status_code is None)
            if error or not ok:
                raise RuntimeError(
                    f"Failed to download {key_str} from S3: {error or status_code}"
                )

            done_event.set()

        # TODO(Jiayi): Need to support offset to enable zero-copy
        # More concretely, we need to get the shared memory offset.
        s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.GET_OBJECT,
            request=req,
            operation_name="GetObject",
            recv_filepath=recv_path,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        # Try non-greedy cache lookup
        obj_size = self._try_get_cached_size(key_str)

        if obj_size is None:
            # Cache miss or unavailable, check S3
            obj_size = self._get_object_size(key_str)
            if obj_size <= 0:
                return None
            # Try to update cache (non-blocking)
            self._update_cache_size(key_str, obj_size)

        await self.inflight_sema.acquire()

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shape,
            self.meta_dtype,
            self.meta_fmt,
        )

        # TODO(Jiayi): Please support this
        assert obj_size == memory_obj.get_size(), (
            "Saving unfull chunk is not supported in S3Connector."
        )

        done_event = threading.Event()

        # TODO(Jiayi): Need to support offset to enable zero-copy
        # We probably need to get the shared memory offset directly from memory object.
        recv_path, shm = self.adhoc_shm_manager.allocate()

        self._s3_download(
            key_str=key_str,
            recv_path=recv_path,
            done_event=done_event,
        )

        while not done_event.is_set():
            await asyncio.sleep(0.005)

        dst_ptr = memory_obj.data_ptr
        ctypes.memmove(dst_ptr, shm, obj_size)

        self.adhoc_shm_manager.free(recv_path, shm)

        self.inflight_sema.release()

        return memory_obj

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        done_events = []
        shms: list[Optional[int]] = []
        recv_paths: list[Optional[str]] = []
        memory_objs: list[Optional[MemoryObj]] = []
        obj_sizes = []

        uuid_str = str(uuid.uuid4())
        logger.info(f"batched get 11111 {uuid_str}")
        # TODO(Jiayi): Need to resolve this
        assert len(keys) <= self.s3_max_inflight_reqs, (
            f"Too many keys {len(keys)} to get in a single pass, "
            f"max is {self.s3_max_inflight_reqs}"
        )

        # TODO(Jiayi): Need some error handling in this loop.
        for key in keys:
            key_str = key.to_string()

            # Try non-greedy cache lookup
            logger.info(f"batched get 22222 {uuid_str}")
            obj_size = self._try_get_cached_size(key_str)
            logger.info(f"batched get 33333 {uuid_str}")

            if obj_size is None:
                # Cache miss or unavailable, check S3
                logger.info(f"batched get 44444 {uuid_str}")
                obj_size = self._get_object_size(key_str)
                logger.info(f"batched get 55555 {uuid_str}")
                if obj_size <= 0:
                    obj_sizes.append(0)
                    memory_objs.append(None)
                # Try to update cache (non-blocking)
                logger.info(f"batched get 66666 {uuid_str}")
                self._update_cache_size(key_str, obj_size)
                logger.info(f"batched get 77777 {uuid_str}")

            # TODO(Jiayi): A caveat of acquire this semaphore
            # is that we might face deadlock when `batched_put`
            # (not supported) is supported in the same fashion.
            logger.info(f"batched get 88888 {uuid_str}")
            await self.inflight_sema.acquire()
            logger.info(f"batched get 99999 {uuid_str}")

            memory_obj = self.local_cpu_backend.allocate(
                self.meta_shape,
                self.meta_dtype,
                self.meta_fmt,
            )

            obj_sizes.append(obj_size)
            memory_objs.append(memory_obj)

            if not memory_obj:
                shms.append(None)
                self.inflight_sema.release()
                continue

            # TODO(Jiayi): Please support this
            assert obj_size == memory_obj.get_size(), (
                "Saving unfull chunk is not supported in S3Connector."
            )

            done_event = threading.Event()
            done_events.append(done_event)

            logger.info(f"batched get 1212121212 {uuid_str}")
            recv_path, shm = self.adhoc_shm_manager.allocate()
            logger.info(f"batched get 1313131313 {uuid_str}")

            recv_paths.append(recv_path)
            logger.info(f"batched get 1010101010 {uuid_str}")
            self._s3_download(
                key_str=key_str,
                recv_path=recv_path,
                done_event=done_event,
            )
            logger.info(f"batched get 1111111111 {uuid_str}")
            shms.append(shm)

        while not all(e.is_set() for e in done_events):
            await asyncio.sleep(0.005)

        for obj_size, memory_obj, shm, recv_path in zip(
            obj_sizes, memory_objs, shms, recv_paths, strict=False
        ):
            if memory_obj is None or shm is None:
                continue

            dst_ptr = memory_obj.data_ptr
            ctypes.memmove(dst_ptr, shm, obj_size)

            self.adhoc_shm_manager.free(recv_path, shm)
            self.inflight_sema.release()

        return memory_objs

    async def _s3_upload(
        self,
        key_str: str,
        send_path: str,
        done_event: threading.Event,
    ):
        """
        Upload a file to S3.
        """
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)

        req = HttpRequest("PUT", self._format_safe_path(key_str), headers)

        done = {"err": None, "status": None}

        def on_done(error=None, status_code=None, **kwargs):
            done["err"] = error
            done["status"] = status_code

            if done["err"] or done["status"] not in (200, 201):
                raise RuntimeError(f"Upload failed in S3Connector: {done}")

            done_event.set()

        s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.PUT_OBJECT,
            request=req,
            operation_name="PutObject",
            send_filepath=send_path,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Store data to S3
        """

        key_str = key.to_string()

        # TODO(Jiayi): Please support this
        assert memory_obj.get_physical_size() == self.s3_part_size, (
            "Saving unfull chunk is not supported in S3Connector."
        )

        await self.inflight_sema.acquire()
        send_path, shm = self.adhoc_shm_manager.allocate()
        logger.debug("Allocated shared memory for S3 upload")

        try:
            buffer_ptr = memory_obj.data_ptr
            ctypes.memmove(shm, buffer_ptr, memory_obj.get_physical_size())
        except Exception as e:
            logger.error(f"Failed to copy data to S3 buffer: {e}")
        logger.debug("Data copy to S3 buffer completed")

        try:
            done_event = threading.Event()
            await self._s3_upload(key_str, send_path, done_event)
            while not done_event.is_set():
                await asyncio.sleep(0.005)

            # Update cache after successful upload
            self._update_cache_size(key_str, memory_obj.get_physical_size())

        except Exception as e:
            logger.error(f"Failed to upload {key_str} to S3: {e}")
            raise
        finally:
            self.inflight_sema.release()
            self.adhoc_shm_manager.free(send_path, shm)
            logger.debug(f"Uploaded {key_str} to S3 successfully")

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # return await self.pq_executor.submit_job(
        #     self._put,
        #     key=key,
        #     memory_obj=memory_obj,
        #     priority=Priorities.PUT,
        # )

        await self._put(key, memory_obj)

    async def list(self) -> List[str]:
        raise NotImplementedError

    def support_ping(self) -> bool:
        return False

    # TODO(Jiayi): This needs to be implemented.
    async def ping(self) -> int:
        raise NotImplementedError

    def support_batched_get(self) -> bool:
        return True

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check if the S3 objects exist for the given keys.
        Returns the number of consecutive hits from the start of the list.
        """
        num_hit_counts = 0

        uuid_str = str(uuid.uuid4())
        logger.info(f"11111 {uuid_str}")
        for key in keys:
            key_str = key.to_string()

            # Try non-greedy cache lookup
            cached_size = self._try_get_cached_size(key_str)
            if cached_size is not None:
                if cached_size > 0:
                    num_hit_counts += 1
                    continue
                else:
                    # Cached as non-existent
                    logger.info(f"22222 {uuid_str}")
                    return num_hit_counts

            # Cache miss or unavailable, check S3
            obj_size = await self._get_object_size_async(key_str)
            logger.info(f"obj_size {obj_size}")
            logger.info(f"33333 {uuid_str}")
            if obj_size <= 0:
                # Try to cache the negative result (non-blocking)
                self._update_cache_size(key_str, 0)
                logger.info(f"44444 {uuid_str}")
                return num_hit_counts

            # Try to cache the positive result (non-blocking)
            self._update_cache_size(key_str, obj_size)
            logger.info(f"55555 {uuid_str}")
            num_hit_counts += 1

        logger.info(f"66666 {uuid_str}")
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        self.batched_async_contains_counter += 1
        logger.info(
            f"batched_async_contains started {self.batched_async_contains_counter}"
        )
        # return await self.pq_executor.submit_job(
        #     self._batched_async_contains,
        #     lookup_id=lookup_id,
        #     keys=keys,
        #     pin=pin,
        #     priority=Priorities.PEEK,
        # )
        return await self._batched_async_contains(lookup_id, keys, pin)

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """
        Non-blocking batched get that reuses the existing batched_get implementation.
        The non-blocking aspect is handled by the StorageManager.
        """
        result = await self.batched_get(keys)
        return [r for r in result if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        self.batched_get_non_blocking_counter += 1
        logger.info(
            f"batched_get_non_blocking started {self.batched_get_non_blocking_counter}"
        )
        # return await self.pq_executor.submit_job(
        #     self._batched_get_non_blocking,
        #     lookup_id=lookup_id,
        #     keys=keys,
        #     priority=Priorities.PREFETCH,
        # )
        return await self._batched_get_non_blocking(lookup_id, keys)

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        # let python's GC clean up mmap inodes
        for mm in self.adhoc_shm_manager.mmaps:
            mm.close()
        # Clean up temporary files
        for shm_name in self.adhoc_shm_manager.shm_names:
            try:
                os.unlink(shm_name)
            except FileNotFoundError:
                pass  # Already deleted
