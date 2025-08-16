# Standard
from typing import List, Optional
from multiprocessing import shared_memory
import abc
import concurrent.futures
import threading

# Third Party
import torch
from awscrt import io, auth, s3, http
from awscrt.http import HttpHeaders, HttpRequest
from awscrt.io import TlsContextOptions, ClientTlsContext, TlsConnectionOptions

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


# TODO(Jiayi): Some pending problems.
# (1) We might need a filesystem-like allocator.
# This could be useful for local disk `LocalDiskBackend` and
# `/dev/shm` in `S3Connector`
# (2) Need to hack amazon python s3 crt library to enable `offset`
# to achieve zero-copy.

class S3Connector(RemoteConnector):
    """
    S3 remote connector
    """

    def __init__(
        self,
        s3_end_point: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        s3_part_size: int,
        s3_max_io_concurrency: int,
        s3_max_inflight_reqs: int,
        s3_prefer_http2: bool,
        s3_region: str,
        s3_enable_s3express: bool,
    ):

        self.s3_end_point = s3_end_point
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.s3_part_size = s3_part_size
        # TODO(Jiayi): Now we only assume S3 part size = chunk size
        # FIXME: Add an assertion here

        self.s3_max_io_concurrency = s3_max_io_concurrency
        self.s3_max_inflight_reqs = s3_max_inflight_reqs
        self.s3_prefer_http2 = s3_prefer_http2
        self.s3_region = s3_region
        self.s3_enable_s3express = s3_enable_s3express

        event_loop_group = io.EventLoopGroup(s3_max_io_concurrency)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        self.credentials_provider = auth.AwsCredentialsProvider.new_default_chain(client_bootstrap)

        tls_opts = None
        if PREFER_HTTP2:
            # Use HTTP/2 multiplexing if possible.
            tls_ctx = ClientTlsContext(TlsContextOptions())
            tls_opts = TlsConnectionOptions(tls_ctx)
            try:
                tls_opts.set_alpn_list(["h2", "http/1.1"])
            except Exception:
                tls_opts = None
        
        self.s3_client = s3.S3Client(
            bootstrap=client_bootstrap,
            region=s3_region,
            credential_provider=self.credentials_provider,
            enable_s3express=True,
            tls_connection_options=tls_opts
        )

        # TODO(Jiayi): We need to handle cache consistency issues in a systematic way
        # across all connectors.
        # We assume S3 cache is never evicted and read-only for now.
        self.object_size_cache = {}

        self.inflight_sema = threading.Semaphore(s3_max_inflight_reqs)
        # NOTE(Jiayi): Threading lock is used here because crt threads
        # are spawned in C and are not managed by python.
        self.inflight_lock = threading.Lock()

        self.num_inflight_reqs = 0
        
        shm_name_prefix = "my_shm"
        self.shms = []
        self.shm_names = []
        for i in range(s3_max_inflight_reqs):
            shm_name = f"{shm_name_prefix}_{i}"
            shm = shared_memory.SharedMemory(
                create=True, size=, name=shm_name)
            self.shms.append(shm)
            self.shm_names.append(shm_name)
        
        # FIXME: Need a buffer menager
        self.adhoc_shm_manager

    

    def _get_object_size(self, key_str: str) -> int:
        headers = HttpHeaders()
        headers.add("Host", ENDPOINT)
        req = HttpRequest("HEAD", f"/{key_str}", headers)

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
            client=s3_client,
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
        except:
            return 0
        if got["err"] or got["status"] != 200:
            logger.warning("Encountering error in S3 HEAD request")
            return 0
        return got["len"]

    # TODO(Jiayi): implement real async
    async def exists(self, key: CacheEngineKey) -> bool:
        return self.exists_sync(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        key_str = key.to_string()
        if key_str in self.object_size_cache:
            return True
        cache_size = self._get_object_size(key_str)
        if cache_size > 0:
            self.object_size_cache[key_str] = cache_size
            return True
        return False

    def _s3_download(
        self, 
        key_str: str,
        obj_size: int,
        memory_obj: MemoryObj,
    ):
        # TODO(Jiayi): Need to support offset to enable zero-copy
        # We probably need to get the shared memory offset directly from memory object.
        with self.inflight_lock:
            recv_path, shm_buffer = self.adhoc_shm_manager.allocate()

        headers = HttpHeaders()
        headers.add("Host", ENDPOINT)

        # TODO(Jiayi): Enbale more finegrained data partition
        # range_header = f"bytes={start_byte}-{end_byte}"
        # headers.add("Range", range_header)

        req = HttpRequest("GET", f"/{key_str}", headers)

        # NOTE(Jiayi): Run in crt threads (not this thread) with GIL
        def on_done(error=None, status_code=None, **kwargs):
            ok = (status_code in (200, 206)) or (status_code is None)
            if error or not ok:
                raise RuntimeError(
                    f"Failed to download {key_str} from S3: {error or status_code}"
                )
            with self.inflight_lock:
                self.num_inflight_reqs -= 1
                self.adhoc_shm_manager.free(recv_path)
            
            # TODO(Jiayi): Need to support offset to enable zero-copy

            done_event.set()

            self.inflight_sema.release()
        
        # TODO(Jiayi): Need to support offset to enable zero-copy
        # More concretely, we need to get the shared memory offset.
        s3_req = s3.S3Request(
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
        
        obj_size = self.object_size_cache.get(key_str, None)

        if obj_size is None:
            obj_size = self._get_object_size(key_str)
            if obj_size <= 0:
                return None
            self.object_size_cache[key_str] = obj_size
        
        self.inflight_sema.acquire()

        memory_obj = self.local_cpu_backend.allocate(
                        self.meta_shape,
                        self.meta_dtype,
                        self.meta_fmt,
                    )
        
        # TODO(Jiayi): Please support this
        assert obj_size == memory_obj.get_size(), \
                "Saving unfull chunk is not supported in S3Connector."

        done_event = threading.Event()

        shm_buffer = self._s3_download(
            key_str=key_str,
            obj_size=obj_size,
            memory_obj=memory_obj,
            done_event=done_event,
        )

        while not done_event.is_set():
            time.sleep(0.005)


        dst_buffer = memory_obj.byte_array
        dst_buffer[:] = shm_buffer[:obj_size]

        return memory_obj


    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:

        done_events = []
        shm_buffers = []
        memory_objs = []
        obj_sizes = []

        # TODO(Jiayi): Need some error handling in this loop.
        for key in keys:
            key_str = key.to_string()
            
            obj_size = self.object_size_cache.get(key_str, None)

            if obj_size is None:
                obj_size = self._get_object_size(key_str)
                if obj_size <= 0:
                    obj_sizes.append(0)
                    memory_objs.append(None)
                    return None
                self.object_size_cache[key_str] = obj_size
            
            self.inflight_sema.acquire()

            memory_obj = self.local_cpu_backend.allocate(
                            self.meta_shape,
                            self.meta_dtype,
                            self.meta_fmt,
                        )

            obj_sizes.append(obj_size)
            memory_objs.append(memory_obj)

            if not memory_obj:
                shm_buffers.append(None)
                continue
            
            # TODO(Jiayi): Please support this
            assert obj_size == memory_obj.get_size(), \
                    "Saving unfull chunk is not supported in S3Connector."

            done_event = threading.Event()
            done_events.append(done_event)

            shm_buffer = self._s3_download(
                key_str=key_str,
                obj_size=obj_size,
                memory_obj=memory_obj,
                done_event=done_event,
            )
            shm_buffers.append(shm_buffer)

        while not all(e.is_set() for e in done_events):
            time.sleep(0.005)

        for obj_size, memory_obj, shm_buffer in zip(
            obj_sizes, memory_objs, shm_buffers):

            if memory_obj is None:
                continue

            dst_buffer = memory_obj.byte_array
            dst_buffer[:] = shm_buffer[:obj_size]
        
        return memory_objs


    def _s3_upload(key_str: str, memory_obj: MemoryObj):
        headers = HttpHeaders()
        headers.add("Host", self.s3_end_point)

        req = HttpRequest("PUT", f"/{key_str}", headers)

        done = {"err": None, "status": None}

        def on_done(error=None, status_code=None, **kwargs):
            done["err"] = error
            done["status"] = status_code

        s3_req = s3.S3Request(
            client=s3_client,
            type=s3.S3RequestType.PUT_OBJECT,
            request=req,
            operation_name="PutObject",
            send_filepath=file_path,     # zero-copy upload from file
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )

        if done["err"] or done["status"] not in (200, 201):
            raise RuntimeError(f"Upload failed in S3Connector: {done}")
        
    @abc.abstractmethod
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        raise NotImplementedError

    async def list(self) -> List[str]:
        raise NotImplementedError

    def support_ping(self) -> bool:
        return False

    # TODO(Jiayi): This needs to be implemented.
    async def ping(self) -> int:
        raise NotImplementedError

    def support_batched_get(self) -> bool:
        return True
    
    async def close(self):
        for shm in self.shms:
            shm.close()
            shm.unlink()

