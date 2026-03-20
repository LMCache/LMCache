# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio

# Third Party
from awscrt import s3
from awscrt.http import HttpHeaders, HttpRequest

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryObj,
    MixedMemoryAllocator,
    PinMemoryAllocator,
)
from lmcache.v1.storage_backend.connector.cuobject_bindings import (
    CuObjClientWrapper,
    CuObjConfig,
)
from lmcache.v1.storage_backend.connector.s3_connector import S3Connector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


def _get_allocator_buffer_info(allocator) -> tuple[int, int]:
    """Extract the base pointer and byte size of the allocator's pinned pool.

    Supports :class:`MixedMemoryAllocator`, :class:`PinMemoryAllocator`,
    and :class:`LazyMemoryAllocator`.  Follows the same dispatch pattern
    used by ``distributed/memory_manager.py:get_vm_space()``.

    Returns:
        ``(base_ptr, size_bytes)``
    """
    if isinstance(allocator, (MixedMemoryAllocator, PinMemoryAllocator)):
        buf = allocator.buffer
        return buf.data_ptr(), buf.numel()
    elif isinstance(allocator, LazyMemoryAllocator):
        buf = allocator.get_underlying_buffer()
        return buf.data_ptr(), buf.numel()
    else:
        raise TypeError(
            f"Cannot extract RDMA-registrable buffer from "
            f"{type(allocator).__name__}. cuObject requires "
            f"MixedMemoryAllocator, PinMemoryAllocator, or "
            f"LazyMemoryAllocator."
        )


class CuObjectS3Connector(S3Connector):
    """S3 connector with RDMA-accelerated data transfer via NVIDIA cuObject.

    Inherits the full CRT-based S3 control plane from
    :class:`S3Connector` (auth, signing, TLS, circuit breaker, object
    size caching, priority-queue executor) and overrides only the data-
    plane methods (``_s3_upload`` / ``_s3_download``) to inject
    ``x-amz-rdma-token`` headers so that a cuObject-enabled storage
    server can transfer data via RDMA instead of the HTTP body.

    Pool-level RDMA registration
    ----------------------------
    At initialisation the entire pinned CPU memory pool backing
    ``local_cpu_backend.memory_allocator`` is registered with cuObject
    once.  Individual ``MemoryObj`` buffers are sub-regions of this pool
    and require no additional registration.

    Graceful fallback
    -----------------
    If the cuObject client library cannot be loaded or initialisation
    fails, the connector transparently falls back to the parent
    :class:`S3Connector` behaviour (pure HTTP body transfer).
    """

    def __init__(
        self,
        s3_endpoint: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        s3_num_io_threads: int,
        s3_prefer_http2: bool,
        s3_region: str,
        s3_enable_s3express: bool,
        disable_tls: bool,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        cuobj_nic_device: Optional[str] = None,
        cuobj_lib_path: Optional[str] = None,
    ):
        # Initialise the parent S3Connector (CRT client, auth, etc.)
        super().__init__(
            s3_endpoint=s3_endpoint,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            s3_num_io_threads=s3_num_io_threads,
            s3_prefer_http2=s3_prefer_http2,
            s3_region=s3_region,
            s3_enable_s3express=s3_enable_s3express,
            disable_tls=disable_tls,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # -- cuObject initialisation -----------------------------------------
        self._rdma_enabled = False
        self._cuobj_client = None
        self._rdma_pool_handle: Optional[tuple[int, int]] = None

        try:
            config = CuObjConfig(
                lib_path=cuobj_lib_path,
                nic_device=cuobj_nic_device,
            )
            self._cuobj_client = CuObjClientWrapper(config)

            # Pool-level RDMA registration
            allocator = local_cpu_backend.memory_allocator
            base_ptr, size_bytes = _get_allocator_buffer_info(allocator)
            if size_bytes > 0:
                self._rdma_pool_handle = self._cuobj_client.register_pool(
                    base_ptr, size_bytes
                )
                logger.info(
                    f"cuObject RDMA pool registered: "
                    f"ptr=0x{base_ptr:x}, size={size_bytes} bytes"
                )
            self._rdma_enabled = True
            logger.info("cuObject RDMA data plane enabled")
        except Exception as exc:
            logger.warning(
                f"cuObject initialisation failed, falling back to "
                f"HTTP data plane: {exc}"
            )
            self._cuobj_client = None
            self._rdma_enabled = False

    # -- Data-plane overrides ------------------------------------------------

    def _s3_upload(
        self,
        key_str: str,
        memory_obj: MemoryObj,
    ):
        """RDMA-accelerated upload.

        If RDMA is enabled, injects ``x-amz-rdma-token`` into the HTTP
        PUT request and sends no body — the cuObject-enabled server
        performs ``RDMA_READ`` directly from our pinned memory.

        Falls back to the parent's HTTP body upload on any RDMA failure.
        """
        if not self._rdma_enabled:
            return super()._s3_upload(key_str, memory_obj)

        try:
            return self._rdma_upload(key_str, memory_obj)
        except Exception as exc:
            logger.warning(
                f"RDMA upload preparation failed for {key_str}, "
                f"falling back to HTTP: {exc}"
            )
            return super()._s3_upload(key_str, memory_obj)

    def _rdma_upload(self, key_str: str, memory_obj: MemoryObj):
        """Build an RDMA-augmented PUT request via CRT."""
        data_size = memory_obj.get_physical_size()

        # Prepare RDMA token (sub-region within the registered pool)
        rdma_token = self._cuobj_client.prepare_put(
            memory_obj.data_ptr, data_size
        )

        # Build HTTP headers
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        headers.add("Content-Type", "application/octet-stream")
        headers.add("Content-Length", str(data_size))
        headers.add("x-amz-rdma-token", rdma_token)

        req = HttpRequest("PUT", self._format_safe_path(key_str), headers)
        # No body_stream — data is transferred via server-initiated RDMA_READ

        # Track RDMA reply from response headers
        rdma_state = {"reply": None, "err": None, "status": None}

        def on_headers(status_code, resp_headers, **kwargs):
            rdma_state["status"] = status_code
            for name, value in resp_headers:
                if name.lower() == "x-amz-rdma-reply":
                    rdma_state["reply"] = value

        def on_done(error=None, status_code=None, **kwargs):
            rdma_state["err"] = error
            final_status = status_code or rdma_state["status"]
            if rdma_state["err"] or final_status not in (200, 201):
                raise RuntimeError(
                    f"RDMA upload failed for {key_str}: "
                    f"error={rdma_state['err']}, status={final_status}"
                )
            # Verify RDMA completion
            if rdma_state["reply"]:
                self._cuobj_client.parse_rdma_reply(rdma_state["reply"])

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            operation_name="PutObject",
            request=req,
            on_headers=on_headers,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )
        return s3_req

    def _s3_download(
        self,
        key_str: str,
        mem_obj: MemoryObj,
    ):
        """RDMA-accelerated download.

        If RDMA is enabled, injects ``x-amz-rdma-token`` into the HTTP
        GET request — the cuObject-enabled server performs
        ``RDMA_WRITE`` directly into our pinned memory buffer.  No
        ``on_body`` callback is needed.

        Falls back to the parent's HTTP body download on any RDMA
        failure.
        """
        if not self._rdma_enabled:
            return super()._s3_download(key_str, mem_obj)

        try:
            return self._rdma_download(key_str, mem_obj)
        except Exception as exc:
            logger.warning(
                f"RDMA download preparation failed for {key_str}, "
                f"falling back to HTTP: {exc}"
            )
            return super()._s3_download(key_str, mem_obj)

    def _rdma_download(self, key_str: str, mem_obj: MemoryObj):
        """Build an RDMA-augmented GET request via CRT."""
        data_size = mem_obj.get_physical_size()

        # Prepare RDMA token (server will RDMA_WRITE into this buffer)
        rdma_token = self._cuobj_client.prepare_get(
            mem_obj.data_ptr, data_size
        )

        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        headers.add("x-amz-rdma-token", rdma_token)

        req = HttpRequest("GET", self._format_safe_path(key_str), headers)

        rdma_state = {"reply": None, "err": None, "status": None}

        def on_headers(status_code, resp_headers, **kwargs):
            rdma_state["status"] = status_code
            for name, value in resp_headers:
                if name.lower() == "x-amz-rdma-reply":
                    rdma_state["reply"] = value

        def on_done(error=None, status_code=None, **kwargs):
            rdma_state["err"] = error
            final_status = status_code or rdma_state["status"]
            ok = (final_status in (200, 206)) or (final_status is None)
            if rdma_state["err"] or not ok:
                raise RuntimeError(
                    f"RDMA download failed for {key_str}: "
                    f"error={rdma_state['err']}, status={final_status}"
                )
            if rdma_state["reply"]:
                self._cuobj_client.parse_rdma_reply(rdma_state["reply"])

        # No on_body — data arrives via RDMA_WRITE, not HTTP body
        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            operation_name="GetObject",
            request=req,
            on_headers=on_headers,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )
        return s3_req

    # -- Lifecycle -----------------------------------------------------------

    async def close(self):
        """Release RDMA resources, then shut down CRT."""
        if self._cuobj_client is not None:
            if self._rdma_pool_handle is not None:
                try:
                    self._cuobj_client.deregister_pool(
                        self._rdma_pool_handle
                    )
                except Exception as exc:
                    logger.warning(
                        f"Error deregistering RDMA pool: {exc}"
                    )
                self._rdma_pool_handle = None
            try:
                self._cuobj_client.close()
            except Exception as exc:
                logger.warning(f"Error closing cuObject client: {exc}")
            self._cuobj_client = None
            self._rdma_enabled = False
        await super().close()
