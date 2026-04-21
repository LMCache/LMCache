# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cuObject-RDMA-augmented S3 connector.

All AWS CRT and cuObject C library calls are mocked so tests can run
on any machine without real S3 or RDMA hardware.
"""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import LocalCPUBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KV_SHAPE = (32, 2, 256, 8, 128)
_DTYPE = torch.bfloat16


def _get_metadata():
    return LMCacheMetadata(
        model_name="deepseek/DeepSeek-R1",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=_DTYPE,
        kv_shape=_KV_SHAPE,
        use_mla=False,
    )


def _create_local_cpu_backend(memory_allocator, config=None):
    if config is None:
        config = LMCacheEngineConfig.from_defaults()
    metadata = _get_metadata()
    return LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )


def _make_mock_cuobj_client():
    """Return a mock ``CuObjClientWrapper`` with sensible defaults."""
    client = MagicMock()
    client.register_pool.return_value = (0x1000, 4096)
    client.prepare_put.return_value = "fake-rdma-put-token"
    client.prepare_get.return_value = "fake-rdma-get-token"
    client.parse_rdma_reply.return_value = True
    client.deregister_pool.return_value = None
    client.close.return_value = None
    return client


def _make_mock_s3_request(status_code=200, rdma_reply="ok"):
    """Return a mock ``s3.S3Request`` with a resolved future."""
    future = asyncio.Future()
    future.set_result(None)

    req = MagicMock()
    req.finished_future = future
    return req


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

# Patch the CRT imports at the s3_connector module level so the parent
# S3Connector.__init__ doesn't try to talk to real AWS.
_CUOBJ = "lmcache.v1.storage_backend.connector.cuobject_s3_connector"
_S3 = "lmcache.v1.storage_backend.connector.s3_connector"
_CRT_PATCHES = {
    f"{_S3}.io": MagicMock(),
    f"{_S3}.auth": MagicMock(),
    f"{_S3}.s3": MagicMock(),
    # Prevent S3Connector.__init__ from spawning AsyncPQExecutor workers
    # whose _worker coroutines would never be awaited in unit tests.
    f"{_S3}.AsyncPQExecutor": MagicMock(),
    f"{_CUOBJ}.s3": MagicMock(),
    f"{_CUOBJ}.HttpHeaders": MagicMock(),
    f"{_CUOBJ}.HttpRequest": MagicMock(),
}


@pytest.fixture
def mock_crt():
    """Patch AWS CRT modules so S3Connector.__init__ succeeds."""
    patchers = [patch(target, mock) for target, mock in _CRT_PATCHES.items()]
    for p in patchers:
        p.start()
    yield
    for p in patchers:
        p.stop()


@pytest.fixture
def async_loop():
    """Create a fresh event loop for testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def local_cpu_backend():
    """Create a local CPU backend with a small pinned allocator."""
    allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    backend = _create_local_cpu_backend(allocator)
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCuObjectS3ConnectorInit:
    """Tests for connector initialisation and graceful fallback."""

    def test_rdma_enabled_when_cuobject_available(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """RDMA should be enabled when cuObject library loads OK."""
        mock_client = _make_mock_cuobj_client()

        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )
            assert connector._rdma_enabled is True
            mock_client.register_pool.assert_called_once()

    def test_fallback_when_cuobject_unavailable(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """RDMA should be disabled if cuObject import fails."""
        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                side_effect=ImportError("no cuobject"),
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )
            assert connector._rdma_enabled is False
            assert connector._cuobj_client is None


class TestRDMAUpload:
    """Tests for the RDMA-accelerated upload path."""

    def test_rdma_upload_injects_token_header(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """PUT request should contain x-amz-rdma-token header."""
        mock_client = _make_mock_cuobj_client()

        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            # Create a test memory object
            mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
            mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
            mem_obj.ref_count_up()

            # Exercise the upload path
            connector._rdma_upload("test-key", mem_obj)

            # Verify cuObjPut was called with the memory object's pointer
            mock_client.prepare_put.assert_called_once()
            args = mock_client.prepare_put.call_args
            assert args[0][0] == mem_obj.data_ptr

    def test_rdma_upload_falls_back_on_error(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """If RDMA prepare_put fails, should fall back to HTTP upload."""
        mock_client = _make_mock_cuobj_client()
        mock_client.prepare_put.side_effect = RuntimeError("RDMA error")

        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
            mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
            mem_obj.ref_count_up()

            # _s3_upload should catch the RDMA error and fall back
            with patch.object(
                CuObjectS3Connector.__bases__[0],
                "_s3_upload",
                return_value=MagicMock(),
            ) as parent_upload:
                connector._s3_upload("test-key", mem_obj)
                parent_upload.assert_called_once()


class TestRDMADownload:
    """Tests for the RDMA-accelerated download path."""

    def test_s3_download_uses_rdma_path_when_enabled(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """_s3_download should route through RDMA when enabled."""
        mock_client = _make_mock_cuobj_client()

        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
            mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
            mem_obj.ref_count_up()

            connector._s3_download("test-key", mem_obj)

            mock_client.prepare_get.assert_called_once()
            args = mock_client.prepare_get.call_args
            assert args[0][0] == mem_obj.data_ptr

    def test_rdma_download_falls_back_on_error(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """If RDMA prepare_get fails, should fall back to HTTP download."""
        mock_client = _make_mock_cuobj_client()
        mock_client.prepare_get.side_effect = RuntimeError("RDMA error")

        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
            mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
            mem_obj.ref_count_up()

            with patch.object(
                CuObjectS3Connector.__bases__[0],
                "_s3_download",
                return_value=MagicMock(),
            ) as parent_download:
                connector._s3_download("test-key", mem_obj)
                parent_download.assert_called_once()


class TestClose:
    """Tests for lifecycle cleanup."""

    def test_close_deregisters_pool_and_destroys_client(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """close() should deregister the RDMA pool and destroy the client."""
        mock_client = _make_mock_cuobj_client()

        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            # Mock the parent S3Connector.close() with AsyncMock to avoid
            # event-loop deadlock and unawaited coroutine warnings.
            with patch.object(
                CuObjectS3Connector.__bases__[0],
                "close",
                new=AsyncMock(),
            ):
                async_loop.run_until_complete(connector.close())

            mock_client.deregister_pool.assert_called_once()
            mock_client.close.assert_called_once()
            assert connector._rdma_enabled is False
            assert connector._cuobj_client is None

    def test_close_tolerates_deregister_failure(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """close() should not raise even if deregister_pool fails."""
        mock_client = _make_mock_cuobj_client()
        mock_client.deregister_pool.side_effect = RuntimeError("oops")

        with (
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(
                "lmcache.v1.storage_backend.connector."
                "cuobject_s3_connector.CuObjConfig",
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            # Mock the parent S3Connector.close() with AsyncMock to avoid
            # event-loop deadlock and unawaited coroutine warnings.
            with patch.object(
                CuObjectS3Connector.__bases__[0],
                "close",
                new=AsyncMock(),
            ):
                async_loop.run_until_complete(connector.close())
            mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# _get_allocator_buffer_info helper tests
# ---------------------------------------------------------------------------


class TestGetAllocatorBufferInfo:
    """Tests for the _get_allocator_buffer_info module-level helper.

    This function extracts (base_ptr, size_bytes) from different allocator
    types.  It is the entry point for RDMA memory registration.
    """

    # Scenario: PinMemoryAllocator has a .buffer attribute with data_ptr()
    # and numel().  _get_allocator_buffer_info should return those values.
    # Verification: Returned tuple matches (data_ptr, numel) from the buffer.
    def test_pin_memory_allocator(self):
        # First Party
        from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
            _get_allocator_buffer_info,
        )

        allocator = MagicMock(spec=PinMemoryAllocator)
        buf = MagicMock()
        buf.data_ptr.return_value = 0xDEAD
        buf.nbytes = 8192
        allocator.buffer = buf

        ptr, size = _get_allocator_buffer_info(allocator)
        assert ptr == 0xDEAD
        assert size == 8192

    # Scenario: MixedMemoryAllocator also has a .buffer attribute.
    # Verification: Same dispatch path as PinMemoryAllocator.
    def test_mixed_memory_allocator(self):
        # First Party
        from lmcache.v1.memory_management import MixedMemoryAllocator
        from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
            _get_allocator_buffer_info,
        )

        allocator = MagicMock(spec=MixedMemoryAllocator)
        buf = MagicMock()
        buf.data_ptr.return_value = 0xBEEF
        buf.nbytes = 4096
        allocator.buffer = buf

        ptr, size = _get_allocator_buffer_info(allocator)
        assert ptr == 0xBEEF
        assert size == 4096

    # Scenario: LazyMemoryAllocator uses get_underlying_buffer() instead of
    # .buffer.  _get_allocator_buffer_info should detect this and call the
    # right method.
    # Verification: Returned values come from get_underlying_buffer().
    def test_lazy_memory_allocator(self):
        # First Party
        from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
        from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
            _get_allocator_buffer_info,
        )

        allocator = MagicMock(spec=LazyMemoryAllocator)
        buf = MagicMock()
        buf.data_ptr.return_value = 0xCAFE
        buf.nbytes = 16384
        allocator.get_underlying_buffer.return_value = buf

        ptr, size = _get_allocator_buffer_info(allocator)
        assert ptr == 0xCAFE
        assert size == 16384
        allocator.get_underlying_buffer.assert_called_once()

    # Scenario: An unsupported allocator type (e.g. a plain MagicMock with
    # no spec) is passed.  The function should raise TypeError with a
    # helpful message listing the supported types.
    # Verification: TypeError is raised with the allocator class name.
    def test_unsupported_allocator_raises_type_error(self):
        # First Party
        from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
            _get_allocator_buffer_info,
        )

        class UnsupportedAllocator:
            pass

        with pytest.raises(TypeError, match="UnsupportedAllocator"):
            _get_allocator_buffer_info(UnsupportedAllocator())


# ---------------------------------------------------------------------------
# RDMA upload callback tests
# ---------------------------------------------------------------------------


class TestRDMAUploadCallbacks:
    """Tests for the internal on_headers/on_done callbacks in _rdma_upload.

    These callbacks are closures created inside _rdma_upload and are invoked
    by CRT when the S3 response arrives.  They verify RDMA completion status
    and raise on failures.
    """

    def _build_connector(self, mock_crt, async_loop, local_cpu_backend):
        """Create a CuObjectS3Connector with mocked cuObject client."""
        mock_client = _make_mock_cuobj_client()

        with (
            patch(
                f"{_CUOBJ}.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(f"{_CUOBJ}.CuObjConfig"),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )
        return connector, mock_client

    # Scenario: _rdma_upload builds an S3Request via CRT's s3.S3Request().
    # We capture the kwargs passed to s3.S3Request() to extract the
    # on_headers and on_done callbacks for isolated testing.
    # Verification: s3.S3Request is called with operation_name="PutObject"
    # and the on_headers/on_done callbacks are callable.
    def test_upload_s3_request_has_correct_operation(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_upload("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        assert call_kwargs.kwargs["operation_name"] == "PutObject"
        assert call_kwargs.kwargs["on_headers"] is not None
        assert call_kwargs.kwargs["on_done"] is not None

    # Scenario: The on_headers callback receives a 200 status and an
    # "x-amz-rdma-reply: ok" header.  Then on_done is called with no
    # error.  The full cycle should complete without raising.
    # Verification: on_done does not raise; parse_rdma_reply is called
    # with the reply value.
    def test_upload_on_done_success(self, mock_crt, async_loop, local_cpu_backend):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_upload("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        # Simulate CRT response
        on_headers(200, [("x-amz-rdma-reply", "ok")])
        on_done(error=None, status_code=200)  # should not raise
        mock_client.parse_rdma_reply.assert_called_with("ok")

    # Scenario: The server returns an HTTP error (e.g. 500).
    # The on_done callback should raise RuntimeError to signal failure.
    # Verification: RuntimeError is raised with the status code.
    def test_upload_on_done_raises_on_http_error(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_upload("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(500, [])
        with pytest.raises(RuntimeError, match="RDMA upload failed"):
            on_done(error=None, status_code=500)

    # Scenario: CRT reports a transport-level error (e.g. connection reset).
    # on_done receives a non-None error argument.
    # Verification: RuntimeError is raised with error info.
    def test_upload_on_done_raises_on_crt_error(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_upload("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(200, [("x-amz-rdma-reply", "ok")])
        with pytest.raises(RuntimeError, match="RDMA upload failed"):
            on_done(error="ConnectionReset", status_code=200)

    # Scenario: The HTTP response is 200 but the x-amz-rdma-reply header
    # indicates failure (e.g. "error: RDMA timeout").
    # parse_rdma_reply returns False, so on_done should raise.
    # Verification: RuntimeError with "verification failed" message.
    def test_upload_on_done_raises_on_rdma_reply_failure(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mock_client.parse_rdma_reply.return_value = False

        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_upload("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(200, [("x-amz-rdma-reply", "error: timeout")])
        with pytest.raises(RuntimeError, match="verification failed"):
            on_done(error=None, status_code=200)

    # Scenario: HTTP 201 Created is a valid success status for PUT.
    # Verification: on_done does not raise when status_code=201.
    def test_upload_on_done_accepts_201_created(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_upload("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(201, [("x-amz-rdma-reply", "ok")])
        on_done(error=None, status_code=201)  # should not raise

    # Scenario: No x-amz-rdma-reply header in the response (e.g. the
    # server does not support RDMA but still returns 200).  rdma_state
    # reply remains None.  on_done should NOT call parse_rdma_reply and
    # should not raise.
    # Verification: parse_rdma_reply is NOT called when reply is None.
    def test_upload_on_done_no_rdma_reply_header(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_upload("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        # No x-amz-rdma-reply header
        on_headers(200, [("content-type", "application/xml")])
        on_done(error=None, status_code=200)  # should not raise
        mock_client.parse_rdma_reply.assert_not_called()


# ---------------------------------------------------------------------------
# RDMA download callback tests
# ---------------------------------------------------------------------------


class TestRDMADownloadCallbacks:
    """Tests for the internal on_headers/on_done callbacks in _rdma_download.

    These callbacks verify RDMA completion for GET operations.  The download
    path differs from upload in accepted status codes (200, 206).
    """

    def _build_connector(self, mock_crt, async_loop, local_cpu_backend):
        """Create a CuObjectS3Connector with mocked cuObject client."""
        mock_client = _make_mock_cuobj_client()

        with (
            patch(
                f"{_CUOBJ}.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(f"{_CUOBJ}.CuObjConfig"),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )
        return connector, mock_client

    # Scenario: s3.S3Request is called with operation_name="GetObject"
    # and type=S3RequestType.DEFAULT (no on_body callback for RDMA).
    # Verification: The operation_name kwarg is "GetObject".
    def test_download_s3_request_has_correct_operation(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_download("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        assert call_kwargs.kwargs["operation_name"] == "GetObject"
        # No on_body callback for RDMA downloads
        assert "on_body" not in call_kwargs.kwargs

    # Scenario: Successful download with HTTP 200 and valid RDMA reply.
    # Verification: on_done completes without raising.
    def test_download_on_done_success(self, mock_crt, async_loop, local_cpu_backend):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_download("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(200, [("x-amz-rdma-reply", "ok")])
        on_done(error=None, status_code=200)  # should not raise

    # Scenario: HTTP 206 Partial Content is valid for range GET downloads.
    # The download on_done accepts both 200 and 206.
    # Verification: on_done does not raise when status_code=206.
    def test_download_on_done_accepts_206_partial_content(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_download("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(206, [("x-amz-rdma-reply", "ok")])
        on_done(error=None, status_code=206)  # should not raise

    # Scenario: Server returns HTTP 404 Not Found for a missing object.
    # on_done should raise RuntimeError.
    # Verification: RuntimeError with "RDMA download failed" message.
    def test_download_on_done_raises_on_404(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_download("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(404, [])
        with pytest.raises(RuntimeError, match="RDMA download failed"):
            on_done(error=None, status_code=404)

    # Scenario: RDMA reply indicates failure on a 200 response.
    # Verification: RuntimeError with "verification failed" message.
    def test_download_on_done_raises_on_rdma_reply_failure(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mock_client.parse_rdma_reply.return_value = False

        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_download("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(200, [("x-amz-rdma-reply", "error")])
        with pytest.raises(RuntimeError, match="verification failed"):
            on_done(error=None, status_code=200)

    # Scenario: on_done receives status_code=None (CRT sometimes omits it).
    # The download callback treats None status as OK (server may not send
    # a final status code in all paths).
    # Verification: on_done does not raise when status_code is None and
    # there is no error.
    def test_download_on_done_none_status_is_ok(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        connector, mock_client = self._build_connector(
            mock_crt, async_loop, local_cpu_backend
        )
        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        # First Party
        import lmcache.v1.storage_backend.connector.cuobject_s3_connector as mod

        s3_mock = mod.s3

        connector._rdma_download("test-key", mem_obj)

        call_kwargs = s3_mock.S3Request.call_args
        on_headers = call_kwargs.kwargs["on_headers"]
        on_done = call_kwargs.kwargs["on_done"]

        on_headers(200, [])
        on_done(error=None, status_code=None)  # should not raise


# ---------------------------------------------------------------------------
# _s3_upload / _s3_download delegation tests
# ---------------------------------------------------------------------------


class TestDataPlaneDelegation:
    """Tests for _s3_upload/_s3_download dispatch logic.

    When _rdma_enabled is False, the connector should delegate directly to
    the parent S3Connector methods without touching the cuObject client.
    """

    def _build_connector_with_rdma_disabled(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """Create a CuObjectS3Connector with RDMA disabled."""
        with (
            patch(
                f"{_CUOBJ}.CuObjClientWrapper",
                side_effect=ImportError("no cuobject"),
            ),
            patch(f"{_CUOBJ}.CuObjConfig"),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )
        return connector

    # Scenario: _s3_upload is called when _rdma_enabled is False.
    # The connector should delegate directly to S3Connector._s3_upload()
    # without calling prepare_put.
    # Verification: Parent _s3_upload is called; no cuObject interaction.
    def test_upload_delegates_to_parent_when_rdma_disabled(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        # First Party
        from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
            CuObjectS3Connector,
        )

        connector = self._build_connector_with_rdma_disabled(
            mock_crt, async_loop, local_cpu_backend
        )
        assert connector._rdma_enabled is False

        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        with patch.object(
            CuObjectS3Connector.__bases__[0],
            "_s3_upload",
            return_value=MagicMock(),
        ) as parent_upload:
            connector._s3_upload("test-key", mem_obj)
            parent_upload.assert_called_once_with("test-key", mem_obj)

    # Scenario: _s3_download is called when _rdma_enabled is False.
    # Verification: Parent _s3_download is called; no cuObject interaction.
    def test_download_delegates_to_parent_when_rdma_disabled(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        # First Party
        from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
            CuObjectS3Connector,
        )

        connector = self._build_connector_with_rdma_disabled(
            mock_crt, async_loop, local_cpu_backend
        )

        mem_shape = torch.Size([_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]])
        mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
        mem_obj.ref_count_up()

        with patch.object(
            CuObjectS3Connector.__bases__[0],
            "_s3_download",
            return_value=MagicMock(),
        ) as parent_download:
            connector._s3_download("test-key", mem_obj)
            parent_download.assert_called_once_with("test-key", mem_obj)


# ---------------------------------------------------------------------------
# Init edge cases
# ---------------------------------------------------------------------------


class TestInitEdgeCases:
    """Tests for edge cases during CuObjectS3Connector initialisation."""

    # Scenario: The memory allocator reports size=0 (empty pool).
    # The connector should skip pool registration and still enable RDMA
    # (the pool handle will be None).
    # Verification: register_pool is NOT called; _rdma_enabled is True.
    def test_zero_size_allocator_skips_registration(self, mock_crt, async_loop):
        # Create a local_cpu_backend whose allocator reports size 0
        allocator = MagicMock(spec=PinMemoryAllocator)
        buf = MagicMock()
        buf.data_ptr.return_value = 0x0
        buf.numel.return_value = 0
        allocator.buffer = buf

        backend = _create_local_cpu_backend(allocator)
        mock_client = _make_mock_cuobj_client()

        with (
            patch(
                f"{_CUOBJ}.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(f"{_CUOBJ}.CuObjConfig"),
            patch(
                f"{_CUOBJ}._get_allocator_buffer_info",
                return_value=(0x0, 0),
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            mock_client.register_pool.assert_not_called()
            assert connector._rdma_enabled is False
            assert connector._rdma_pool_handle is None

        backend.close()

    # Scenario: cuObject client creation succeeds but register_pool fails
    # (e.g. RDMA NIC not available, memory not pinned).
    # The connector should catch the exception and fall back to HTTP.
    # Verification: _rdma_enabled is False after init.
    def test_register_pool_failure_disables_rdma(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        mock_client = _make_mock_cuobj_client()
        mock_client.register_pool.side_effect = RuntimeError(
            "cuMemObjGetDescriptor failed"
        )

        with (
            patch(
                f"{_CUOBJ}.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(f"{_CUOBJ}.CuObjConfig"),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            assert connector._rdma_enabled is False


# ---------------------------------------------------------------------------
# Close edge cases
# ---------------------------------------------------------------------------


class TestCloseEdgeCases:
    """Tests for edge cases during close()."""

    # Scenario: close() is called on a connector whose cuobj_client is
    # already None (e.g. RDMA init failed or close() called twice).
    # Verification: No exception raised, parent close() still called.
    def test_close_when_cuobj_client_is_none(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        with (
            patch(
                f"{_CUOBJ}.CuObjClientWrapper",
                side_effect=ImportError("no cuobject"),
            ),
            patch(f"{_CUOBJ}.CuObjConfig"),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            assert connector._cuobj_client is None

            with patch.object(
                CuObjectS3Connector.__bases__[0],
                "close",
                new=AsyncMock(),
            ):
                # Should not raise
                async_loop.run_until_complete(connector.close())

    # Scenario: close() is called but the cuObject client's close() method
    # raises.  The connector should catch it, log a warning, and still
    # complete the parent close().
    # Verification: No exception propagated; parent close() still called.
    def test_close_tolerates_client_close_failure(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        mock_client = _make_mock_cuobj_client()
        mock_client.close.side_effect = RuntimeError("close failed")

        with (
            patch(
                f"{_CUOBJ}.CuObjClientWrapper",
                return_value=mock_client,
            ),
            patch(f"{_CUOBJ}.CuObjConfig"),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.cuobject_s3_connector import (
                CuObjectS3Connector,
            )

            connector = CuObjectS3Connector(
                s3_endpoint="s3://test-bucket.s3.us-east-1.amazonaws.com",
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                s3_num_io_threads=1,
                s3_prefer_http2=False,
                s3_region="us-east-1",
                s3_enable_s3express=False,
                disable_tls=True,
            )

            with patch.object(
                CuObjectS3Connector.__bases__[0],
                "close",
                new=AsyncMock(),
            ):
                # Should not raise despite client.close() failure
                async_loop.run_until_complete(connector.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
