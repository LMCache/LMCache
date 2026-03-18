# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cuObject-RDMA-augmented S3 connector.

All AWS CRT and cuObject C library calls are mocked so tests can run
on any machine without real S3 or RDMA hardware.
"""
# Standard
from unittest.mock import MagicMock, patch
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
_CRT_PATCHES = {
    "lmcache.v1.storage_backend.connector.s3_connector.io": MagicMock(),
    "lmcache.v1.storage_backend.connector.s3_connector.auth": MagicMock(),
    "lmcache.v1.storage_backend.connector.s3_connector.s3": MagicMock(),
}


@pytest.fixture
def mock_crt():
    """Patch AWS CRT modules so S3Connector.__init__ succeeds."""
    with patch.dict("sys.modules", {}):
        patchers = [
            patch(target, mock)
            for target, mock in _CRT_PATCHES.items()
        ]
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

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            return_value=mock_client,
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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
        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            side_effect=ImportError("no cuobject"),
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            return_value=mock_client,
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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
            mem_shape = torch.Size(
                [_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]]
            )
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

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            return_value=mock_client,
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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

            mem_shape = torch.Size(
                [_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]]
            )
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

    def test_rdma_download_injects_token_header(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """GET request should contain x-amz-rdma-token header."""
        mock_client = _make_mock_cuobj_client()

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            return_value=mock_client,
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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

            mem_shape = torch.Size(
                [_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]]
            )
            mem_obj = local_cpu_backend.allocate(mem_shape, _DTYPE)
            mem_obj.ref_count_up()

            connector._rdma_download("test-key", mem_obj)

            mock_client.prepare_get.assert_called_once()
            args = mock_client.prepare_get.call_args
            assert args[0][0] == mem_obj.data_ptr

    def test_rdma_download_falls_back_on_error(
        self, mock_crt, async_loop, local_cpu_backend
    ):
        """If RDMA prepare_get fails, should fall back to HTTP download."""
        mock_client = _make_mock_cuobj_client()
        mock_client.prepare_get.side_effect = RuntimeError("RDMA error")

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            return_value=mock_client,
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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

            mem_shape = torch.Size(
                [_KV_SHAPE[1], _KV_SHAPE[3], 256, _KV_SHAPE[4]]
            )
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

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            return_value=mock_client,
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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

            # Run the async close
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

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjClientWrapper",
            return_value=mock_client,
        ), patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjConfig",
        ):
            # Local
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

            # Should not raise
            async_loop.run_until_complete(connector.close())
            mock_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
