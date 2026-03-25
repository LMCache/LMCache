# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cuObject S3 connector adapter.

Verifies that:
- The adapter registers itself with the ``cuobj+s3://`` prefix.
- ``create_connector`` extracts cuObject-specific config and strips the
  ``cuobj+`` prefix before passing the URL to the connector.
- Auto-discovery picks up the adapter from the connector package.
"""

# Standard
from types import ModuleType
from unittest.mock import MagicMock, patch
import asyncio
import importlib
import pkgutil

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorManager,
)
from lmcache.v1.storage_backend.connector.cuobject_s3_adapter import (
    CuObjectS3ConnectorAdapter,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdapterRegistration:
    """Verify adapter metadata."""

    def test_prefix(self):
        adapter = CuObjectS3ConnectorAdapter()
        assert adapter.schema == "cuobj+s3://"

    def test_is_connector_adapter(self):
        adapter = CuObjectS3ConnectorAdapter()
        assert isinstance(adapter, ConnectorAdapter)


class TestCreateConnector:
    """Verify that create_connector builds the right connector."""

    def test_strips_cuobj_prefix(self):
        """The URL passed to CuObjectS3Connector should be s3://..."""
        adapter = CuObjectS3ConnectorAdapter()

        mock_config = MagicMock()
        mock_config.extra_config = {
            "s3_region": "us-east-1",
            "s3_num_io_threads": 4,
            "s3_prefer_http2": False,
            "s3_enable_s3express": False,
            "disable_tls": True,
            "cuobj_nic_device": "mlx5_0",
        }

        mock_metadata = MagicMock()
        mock_backend = MagicMock()
        mock_loop = asyncio.new_event_loop()

        mock_context = MagicMock()
        mock_context.url = "cuobj+s3://bucket.s3.us-east-1.amazonaws.com"
        mock_context.config = mock_config
        mock_context.metadata = mock_metadata
        mock_context.loop = mock_loop
        mock_context.local_cpu_backend = mock_backend

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjectS3Connector"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter.create_connector(mock_context)
            # Verify the s3_endpoint was stripped of "cuobj+"
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs["s3_endpoint"] == (
                "s3://bucket.s3.us-east-1.amazonaws.com"
            )
            assert call_kwargs.kwargs["s3_region"] == "us-east-1"
            assert call_kwargs.kwargs["cuobj_nic_device"] == "mlx5_0"

        mock_loop.close()

    def test_requires_s3_region(self):
        """Should assert if s3_region is not provided."""
        adapter = CuObjectS3ConnectorAdapter()
        mock_config = MagicMock()
        mock_config.extra_config = {}

        mock_context = MagicMock()
        mock_context.url = "cuobj+s3://bucket"
        mock_context.config = mock_config
        mock_context.metadata = MagicMock()

        with pytest.raises(AssertionError, match="s3_region is required"):
            adapter.create_connector(mock_context)

    def test_requires_metadata(self):
        """Should raise if metadata is None."""
        adapter = CuObjectS3ConnectorAdapter()
        mock_config = MagicMock()
        mock_config.extra_config = {"s3_region": "us-east-1"}

        mock_context = MagicMock()
        mock_context.url = "cuobj+s3://bucket"
        mock_context.config = mock_config
        mock_context.metadata = None

        with pytest.raises(ValueError, match="metadata is required"):
            adapter.create_connector(mock_context)

    # Scenario: save_chunk_meta is set to True in extra_config.
    # The adapter asserts this must be False for cuObject+S3 because
    # RDMA transfers do not support chunk metadata.
    # Verification: AssertionError is raised with descriptive message.
    def test_rejects_save_chunk_meta_true(self):
        adapter = CuObjectS3ConnectorAdapter()
        mock_config = MagicMock()
        mock_config.extra_config = {
            "s3_region": "us-east-1",
            "save_chunk_meta": True,
        }
        mock_context = MagicMock()
        mock_context.url = "cuobj+s3://bucket"
        mock_context.config = mock_config
        mock_context.metadata = MagicMock()

        with pytest.raises(AssertionError, match="save_chunk_meta must be False"):
            adapter.create_connector(mock_context)

    # Scenario: extra_config is None (not provided in the LMCacheEngineConfig).
    # The adapter should treat it as an empty dict and use defaults.
    # Verification: AssertionError for missing s3_region (the first required
    # param check) rather than a TypeError from iterating None.
    def test_none_extra_config_uses_empty_defaults(self):
        adapter = CuObjectS3ConnectorAdapter()
        mock_config = MagicMock()
        mock_config.extra_config = None

        mock_context = MagicMock()
        mock_context.url = "cuobj+s3://bucket"
        mock_context.config = mock_config
        mock_context.metadata = MagicMock()

        with pytest.raises(AssertionError, match="s3_region is required"):
            adapter.create_connector(mock_context)

    # Scenario: Default values for optional config keys.
    # When only s3_region is provided, the adapter should pass correct
    # defaults to the CuObjectS3Connector constructor.
    # Verification: Check kwargs for default values of s3_num_io_threads,
    # s3_prefer_http2, disable_tls, cuobj_nic_device, etc.
    def test_default_optional_config_values(self):
        adapter = CuObjectS3ConnectorAdapter()
        mock_config = MagicMock()
        mock_config.extra_config = {"s3_region": "eu-west-1"}

        mock_context = MagicMock()
        mock_context.url = "cuobj+s3://bucket.eu-west-1.amazonaws.com"
        mock_context.config = mock_config
        mock_context.metadata = MagicMock()
        mock_context.loop = asyncio.new_event_loop()
        mock_context.local_cpu_backend = MagicMock()

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjectS3Connector"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter.create_connector(mock_context)

            kw = mock_cls.call_args.kwargs
            assert kw["s3_num_io_threads"] == 64
            assert kw["s3_prefer_http2"] is True
            assert kw["s3_enable_s3express"] is False
            assert kw["disable_tls"] is False
            assert kw["aws_access_key_id"] is None
            assert kw["aws_secret_access_key"] is None
            assert kw["cuobj_nic_device"] is None
            assert kw["s3_region"] == "eu-west-1"

        mock_context.loop.close()

    # Scenario: URL that does NOT start with "cuobj+" should be passed
    # through unchanged (defensive — normally the adapter is only invoked
    # for matching URLs).
    # Verification: s3_endpoint is the original URL without stripping.
    def test_non_cuobj_url_passed_unchanged(self):
        adapter = CuObjectS3ConnectorAdapter()
        mock_config = MagicMock()
        mock_config.extra_config = {"s3_region": "us-east-1"}

        mock_context = MagicMock()
        mock_context.url = "s3://bucket.us-east-1.amazonaws.com"
        mock_context.config = mock_config
        mock_context.metadata = MagicMock()
        mock_context.loop = asyncio.new_event_loop()
        mock_context.local_cpu_backend = MagicMock()

        with patch(
            "lmcache.v1.storage_backend.connector."
            "cuobject_s3_connector.CuObjectS3Connector"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter.create_connector(mock_context)

            kw = mock_cls.call_args.kwargs
            assert kw["s3_endpoint"] == "s3://bucket.us-east-1.amazonaws.com"

        mock_context.loop.close()


class TestAutoDiscovery:
    """Verify that the adapter is discoverable by ConnectorManager."""

    def test_adapter_discovered(self, monkeypatch):
        """ConnectorManager should find CuObjectS3ConnectorAdapter."""
        # Create a fake module containing our adapter
        cuobj_module = ModuleType("cuobject_s3_adapter")
        cuobj_module.CuObjectS3ConnectorAdapter = CuObjectS3ConnectorAdapter

        def fake_iter_modules(_path):
            yield None, "cuobject_s3_adapter", False

        def fake_import_module(name):
            if name.endswith(".cuobject_s3_adapter"):
                return cuobj_module
            raise ImportError(f"unexpected import: {name}")

        monkeypatch.setattr(pkgutil, "iter_modules", fake_iter_modules)
        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        loop = asyncio.new_event_loop()
        try:
            manager = ConnectorManager("cuobj+s3://test-bucket", loop, None)
        finally:
            loop.close()

        assert len(manager.adapters) == 1
        assert isinstance(manager.adapters[0], CuObjectS3ConnectorAdapter)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
