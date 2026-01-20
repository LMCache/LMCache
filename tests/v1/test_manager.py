# SPDX-License-Identifier: Apache-2.0
"""
Tests for LMCacheManager.
"""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.manager import LMCacheManager
from tests.v1.utils import create_test_metadata


class TestLMCacheManagerInit:
    """Tests for LMCacheManager initialization."""

    def test_init_stores_config(self):
        """Test that __init__ stores config correctly."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        assert manager._config is config
        assert manager._metadata is metadata
        assert manager._northbound is northbound

    def test_init_calls_init_components(self):
        """Test that __init__ calls _init_components."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components") as mock_init:
            LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

            mock_init.assert_called_once()


class TestLMCacheManagerProperties:
    """Tests for LMCacheManager property accessors."""

    @pytest.fixture
    def manager_with_mocked_init(self):
        """Create a manager with mocked initialization."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )
        return manager

    def test_lmcache_engine_property(self, manager_with_mocked_init):
        """Test lmcache_engine property returns the engine."""
        manager = manager_with_mocked_init
        mock_engine = MagicMock()
        manager._lmcache_engine = mock_engine

        assert manager.lmcache_engine is mock_engine

    def test_lookup_client_property(self, manager_with_mocked_init):
        """Test lookup_client property returns the client."""
        manager = manager_with_mocked_init
        mock_client = MagicMock()
        manager._lookup_client = mock_client

        assert manager.lookup_client is mock_client

    def test_lookup_server_property(self, manager_with_mocked_init):
        """Test lookup_server property returns the server."""
        manager = manager_with_mocked_init
        mock_server = MagicMock()
        manager._lookup_server = mock_server

        assert manager.lookup_server is mock_server

    def test_offload_server_property(self, manager_with_mocked_init):
        """Test offload_server property returns the server."""
        manager = manager_with_mocked_init
        mock_server = MagicMock()
        manager._offload_server = mock_server

        assert manager.offload_server is mock_server

    def test_config_property(self, manager_with_mocked_init):
        """Test config property returns the config."""
        manager = manager_with_mocked_init
        assert manager.config is manager._config


class TestLMCacheManagerStart:
    """Tests for LMCacheManager start method."""

    def test_start_calls_api_server_start(self):
        """Test start() calls api_server.start() when api_server exists."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        mock_api_server = MagicMock()
        mock_plugin_launcher = MagicMock()
        manager._api_server = mock_api_server
        manager._runtime_plugin_launcher = mock_plugin_launcher

        manager.start_services()

        mock_api_server.start.assert_called_once()
        mock_plugin_launcher.launch_plugins.assert_called_once()

    def test_start_handles_none_api_server(self):
        """Test start() handles None api_server gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        manager._api_server = None
        manager._runtime_plugin_launcher = None

        # Should not raise any exception
        manager.start_services()


class TestLMCacheManagerPostInit:
    """Tests for LMCacheManager post_init method."""

    def test_post_init_without_engine(self):
        """Test post_init returns early when engine is None."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        manager._lmcache_engine = None

        # Should not raise any exception
        manager.post_init()

    def test_post_init_with_engine_and_async_loading(self):
        """Test post_init calls engine.post_init with async_lookup_server."""
        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        mock_engine = MagicMock()
        manager._lmcache_engine = mock_engine
        manager._lookup_server = None

        manager.post_init()

        # When lookup_server is None, engine.post_init should be called
        # with async_lookup_server=None
        mock_engine.post_init.assert_called_once_with(async_lookup_server=None)

    def test_post_init_with_engine_and_async_server(self):
        """Test post_init calls engine.post_init when async lookup server exists."""
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        mock_engine = MagicMock()
        mock_lookup_server = MagicMock(spec=LMCacheAsyncLookupServer)
        manager._lmcache_engine = mock_engine
        manager._lookup_server = mock_lookup_server

        manager.post_init()

        # When lookup_server is LMCacheAsyncLookupServer, it should be passed
        mock_engine.post_init.assert_called_once_with(
            async_lookup_server=mock_lookup_server
        )


class TestLMCacheManagerShutdown:
    """Tests for LMCacheManager shutdown method."""

    def test_shutdown_closes_all_components(self):
        """Test shutdown() closes all components."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        # Setup mock components
        mock_offload_server = MagicMock()
        mock_plugin_launcher = MagicMock()
        mock_api_server = MagicMock()
        mock_lookup_server = MagicMock()
        mock_lookup_client = MagicMock()

        manager._offload_server = mock_offload_server
        manager._runtime_plugin_launcher = mock_plugin_launcher
        manager._api_server = mock_api_server
        manager._lookup_server = mock_lookup_server
        manager._lookup_client = mock_lookup_client

        with patch("lmcache.v1.manager.LMCacheEngineBuilder") as mock_builder:
            manager.stop_services()

            # Verify all components were closed
            mock_offload_server.close.assert_called_once()
            mock_plugin_launcher.stop_plugins.assert_called_once()
            mock_api_server.stop.assert_called_once()
            mock_lookup_server.close.assert_called_once()
            mock_lookup_client.close.assert_called_once()
            mock_builder.destroy.assert_called_once()

    def test_shutdown_handles_none_components(self):
        """Test shutdown() handles None components gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        # All components are None
        manager._offload_server = None
        manager._runtime_plugin_launcher = None
        manager._api_server = None
        manager._lookup_server = None
        manager._lookup_client = None

        with patch("lmcache.v1.manager.LMCacheEngineBuilder"):
            # Should not raise any exception
            manager.stop_services()

    def test_shutdown_handles_component_errors(self):
        """Test shutdown() handles errors from components gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        # Setup mock component that raises exception
        mock_offload_server = MagicMock()
        mock_offload_server.close.side_effect = RuntimeError("Test error")
        manager._offload_server = mock_offload_server

        # Setup normal components
        mock_lookup_client = MagicMock()
        manager._lookup_client = mock_lookup_client

        with patch("lmcache.v1.manager.LMCacheEngineBuilder"):
            # Should not raise exception, but should continue shutdown
            manager.stop_services()

            # lookup_client should still be closed even if offload_server failed
            mock_lookup_client.close.assert_called_once()


class TestLMCacheManagerHelpers:
    """Tests for LMCacheManager helper methods."""

    @pytest.mark.skip(
        reason="_need_gpu_interm_buffer moved from LMCacheManager to LMCacheEngine"
    )
    def test_need_gpu_interm_buffer_returns_not_enable_pd(self):
        """Test _need_gpu_interm_buffer returns opposite of enable_pd."""
        config = LMCacheEngineConfig.from_defaults()
        config.enable_pd = False
        metadata = create_test_metadata()
        northbound = MagicMock()

        with patch.object(LMCacheManager, "_init_components"):
            manager = LMCacheManager(
                config=config,
                metadata=metadata,
                northbound=northbound,
            )

        assert manager._need_gpu_interm_buffer() is True

        config.enable_pd = True
        assert manager._need_gpu_interm_buffer() is False
