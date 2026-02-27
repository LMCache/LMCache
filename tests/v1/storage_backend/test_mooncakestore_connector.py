# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch
import asyncio
import json
import os
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
)


class TestMooncakeStoreConfigDummyClient:
    """Test MooncakeStoreConfig parsing of dummy client fields."""

    def _make_config_dict(self, **overrides):
        """Create a base config dict with required fields."""
        base = {
            "local_hostname": "127.0.0.1",
            "metadata_server": "127.0.0.1:2379",
            "master_server_address": "127.0.0.1:50051",
        }
        base.update(overrides)
        return base

    def _write_config_file(self, config_dict, tmpdir):
        """Write config dict to a temp JSON file and return the path."""
        path = os.path.join(tmpdir, "mooncake_config.json")
        with open(path, "w") as f:
            json.dump(config_dict, f)
        return path

    # --- Tests for default values ---

    def test_from_file_defaults_no_dummy_fields(self, tmp_path):
        """Old config without dummy fields should default to False/empty."""
        config_dict = self._make_config_dict()
        path = self._write_config_file(config_dict, str(tmp_path))

        config = MooncakeStoreConfig.from_file(path)

        assert config.use_dummy_client is False
        assert config.dummy_server_address == ""

    def test_from_file_with_dummy_enabled(self, tmp_path):
        """Config with use_dummy_client=True and dummy_server_address set."""
        config_dict = self._make_config_dict(
            use_dummy_client=True,
            dummy_server_address="127.0.0.1:50052",
        )
        path = self._write_config_file(config_dict, str(tmp_path))

        config = MooncakeStoreConfig.from_file(path)

        assert config.use_dummy_client is True
        assert config.dummy_server_address == "127.0.0.1:50052"

    def test_from_file_with_dummy_disabled_explicitly(self, tmp_path):
        """Config with use_dummy_client=False explicitly set."""
        config_dict = self._make_config_dict(use_dummy_client=False)
        path = self._write_config_file(config_dict, str(tmp_path))

        config = MooncakeStoreConfig.from_file(path)

        assert config.use_dummy_client is False
        assert config.dummy_server_address == ""

    # --- Tests for lmcache extra_config path ---

    def test_load_from_lmcache_config_defaults(self):
        """lmcache extra_config without dummy fields should default."""
        mock_config = MagicMock()
        mock_config.extra_config = self._make_config_dict()

        config = MooncakeStoreConfig.load_from_lmcache_config(mock_config)

        assert config.use_dummy_client is False
        assert config.dummy_server_address == ""

    def test_load_from_lmcache_config_with_dummy(self):
        """lmcache extra_config with dummy fields set."""
        mock_config = MagicMock()
        mock_config.extra_config = self._make_config_dict(
            use_dummy_client=True,
            dummy_server_address="10.0.0.1:50052",
        )

        config = MooncakeStoreConfig.load_from_lmcache_config(mock_config)

        assert config.use_dummy_client is True
        assert config.dummy_server_address == "10.0.0.1:50052"

    # --- Tests for env path ---

    def test_load_from_env_defaults(self, tmp_path):
        """load_from_env with old config should default dummy fields."""
        config_dict = self._make_config_dict()
        path = self._write_config_file(config_dict, str(tmp_path))

        with patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": path}):
            config = MooncakeStoreConfig.load_from_env()

        assert config.use_dummy_client is False
        assert config.dummy_server_address == ""

    def test_load_from_env_with_dummy(self, tmp_path):
        """load_from_env with dummy fields in config file."""
        config_dict = self._make_config_dict(
            use_dummy_client=True,
            dummy_server_address="192.168.1.1:50052",
        )
        path = self._write_config_file(config_dict, str(tmp_path))

        with patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": path}):
            config = MooncakeStoreConfig.load_from_env()

        assert config.use_dummy_client is True
        assert config.dummy_server_address == "192.168.1.1:50052"


# ---- Helpers for connector init tests ----


def _make_mooncake_store_config(**overrides):
    """Create a MooncakeStoreConfig with sensible defaults."""
    defaults = dict(
        local_hostname="127.0.0.1",
        metadata_server="127.0.0.1:2379",
        global_segment_size=3355443200,
        local_buffer_size=1073741824,
        protocol="tcp",
        device_name="",
        master_server_address="127.0.0.1:50051",
        transfer_timeout=1,
        storage_root_dir="",
        prefer_local_alloc=False,
        use_dummy_client=False,
        dummy_server_address="",
    )
    defaults.update(overrides)
    return MooncakeStoreConfig(**defaults)


@pytest.fixture
def mock_mooncake_module():
    """Mock the mooncake.store module to avoid real C++ imports."""
    mock_store_instance = MagicMock()
    mock_store_instance.setup.return_value = None
    mock_store_instance.setup_dummy.return_value = 0
    mock_store_instance.register_buffer.return_value = 0
    mock_store_instance.unregister_buffer.return_value = 0
    mock_store_instance.get_hostname.return_value = "test-host"

    mock_store_class = MagicMock(return_value=mock_store_instance)
    mock_replicate_config = MagicMock()

    mock_module = MagicMock()
    mock_module.MooncakeDistributedStore = mock_store_class
    mock_module.ReplicateConfig = mock_replicate_config

    with patch.dict(
        sys.modules,
        {"mooncake": MagicMock(), "mooncake.store": mock_module},
    ):
        yield mock_store_instance, mock_store_class, mock_replicate_config


@pytest.fixture
def mock_local_cpu_backend():
    """Mock LocalCPUBackend with memory allocator."""
    backend = MagicMock()
    backend.config = MagicMock()
    backend.config.extra_config = None
    backend.config.use_layerwise = False
    backend.metadata = MagicMock()
    backend.metadata.get_shapes.return_value = []
    backend.metadata.get_dtypes.return_value = []
    backend.metadata.use_mla = False
    backend.metadata.chunk_size = 256
    backend.metadata.get_num_groups.return_value = 1
    # Memory allocator with pin_allocator.buffer
    mock_buffer = MagicMock()
    mock_buffer.data_ptr.return_value = 0x7F000000
    mock_buffer.numel.return_value = 1073741824
    backend.memory_allocator.pin_allocator.buffer = mock_buffer
    backend.memory_allocator.numa_mapping = None
    return backend


@pytest.fixture
def mock_async_loop():
    """Create a simple event loop for testing."""
    return asyncio.new_event_loop()


def _create_connector(
    mock_mooncake_module,
    mock_local_cpu_backend,
    mock_async_loop,
    mooncake_config,
    lmcache_config=None,
):
    """Helper to create MooncakestoreConnector with all mocks in place."""
    # We need to patch multiple things:
    # 1. The mooncake import inside __init__
    # 2. The RemoteConnector.__init__ (base class has complex deps)
    # 3. The config loading

    store_instance, store_class, replicate_config_cls = mock_mooncake_module

    # Patch MOONCAKE_CONFIG_PATH to None to force lmcache_config path
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)

        # Patch the base class __init__ to avoid its complex dependencies
        with patch(
            "lmcache.v1.storage_backend.connector."
            "mooncakestore_connector."
            "MooncakestoreConnector.__init__"
        ):
            # We can't easily call the real __init__ with mocks,
            # so let's test the setup path logic directly instead.
            pass

    # Instead of trying to instantiate the full class, test the setup logic
    # by importing and calling with appropriate mocks on the config loading
    return store_instance


class TestMooncakestoreConnectorInit:
    """Test MooncakestoreConnector initialization paths (real vs dummy)."""

    def test_real_client_calls_setup(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """When use_dummy_client=False, store.setup() should be called."""
        store_instance, _, _ = mock_mooncake_module
        config = _make_mooncake_store_config(use_dummy_client=False)

        # Patch config loading to return our config
        with (
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.MooncakeStoreConfig.load_from_lmcache_config",
                return_value=config,
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.RemoteConnector.__init__"
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.NUMADetector"
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
                MooncakestoreConnector,
            )

            MooncakestoreConnector(
                host="127.0.0.1",
                port=50051,
                dev_name="",
                loop=mock_async_loop,
                local_cpu_backend=mock_local_cpu_backend,
                lmcache_config=mock_local_cpu_backend.config,
            )

            store_instance.setup.assert_called_once()
            store_instance.setup_dummy.assert_not_called()

    def test_dummy_client_calls_setup_dummy(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """When use_dummy_client=True, store.setup_dummy() should be called."""
        store_instance, _, _ = mock_mooncake_module
        config = _make_mooncake_store_config(
            use_dummy_client=True,
            dummy_server_address="127.0.0.1:50052",
        )

        with (
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.MooncakeStoreConfig.load_from_lmcache_config",
                return_value=config,
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.RemoteConnector.__init__"
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.NUMADetector"
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
                MooncakestoreConnector,
            )

            MooncakestoreConnector(
                host="127.0.0.1",
                port=50051,
                dev_name="",
                loop=mock_async_loop,
                local_cpu_backend=mock_local_cpu_backend,
                lmcache_config=mock_local_cpu_backend.config,
            )

            store_instance.setup_dummy.assert_called_once_with(
                config.global_segment_size,
                config.local_buffer_size,
                config.dummy_server_address,
            )
            store_instance.setup.assert_not_called()

    def test_dummy_client_not_supported_raises_error(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """When use_dummy_client=True but mooncake has no setup_dummy, raise error."""
        store_instance, _, _ = mock_mooncake_module
        # Remove setup_dummy to simulate old mooncake version
        del store_instance.setup_dummy
        config = _make_mooncake_store_config(
            use_dummy_client=True,
            dummy_server_address="127.0.0.1:50052",
        )

        with (
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.MooncakeStoreConfig.load_from_lmcache_config",
                return_value=config,
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.RemoteConnector.__init__"
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.NUMADetector"
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
                MooncakestoreConnector,
            )

            with pytest.raises(RuntimeError, match="does not support.*dummy"):
                MooncakestoreConnector(
                    host="127.0.0.1",
                    port=50051,
                    dev_name="",
                    loop=mock_async_loop,
                    local_cpu_backend=mock_local_cpu_backend,
                    lmcache_config=mock_local_cpu_backend.config,
                )

    def test_dummy_client_missing_address_raises_error(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """When use_dummy_client=True but dummy_server_address is empty, raise error."""
        store_instance, _, _ = mock_mooncake_module
        config = _make_mooncake_store_config(
            use_dummy_client=True,
            dummy_server_address="",
        )

        with (
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.MooncakeStoreConfig.load_from_lmcache_config",
                return_value=config,
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.RemoteConnector.__init__"
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.NUMADetector"
            ),
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
                MooncakestoreConnector,
            )

            with pytest.raises(
                ValueError, match="dummy_server_address must be provided"
            ):
                MooncakestoreConnector(
                    host="127.0.0.1",
                    port=50051,
                    dev_name="",
                    loop=mock_async_loop,
                    local_cpu_backend=mock_local_cpu_backend,
                    lmcache_config=mock_local_cpu_backend.config,
                )

    def test_dummy_client_skips_numa_binding(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """When use_dummy_client=True, NUMA binding should be skipped."""
        store_instance, _, _ = mock_mooncake_module
        config = _make_mooncake_store_config(
            use_dummy_client=True,
            dummy_server_address="127.0.0.1:50052",
        )

        with (
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.MooncakeStoreConfig.load_from_lmcache_config",
                return_value=config,
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.RemoteConnector.__init__"
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.NUMADetector"
            ) as mock_numa,
        ):
            # First Party
            from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
                MooncakestoreConnector,
            )

            MooncakestoreConnector(
                host="127.0.0.1",
                port=50051,
                dev_name="",
                loop=mock_async_loop,
                local_cpu_backend=mock_local_cpu_backend,
                lmcache_config=mock_local_cpu_backend.config,
            )

            # NUMADetector.get_numa_mapping should NOT be called in dummy mode
            mock_numa.get_numa_mapping.assert_not_called()

    def test_real_client_does_numa_binding(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """When use_dummy_client=False, NUMA binding should proceed normally."""
        store_instance, _, _ = mock_mooncake_module
        config = _make_mooncake_store_config(use_dummy_client=False)

        with (
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.MooncakeStoreConfig.load_from_lmcache_config",
                return_value=config,
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.RemoteConnector.__init__"
            ),
            patch(
                "lmcache.v1.storage_backend.connector.mooncakestore_connector.NUMADetector"
            ) as mock_numa,
        ):
            mock_numa.get_numa_mapping.return_value = None

            # First Party
            from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
                MooncakestoreConnector,
            )

            MooncakestoreConnector(
                host="127.0.0.1",
                port=50051,
                dev_name="",
                loop=mock_async_loop,
                local_cpu_backend=mock_local_cpu_backend,
                lmcache_config=mock_local_cpu_backend.config,
            )

            # NUMADetector.get_numa_mapping should be called in real mode
            mock_numa.get_numa_mapping.assert_called_once()

    def test_register_buffer_called_in_both_modes(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """register_buffer should be called in both real and dummy modes."""
        store_instance, _, _ = mock_mooncake_module

        for use_dummy in [False, True]:
            store_instance.reset_mock()
            config = _make_mooncake_store_config(
                use_dummy_client=use_dummy,
                dummy_server_address="127.0.0.1:50052" if use_dummy else "",
            )

            # First Party
            from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
                MooncakestoreConnector,
            )

            with (
                patch(
                    "lmcache.v1.storage_backend.connector.mooncakestore_connector.MooncakeStoreConfig.load_from_lmcache_config",
                    return_value=config,
                ),
                patch(
                    "lmcache.v1.storage_backend.connector.mooncakestore_connector.RemoteConnector.__init__"
                ),
                patch(
                    "lmcache.v1.storage_backend.connector.mooncakestore_connector.NUMADetector"
                ),
            ):
                MooncakestoreConnector(
                    host="127.0.0.1",
                    port=50051,
                    dev_name="",
                    loop=mock_async_loop,
                    local_cpu_backend=mock_local_cpu_backend,
                    lmcache_config=mock_local_cpu_backend.config,
                )

                store_instance.register_buffer.assert_called_once_with(
                    0x7F000000, 1073741824
                )
