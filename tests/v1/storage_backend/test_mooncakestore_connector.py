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

    @pytest.mark.parametrize(
        "use_dummy, dummy_addr",
        [(False, ""), (True, "127.0.0.1:50052")],
        ids=["real_client", "dummy_client"],
    )
    def test_register_buffer_called_in_both_modes(
        self,
        mock_mooncake_module,
        mock_local_cpu_backend,
        mock_async_loop,
        use_dummy,
        dummy_addr,
    ):
        """register_buffer should be called in both real and dummy modes."""
        store_instance, _, _ = mock_mooncake_module
        config = _make_mooncake_store_config(
            use_dummy_client=use_dummy,
            dummy_server_address=dummy_addr,
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


class TestDummyClientShmBufferReplacement:
    """Test that dummy client mode replaces pin_allocator with shm buffer."""

    def _create_dummy_connector(
        self,
        mock_mooncake_module,
        mock_local_cpu_backend,
        mock_async_loop,
        shm_ptr=0x7F5500000000,
    ):
        """Helper: create connector in dummy mode with all mocks."""
        store_instance, _, _ = mock_mooncake_module
        store_instance.alloc_from_mem_pool.return_value = shm_ptr
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

            connector = MooncakestoreConnector(
                host="127.0.0.1",
                port=50051,
                dev_name="",
                loop=mock_async_loop,
                local_cpu_backend=mock_local_cpu_backend,
                lmcache_config=mock_local_cpu_backend.config,
            )
            return connector, store_instance

    def test_dummy_calls_alloc_from_mem_pool(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """Dummy mode should allocate shm from mooncake memory pool."""
        _, store = self._create_dummy_connector(
            mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
        )
        store.alloc_from_mem_pool.assert_called_once_with(1073741824)

    def test_dummy_calls_replace_buffer(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """Dummy mode should call allocator.replace_buffer(ptr, size)."""
        self._create_dummy_connector(
            mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
        )
        allocator = mock_local_cpu_backend.memory_allocator
        allocator.replace_buffer.assert_called_once_with(0x7F5500000000, 1073741824)

    def test_dummy_registers_buffer_after_replace(
        self,
        mock_mooncake_module,
        mock_local_cpu_backend,
        mock_async_loop,
    ):
        """register_buffer should be called after replace_buffer."""
        _, store = self._create_dummy_connector(
            mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
        )
        store.register_buffer.assert_called_once()

    def test_real_mode_no_alloc_from_mem_pool(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """Real mode should NOT call alloc_from_mem_pool."""
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
            store_instance.alloc_from_mem_pool.assert_not_called()

    def test_real_mode_no_replace_buffer(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """Real mode should NOT call replace_buffer."""
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
            mock_local_cpu_backend.memory_allocator.replace_buffer.assert_not_called()

    def test_dummy_register_buffer_uses_new_shm_ptr(
        self, mock_mooncake_module, mock_local_cpu_backend, mock_async_loop
    ):
        """register_buffer should see the new buffer ptr/size after replace_buffer."""
        store_instance, _, _ = mock_mooncake_module
        shm_ptr = 0x7F5500000000
        shm_size = 1073741824
        store_instance.alloc_from_mem_pool.return_value = shm_ptr

        # After replace_buffer, the mock allocator's pin_allocator.buffer
        # should reflect the new ptr. We track call order to verify
        # replace_buffer is called BEFORE register_buffer.
        call_order = []

        def track_replace_buffer(ptr, size):
            call_order.append(("replace_buffer", ptr, size))
            # Simulate what replace_buffer does: update the mock buffer
            new_mock_buffer = MagicMock()
            new_mock_buffer.data_ptr.return_value = ptr
            new_mock_buffer.numel.return_value = size
            mock_local_cpu_backend.memory_allocator.pin_allocator.buffer = (
                new_mock_buffer
            )

        def track_register_buffer(ptr, size):
            call_order.append(("register_buffer", ptr, size))
            return 0

        mock_local_cpu_backend.memory_allocator.replace_buffer.side_effect = (
            track_replace_buffer
        )
        store_instance.register_buffer.side_effect = track_register_buffer

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

        # Verify replace_buffer was called before register_buffer
        assert len(call_order) == 2
        assert call_order[0][0] == "replace_buffer"
        assert call_order[1][0] == "register_buffer"
        # register_buffer should see the new shm ptr and size
        assert call_order[1][1] == shm_ptr
        assert call_order[1][2] == shm_size
