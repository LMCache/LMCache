# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LMCacheConnectorV1Impl class."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
    RequestTracker,
    SaveSpec,
)
from lmcache.v1.config import LMCacheEngineConfig


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig object."""
    config = Mock()

    # Model config
    config.model_config = Mock()
    config.model_config.model = "test-model"
    config.model_config.dtype = torch.bfloat16
    config.model_config.max_model_len = 2048
    config.model_config.vocab_size = 32000
    config.model_config.get_num_layers = Mock(return_value=32)
    config.model_config.get_num_attention_heads = Mock(return_value=32)
    config.model_config.get_num_kv_heads = Mock(return_value=8)
    config.model_config.get_head_size = Mock(return_value=128)
    config.model_config.hf_config = SimpleNamespace()

    # Cache config
    config.cache_config = Mock()
    config.cache_config.block_size = 16
    config.cache_config.cache_dtype = "auto"
    config.cache_config.gpu_memory_utilization = 0.9
    config.cache_config.swap_space = 4
    config.cache_config.enable_prefix_caching = False

    # Parallel config
    config.parallel_config = Mock()
    config.parallel_config.world_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.data_parallel_rank_local = 0

    # KV transfer config
    config.kv_transfer_config = Mock()
    config.kv_transfer_config.kv_role = "kv_both"
    config.kv_transfer_config.kv_connector_extra_config = {}
    config.kv_transfer_config.get_from_extra_config = Mock(
        side_effect=lambda key, default: default
    )

    # Speculative config
    config.speculative_config = None

    return config


@pytest.fixture
def mock_lmcache_config():
    """Create a mock LMCacheEngineConfig object."""
    return LMCacheEngineConfig.from_defaults()


@pytest.fixture
def mock_parent_connector():
    """Create a mock parent KVConnectorBase_V1 object."""
    parent = Mock()
    parent._connector_metadata = None
    parent._get_connector_metadata = Mock(return_value=None)
    return parent


class TestLMCacheConnectorV1ImplInit:
    """Test initialization of LMCacheConnectorV1Impl."""

    @patch("lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.LookupClientFactory")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.InternalAPIServer")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.PluginLauncher")
    def test_init_scheduler_role(
        self,
        mock_plugin_launcher,
        mock_api_server,
        mock_lookup_factory,
        mock_get_config,
        mock_vllm_config,
        mock_lmcache_config,
        mock_parent_connector,
    ):
        """Test initialization with scheduler role."""
        # Third Party
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        mock_get_config.return_value = mock_lmcache_config
        mock_lookup_factory.create_lookup_client.return_value = Mock()

        connector = LMCacheConnectorV1Impl(
            vllm_config=mock_vllm_config,
            role=KVConnectorRole.SCHEDULER,
            parent=mock_parent_connector,
        )

        assert connector.kv_role == "kv_both"
        assert connector.worker_count == 1
        assert connector._block_size == 16
        assert connector.lmcache_engine is None
        assert hasattr(connector, "lookup_client")
        mock_lookup_factory.create_lookup_client.assert_called_once()

    @patch("lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config")
    @patch("lmcache.integration.vllm.vllm_v1_adapter._init_lmcache_engine")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.LookupClientFactory")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.ZMQOffloadServer")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.InternalAPIServer")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.PluginLauncher")
    @patch("lmcache.integration.vllm.vllm_v1_adapter.get_tensor_model_parallel_rank")
    def test_init_worker_role(
        self,
        mock_get_rank,
        mock_plugin_launcher,
        mock_api_server,
        mock_offload_server,
        mock_lookup_factory,
        mock_init_engine,
        mock_get_config,
        mock_vllm_config,
        mock_lmcache_config,
        mock_parent_connector,
    ):
        """Test initialization with worker role."""
        # Third Party
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        mock_get_config.return_value = mock_lmcache_config
        mock_engine = Mock()
        mock_engine.gpu_connector = None
        mock_init_engine.return_value = mock_engine
        mock_lookup_factory.create_lookup_server.return_value = None
        mock_get_rank.return_value = 0

        connector = LMCacheConnectorV1Impl(
            vllm_config=mock_vllm_config,
            role=KVConnectorRole.WORKER,
            parent=mock_parent_connector,
        )

        assert connector.lmcache_engine is not None
        assert hasattr(connector, "offload_server")
        mock_init_engine.assert_called_once()


class TestLMCacheConnectorV1ImplMethods:
    """Test methods of LMCacheConnectorV1Impl."""

    @pytest.fixture
    def connector(self, mock_vllm_config, mock_lmcache_config, mock_parent_connector):
        """Create a connector instance for testing."""
        # Third Party
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        with patch(
            "lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config"
        ) as mock_get_config:
            mock_get_config.return_value = mock_lmcache_config
            with patch("lmcache.integration.vllm.vllm_v1_adapter.LookupClientFactory"):
                with patch(
                    "lmcache.integration.vllm.vllm_v1_adapter.InternalAPIServer"
                ):
                    with patch(
                        "lmcache.integration.vllm.vllm_v1_adapter.PluginLauncher"
                    ):
                        connector = LMCacheConnectorV1Impl(
                            vllm_config=mock_vllm_config,
                            role=KVConnectorRole.SCHEDULER,
                            parent=mock_parent_connector,
                        )
                        return connector

    def test_get_inference_info(self, connector):
        """Test get_inference_info method."""
        info = connector.get_inference_info()

        assert "vllm_version" in info
        assert "lmcache_version" in info
        assert "vllm_config" in info
        assert "model_config" in info
        assert "cache_config" in info

        # Check model_config details
        assert info["model_config"]["model"] == "test-model"
        assert info["model_config"]["num_layers"] == 32
        assert info["model_config"]["num_kv_heads"] == 8

        # Check cache_config details
        assert info["cache_config"]["block_size"] == 16

    def test_get_inference_version(self, connector):
        """Test get_inference_version method."""
        version = connector.get_inference_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_finished(self, connector):
        """Test get_finished method."""
        result = connector.get_finished({"req1", "req2"})
        assert result == (None, None)

    def test_get_block_ids_with_load_errors_empty(self, connector):
        """Test get_block_ids_with_load_errors when no errors."""
        result = connector.get_block_ids_with_load_errors()
        assert result == set()

    def test_get_block_ids_with_load_errors_with_data(self, connector):
        """Test get_block_ids_with_load_errors with error data."""
        connector._invalid_block_ids = {1, 2, 3}
        result = connector.get_block_ids_with_load_errors()

        assert result == {1, 2, 3}
        # Should be cleared after retrieval
        assert connector._invalid_block_ids == set()

    def test_record_failed_blocks_empty_mask(self, connector):
        """Test record_failed_blocks with empty mask."""
        expected_mask = torch.tensor([], dtype=torch.bool)
        ret_mask = torch.tensor([], dtype=torch.bool)
        slot_mapping = torch.tensor([], dtype=torch.long)

        result = connector.record_failed_blocks(
            "test_req", expected_mask, ret_mask, slot_mapping
        )

        assert result == set()

    def test_record_failed_blocks_all_success(self, connector):
        """Test record_failed_blocks when all loads succeed."""
        expected_mask = torch.tensor([False, True, True, True], dtype=torch.bool)
        ret_mask = torch.tensor([False, True, True, True], dtype=torch.bool)
        slot_mapping = torch.tensor([0, 16, 32, 48], dtype=torch.long)

        result = connector.record_failed_blocks(
            "test_req", expected_mask, ret_mask, slot_mapping
        )

        assert result == set()

    def test_record_failed_blocks_partial_failure(self, connector):
        """Test record_failed_blocks with partial failure."""
        # Token 0 is in vLLM cache (False), tokens 1-3 should be loaded (True)
        # But tokens 2-3 failed to load
        expected_mask = torch.tensor([False, True, True, True], dtype=torch.bool)
        ret_mask = torch.tensor([False, True, False, False], dtype=torch.bool)
        slot_mapping = torch.tensor([0, 16, 32, 48], dtype=torch.long)

        result = connector.record_failed_blocks(
            "test_req", expected_mask, ret_mask, slot_mapping
        )

        # Tokens at slots 32 and 48 failed
        # Block IDs: 32 // 16 = 2, 48 // 16 = 3
        assert result == {2, 3}

    def test_record_failed_blocks_shape_mismatch(self, connector):
        """Test record_failed_blocks with shape mismatch."""
        expected_mask = torch.tensor([True, True, True], dtype=torch.bool)
        ret_mask = torch.tensor([True, True], dtype=torch.bool)
        slot_mapping = torch.tensor([0, 16, 32], dtype=torch.long)

        result = connector.record_failed_blocks(
            "test_req", expected_mask, ret_mask, slot_mapping
        )

        assert result == set()

    def test_request_finished_without_params(self, connector):
        """Test request_finished without kv_transfer_params."""
        request = Mock()
        request.kv_transfer_params = None

        should_continue, params = connector.request_finished(request, [1, 2, 3])

        assert should_continue is False
        assert params is None

    def test_request_finished_with_ret_first_tok(self, connector):
        """Test request_finished with ret_first_tok param."""
        request = Mock()
        request.kv_transfer_params = {"ret_first_tok": True}
        request._output_token_ids = [42, 43, 44]

        should_continue, params = connector.request_finished(request, [1, 2, 3])

        assert should_continue is False
        assert params == {"first_tok": 42}

    @patch("lmcache.integration.vllm.vllm_v1_adapter.LMCacheEngineBuilder.destroy")
    def test_shutdown(self, mock_destroy, connector):
        """Test shutdown method."""
        # Mock the various components that need to be shut down
        connector.api_server = Mock()
        connector.api_server.stop = Mock()

        connector.plugin_launcher = Mock()
        connector.plugin_launcher.stop_plugins = Mock()

        connector.offload_server = Mock()
        connector.offload_server.close = Mock()

        connector.lookup_server = Mock()
        connector.lookup_server.close = Mock()

        connector.lookup_client = Mock()
        connector.lookup_client.close = Mock()

        connector.shutdown()

        # Verify destroy was called
        mock_destroy.assert_called_once()


class TestRequestTracker:
    """Test RequestTracker dataclass and methods."""

    def test_from_new_request_basic(self):
        """Test RequestTracker.from_new_request basic functionality."""
        # Create mock new_request
        new_request = Mock()
        new_request.req_id = "test_req_123"
        new_request.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        new_request.block_ids = [0, 1]
        new_request.sampling_params = Mock()
        new_request.sampling_params.extra_args = None
        new_request.mm_inputs = None
        new_request.mm_positions = None

        config = LMCacheEngineConfig.from_defaults()

        with patch(
            "lmcache.integration.vllm.vllm_v1_adapter.extract_mm_features"
        ) as mock_extract:
            mock_extract.return_value = (None, None)

            tracker = RequestTracker.from_new_request(
                lmcache_config=config,
                new_request=new_request,
                num_tokens_to_compute=5,
                lmcache_cached_tokens=0,
                skip_save=False,
            )

        assert tracker.req_id == "test_req_123"
        assert tracker.prompt_len == 8
        assert tracker.token_ids == [1, 2, 3, 4, 5]
        assert tracker.allocated_block_ids == [0, 1]
        assert tracker.num_saved_tokens == 0
        assert tracker.is_decode_phase is False

    def test_from_new_request_with_nested_block_ids(self):
        """Test RequestTracker.from_new_request with nested block_ids."""
        new_request = Mock()
        new_request.req_id = "test_req_456"
        new_request.prompt_token_ids = [1, 2, 3, 4]
        new_request.block_ids = [[0, 1, 2]]  # Nested list
        new_request.sampling_params = Mock()
        new_request.sampling_params.extra_args = None
        new_request.mm_inputs = None
        new_request.mm_positions = None

        config = LMCacheEngineConfig.from_defaults()

        with patch(
            "lmcache.integration.vllm.vllm_v1_adapter.extract_mm_features"
        ) as mock_extract:
            mock_extract.return_value = (None, None)

            tracker = RequestTracker.from_new_request(
                lmcache_config=config,
                new_request=new_request,
                num_tokens_to_compute=4,
                lmcache_cached_tokens=0,
                skip_save=False,
            )

        assert tracker.allocated_block_ids == [0, 1, 2]

    def test_update_with_new_tokens(self):
        """Test RequestTracker.update method."""
        new_request = Mock()
        new_request.req_id = "test_req"
        new_request.prompt_token_ids = [1, 2, 3, 4]
        new_request.block_ids = [0, 1]
        new_request.sampling_params = Mock()
        new_request.sampling_params.extra_args = None
        new_request.mm_inputs = None
        new_request.mm_positions = None

        config = LMCacheEngineConfig.from_defaults()

        with patch(
            "lmcache.integration.vllm.vllm_v1_adapter.extract_mm_features"
        ) as mock_extract:
            mock_extract.return_value = (None, None)

            tracker = RequestTracker.from_new_request(
                lmcache_config=config,
                new_request=new_request,
                num_tokens_to_compute=4,
                lmcache_cached_tokens=0,
                skip_save=False,
            )

        # Update with new tokens
        tracker.update([5, 6], [2, 3])

        assert tracker.token_ids == [1, 2, 3, 4, 5, 6]
        assert tracker.allocated_block_ids == [0, 1, 2, 3]
        assert tracker.is_decode_phase is False

    def test_update_single_token_decode_phase(self):
        """Test RequestTracker.update enters decode phase with single token."""
        new_request = Mock()
        new_request.req_id = "test_req"
        new_request.prompt_token_ids = [1, 2, 3, 4]
        new_request.block_ids = [0, 1]
        new_request.sampling_params = Mock()
        new_request.sampling_params.extra_args = None
        new_request.mm_inputs = None
        new_request.mm_positions = None

        config = LMCacheEngineConfig.from_defaults()

        with patch(
            "lmcache.integration.vllm.vllm_v1_adapter.extract_mm_features"
        ) as mock_extract:
            mock_extract.return_value = (None, None)

            tracker = RequestTracker.from_new_request(
                lmcache_config=config,
                new_request=new_request,
                num_tokens_to_compute=4,
                lmcache_cached_tokens=0,
                skip_save=False,
            )

        # Update with single token (decode phase)
        tracker.update([5], [2])

        assert tracker.is_decode_phase is True


class TestLoadSpecAndSaveSpec:
    """Test LoadSpec and SaveSpec dataclasses."""

    def test_load_spec_creation(self):
        """Test LoadSpec creation."""
        spec = LoadSpec(
            vllm_cached_tokens=100,
            lmcache_cached_tokens=200,
            can_load=True,
        )

        assert spec.vllm_cached_tokens == 100
        assert spec.lmcache_cached_tokens == 200
        assert spec.can_load is True

    def test_save_spec_creation(self):
        """Test SaveSpec creation."""
        spec = SaveSpec(
            skip_leading_tokens=50,
            can_save=True,
        )

        assert spec.skip_leading_tokens == 50
        assert spec.can_save is True
