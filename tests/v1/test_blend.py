# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import Mock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig


class TestLMCBlendMetadata:
    def test_metadata_initialization(self):
        """Test LMCBlendMetadata initialization."""
        metadata = LMCBlendMetadata()
        assert metadata.imp_indices is None
        assert metadata.attn_mask is None
        assert metadata.positions is None

    def test_metadata_clean(self):
        """Test LMCBlendMetadata clean method."""
        metadata = LMCBlendMetadata()
        metadata.imp_indices = torch.tensor([1, 2, 3])
        metadata.attn_mask = torch.tensor([True, False, True])
        metadata.positions = torch.tensor([0, 1, 2])

        metadata.clean()

        assert metadata.imp_indices is None
        assert metadata.attn_mask is None
        assert metadata.positions is None


class TestLMCBlendCommonMetadata:
    def test_common_metadata_initialization(self):
        """Test LMCBlendCommonMetadata initialization."""
        metadata = LMCBlendCommonMetadata(
            check_layers=[1, 3, 5],
            recomp_ratios=[0.1, 0.2, 0.3],
            thresholds=[0.5, 0.6, 0.7],
        )
        assert metadata.check_layers == [1, 3, 5]
        assert metadata.recomp_ratios == [0.1, 0.2, 0.3]
        assert metadata.thresholds == [0.5, 0.6, 0.7]

    def test_common_metadata_defaults(self):
        """Test LMCBlendCommonMetadata with default values."""
        metadata = LMCBlendCommonMetadata(check_layers=[1])
        assert metadata.check_layers == [1]
        assert metadata.recomp_ratios is None
        assert metadata.thresholds is None


class TestLMCBlender:
    @pytest.fixture
    def mock_vllm_model(self):
        """Create a mock vLLM model."""
        model = Mock()
        model.model.layers = [Mock() for _ in range(4)]

        def mock_rotary_emb(positions, q, k):
            return q, k

        for layer in model.model.layers:
            layer.self_attn = Mock()
            layer.self_attn.rotary_emb = Mock(side_effect=mock_rotary_emb)

        return model

    @pytest.fixture
    def mock_cache_engine(self):
        """Create a mock cache engine."""
        return Mock()

    @pytest.fixture
    def mock_gpu_connector(self):
        """Create a mock GPU connector."""
        connector = Mock()
        connector.get_kv = Mock(
            return_value=(torch.randn(10, 32, 128), torch.randn(10, 32, 128))
        )
        return connector

    @pytest.fixture
    def mock_config(self):
        """Create a mock LMCache config."""
        return LMCacheEngineConfig.from_legacy()

    @pytest.fixture
    @patch("lmcache.v1.compute.blend.blender.infer_model_from_vllm")
    def blender(
        self,
        mock_infer_model,
        mock_cache_engine,
        mock_gpu_connector,
        mock_vllm_model,
        mock_config,
    ):
        """Create a LMCBlender instance with mocked dependencies."""
        mock_layerwise_model = Mock()
        mock_layerwise_model.vllm_model = mock_vllm_model
        mock_infer_model.return_value = mock_layerwise_model

        return LMCBlender(
            cache_engine=mock_cache_engine,
            gpu_connector=mock_gpu_connector,
            vllm_model=mock_vllm_model,
            config=mock_config,
        )

    def test_blender_initialization(self, blender):
        """Test LMCBlender initialization."""
        assert blender.cache_engine is not None
        assert blender.gpu_connector is not None
        assert blender.layerwise_model is not None
        assert blender.num_layers == 4
        assert isinstance(blender.common_metadata, LMCBlendCommonMetadata)
        assert isinstance(blender.metadata, LMCBlendMetadata)

    def test_process_qkv_without_check_layer(self, blender):
        """Test process_qkv method when layer is not in check_layers."""
        device = "cpu"  # Use CPU for testing
        q = torch.randn(10, 32, 128, device=device)
        k = torch.randn(10, 32, 128, device=device)
        v = torch.randn(10, 32, 128, device=device)
        residual = torch.randn(10, 4096, device=device)
        attn_output = torch.randn(10, 32, 128, device=device)

        # Mock attention metadata
        attn_metadata = Mock(spec=LMCAttnMetadata)

        # Mock GPU connector to return KV caches
        blender.gpu_connector.get_kv.return_value = (
            torch.randn(10, 32, 128, device=device),
            torch.randn(10, 32, 128, device=device),
        )

        # Set layer_id to 0 (not in check_layers=[1])
        layer_id = 0

        result = blender.process_qkv(
            q, k, v, residual, layer_id, attn_output, attn_metadata
        )

        assert len(result) == 6
        q_res, k_res, v_res, residual_res, attn_output_res, _ = result
        assert q_res.shape == q.shape
        assert k_res.shape == k.shape
        assert v_res.shape == v.shape
        assert residual_res.shape == residual.shape
        assert attn_output_res.shape == attn_output.shape

    def test_process_qkv_with_check_layer(self, blender):
        """Test process_qkv method when layer is in check_layers."""
        device = "cpu"
        # Use larger num_tokens to accommodate the way topk works with dim=[1]
        num_tokens = 200  # Increased to handle potential large indices from topk
        q = torch.randn(num_tokens, 32, 128, device=device)
        k = torch.randn(num_tokens, 32, 128, device=device)
        v = torch.randn(num_tokens, 32, 128, device=device)
        residual = torch.randn(num_tokens, 4096, device=device)
        attn_output = torch.randn(num_tokens, 32, 128, device=device)
        attn_metadata = Mock(spec=LMCAttnMetadata)
        attn_metadata.update_from_top_indices = Mock()

        # Mock GPU connector to return similar KV caches
        old_k = k + torch.randn_like(k) * 0.001  # Very small perturbation
        old_v = v + torch.randn_like(v) * 0.001
        blender.gpu_connector.get_kv.return_value = (old_k, old_v)

        # The blender fixture has check_layers=[1] by default
        layer_id = 1

        result = blender.process_qkv(
            q, k, v, residual, layer_id, attn_output, attn_metadata
        )

        assert len(result) == 6
        q_res, k_res, v_res, residual_res, attn_output_res, _ = result

        # Check that we get valid tensors back and their shapes
        recomp_ratio = blender.common_metadata.recomp_ratios[0]
        topk_num = int(num_tokens * recomp_ratio)
        assert q_res.shape == (num_tokens, topk_num, 32, 128)
        assert k_res.shape == k.shape
        assert v_res.shape == v.shape
        assert residual_res.shape == (num_tokens, topk_num, 4096)
        assert attn_output_res.shape == (topk_num, 32, 128)

        # Check metadata updates
        assert blender.metadata.imp_indices is not None
        assert blender.metadata.imp_indices.shape == (num_tokens, topk_num)
        assert blender.metadata.positions is not None
        assert blender.metadata.positions.shape == (num_tokens, topk_num)

    def test_blend_layer_generator(self, blender):
        """Test blend_layer method returns generator."""
        tokens = torch.randn(10, 128)

        # Mock the generators
        blender.layerwise_model.compute_layer = Mock(
            return_value=iter(range(6))
        )  # num_layers + 2
        blender.cache_engine.retrieve_layer = Mock(return_value=iter(range(6)))

        blender_gen = blender.blend_layer(tokens)

        # Should be a generator
        assert hasattr(blender_gen, "__next__")

        # Should be able to iterate through all layers without stopping prematurely
        for i in range(6):  # num_layers (4) + 2
            try:
                next(blender_gen)
            except StopIteration:
                pytest.fail(f"Generator stopped prematurely at iteration {i + 1}")

        # Should raise StopIteration after all layers are consumed
        with pytest.raises(StopIteration):
            next(blender_gen)

    def test_blend_method(self, blender):
        """Test blend method."""
        tokens = torch.randn(10, 128)

        # Mock blend_layer to return a simple generator
        def mock_blend_layer(*args, **kwargs):
            for i in range(6):  # num_layers + 2
                yield i

        blender.blend_layer = Mock(side_effect=mock_blend_layer)

        # Should complete without error
        blender.blend(tokens)

        # Verify blend_layer was called with correct arguments
        blender.blend_layer.assert_called_once_with(tokens, None)


class TestLMCBlenderBuilder:
    @patch("lmcache.v1.compute.blend.blender.infer_model_from_vllm")
    def test_builder_singleton_pattern(self, mock_infer_model):
        """Test that LMCBlenderBuilder maintains singleton instances."""
        LMCBlenderBuilder._blenders.clear()

        instance_id = "test_instance"

        mock_layerwise_model = Mock()
        mock_layerwise_model.vllm_model.model.layers = [Mock(), Mock()]
        mock_infer_model.return_value = mock_layerwise_model

        with patch(
            "lmcache.v1.compute.blend.utils.VLLMModelTracker.get_model"
        ) as mock_get_model:
            mock_model = Mock()
            mock_model.model.layers = [Mock(), Mock()]
            mock_get_model.return_value = mock_model

            mock_cache_engine = Mock()
            mock_gpu_connector = Mock()
            mock_config = LMCacheEngineConfig.from_legacy()

            blender1 = LMCBlenderBuilder.get_or_create(
                instance_id, mock_cache_engine, mock_gpu_connector, mock_config
            )

            blender2 = LMCBlenderBuilder.get_or_create(
                instance_id, mock_cache_engine, mock_gpu_connector, mock_config
            )

            assert blender1 is blender2
            assert len(LMCBlenderBuilder._blenders) == 1

    def test_builder_get_nonexistent(self):
        """Test getting non-existent blender raises ValueError."""
        LMCBlenderBuilder._blenders.clear()

        with pytest.raises(ValueError, match="Blender for nonexistent not found"):
            LMCBlenderBuilder.get("nonexistent")

    def test_builder_get_existing(self):
        """Test getting existing blender."""
        LMCBlenderBuilder._blenders.clear()

        instance_id = "test_instance"
        mock_blender = Mock()
        LMCBlenderBuilder._blenders[instance_id] = mock_blender

        result = LMCBlenderBuilder.get(instance_id)
        assert result is mock_blender
