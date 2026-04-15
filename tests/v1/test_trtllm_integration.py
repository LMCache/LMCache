# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TRT-LLM integration components.

These tests verify the LMCache-side components (EngineType, GPU connector,
utils) without requiring TRT-LLM to be installed.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType


def _has_tensorrt_llm():
    try:
        # Third Party
        import tensorrt_llm  # noqa: F401

        return True
    except ImportError:
        return False


def _has_lmc_ops():
    try:
        # First Party
        import lmcache.c_ops  # noqa: F401

        return True
    except ImportError:
        return False


class TestEngineType:
    """Tests for EngineType enum."""

    def test_trtllm_exists(self):
        assert hasattr(EngineType, "TRTLLM")
        assert EngineType.TRTLLM == "trtllm"

    def test_all_engine_types(self):
        expected = {"vllm", "sglang", "trtllm", "mock"}
        actual = {e.value for e in EngineType}
        assert expected == actual


@pytest.mark.skipif(not _has_lmc_ops(), reason="lmcache C ops not built")
class TestTRTLLMGPUConnector:
    """Tests for the TRTLLMGPUConnector class."""

    def test_constructor(self):
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        connector = TRTLLMGPUConnector(
            hidden_dim_size=128,
            num_layers=4,
            chunk_size=32,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            num_kv_heads=8,
            head_dim=16,
        )
        assert connector.hidden_dim_size == 128
        assert connector.num_layers == 4
        assert connector.chunk_size == 32
        assert connector.num_kv_heads == 8
        assert connector.head_dim == 16
        assert connector.kv_cache_tensor is None

    def test_get_shape(self):
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        connector = TRTLLMGPUConnector(
            hidden_dim_size=128,
            num_layers=4,
            chunk_size=32,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            num_kv_heads=8,
            head_dim=16,
        )
        shape = connector.get_shape(32)
        assert shape == torch.Size([2, 4, 32, 128])

    def test_get_shape_different_tokens(self):
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        connector = TRTLLMGPUConnector(
            hidden_dim_size=256,
            num_layers=8,
            chunk_size=64,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            num_kv_heads=4,
            head_dim=64,
        )
        shape = connector.get_shape(64)
        assert shape == torch.Size([2, 8, 64, 256])

    def test_register_kv_caches(self):
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        num_kv_heads = 8
        head_dim = 16
        tokens_per_block = 32
        num_layers = 2
        block_size_flat = tokens_per_block * num_kv_heads * head_dim

        connector = TRTLLMGPUConnector(
            hidden_dim_size=num_kv_heads * head_dim,
            num_layers=num_layers,
            chunk_size=32,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        fake_tensor = torch.zeros(
            10,
            num_layers,
            2,
            block_size_flat,
            dtype=torch.float16,
            device="cuda:0",
        )
        connector.register_kv_caches(fake_tensor)
        assert connector.kv_cache_tensor is not None
        assert connector.tokens_per_block == tokens_per_block
        assert connector.blocks_per_chunk == 1  # chunk_size == tokens_per_block
        assert connector.shape_desc is not None
        assert connector.paged_buffer_ptrs is not None
        assert connector.paged_buffer_ptrs.shape == (1,)

    def test_register_kv_caches_multi_block_chunk(self):
        """Test with chunk_size = 4 * tokens_per_block."""
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        num_kv_heads = 4
        head_dim = 32
        tokens_per_block = 16
        chunk_size = 64  # 4 blocks per chunk

        connector = TRTLLMGPUConnector(
            hidden_dim_size=num_kv_heads * head_dim,
            num_layers=4,
            chunk_size=chunk_size,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        block_size_flat = tokens_per_block * num_kv_heads * head_dim
        fake_tensor = torch.zeros(
            100,
            4,
            2,
            block_size_flat,
            dtype=torch.float16,
            device="cuda:0",
        )
        connector.register_kv_caches(fake_tensor)
        assert connector.tokens_per_block == tokens_per_block
        assert connector.blocks_per_chunk == 4

    def test_streams_exist(self):
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        connector = TRTLLMGPUConnector(
            hidden_dim_size=128,
            num_layers=2,
            chunk_size=32,
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            num_kv_heads=8,
            head_dim=16,
        )
        assert connector.load_stream is not None
        assert connector.store_stream is not None


class TestUtils:
    """Tests for integration utils."""

    def test_engine_name(self):
        # First Party
        from lmcache.integration.tensorrt_llm.utils import ENGINE_NAME

        assert ENGINE_NAME == "trtllm-instance"

    def test_lmcache_get_config_from_env(self, monkeypatch):
        """Test that config can be loaded from environment variables."""
        # First Party
        from lmcache.integration.tensorrt_llm.utils import lmcache_get_config

        # Remove LMCACHE_CONFIG_FILE to force env-based config
        monkeypatch.delenv("LMCACHE_CONFIG_FILE", raising=False)
        config = lmcache_get_config()
        assert config is not None

    @pytest.mark.skipif(
        not _has_tensorrt_llm(), reason="create_trtllm_metadata requires tensorrt_llm"
    )
    def test_create_trtllm_metadata(self):
        # First Party
        from lmcache.integration.tensorrt_llm.utils import create_trtllm_metadata
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_env()
        chunk_size = config.chunk_size

        num_kv_heads = 8
        head_dim = 64
        tokens_per_block = 16
        block_size_flat = tokens_per_block * num_kv_heads * head_dim
        num_layers = 4

        kv_tensor = torch.zeros(
            10,
            num_layers,
            2,
            block_size_flat,
            dtype=torch.float16,
            device="cuda:0",
        )

        # Standard
        from unittest.mock import MagicMock

        mock_llm_args = MagicMock()
        mock_llm_args.tensor_parallel_size = 1
        mock_llm_args.pipeline_parallel_size = 1
        mock_llm_args.model = "test-model"

        metadata = create_trtllm_metadata(
            mock_llm_args,
            kv_tensor,
            config,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        assert metadata.model_name == "test-model"
        assert metadata.kv_shape[0] == num_layers
        assert metadata.kv_shape[1] == 2  # kv_factor
        assert metadata.kv_shape[2] == chunk_size
        assert metadata.kv_shape[3] == num_kv_heads
        assert metadata.kv_shape[4] == head_dim
        assert metadata.kv_dtype == torch.float16
        assert metadata.world_size == 1


@pytest.mark.skipif(not _has_lmc_ops(), reason="lmcache C ops not built")
class TestCreateGPUConnector:
    """Test that CreateGPUConnector works with TRTLLM engine type."""

    def test_create_trtllm_connector(self):
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.gpu_connector import CreateGPUConnector
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector
        from lmcache.v1.metadata import LMCacheMetadata

        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(4, 2, 32, 8, 64),  # (layers, kv, chunk, heads, head_dim)
        )
        config = LMCacheEngineConfig.from_env()
        connector = CreateGPUConnector(config, metadata, "trtllm")
        assert isinstance(connector, TRTLLMGPUConnector)
        assert connector.hidden_dim_size == 8 * 64  # num_kv_heads * head_dim = 512
        assert connector.num_layers == 4
        assert connector.num_kv_heads == 8
        assert connector.head_dim == 64


class _FakeMemoryObj:
    """Minimal stand-in for MemoryObj in roundtrip tests.

    The connector only accesses ``tensor`` and ``tensor.data_ptr()``.
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


@pytest.mark.skipif(not _has_lmc_ops(), reason="lmcache C ops not built")
class TestKernelRoundtrip:
    """Test data integrity through the kernel-based transfer path."""

    def test_single_chunk_roundtrip(self):
        """Write random data to GPU pool, from_gpu to CPU, to_gpu to different
        blocks, verify the data matches."""
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        num_kv_heads = 4
        head_dim = 32
        tokens_per_block = 16
        chunk_size = 16  # 1 block per chunk
        num_layers = 2
        num_blocks = 100
        hidden_dim = num_kv_heads * head_dim
        block_size_flat = tokens_per_block * hidden_dim

        device = torch.device("cuda:0")
        connector = TRTLLMGPUConnector(
            hidden_dim_size=hidden_dim,
            num_layers=num_layers,
            chunk_size=chunk_size,
            dtype=torch.bfloat16,
            device=device,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Create a pool tensor with random data
        pool = torch.randn(
            num_blocks,
            num_layers,
            2,
            block_size_flat,
            dtype=torch.bfloat16,
            device=device,
        )
        connector.register_kv_caches(pool)

        # Save original data from source block for later comparison
        src_block = 5
        orig_data = pool[src_block].clone()

        # Create a CPU tensor matching LMCache intermediate shape
        shape = connector.get_shape(chunk_size)
        cpu_tensor = torch.zeros(shape, dtype=torch.bfloat16, device="cpu").pin_memory()
        mem_obj = _FakeMemoryObj(tensor=cpu_tensor)

        # D2H: copy block 5 to CPU
        connector.from_gpu(
            mem_obj,
            start=0,
            end=chunk_size,
            block_ids=[src_block],
        )
        connector.store_stream.synchronize()

        # H2D: copy CPU data to block 50
        dst_block = 50
        connector.to_gpu(
            mem_obj,
            start=0,
            end=chunk_size,
            block_ids=[dst_block],
        )
        connector.load_stream.synchronize()

        # Verify: block 50 should now contain the same data as the original block 5
        torch.testing.assert_close(
            pool[dst_block],
            orig_data,
            msg="Kernel roundtrip data mismatch",
        )

    def test_multi_block_chunk_roundtrip(self):
        """Test with chunk_size = 4 * tokens_per_block."""
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        num_kv_heads = 4
        head_dim = 32
        tokens_per_block = 16
        chunk_size = 64  # 4 blocks per chunk
        num_layers = 2
        num_blocks = 200
        hidden_dim = num_kv_heads * head_dim
        block_size_flat = tokens_per_block * hidden_dim

        device = torch.device("cuda:0")
        connector = TRTLLMGPUConnector(
            hidden_dim_size=hidden_dim,
            num_layers=num_layers,
            chunk_size=chunk_size,
            dtype=torch.bfloat16,
            device=device,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        pool = torch.randn(
            num_blocks,
            num_layers,
            2,
            block_size_flat,
            dtype=torch.bfloat16,
            device=device,
        )
        connector.register_kv_caches(pool)

        # Save original data from source blocks
        src_blocks = [10, 11, 12, 13]
        orig_data = [pool[b].clone() for b in src_blocks]

        shape = connector.get_shape(chunk_size)
        cpu_tensor = torch.zeros(shape, dtype=torch.bfloat16, device="cpu").pin_memory()
        mem_obj = _FakeMemoryObj(tensor=cpu_tensor)

        # D2H: blocks [10, 11, 12, 13] -> CPU
        connector.from_gpu(
            mem_obj,
            start=0,
            end=chunk_size,
            block_ids=src_blocks,
        )
        connector.store_stream.synchronize()

        # H2D: CPU -> blocks [50, 51, 52, 53]
        dst_blocks = [50, 51, 52, 53]
        connector.to_gpu(
            mem_obj,
            start=0,
            end=chunk_size,
            block_ids=dst_blocks,
        )
        connector.load_stream.synchronize()

        # Verify each destination block matches the corresponding source
        for i, (dst, orig) in enumerate(zip(dst_blocks, orig_data, strict=False)):
            torch.testing.assert_close(
                pool[dst],
                orig,
                msg=f"Block {i}: dst={dst} does not match source",
            )
