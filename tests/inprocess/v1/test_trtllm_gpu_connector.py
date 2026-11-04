# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the InProcess TensorRT-LLM GPU connector."""

# Standard
from typing import cast
from unittest.mock import Mock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.memory_management import MemoryObj

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
]


def _has_cuda_ops() -> bool:
    try:
        # First Party
        import lmcache.cuda_ops  # noqa: F401

        return True
    except ImportError:
        return False


def _has_cuda() -> bool:
    return torch_dev.is_available() and torch_device_type == "cuda"


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for connector init")
@pytest.mark.skipif(not _has_cuda_ops(), reason="lmcache C ops not built")
class TestTRTLLMGPUConnector:
    def test_construct(self) -> None:
        # First Party
        from lmcache.inprocess.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        device = torch.device(f"{torch_device_type}:0")
        c = TRTLLMGPUConnector(
            num_kv_heads=2,
            head_dim=64,
            hidden_dim_size=128,
            num_layers=4,
            chunk_size=256,
            dtype=torch.bfloat16,
            device=device,
        )
        assert c.num_kv_heads == 2
        assert c.head_dim == 64
        assert c.kv_cache_tensor is None  # not registered yet

    def test_get_shape(self) -> None:
        # First Party
        from lmcache.inprocess.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        device = torch.device(f"{torch_device_type}:0")
        c = TRTLLMGPUConnector(
            num_kv_heads=2,
            head_dim=64,
            hidden_dim_size=128,
            num_layers=4,
            chunk_size=256,
            dtype=torch.bfloat16,
            device=device,
        )
        # Memory-obj layout: [2, num_layers, num_tokens, hidden_dim]
        assert tuple(c.get_shape(256)) == (2, 4, 256, 128)

    def test_register_kv_caches_with_4d_tensor(self) -> None:
        # First Party
        from lmcache.inprocess.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        device = torch.device(f"{torch_device_type}:0")
        nh, bs, hs = 2, 16, 64
        nb, nl, kv = 4, 4, 2
        flat = nh * bs * hs
        c = TRTLLMGPUConnector(
            num_kv_heads=nh,
            head_dim=hs,
            hidden_dim_size=nh * hs,
            num_layers=nl,
            chunk_size=256,
            dtype=torch.bfloat16,
            device=device,
        )
        t = torch.zeros(nb, nl, kv, flat, dtype=torch.bfloat16, device=device)
        c.register_kv_caches(t)
        assert c.kv_cache_tensor is not None
        assert tuple(c.kv_cache_tensor.shape) == (nb, nl, kv, nh, bs, hs)
        assert c.tokens_per_block == bs
        assert c.blocks_per_chunk == 256 // bs
        assert c.shape_desc is not None
        assert c.shape_desc.nl == nl
        assert c.shape_desc.nb == nb
        assert c.shape_desc.bs == bs
        assert c.shape_desc.nh == nh
        assert c.shape_desc.hs == hs

    def test_batched_store_respects_kernel_object_limit(self) -> None:
        # First Party
        from lmcache.inprocess.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        device = torch.device(f"{torch_device_type}:0")
        num_chunks = 5
        chunk_size = 16
        c = TRTLLMGPUConnector(
            num_kv_heads=1,
            head_dim=16,
            hidden_dim_size=16,
            num_layers=1,
            chunk_size=chunk_size,
            dtype=torch.bfloat16,
            device=device,
        )
        kv_cache = torch.zeros(
            num_chunks,
            1,
            2,
            chunk_size * 16,
            dtype=torch.bfloat16,
            device=device,
        )
        c.register_kv_caches(kv_cache)

        memory_objs = [
            cast(MemoryObj, Mock(tensor=torch.empty(1))) for _ in range(num_chunks)
        ]
        starts = [i * chunk_size for i in range(num_chunks)]
        ends = [(i + 1) * chunk_size for i in range(num_chunks)]

        with patch(
            "lmcache.inprocess.v1.gpu_connector.gpu_connectors."
            "device_ops.multi_layer_block_kv_transfer"
        ) as transfer:
            c.batched_from_gpu(
                memory_objs,
                starts,
                ends,
                block_ids=list(range(num_chunks)),
            )

        assert [len(call.args[1]) for call in transfer.call_args_list] == [4, 1]
