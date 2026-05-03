# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TurboQuant serde skeleton.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde import (
    SerdeConfig,
    create_serde_processor,
    get_registered_serde_types,
)
from lmcache.v1.distributed.serde.turboquant import (
    TurboQuantSerdeConfig,
    TurboQuantSerializer,
)


def test_turboquant_registered() -> None:
    assert "turboquant" in get_registered_serde_types()


def test_create_turboquant_serde_processor() -> None:
    processor = create_serde_processor(
        SerdeConfig(
            type="turboquant",
            kwargs={
                "preset": "turboquant_k8v4",
                "head_dim": 128,
                "block_size": 16,
                "max_workers": 1,
            },
        )
    )
    processor.close()


def test_k8v4_config_sizes_head_dim_128() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_k8v4",
        head_dim=128,
        block_size=16,
    )

    assert cfg.key_fp8 is True
    assert cfg.key_quant_bits == 8
    assert cfg.key_mse_bits == 0
    assert cfg.value_quant_bits == 4
    assert cfg.key_packed_size == 128
    assert cfg.value_packed_size == 68
    assert cfg.slot_size == 196
    assert cfg.slot_size_aligned == 196


def test_estimate_serialized_size_k8v4() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_k8v4",
        head_dim=128,
        block_size=16,
    )
    serializer = TurboQuantSerializer(cfg)

    # LMCache KV layout: [2, num_layers, num_tokens, hidden_dim]
    # hidden_dim = num_heads * head_dim = 4 * 128 = 512
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 3, 20, 512])],
        dtypes=[torch.bfloat16],
    )

    # num_layers = 3
    # num_tokens = 20
    # num_heads = 4
    # num_blocks = ceil(20 / 16) = 2
    # slot_size_aligned = 196
    expected = 3 * 2 * 16 * 4 * 196

    assert serializer.estimate_serialized_size(layout) == expected


def test_estimate_serialized_size_rejects_invalid_kv_size() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_k8v4",
        head_dim=128,
        block_size=16,
    )
    serializer = TurboQuantSerializer(cfg)

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([1, 3, 20, 512])],
        dtypes=[torch.bfloat16],
    )

    with pytest.raises(ValueError, match="kv_size=2"):
        serializer.estimate_serialized_size(layout)


def test_estimate_serialized_size_rejects_bad_head_dim() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_k8v4",
        head_dim=128,
        block_size=16,
    )
    serializer = TurboQuantSerializer(cfg)

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 3, 20, 500])],
        dtypes=[torch.bfloat16],
    )

    with pytest.raises(ValueError, match="must be divisible"):
        serializer.estimate_serialized_size(layout)
        
