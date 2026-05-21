# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.utils import (
    filter_attention_kv_cache_dict,
    filter_attention_kv_caches,
)


def test_filter_attention_kv_caches_skips_recurrent_entries() -> None:
    """Test that hybrid models keep attention tensors only."""
    attn_0 = torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)
    attn_1 = torch.randn(32, 256, 512, dtype=torch.float16)
    mamba_state = [
        torch.randn(32, 16, 128, dtype=torch.float16),
        torch.randn(32, 4, 64, dtype=torch.float16),
    ]

    filtered = filter_attention_kv_caches([attn_0, mamba_state, attn_1])

    assert len(filtered) == 2
    assert filtered[0] is attn_0
    assert filtered[1] is attn_1


def test_filter_attention_kv_cache_dict_counts_skipped_entries() -> None:
    """Test dict filtering used by the vLLM adapter."""
    attn_0 = torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)
    attn_1 = torch.randn(32, 256, 512, dtype=torch.float16)
    recurrent_state = [
        torch.randn(32, 16, 128, dtype=torch.float16),
        torch.randn(32, 4, 64, dtype=torch.float16),
    ]

    filtered, skipped = filter_attention_kv_cache_dict(
        {
            "model.layers.0.self_attn": attn_0,
            "model.layers.1.mamba": recurrent_state,
            "model.layers.2.self_attn": attn_1,
        }
    )

    assert skipped == 1
    assert list(filtered) == [
        "model.layers.0.self_attn",
        "model.layers.2.self_attn",
    ]
    assert filtered["model.layers.0.self_attn"] is attn_0
    assert filtered["model.layers.2.self_attn"] is attn_1
