# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server configuration discovery and shape parsing."""

# Standard
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache_client_sdk.server_config import (
    LMCacheClientError,
    _parse_concrete_shape,
    fetch_cache_layout,
    fetch_server_config,
)


@pytest.mark.parametrize(
    ("format_name", "shape_str", "expected"),
    [
        # Standard split-KV: 2 x NL x NB x BS x NH x HS
        ("NL_X_TWO_NB_BS_NH_HS", "32 x [2, 2048, 16, 4, 64]", (16, 4, 64)),
        # CROSS_LAYER: NB x NL x 2 x BS x NH x HS (no list level)
        ("NB_NL_TWO_BS_NH_HS", "2048 x [2048, 32, 2, 16, 4, 64]", (16, 4, 64)),
        # MLA single-plane: NL x NB x BS x HS (no NH axis)
        ("NL_X_NB_BS_HS", "32 x [2048, 16, 128]", (16, 1, 128)),
        # DeepSeek MLA with context-scaling: NL x NB x BS x CS
        ("NL_X_NB_BS_CS", "32 x [2048, 16, 128]", (16, 1, 128)),
    ],
)
def test_parse_concrete_shape(
    format_name: str, shape_str: str, expected: tuple[int, int, int]
) -> None:
    """Check that the template reverse-parser recovers geometry."""
    assert _parse_concrete_shape(shape_str, format_name) == expected


def test_parse_concrete_shape_rejects_mismatched_template() -> None:
    """Check that a shape not matching the format template raises."""
    with pytest.raises(LMCacheClientError):
        _parse_concrete_shape("32 x [2, 2048, 16, 4, 64]", "NL_X_NB_BS_HS")


def test_parse_concrete_shape_rejects_non_numeric() -> None:
    """Check that a non-numeric shape raises."""
    with pytest.raises(LMCacheClientError):
        _parse_concrete_shape("32 x [2, N, 16]", "NL_X_NB_BS_HS")


def test_parse_concrete_shape_rejects_missing_block_axis() -> None:
    """Check that a format without a BS axis raises."""
    with pytest.raises(LMCacheClientError):
        _parse_concrete_shape("32 x [2, 16, 4]", "NL_X_NB_NH_HS")


_META_CONFIG = {
    "cache_context_meta": {
        "Qwen/Qwen3-8B": {
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 2,
            "num_kv_readers": 1,
            "kv_cache_layout": {
                "num_layers": 32,
                "kernel_groups": [
                    {
                        "kernel_group_idx": 0,
                        "dtype": "torch.float16",
                        "tokens_per_block": 16,
                        "engine_kv_concrete_shape": "32 x [2, 2048, 16, 4, 64]",
                        "engine_kv_format": "NL_X_TWO_NB_BS_NH_HS",
                        "is_mla": False,
                    }
                ],
            },
        }
    }
}


def test_fetch_server_config_parses_chunk_size() -> None:
    """Check that /config is parsed into the chunk size."""
    with patch("lmcache_client_sdk.server_config.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"mp": {"chunk_size": 256}}
        config = fetch_server_config("http://127.0.0.1:1", timeout=10)
    assert config.chunk_size == 256


def test_fetch_server_config_raises_on_missing_field() -> None:
    """Check that a malformed /config raises a client error."""
    with patch("lmcache_client_sdk.server_config.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {}
        with pytest.raises(LMCacheClientError):
            fetch_server_config("http://127.0.0.1:1", timeout=10)


def test_fetch_cache_layout_decodes_geometry() -> None:
    """Check that /status is parsed into the cache layout."""
    with patch("lmcache_client_sdk.server_config.requests.get") as mock_get:
        mock_get.return_value.json.return_value = _META_CONFIG
        layout = fetch_cache_layout(
            "http://127.0.0.1:1", "Qwen/Qwen3-8B", timeout=10
        )
    assert layout.world_size == 2
    assert layout.num_layers == 32
    assert layout.block_size == 16
    assert layout.num_kv_heads == 4
    assert layout.head_dim == 64
    assert layout.hidden_dim_size == 256
    assert layout.dtype_str == "float16"
    assert layout.use_mla is False


def test_fetch_cache_layout_raises_when_model_missing() -> None:
    """Check that an unregistered model raises a client error."""
    with patch("lmcache_client_sdk.server_config.requests.get") as mock_get:
        mock_get.return_value.json.return_value = _META_CONFIG
        with pytest.raises(LMCacheClientError):
            fetch_cache_layout("http://127.0.0.1:1", "Other/Model", timeout=10)


def test_fetch_cache_layout_raises_on_block_size_mismatch() -> None:
    """Check that a tokens_per_block mismatch raises a client error."""
    meta = {
        "cache_context_meta": {
            "M": {
                "model_name": "M",
                "world_size": 1,
                "num_kv_readers": 1,
                "kv_cache_layout": {
                    "num_layers": 32,
                    "kernel_groups": [
                        {
                            "kernel_group_idx": 0,
                            "dtype": "torch.float16",
                            "tokens_per_block": 999,
                            "engine_kv_concrete_shape": "32 x [2, 2048, 16, 4, 64]",
                            "engine_kv_format": "NL_X_TWO_NB_BS_NH_HS",
                            "is_mla": False,
                        }
                    ],
                },
            }
        }
    }
    with patch("lmcache_client_sdk.server_config.requests.get") as mock_get:
        mock_get.return_value.json.return_value = meta
        with pytest.raises(LMCacheClientError):
            fetch_cache_layout("http://127.0.0.1:1", "M", timeout=10)
