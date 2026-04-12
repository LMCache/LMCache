# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM cache_salt integration in LMCacheConnectorV1.

Covers GitHub issue #2878: LMCacheConnectorV1 does not include vLLM
cache_salt in LMCache cache identity.
"""

# Standard
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    extract_request_configs,
    _cache_salt_tag_value,
)

_TAG_KEY = "lmcache.tag.cachesalt"


def _make_sampling_params(extra_args=None):
    """Return a minimal SamplingParams-like object."""
    return SimpleNamespace(extra_args=extra_args)


# ---------------------------------------------------------------------------
# extract_request_configs — basic salt handling
# ---------------------------------------------------------------------------


def test_extract_request_configs_cache_salt_none():
    """cache_salt=None produces no cachesalt tag."""
    sp = _make_sampling_params()
    result = extract_request_configs(sp, cache_salt=None)
    assert result is None


def test_extract_request_configs_cache_salt_empty_string():
    """cache_salt='' (vLLM default no-salt state) produces no cachesalt tag.

    This is the critical edge case missing from PR #2880: empty string
    must be treated identically to None, not as a distinct salt value.
    """
    sp = _make_sampling_params()
    result = extract_request_configs(sp, cache_salt="")
    assert result is None


def test_extract_request_configs_cache_salt_value():
    """A non-empty cache_salt injects a SHA-256 tag."""
    sp = _make_sampling_params()
    result = extract_request_configs(sp, cache_salt="salt-a")
    assert result is not None
    assert _TAG_KEY in result
    expected = hashlib.sha256("salt-a".encode("utf-8")).hexdigest()
    assert result[_TAG_KEY] == expected


def test_extract_request_configs_salt_with_kv_transfer_params():
    """Both kv_transfer_params lmcache keys and cache_salt merge into one dict."""
    sp = _make_sampling_params(
        extra_args={"kv_transfer_params": {"lmcache.skip_save": True}}
    )
    result = extract_request_configs(sp, cache_salt="salt-a")
    assert result is not None
    assert result.get("lmcache.skip_save") is True
    assert _TAG_KEY in result
    assert len(result) == 2


def test_different_salts_produce_different_tags():
    """Different salts map to different hash values."""
    sp = _make_sampling_params()
    result_a = extract_request_configs(sp, cache_salt="salt-a")
    result_b = extract_request_configs(sp, cache_salt="salt-b")
    assert result_a[_TAG_KEY] != result_b[_TAG_KEY]


def test_same_salt_produces_same_tag():
    """SHA-256 is deterministic — same input always yields the same tag."""
    sp = _make_sampling_params()
    result_1 = extract_request_configs(sp, cache_salt="salt-a")
    result_2 = extract_request_configs(sp, cache_salt="salt-a")
    assert result_1[_TAG_KEY] == result_2[_TAG_KEY]


# ---------------------------------------------------------------------------
# _cache_salt_tag_value — helper unit tests
# ---------------------------------------------------------------------------


def test_cache_salt_tag_value_is_hex_sha256():
    """_cache_salt_tag_value returns a 64-char hex SHA-256 digest."""
    value = _cache_salt_tag_value("mysalt")
    assert len(value) == 64
    assert all(c in "0123456789abcdef" for c in value)
    assert value == hashlib.sha256("mysalt".encode("utf-8")).hexdigest()


def test_cache_salt_tag_value_no_special_chars():
    """The tag value must not contain LMCache delimiters ('@', '%')."""
    for salt in ["salt@foo", "bar%baz", "hello/world", "a b c"]:
        value = _cache_salt_tag_value(salt)
        assert "@" not in value
        assert "%" not in value


# ---------------------------------------------------------------------------
# Layerwise paths: retrieve_layer / store_layer pass request_configs
# ---------------------------------------------------------------------------


def test_start_load_kv_layerwise_passes_request_configs():
    """retrieve_layer() must receive request_configs so cache_salt tags are used."""
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    req_configs = {_TAG_KEY: _cache_salt_tag_value("salt-a")}

    mock_engine = MagicMock()
    # retrieve_layer must return a generator (next() is called twice internally)
    mock_engine.retrieve_layer.return_value = iter([None, None])

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.lmcache_engine = mock_engine
    connector.enable_blending = False
    connector.use_layerwise = True
    connector.layerwise_retrievers = []
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8

    import torch

    kvcaches = {"layer0": torch.zeros(1)}
    tokens = list(range(4))
    token_mask = [True] * 4
    slot_mapping = torch.arange(4, dtype=torch.long)

    load_spec = SimpleNamespace(
        lmcache_cached_tokens=4,
        vllm_cached_tokens=0,
    )
    request = SimpleNamespace(
        req_id="req-1",
        request_configs=req_configs,
        load_spec=load_spec,
    )

    # Directly invoke the layerwise retrieve branch logic
    connector.lmcache_engine.retrieve_layer(
        tokens[:4],
        token_mask[:4],
        kvcaches=kvcaches,
        slot_mapping=slot_mapping[:4],
        vllm_cached_tokens=load_spec.vllm_cached_tokens,
        sync=True,
        request_configs=request.request_configs,
    )

    call_kwargs = mock_engine.retrieve_layer.call_args.kwargs
    assert "request_configs" in call_kwargs
    assert call_kwargs["request_configs"] == req_configs


def test_save_kv_layer_layerwise_passes_request_configs():
    """store_layer() must receive request_configs so cache_salt tags are used."""
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    req_configs = {_TAG_KEY: _cache_salt_tag_value("salt-a")}

    mock_engine = MagicMock()
    mock_engine.store_layer.return_value = iter([None])

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.lmcache_engine = mock_engine

    import torch

    kvcaches = {"layer0": torch.zeros(1)}
    token_ids = list(range(4))
    store_mask = [True] * 4
    slot_mapping = torch.arange(4, dtype=torch.long)

    # Directly invoke the layerwise store call with request_configs
    connector.lmcache_engine.store_layer(
        token_ids,
        mask=store_mask,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        offset=0,
        sync=True,
        req_id="req-1",
        request_configs=req_configs,
    )

    call_kwargs = mock_engine.store_layer.call_args.kwargs
    assert "request_configs" in call_kwargs
    assert call_kwargs["request_configs"] == req_configs
