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
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    RequestTracker,
    _cache_salt_tag_value,
    extract_request_configs,
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
    """retrieve_layer() must receive request_configs when called via start_load_kv()."""
    import torch

    req_configs = {_TAG_KEY: _cache_salt_tag_value("salt-a")}
    recorded: dict = {}

    def _fake_retrieve_layer(token_ids, token_mask, **kwargs):
        recorded.update(kwargs)
        return iter([None, None, None])  # next() is called twice internally

    req = SimpleNamespace(
        req_id="req-1",
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        load_spec=SimpleNamespace(
            lmcache_cached_tokens=4,
            vllm_cached_tokens=0,
            can_load=True,
        ),
        request_configs=req_configs,
    )
    metadata = LMCacheConnectorMetadata(requests=[req])
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
    )
    connector.lmcache_engine = SimpleNamespace(retrieve_layer=_fake_retrieve_layer)
    connector.enable_blending = False
    connector.use_layerwise = True
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector.layerwise_retrievers = []
    connector._stats_monitor = SimpleNamespace(
        update_interval_vllm_hit_tokens=lambda x: None,
        update_interval_prompt_tokens=lambda x: None,
    )
    forward_context = SimpleNamespace(attn_metadata=object())  # non-None triggers load

    connector.start_load_kv(forward_context)

    assert "request_configs" in recorded, (
        "retrieve_layer() was not called with request_configs"
    )
    assert recorded["request_configs"] == req_configs


def test_save_kv_layer_layerwise_passes_request_configs():
    """store_layer() must receive request_configs when called via save_kv_layer()."""
    import torch

    req_configs = {_TAG_KEY: _cache_salt_tag_value("salt-a")}
    recorded: dict = {}

    def _fake_store_layer(token_ids, **kwargs):
        recorded.update(kwargs)
        return iter([None])

    req = SimpleNamespace(
        req_id="req-1",
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        save_spec=SimpleNamespace(skip_leading_tokens=0, can_save=True),
        request_configs=req_configs,
    )
    metadata = LMCacheConnectorMetadata(requests=[req])
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = SimpleNamespace(
        _connector_metadata=metadata,  # not-None guard at line 1049
        _get_connector_metadata=lambda: metadata,
    )
    connector.lmcache_engine = SimpleNamespace(store_layer=_fake_store_layer)
    connector.kv_role = "kv_producer"
    connector.use_layerwise = True
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._layerwise_save_storers = {}

    connector.save_kv_layer("layer0", torch.zeros(1), None)

    assert "request_configs" in recorded, (
        "store_layer() was not called with request_configs"
    )
    assert recorded["request_configs"] == req_configs


# ---------------------------------------------------------------------------
# from_new_request: cache_salt flows into RequestTracker.request_configs
# ---------------------------------------------------------------------------


def test_from_new_request_with_cache_salt():
    """RequestTracker.from_new_request() stores the hashed salt in request_configs."""
    new_request = SimpleNamespace(
        req_id="req-1",
        block_ids=[0, 1, 2, 3],
        sampling_params=_make_sampling_params(),
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    tracker = RequestTracker.from_new_request(
        None,  # lmcache_config is not used inside the function
        new_request,
        num_tokens_to_compute=8,
        lmcache_cached_tokens=0,
        skip_save=False,
        cache_salt="salt-a",
    )
    assert tracker.request_configs is not None
    assert _TAG_KEY in tracker.request_configs
    expected = hashlib.sha256("salt-a".encode("utf-8")).hexdigest()
    assert tracker.request_configs[_TAG_KEY] == expected


def test_from_new_request_no_salt_leaves_request_configs_none():
    """Without a cache_salt, request_configs should remain None."""
    new_request = SimpleNamespace(
        req_id="req-2",
        block_ids=[0, 1, 2, 3],
        sampling_params=_make_sampling_params(),
        prompt_token_ids=[1, 2, 3, 4],
    )
    tracker = RequestTracker.from_new_request(
        None,
        new_request,
        num_tokens_to_compute=4,
        lmcache_cached_tokens=0,
        skip_save=False,
        cache_salt=None,
    )
    assert tracker.request_configs is None


# ---------------------------------------------------------------------------
# get_num_new_matched_tokens: request_configs with cache_salt reaches lookup()
# ---------------------------------------------------------------------------


def test_get_num_new_matched_tokens_passes_cache_salt_to_lookup():
    """lookup() must be called with request_configs containing the cache_salt tag."""
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = -1  # first-time lookup
    lookup_client.lookup.return_value = 4

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.kv_role = "kv_consumer"
    connector.lookup_client = lookup_client
    connector.skip_last_n_tokens = 0
    connector._requests_priority = {}
    connector.config = SimpleNamespace(min_retrieve_tokens=0)
    connector.load_specs = {}

    request = SimpleNamespace(
        request_id="req-1",
        all_token_ids=[1, 2, 3, 4],
        num_tokens=4,
        sampling_params=_make_sampling_params(),
        cache_salt="salt-a",
    )

    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    lookup_client.lookup.assert_called_once()
    call_kwargs = lookup_client.lookup.call_args.kwargs
    assert "request_configs" in call_kwargs
    assert call_kwargs["request_configs"] is not None
    assert _TAG_KEY in call_kwargs["request_configs"]
    expected = hashlib.sha256("salt-a".encode("utf-8")).hexdigest()
    assert call_kwargs["request_configs"][_TAG_KEY] == expected
