# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import hashlib

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.sampling_params import SamplingParams

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    SaveSpec,
    extract_request_configs,
)


class _FakeParent:
    def __init__(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeLayerwiseEngine:
    def __init__(self) -> None:
        self.retrieve_layer_calls: list[dict] = []
        self.store_layer_calls: list[dict] = []

    def retrieve_layer(self, *args, **kwargs):
        self.retrieve_layer_calls.append(kwargs)

        def _retriever():
            while True:
                yield None

        return _retriever()

    def store_layer(self, *args, **kwargs):
        self.store_layer_calls.append(kwargs)

        def _storer():
            while True:
                yield None

        return _storer()


class _FakeBlender:
    def __init__(self) -> None:
        self.blend_calls: list[dict] = []

    def blend(self, *args, **kwargs) -> None:
        self.blend_calls.append(kwargs)


class _FakeManager:
    def __init__(self, lookup_client=None, lmcache_engine=None) -> None:
        self.lookup_client = lookup_client
        self.lmcache_engine = lmcache_engine
        self.lookup_server = None


def _hash_cache_salt(cache_salt: str) -> str:
    return hashlib.sha256(cache_salt.encode("utf-8")).hexdigest()


def _make_connector() -> LMCacheConnectorV1Impl:
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = -1
    cast(Any, connector)._manager = _FakeManager(lookup_client=lookup_client)
    connector.kv_role = "kv_both"
    connector.skip_last_n_tokens = 0
    connector.load_specs = {}
    connector._requests_priority = {}
    connector._unfinished_requests = {}
    connector._request_trackers = {}
    connector._block_size = 16
    connector._lmcache_chunk_size = 8
    connector._max_tokens_per_load = 0
    connector._discard_partial_chunks = False
    connector.force_skip_save = False
    connector.config = SimpleNamespace(
        priority_limit=None,
        save_decode_cache=False,
        min_retrieve_tokens=0,
    )
    return connector


def _make_layerwise_connector(
    requests,
) -> tuple[LMCacheConnectorV1Impl, _FakeLayerwiseEngine]:
    connector = _make_connector()
    engine = _FakeLayerwiseEngine()
    connector.use_layerwise = True
    connector.enable_blending = False
    cast(Any, connector)._manager.lmcache_engine = engine
    connector._parent = _FakeParent(LMCacheConnectorMetadata(requests=requests))
    connector._stats_monitor = MagicMock()
    connector.device = "cpu"
    connector.kv_caches = {"layer0": MagicMock()}
    connector.layerwise_retrievers = []
    connector._layerwise_save_storers = {}
    return connector, engine


def test_extract_request_configs_includes_cache_salt_tag() -> None:
    sampling_params = SamplingParams(
        extra_args={
            "kv_transfer_params": {
                "lmcache.tag.user": "example-user",
                "lmcache.ttl": 60,
                "ignored": "value",
            }
        }
    )

    assert extract_request_configs(sampling_params, "salt-a") == {
        "lmcache.tag.user": "example-user",
        "lmcache.ttl": 60,
        "lmcache.tag.cachesalt": _hash_cache_salt("salt-a"),
    }


def test_extract_request_configs_hashes_underscore_salt() -> None:
    request_configs = extract_request_configs(SamplingParams(), "salt_with_underscore")

    assert request_configs == {
        "lmcache.tag.cachesalt": _hash_cache_salt("salt_with_underscore"),
    }


def test_lookup_uses_cache_salt_in_request_configs() -> None:
    connector = _make_connector()
    lookup_client = cast(Any, connector.lookup_client)
    lookup_client.lookup.return_value = 4

    request = MagicMock()
    request.request_id = "req-1"
    request.priority = 0
    request.cache_salt = "salt-a"
    request.sampling_params = SamplingParams(
        extra_args={
            "kv_transfer_params": {
                "lmcache.tag.user": "example-user",
                "lmcache.ttl": 60,
            }
        }
    )
    request.all_token_ids = [1, 2, 3, 4]
    request.prompt_token_ids = [1, 2, 3, 4]
    request.num_tokens = 4
    request.mm_features = []

    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    lookup_client.lookup.assert_called_once_with(
        [1, 2, 3, 4],
        lookup_id="req-1",
        request_configs={
            "lmcache.tag.user": "example-user",
            "lmcache.ttl": 60,
            "lmcache.tag.cachesalt": _hash_cache_salt("salt-a"),
        },
    )


def test_build_connector_meta_preserves_cache_salt_for_req_meta() -> None:
    connector = _make_connector()
    connector._unfinished_requests["req-1"] = MagicMock(cache_salt="salt-a")

    scheduler_output = SimpleNamespace(
        finished_req_ids=[],
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="req-1",
                num_computed_tokens=0,
                block_ids=[0],
                prompt_token_ids=[1, 2, 3, 4],
                sampling_params=SamplingParams(
                    extra_args={"kv_transfer_params": {"lmcache.ttl": 60}}
                ),
            )
        ],
        num_scheduled_tokens={"req-1": 4},
        scheduled_cached_reqs=[],
    )

    meta = connector.build_connector_meta(scheduler_output)

    assert len(meta.requests) == 1
    assert meta.requests[0].request_configs == {
        "lmcache.ttl": 60,
        "lmcache.tag.cachesalt": _hash_cache_salt("salt-a"),
    }


def test_start_load_kv_layerwise_passes_request_configs() -> None:
    request_configs = {
        "lmcache.ttl": 60,
        "lmcache.tag.cachesalt": _hash_cache_salt("salt-a"),
    }
    request = SimpleNamespace(
        req_id="req-1",
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4,
            can_load=True,
        ),
        request_configs=request_configs,
    )

    connector, engine = _make_layerwise_connector([request])
    forward_context = SimpleNamespace(attn_metadata=object())

    connector.start_load_kv(forward_context)

    assert len(engine.retrieve_layer_calls) == 1
    assert engine.retrieve_layer_calls[0]["request_configs"] == request_configs


def test_start_load_kv_blending_passes_request_configs() -> None:
    request_configs = {
        "lmcache.ttl": 60,
        "lmcache.tag.cachesalt": _hash_cache_salt("salt-a"),
    }
    request = SimpleNamespace(
        req_id="req-1",
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4,
            can_load=True,
        ),
        request_configs=request_configs,
    )

    connector, _ = _make_layerwise_connector([request])
    connector.enable_blending = True
    connector.blender = _FakeBlender()
    forward_context = SimpleNamespace(attn_metadata=object())

    connector.start_load_kv(forward_context)

    assert len(connector.blender.blend_calls) == 1
    assert connector.blender.blend_calls[0]["request_configs"] == request_configs


def test_save_kv_layer_layerwise_passes_request_configs() -> None:
    request_configs = {
        "lmcache.ttl": 60,
        "lmcache.tag.cachesalt": _hash_cache_salt("salt-a"),
    }
    request = SimpleNamespace(
        req_id="req-1",
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        request_configs=request_configs,
    )

    connector, engine = _make_layerwise_connector([request])

    connector.save_kv_layer("layer0", MagicMock(), MagicMock())

    assert len(engine.store_layer_calls) == 1
    assert engine.store_layer_calls[0]["request_configs"] == request_configs
