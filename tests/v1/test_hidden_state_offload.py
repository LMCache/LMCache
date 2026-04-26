# SPDX-License-Identifier: Apache-2.0
"""Tests for hidden state offload alongside KV cache in LMCache."""

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey

# ---------------------------------------------------------------------------
# CacheEngineKey.to_hs_key()
# ---------------------------------------------------------------------------


def test_to_hs_key_produces_distinct_key():
    key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=12345,
        dtype=torch.bfloat16,
    )
    hs_key = key.to_hs_key()

    assert hs_key.model_name == "test-model:hs"
    assert hs_key.chunk_hash == key.chunk_hash
    assert hs_key.world_size == key.world_size
    assert hs_key.worker_id == key.worker_id
    assert hs_key != key
    assert hash(hs_key) != hash(key)


def test_to_hs_key_preserves_request_configs():
    key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=99999,
        dtype=torch.bfloat16,
        request_configs={"lmcache.tag.user": "alice"},
    )
    hs_key = key.to_hs_key()
    assert hs_key.request_configs == key.request_configs
    assert hs_key.tags == key.tags


def test_to_hs_key_is_idempotent_hash():
    key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=42,
        dtype=torch.bfloat16,
    )
    hs_key1 = key.to_hs_key()
    hs_key2 = key.to_hs_key()
    assert hs_key1 == hs_key2
    assert hash(hs_key1) == hash(hs_key2)


def test_to_hs_key_string_representation():
    key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=42,
        dtype=torch.bfloat16,
    )
    hs_key = key.to_hs_key()
    assert ":hs" in hs_key.to_string()
    assert hs_key.to_string() != key.to_string()


def test_hs_key_does_not_collide_with_kv_key():
    """HS and KV keys for the same chunk must never match in a hash table."""
    key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=12345,
        dtype=torch.bfloat16,
    )
    hs_key = key.to_hs_key()

    # Must not be equal
    assert hs_key != key
    # Must not collide in dict/set
    d = {key: "kv", hs_key: "hs"}
    assert len(d) == 2
    assert d[key] == "kv"
    assert d[hs_key] == "hs"
