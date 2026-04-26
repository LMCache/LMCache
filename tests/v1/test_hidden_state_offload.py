# SPDX-License-Identifier: Apache-2.0
"""Tests for hidden state offload alongside KV cache in LMCache."""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import (
    CacheEngineKey,
    mock_up_broadcast_fn,
    mock_up_broadcast_object_fn,
)
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig

# Local
from .utils import (
    create_gpu_connector,
    dumb_metadata,
    generate_tokens,
)

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


# ---------------------------------------------------------------------------
# store_hidden_states / retrieve_hidden_states roundtrip
# ---------------------------------------------------------------------------


def _build_engine(autorelease_v1, chunk_size: int = 64):
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        remote_url=None,
        save_unfull_chunk=True,
    )
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)
    return autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LMCacheEngine init currently requires CUDA-capable GPU connector",
)
def test_store_then_retrieve_hidden_states_roundtrip(autorelease_v1):
    chunk_size = 64
    num_tokens = 192  # exactly 3 full chunks
    hidden_dim = 16

    engine = _build_engine(autorelease_v1, chunk_size=chunk_size)
    tokens = generate_tokens(num_tokens, device="cuda")
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)

    engine.store_hidden_states(tokens, hidden_states=hidden_states)
    retrieved = engine.retrieve_hidden_states(tokens)

    assert retrieved is not None
    assert retrieved.shape == hidden_states.shape
    assert torch.equal(retrieved, hidden_states)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LMCacheEngine init currently requires CUDA-capable GPU connector",
)
def test_store_hidden_states_none_is_noop(autorelease_v1):
    """Passing hidden_states=None should be a silent no-op."""
    engine = _build_engine(autorelease_v1)
    tokens = generate_tokens(128, device="cuda")
    # No raise; nothing stored.
    engine.store_hidden_states(tokens, hidden_states=None)
    assert engine.retrieve_hidden_states(tokens) is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LMCacheEngine init currently requires CUDA-capable GPU connector",
)
def test_retrieve_hidden_states_returns_none_on_miss(autorelease_v1):
    """Retrieving without a prior store should return None (not crash)."""
    engine = _build_engine(autorelease_v1)
    tokens = generate_tokens(128, device="cuda")
    assert engine.retrieve_hidden_states(tokens) is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LMCacheEngine init currently requires CUDA-capable GPU connector",
)
def test_retrieve_hidden_states_partial_returns_none(autorelease_v1):
    """If only the first chunk's tokens were stored, retrieving a longer
    sequence must return None (all-or-nothing semantics)."""
    chunk_size = 64
    hidden_dim = 16

    engine = _build_engine(autorelease_v1, chunk_size=chunk_size)
    short_tokens = generate_tokens(chunk_size, device="cuda")
    long_tokens = torch.cat([short_tokens, generate_tokens(chunk_size, device="cuda")])
    short_hs = torch.randn(chunk_size, hidden_dim, dtype=torch.bfloat16)

    engine.store_hidden_states(short_tokens, hidden_states=short_hs)
    # First chunk is stored, second is not -> overall miss.
    assert engine.retrieve_hidden_states(long_tokens) is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LMCacheEngine init currently requires CUDA-capable GPU connector",
)
def test_store_hidden_states_skipped_when_frozen(autorelease_v1):
    """In freeze mode, store_hidden_states must not write anything."""
    engine = _build_engine(autorelease_v1)
    tokens = generate_tokens(128, device="cuda")
    hs = torch.randn(128, 16, dtype=torch.bfloat16)

    engine.freeze(True)
    try:
        engine.store_hidden_states(tokens, hidden_states=hs)
    finally:
        engine.freeze(False)
    assert engine.retrieve_hidden_states(tokens) is None
