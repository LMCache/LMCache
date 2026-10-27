# SPDX-License-Identifier: Apache-2.0
"""End-to-end unit tests for the client context against a mock server."""

# Third Party
import pytest
import torch

# First Party
from lmcache_client_sdk import connect
from lmcache_client_sdk.server_config import LMCacheClientError

# Local
from .mock_server import (
    CHUNK_SIZE,
    NUM_LAYERS,
    MockLMCacheServer,
)


@pytest.fixture
def server() -> MockLMCacheServer:
    """A running in-memory mock server for one test."""
    mock = MockLMCacheServer(model_name="Qwen/Qwen3-8B", world_size=1)
    yield mock
    mock.close()


def test_connect_registers_and_returns_context(server: MockLMCacheServer) -> None:
    """Check that connect() fetches config and registers the client."""
    ctx = connect(
        url=server.mq_url,
        http_url=server.http_url,
        model_name="Qwen/Qwen3-8B",
        timeout=10,
    )
    try:
        assert ctx.chunk_size == CHUNK_SIZE
        assert "register_kv_cache_engine_driven_context" in server.operations
    finally:
        ctx.close()
    assert "unregister_kv_cache_engine_driven_context" in server.operations


def test_connect_fails_when_model_not_registered(
    server: MockLMCacheServer,
) -> None:
    """Check that connect() fails cleanly for an unregistered model."""
    with pytest.raises(LMCacheClientError):
        connect(
            url=server.mq_url,
            http_url=server.http_url,
            model_name="Unknown/Model",
            timeout=10,
        )


def _kv_tensor(num_tokens: int = CHUNK_SIZE) -> torch.Tensor:
    """A deterministic [2, L, T, D] float16 CPU tensor."""
    shape = (2, NUM_LAYERS, num_tokens, 256)
    return torch.arange(torch.tensor(shape).prod(), dtype=torch.float16).view(shape)


def test_store_then_retrieve_roundtrip(server: MockLMCacheServer) -> None:
    """Check that store() followed by retrieve() returns the same tensors."""
    ctx = connect(
        url=server.mq_url,
        http_url=server.http_url,
        model_name="Qwen/Qwen3-8B",
        timeout=10,
    )
    tokens = list(range(CHUNK_SIZE))
    kv = _kv_tensor()
    try:
        assert ctx.store(kv=kv, tokens=tokens) is True
        retrieved = ctx.retrieve(tokens)
    finally:
        ctx.close()
    assert retrieved is not None
    assert retrieved.shape == torch.Size((2, NUM_LAYERS, CHUNK_SIZE, 256))
    assert torch.equal(retrieved, kv)


def test_retrieve_miss_returns_none(server: MockLMCacheServer) -> None:
    """Check that a retrieval of never-stored tokens returns None."""
    ctx = connect(
        url=server.mq_url,
        http_url=server.http_url,
        model_name="Qwen/Qwen3-8B",
        timeout=10,
    )
    tokens = list(range(1000, 1000 + CHUNK_SIZE))
    try:
        retrieved = ctx.retrieve(tokens)
    finally:
        ctx.close()
    assert retrieved is None


def test_store_rejects_token_count_mismatch(
    server: MockLMCacheServer,
) -> None:
    """Check that a token/KV length mismatch raises a client error."""
    ctx = connect(
        url=server.mq_url,
        http_url=server.http_url,
        model_name="Qwen/Qwen3-8B",
        timeout=10,
    )
    try:
        with pytest.raises(LMCacheClientError):
            ctx.store(kv=_kv_tensor(), tokens=list(range(CHUNK_SIZE + 1)))
    finally:
        ctx.close()


def test_retrieve_empty_tokens_returns_none(server: MockLMCacheServer) -> None:
    """Check that retrieving no tokens returns None without an RPC."""
    ctx = connect(
        url=server.mq_url,
        http_url=server.http_url,
        model_name="Qwen/Qwen3-8B",
        timeout=10,
    )
    try:
        assert ctx.retrieve([]) is None
    finally:
        ctx.close()
    assert "lookup" not in server.operations
