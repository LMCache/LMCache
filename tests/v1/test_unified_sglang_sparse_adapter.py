# SPDX-License-Identifier: Apache-2.0
"""Sparse lease ownership through the current unified SGLang connector."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.sglang import unified_lmcache_mp_connector as adapter_mod
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)
from lmcache.v1.multiprocess.futures import MessagingFuture

_CHUNK_SIZE = 256


def _make_connector(healthy=True):
    conn = object.__new__(UnifiedLMCacheMPConnector)
    conn._closed = not healthy
    conn.chunk_size = _CHUNK_SIZE
    conn.page_size = 1
    conn._sparse_mha_layer_count = 4
    conn.instance_id = 1
    conn.pp_size = 1
    conn._kv_groups = (SimpleNamespace(recurrent_state=False),)
    conn._mq_timeout = 5.0
    conn._sparse_handles = set()
    conn._sparse_handles_lock = threading.Lock()
    conn._sparse_key_cache = {}
    conn._sparse_hash_cache = {}
    conn._sparse_key_cache_lock = threading.Lock()
    conn._active_sessions = set()
    conn._control_futures = []
    conn._heartbeat_stop = threading.Event()
    conn._heartbeat_interval = 1.0
    return conn


def test_sparse_release_keeps_cleanup_record_until_remote_success() -> None:
    conn = _make_connector(healthy=True)
    conn._req_client = MagicMock(name="rpc_client")
    key = ("request", 2, 3)
    conn._sparse_handles = {key}

    failed: MessagingFuture = MessagingFuture()
    failed.set_result(False)
    conn._req_client.sparse_release_prefetch.return_value = failed

    future = conn.sparse_release_prefetch(*key)
    assert future.result(timeout=0) is False
    assert key in conn._sparse_handles

    succeeded: MessagingFuture = MessagingFuture()
    succeeded.set_result(True)
    conn._req_client.sparse_release_prefetch.return_value = succeeded
    future = conn.sparse_release_prefetch(*key)
    assert future.result(timeout=0) is True
    assert key not in conn._sparse_handles


def test_sparse_object_keys_reuse_hashes_with_request_scoped_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated layer lookups reuse hashes without crossing generations."""
    conn: Any = object.__new__(UnifiedLMCacheMPConnector)
    conn._closed = False
    conn.chunk_size = _CHUNK_SIZE
    conn.page_size = 1
    conn._sparse_mha_layer_count = 4
    conn.model_name = "test-model"
    conn.tp_size = 1
    conn.tp_rank = 0
    conn.pp_size = 1
    conn._kv_groups = (SimpleNamespace(recurrent_state=False),)
    conn._sparse_key_cache = {}
    conn._sparse_hash_cache = {}
    conn._sparse_key_cache_lock = threading.Lock()

    class _Hasher:
        def __init__(self) -> None:
            self.calls = 0

        def compute_chunk_hashes(self, token_ids, end):
            self.calls += 1
            return [bytes([index]) * 32 for index in range(end // _CHUNK_SIZE)]

    hasher = _Hasher()
    conn._sparse_token_hasher = hasher
    token_ids = list(range(2 * _CHUNK_SIZE))

    first = conn.create_sparse_object_keys(
        token_ids,
        [0],
        cache_salt="salt-a",
        request_id="request-1",
        generation=3,
        layer_id=0,
    )
    second = conn.create_sparse_object_keys(
        token_ids,
        [0],
        cache_salt="salt-a",
        request_id="request-1",
        generation=3,
        layer_id=1,
    )
    assert first == second
    assert hasher.calls == 1

    conn.create_sparse_object_keys(
        token_ids,
        [0],
        cache_salt="salt-a",
        request_id="request-1",
        generation=4,
        layer_id=0,
    )
    assert hasher.calls == 2


def test_sparse_lease_future_retries_local_cleanup_after_failure() -> None:
    raw_future: MessagingFuture = MessagingFuture()
    raw_future.set_result(True)
    cleanup_calls = []

    def cleanup() -> None:
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            raise RuntimeError("local cleanup failed")

    future = adapter_mod._SparseLeaseFuture(
        raw_future,
        cleanup,
        lambda result: result is True,
    )

    with pytest.raises(RuntimeError, match="local cleanup failed"):
        future.result(timeout=0)
    assert future.result(timeout=0) is True
    assert len(cleanup_calls) == 2


def test_closed_sparse_cleanup_is_not_reported_as_remote_success() -> None:
    conn = _make_connector(healthy=False)
    conn._req_client = MagicMock(name="rpc_client")
    key = ("request", 4, 5)
    conn._sparse_handles = {key}

    cancel_future = conn.sparse_cancel_prefetch(*key)
    release_future = conn.sparse_release_prefetch(*key)

    assert cancel_future.result(timeout=0) is False
    assert release_future.result(timeout=0) is False
    assert key in conn._sparse_handles
    conn._req_client.sparse_cancel_prefetch.assert_not_called()
    conn._req_client.sparse_release_prefetch.assert_not_called()


def test_close_keeps_sparse_cleanup_record_until_cancel_succeeds() -> None:
    conn = _make_connector(healthy=True)
    conn._req_client = MagicMock(name="rpc_client")
    conn._heartbeat_thread = None
    conn._registered = False
    conn._transfer_ctx = None
    key = ("request", 5, 6)
    conn._sparse_handles = {key}

    failed: MessagingFuture = MessagingFuture()
    failed.set_result(False)
    succeeded: MessagingFuture = MessagingFuture()
    succeeded.set_result(True)
    conn._req_client.sparse_cancel_prefetch.side_effect = [failed, succeeded]

    conn.close()
    assert key in conn._sparse_handles
    conn._req_client.close.assert_not_called()

    conn.close()
    assert key not in conn._sparse_handles
    assert conn._req_client.sparse_cancel_prefetch.call_count == 2
    conn._req_client.close.assert_called_once()
