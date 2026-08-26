# SPDX-License-Identifier: Apache-2.0
"""
Tests for the FREE_LOOKUP_LOCKS protocol: enum registration, protocol definition,
message-queue round-trip, server handler, and client-side adapter API.
"""

# Standard
from unittest.mock import MagicMock, patch
import threading

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType

# Test helpers
from tests.v1.multiprocess import test_mq_handler_helpers
from tests.v1.multiprocess.test_mq import (
    MessageQueueTestHelper,
    create_cache_key,
)

# ============================================================================
# Protocol definition tests
# ============================================================================


def test_free_locks_in_request_type():
    """FREE_LOOKUP_LOCKS should be a member of RequestType."""
    assert hasattr(RequestType, "FREE_LOOKUP_LOCKS")
    assert isinstance(RequestType.FREE_LOOKUP_LOCKS, RequestType)


def test_free_locks_payload_classes():
    """FREE_LOOKUP_LOCKS payload should be [IPCCacheServerKey, int]."""
    payload_classes = get_payload_classes(RequestType.FREE_LOOKUP_LOCKS)
    assert len(payload_classes) == 2
    assert payload_classes[0] is IPCCacheServerKey
    assert payload_classes[1] is int


def test_free_locks_response_class():
    """FREE_LOOKUP_LOCKS should have no response (None)."""
    response_class = get_response_class(RequestType.FREE_LOOKUP_LOCKS)
    assert response_class is None


def test_free_locks_handler_type():
    """FREE_LOOKUP_LOCKS should use BLOCKING handler type."""
    handler_type = get_handler_type(RequestType.FREE_LOOKUP_LOCKS)
    assert handler_type == HandlerType.BLOCKING


# ============================================================================
# Message-queue round-trip test
# ============================================================================


def test_mq_free_locks():
    """
    Test MessageQueue with FREE_LOOKUP_LOCKS request type.
    FREE_LOOKUP_LOCKS takes (key: KeyType) and returns None.
    """
    key = create_cache_key(0)

    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5570")
    helper.register_handler(
        RequestType.FREE_LOOKUP_LOCKS, test_mq_handler_helpers.free_locks_handler
    )

    helper.run_test(
        request_type=RequestType.FREE_LOOKUP_LOCKS,
        payloads=[key, 1],
        expected_response=None,
        num_requests=1,
    )


# ============================================================================
# Server handler tests
# ============================================================================


def _make_free_locks_ctx(
    chunk_hashes: list[bytes],
    windows: list[int],
    hit_chunks: int,
    locked_gids: tuple = (),
) -> MagicMock:
    """Build a mock engine context for free_lookup_locks tests.

    Args:
        chunk_hashes: Hashes returned for the freed [start, end) range.
        windows: Per-object-group windows for the registered AttnWindowDesc.
        hit_chunks: Prefetch hit length recorded on the session (-1 for
            "never recorded").
        locked_gids: The lock model recorded on the session; empty means
            "every group" (legacy std-lookup behavior).

    Returns:
        The configured MagicMock context.
    """
    ctx = MagicMock()
    ctx.chunk_size = 256
    ctx.token_hasher.chunk_size = 256
    ctx.token_hasher.compute_chunk_hashes.return_value = chunk_hashes
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=windows
    )
    session = ctx.session_manager.get_or_create.return_value
    session.prefetch_hit_chunks = hit_chunks
    session.prefetch_locked_gids = tuple(locked_gids)
    return ctx


def test_server_free_lookup_locks_calls_finish_read_prefetched():
    """LookupModule.free_lookup_locks should resolve hash keys and call
    finish_read_prefetched on the storage manager."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    ctx = _make_free_locks_ctx([b"hash0"], windows=[-1], hit_chunks=1)

    module = LookupModule(ctx)

    # Build a key
    key = create_cache_key(0).no_worker_id_version()

    sentinel_obj_keys = [MagicMock()]
    with patch(
        "lmcache.v1.multiprocess.modules.lookup.ipc_key_to_object_keys",
        return_value=[sentinel_obj_keys],
    ):
        module.free_lookup_locks(key, 1)

    module.context.storage_manager.finish_read_prefetched.assert_called_once_with(
        sentinel_obj_keys, read_locks=1
    )


def _free_locks_key(num_tokens: int, start: int, end: int) -> IPCCacheServerKey:
    """Build a lookup-style (worker_id=None) key over [start, end)."""
    return IPCCacheServerKey(
        model_name="testmodel",
        world_size=1,
        num_kv_readers=1,
        worker_id=None,
        token_ids=tuple(range(num_tokens)),
        start=start,
        end=end,
        request_id="req-sw",
    )


def _released_chunks(finish_read_mock: MagicMock) -> set[tuple[int, bytes]]:
    """Collect (object_group_id, chunk_hash) pairs released by the module."""
    (obj_keys,), kwargs = finish_read_mock.call_args
    assert kwargs == {"read_locks": 1}
    return {(k.object_group_id, k.chunk_hash) for k in obj_keys}


def test_server_free_lookup_locks_sliding_window_skips_unlocked_chunks():
    """A sliding-window group must release only its locked suffix.

    Hit of 4 chunks, windows [1, -1], freed range [0, 3): the window group
    locked only chunk 3, which is outside the range, so only the
    full-attention group's chunks 0-2 are released.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    hashes = [b"h0", b"h1", b"h2"]
    ctx = _make_free_locks_ctx(hashes, windows=[1, -1], hit_chunks=4)
    module = LookupModule(ctx)

    module.free_lookup_locks(_free_locks_key(1024, start=0, end=768), 1)

    released = _released_chunks(ctx.storage_manager.finish_read_prefetched)
    assert released == {(1, b"h0"), (1, b"h1"), (1, b"h2")}


def test_server_free_lookup_locks_sliding_window_releases_locked_suffix():
    """Freeing the whole hit range releases the window group's suffix.

    Hit of 4 chunks, windows [1, -1], freed range [0, 4): the window group
    releases chunk 3, the full-attention group releases chunks 0-3.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    hashes = [b"h0", b"h1", b"h2", b"h3"]
    ctx = _make_free_locks_ctx(hashes, windows=[1, -1], hit_chunks=4)
    module = LookupModule(ctx)

    module.free_lookup_locks(_free_locks_key(1024, start=0, end=1024), 1)

    released = _released_chunks(ctx.storage_manager.finish_read_prefetched)
    assert released == {
        (0, b"h3"),
        (1, b"h0"),
        (1, b"h1"),
        (1, b"h2"),
        (1, b"h3"),
    }


def test_server_free_lookup_locks_caps_release_at_hit_length():
    """Chunks beyond the hit length were never locked and must not be freed."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    hashes = [b"h0", b"h1", b"h2", b"h3"]
    ctx = _make_free_locks_ctx(hashes, windows=[-1], hit_chunks=2)
    module = LookupModule(ctx)

    module.free_lookup_locks(_free_locks_key(1024, start=0, end=1024), 1)

    released = _released_chunks(ctx.storage_manager.finish_read_prefetched)
    assert released == {(0, b"h0"), (0, b"h1")}


def test_server_free_lookup_locks_unknown_hit_skips_window_groups():
    """Without a recorded hit length, window groups release nothing.

    Full-attention groups keep the legacy full-range release; a
    sliding-window group's locked suffix is unknown, so it is skipped
    (leaked locks expire with the TTL, over-releasing could strip a
    concurrent reader's lock).
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    hashes = [b"h0", b"h1", b"h2"]
    ctx = _make_free_locks_ctx(hashes, windows=[1, -1], hit_chunks=-1)
    module = LookupModule(ctx)

    module.free_lookup_locks(_free_locks_key(1024, start=0, end=768), 1)

    released = _released_chunks(ctx.storage_manager.finish_read_prefetched)
    assert released == {(1, b"h0"), (1, b"h1"), (1, b"h2")}


def test_server_free_lookup_locks_no_matching_chunks():
    """LookupModule.free_lookup_locks with no chunks in range should be a no-op."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 256
    ctx.token_hasher.compute_chunk_hashes.return_value = []

    module = LookupModule(ctx)

    # Key with start == end means no chunks to free
    key = IPCCacheServerKey(
        model_name="testmodel",
        world_size=1,
        num_kv_readers=1,
        worker_id=None,
        token_ids=tuple(range(256)),
        start=0,
        end=0,
        request_id="req-empty",
    )

    module.free_lookup_locks(key, 1)

    module.context.storage_manager.finish_read_prefetched.assert_not_called()


def test_server_handler_registered():
    """LookupModule should have a free_lookup_locks method."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    assert hasattr(LookupModule, "free_lookup_locks")
    assert callable(LookupModule.free_lookup_locks)


# ============================================================================
# Client adapter tests
# ============================================================================


def test_adapter_free_lookup_locks_sends_request():
    """LMCacheMPSchedulerAdapter.free_lookup_locks should send a FREE_LOOKUP_LOCKS
    request with the correct key payload."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        ParallelStrategy,
    )

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.lmcache_tokens_per_chunk = 256
    adapter.blocks_in_chunk = 16
    adapter.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 1, 1)
    adapter._health_events = {"tcp://test:0": threading.Event()}
    adapter._health_events["tcp://test:0"].set()
    adapter._server_urls = ["tcp://test:0"]
    adapter._mq_timeout = 30.0

    mock_client = MagicMock(spec=MessageQueueClient)
    mock_future = MagicMock()
    mock_client.submit_request.return_value = mock_future
    adapter.mq_clients = {"tcp://test:0": mock_client}
    adapter._pending_lookups = set()

    token_ids = list(range(512))
    adapter.free_lookup_locks(
        token_ids=token_ids,
        start=0,
        end=512,
        request_id="req-1",
    )

    mock_client.submit_request.assert_called_once()
    call_args = mock_client.submit_request.call_args
    req_type = call_args[0][0]
    payloads = call_args[0][1]
    assert req_type == RequestType.FREE_LOOKUP_LOCKS

    # Payload should be [key, tp_size]
    assert isinstance(payloads, list)
    assert len(payloads) == 2

    key = payloads[0]
    assert isinstance(key, IPCCacheServerKey)
    assert key.worker_id is None
    assert key.model_name == "test_model"
    assert key.request_id == "req-1"
    assert payloads[1] == 1  # tp_size


def test_adapter_free_lookup_locks_key_matches_lookup():
    """The key created by free_lookup_locks should match the key created by
    maybe_submit_lookup_request (no_worker_id_version, same start/end)."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        ParallelStrategy,
    )

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.lmcache_tokens_per_chunk = 256
    adapter.blocks_in_chunk = 16
    adapter.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 1, 1)
    adapter._server_urls = ["tcp://test:0"]
    adapter._health_events = {"tcp://test:0": threading.Event()}
    adapter._health_events["tcp://test:0"].set()
    adapter._mq_timeout = 30.0
    adapter._heartbeats: dict[str, object] = {}
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeat_interval = 5.0

    mock_client = MagicMock(spec=MessageQueueClient)
    mock_future = MagicMock()
    mock_future.result.return_value = None  # LOOKUP returns None
    mock_client.submit_request.return_value = mock_future
    adapter.mq_clients = {"tcp://test:0": mock_client}
    adapter._pending_lookups = set()
    adapter._lookup_params = {}

    token_ids = list(range(512))

    # Submit lookup – patch heartbeat to avoid spawning a real thread
    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.maybe_submit_lookup_request("req-1", token_ids)
    lookup_call = mock_client.submit_request.call_args
    lookup_payloads = lookup_call[0][1]
    lookup_key = lookup_payloads[0]

    mock_client.submit_request.reset_mock()

    # Submit free_lookup_locks with aligned end
    tokens_per_chunk = adapter.lmcache_tokens_per_chunk
    aligned_end = (len(token_ids) // tokens_per_chunk) * tokens_per_chunk
    adapter.free_lookup_locks(
        token_ids=token_ids,
        start=0,
        end=aligned_end,
        request_id="req-1",
    )
    free_call = mock_client.submit_request.call_args
    free_payloads = free_call[0][1]
    assert len(free_payloads) == 2
    free_key = free_payloads[0]
    assert free_payloads[1] == 1  # tp_size

    # Keys should be identical
    assert lookup_key.model_name == free_key.model_name
    assert lookup_key.world_size == free_key.world_size
    assert lookup_key.worker_id == free_key.worker_id
    assert lookup_key.worker_id is None
    assert lookup_key.start == free_key.start
    assert lookup_key.end == free_key.end
    assert lookup_key.request_id == free_key.request_id
    assert lookup_key.token_ids == free_key.token_ids


def test_server_free_lookup_locks_honors_the_session_lock_model():
    """A prefetch that locked only a subset of groups (the CB prefix leg:
    recurrent + attention, never aux) must release exactly that subset --
    releasing an unlocked group would drop another request's lock on the
    shared object key."""
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    hashes = [b"h0", b"h1", b"h2"]
    ctx = _make_free_locks_ctx(
        hashes, windows=[1, -1, -1], hit_chunks=3, locked_gids=(0, 1)
    )
    module = LookupModule(ctx)
    key = _free_locks_key(768, 0, 768)

    with patch(
        "lmcache.v1.multiprocess.modules.lookup.ipc_key_to_object_keys",
        side_effect=lambda k, hs, gids: [[f"g{gids[0]}-{h.decode()}" for h in hs]],
    ):
        module.free_lookup_locks(key, 1)

    released = ctx.storage_manager.finish_read_prefetched.call_args[0][0]
    # Group 0 (recurrent, window 1): only the boundary chunk. Group 1
    # (attention): the whole hit prefix. Group 2 (standalone aux): NOTHING.
    assert released == ["g0-h2", "g1-h0", "g1-h1", "g1-h2"]
