# SPDX-License-Identifier: Apache-2.0
"""
Tests for the FREE_LOCKS protocol: enum registration, protocol definition,
message-queue round-trip, server handler, and client-side adapter API.
"""

# Standard
from unittest.mock import MagicMock, patch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
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
    """FREE_LOCKS should be a member of RequestType."""
    assert hasattr(RequestType, "FREE_LOCKS")
    assert isinstance(RequestType.FREE_LOCKS, RequestType)


def test_free_locks_payload_classes():
    """FREE_LOCKS payload should be a single list[IPCCacheEngineKey]."""
    payload_classes = get_payload_classes(RequestType.FREE_LOCKS)
    assert len(payload_classes) == 1
    assert payload_classes[0] == list[IPCCacheEngineKey]


def test_free_locks_response_class():
    """FREE_LOCKS should have no response (None)."""
    response_class = get_response_class(RequestType.FREE_LOCKS)
    assert response_class is None


def test_free_locks_handler_type():
    """FREE_LOCKS should use BLOCKING handler type."""
    handler_type = get_handler_type(RequestType.FREE_LOCKS)
    assert handler_type == HandlerType.BLOCKING


# ============================================================================
# Message-queue round-trip test
# ============================================================================


def test_mq_free_locks():
    """
    Test MessageQueue with FREE_LOCKS request type.
    FREE_LOCKS takes (keys: list[KeyType]) and returns None.
    """
    keys = [create_cache_key(i) for i in range(4)]

    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5570")
    helper.register_handler(
        RequestType.FREE_LOCKS, test_mq_handler_helpers.free_locks_handler
    )

    helper.run_test(
        request_type=RequestType.FREE_LOCKS,
        payloads=[keys],
        expected_response=None,
        num_requests=1,
    )


# ============================================================================
# Server handler tests
# ============================================================================


CHUNK_SIZE = 256


def _make_engine_mock():
    """Create a MagicMock that behaves like MPCacheEngine for free_locks tests."""
    engine = MagicMock()
    engine.token_hasher = MagicMock()
    engine.token_hasher.chunk_size = CHUNK_SIZE
    engine.storage_manager = MagicMock()
    return engine


def test_server_free_locks_calls_finish_read_prefetched():
    """MPCacheEngine.free_locks should resolve hash keys and call
    finish_read_prefetched on the storage manager."""
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = _make_engine_mock()
    key = create_cache_key(0).no_worker_id_version()
    hash_key = create_cache_key(0)
    sentinel_obj_keys = [MagicMock()]

    with (
        patch.object(IPCCacheEngineKey, "to_hash_keys", return_value=[hash_key]),
        patch(
            "lmcache.v1.multiprocess.server.ipc_keys_to_object_keys",
            return_value=sentinel_obj_keys,
        ) as mock_convert,
    ):
        MPCacheEngine.free_locks(engine, [key])

    mock_convert.assert_called_once()
    engine.storage_manager.finish_read_prefetched.assert_called_once_with(
        sentinel_obj_keys
    )


def test_server_free_locks_empty_keys():
    """MPCacheEngine.free_locks with empty keys should be a no-op."""
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = _make_engine_mock()

    MPCacheEngine.free_locks(engine, [])

    engine.storage_manager.finish_read_prefetched.assert_not_called()


def test_server_free_locks_multiple_keys():
    """MPCacheEngine.free_locks should expand each key via to_hash_keys."""
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = _make_engine_mock()
    keys = [create_cache_key(i).no_worker_id_version() for i in range(3)]
    hash_key = create_cache_key(0)
    sentinel_obj_keys = [MagicMock(), MagicMock(), MagicMock()]

    with (
        patch.object(IPCCacheEngineKey, "to_hash_keys", return_value=[hash_key]),
        patch(
            "lmcache.v1.multiprocess.server.ipc_keys_to_object_keys",
            return_value=sentinel_obj_keys,
        ),
    ):
        MPCacheEngine.free_locks(engine, keys)

    engine.storage_manager.finish_read_prefetched.assert_called_once_with(
        sentinel_obj_keys
    )


def test_server_free_locks_filters_by_start_end():
    """free_locks should only release hash keys within [start, end),
    not all hash keys expanded from token_ids."""
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = _make_engine_mock()

    # Simulate 4 chunks worth of tokens (1024 tokens at chunk_size=256)
    # but the key only covers chunks 1..3 (start=256, end=768)
    token_ids = tuple(range(1024))
    key = IPCCacheEngineKey(
        model_name="testmodel",
        world_size=1,
        worker_id=None,
        token_ids=token_ids,
        start=256,
        end=768,
        request_id="req-filter",
    )

    # to_hash_keys returns 4 hash keys (one per chunk over all token_ids)
    hash_keys = [MagicMock(name=f"hk{i}") for i in range(4)]
    sentinel_obj_keys = [MagicMock(name="obj1"), MagicMock(name="obj2")]

    with (
        patch.object(IPCCacheEngineKey, "to_hash_keys", return_value=hash_keys),
        patch(
            "lmcache.v1.multiprocess.server.ipc_keys_to_object_keys",
            return_value=sentinel_obj_keys,
        ) as mock_convert,
    ):
        MPCacheEngine.free_locks(engine, [key])

    # Only hash_keys[1] and hash_keys[2] should be passed (chunks 1 and 2)
    passed_ipc_keys = mock_convert.call_args[0][0]
    assert len(passed_ipc_keys) == 2
    assert passed_ipc_keys == hash_keys[1:3]

    engine.storage_manager.finish_read_prefetched.assert_called_once_with(
        sentinel_obj_keys
    )


def test_server_free_locks_unaligned_start_end():
    """When start or end is not aligned to chunk_size, the entire chunk
    containing that boundary should be freed."""
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = _make_engine_mock()

    # 4 chunks of tokens (1024 tokens at chunk_size=256).
    # start=100 falls in chunk 0, end=700 falls in chunk 2.
    # Expected freed chunks: 0, 1, 2  (indices [0:3])
    token_ids = tuple(range(1024))
    key = IPCCacheEngineKey(
        model_name="testmodel",
        world_size=1,
        worker_id=None,
        token_ids=token_ids,
        start=100,
        end=700,
        request_id="req-unaligned",
    )

    hash_keys = [MagicMock(name=f"hk{i}") for i in range(4)]
    sentinel_obj_keys = [MagicMock(name=f"obj{i}") for i in range(3)]

    with (
        patch.object(IPCCacheEngineKey, "to_hash_keys", return_value=hash_keys),
        patch(
            "lmcache.v1.multiprocess.server.ipc_keys_to_object_keys",
            return_value=sentinel_obj_keys,
        ) as mock_convert,
    ):
        MPCacheEngine.free_locks(engine, [key])

    # Chunks 0, 1, 2 should be freed (hash_keys[0:3])
    passed_ipc_keys = mock_convert.call_args[0][0]
    assert len(passed_ipc_keys) == 3
    assert passed_ipc_keys == hash_keys[0:3]

    engine.storage_manager.finish_read_prefetched.assert_called_once_with(
        sentinel_obj_keys
    )


def test_server_handler_registered():
    """run_cache_server should register a FREE_LOCKS handler."""
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = MPCacheEngine.__new__(MPCacheEngine)
    assert hasattr(engine, "free_locks")
    assert callable(engine.free_locks)


# ============================================================================
# Client adapter tests
# ============================================================================


def test_adapter_free_locks_sends_request():
    """LMCacheMPSchedulerAdapter.free_locks should send a FREE_LOCKS request
    with the correct key payload."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.world_size = 1
    adapter.worker_id = 0
    adapter.chunk_size = 256
    adapter.blocks_in_chunk = 16

    mock_client = MagicMock(spec=MessageQueueClient)
    mock_future = MagicMock()
    mock_client.submit_request.return_value = mock_future
    adapter.mq_client = mock_client
    adapter.lookup_futures = {}

    token_ids = list(range(512))
    adapter.free_locks(
        token_ids=token_ids,
        start=0,
        end=512,
        request_id="req-1",
    )

    mock_client.submit_request.assert_called_once()
    call_args = mock_client.submit_request.call_args
    req_type = call_args[0][0]
    payloads = call_args[0][1]
    assert req_type == RequestType.FREE_LOCKS

    # Payload should be a list containing a single-element list of keys
    assert isinstance(payloads, list)
    assert len(payloads) == 1
    key_list = payloads[0]
    assert isinstance(key_list, list)
    assert len(key_list) == 1

    key = key_list[0]
    assert isinstance(key, IPCCacheEngineKey)
    assert key.worker_id is None
    assert key.model_name == "test_model"
    assert key.request_id == "req-1"


def test_adapter_free_locks_key_matches_lookup():
    """The key created by free_locks should match the key created by
    maybe_submit_lookup_request (no_worker_id_version, same start/end)."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.world_size = 1
    adapter.worker_id = 0
    adapter.chunk_size = 256
    adapter.blocks_in_chunk = 16

    mock_client = MagicMock(spec=MessageQueueClient)
    mock_future = MagicMock()
    mock_future.query.return_value = False
    mock_client.submit_request.return_value = mock_future
    adapter.mq_client = mock_client
    adapter.lookup_futures = {}

    token_ids = list(range(512))

    # Submit lookup
    adapter.maybe_submit_lookup_request("req-1", token_ids)
    lookup_call = mock_client.submit_request.call_args
    lookup_payloads = lookup_call[0][1]
    # Upstream lookup sends a single key: payloads = [key]
    lookup_key = lookup_payloads[0]

    mock_client.submit_request.reset_mock()

    # Submit free_locks with same range as lookup
    aligned_end = (len(token_ids) // adapter.chunk_size) * adapter.chunk_size
    adapter.free_locks(
        token_ids=token_ids,
        start=0,
        end=aligned_end,
        request_id="req-1",
    )
    free_call = mock_client.submit_request.call_args
    free_payloads = free_call[0][1]
    # free_locks sends: payloads = [[key]]
    free_key = free_payloads[0][0]

    # Keys should be identical
    assert lookup_key.model_name == free_key.model_name
    assert lookup_key.world_size == free_key.world_size
    assert lookup_key.worker_id == free_key.worker_id
    assert lookup_key.worker_id is None
    assert lookup_key.start == free_key.start
    assert lookup_key.end == free_key.end
    assert lookup_key.request_id == free_key.request_id
    assert lookup_key.token_ids == free_key.token_ids
