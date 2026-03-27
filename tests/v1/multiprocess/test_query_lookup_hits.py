# SPDX-License-Identifier: Apache-2.0
"""
Tests for the QUERY_PREFETCH_LOOKUP_HITS protocol: enum registration,
protocol definition, message-queue round-trip, and server handler.
"""

# Standard
from unittest.mock import MagicMock
import threading
import time

# First Party
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType
from lmcache.v1.multiprocess.server import MPCacheEngine, _PrefetchJob

# Test helpers
from tests.v1.multiprocess.test_mq import (
    MessageQueueTestHelper,
)

# ============================================================================
# Protocol definition tests
# ============================================================================


def test_query_prefetch_lookup_hits_in_request_type():
    """QUERY_PREFETCH_LOOKUP_HITS should be a member of RequestType."""
    assert hasattr(RequestType, "QUERY_PREFETCH_LOOKUP_HITS")
    assert isinstance(RequestType.QUERY_PREFETCH_LOOKUP_HITS, RequestType)


def test_query_prefetch_lookup_hits_payload_classes():
    """QUERY_PREFETCH_LOOKUP_HITS payload should be [int]."""
    payload_classes = get_payload_classes(RequestType.QUERY_PREFETCH_LOOKUP_HITS)
    assert len(payload_classes) == 1
    assert payload_classes[0] is int


def test_query_prefetch_lookup_hits_response_class():
    """QUERY_PREFETCH_LOOKUP_HITS response should be int | None."""
    response_class = get_response_class(RequestType.QUERY_PREFETCH_LOOKUP_HITS)
    assert response_class == int | None


def test_query_prefetch_lookup_hits_handler_type():
    """QUERY_PREFETCH_LOOKUP_HITS should use BLOCKING handler type."""
    handler_type = get_handler_type(RequestType.QUERY_PREFETCH_LOOKUP_HITS)
    assert handler_type == HandlerType.BLOCKING


# ============================================================================
# Message-queue round-trip test
# ============================================================================


def _query_lookup_hits_handler(job_id: int) -> int | None:
    """Dummy handler for QUERY_PREFETCH_LOOKUP_HITS requests."""
    assert isinstance(job_id, int)
    return 42


def test_mq_query_prefetch_lookup_hits():
    """Test MessageQueue with QUERY_PREFETCH_LOOKUP_HITS request type."""
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5575")
    helper.register_handler(
        RequestType.QUERY_PREFETCH_LOOKUP_HITS, _query_lookup_hits_handler
    )

    helper.run_test(
        request_type=RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        payloads=[1],
        expected_response=42,
        num_requests=1,
    )


def _query_lookup_hits_none_handler(job_id: int) -> int | None:
    """Dummy handler that returns None (lookup still in progress)."""
    assert isinstance(job_id, int)
    return None


def test_mq_query_prefetch_lookup_hits_none_response():
    """Test MessageQueue returns None when lookup is still in progress."""
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5576")
    helper.register_handler(
        RequestType.QUERY_PREFETCH_LOOKUP_HITS, _query_lookup_hits_none_handler
    )

    helper.run_test(
        request_type=RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        payloads=[1],
        expected_response=None,
        num_requests=1,
    )


# ============================================================================
# Server handler tests
# ============================================================================


def _make_engine_with_job(
    world_size: int, storage_return: int | None
) -> tuple[MagicMock, int]:
    """Create a mock MPCacheEngine with a single prefetch job.

    Returns:
        (engine_mock, job_id)
    """
    engine = MagicMock()
    engine._prefetch_job_lock = threading.Lock()

    handle = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req-0",
        l1_prefix_hit_count=0,
        total_requested_keys=10,
        submit_time=time.monotonic(),
    )
    job = _PrefetchJob(handle=handle, world_size=world_size, request_id="req-1")
    job_id = 42
    engine._prefetch_jobs = {job_id: job}
    engine.storage_manager.query_prefetch_lookup_hits.return_value = storage_return

    return engine, job_id


def test_server_lookup_hits_returns_count():
    """query_prefetch_lookup_hits returns chunk count when lookup is done."""
    engine, job_id = _make_engine_with_job(world_size=1, storage_return=5)

    result = MPCacheEngine.query_prefetch_lookup_hits(engine, job_id)

    assert result == 5
    engine.storage_manager.query_prefetch_lookup_hits.assert_called_once()


def test_server_lookup_hits_divides_by_world_size():
    """Result should be divided by world_size for tensor parallelism."""
    engine, job_id = _make_engine_with_job(world_size=2, storage_return=10)

    result = MPCacheEngine.query_prefetch_lookup_hits(engine, job_id)

    assert result == 5  # 10 // 2


def test_server_lookup_hits_returns_none_when_in_progress():
    """Returns None when storage manager lookup is still in progress."""
    engine, job_id = _make_engine_with_job(world_size=1, storage_return=None)

    result = MPCacheEngine.query_prefetch_lookup_hits(engine, job_id)

    assert result is None


def test_server_lookup_hits_returns_none_for_invalid_job():
    """Returns None for a job ID that doesn't exist."""
    engine = MagicMock()
    engine._prefetch_job_lock = threading.Lock()
    engine._prefetch_jobs = {}

    result = MPCacheEngine.query_prefetch_lookup_hits(engine, 999)

    assert result is None
    engine.storage_manager.query_prefetch_lookup_hits.assert_not_called()


def test_server_lookup_hits_returns_none_after_prefetch_consumed():
    """Returns None after query_prefetch_status has consumed the job."""
    engine, job_id = _make_engine_with_job(world_size=1, storage_return=5)

    # Simulate query_prefetch_status consuming the job
    del engine._prefetch_jobs[job_id]

    result = MPCacheEngine.query_prefetch_lookup_hits(engine, job_id)

    assert result is None


def test_server_lookup_hits_zero_count():
    """Returns 0 when no keys matched (not None)."""
    engine, job_id = _make_engine_with_job(world_size=1, storage_return=0)

    result = MPCacheEngine.query_prefetch_lookup_hits(engine, job_id)

    assert result == 0
    assert result is not None


def test_server_handler_registered():
    """MPCacheEngine should have a query_prefetch_lookup_hits method."""
    engine = MPCacheEngine.__new__(MPCacheEngine)
    assert hasattr(engine, "query_prefetch_lookup_hits")
    assert callable(engine.query_prefetch_lookup_hits)
