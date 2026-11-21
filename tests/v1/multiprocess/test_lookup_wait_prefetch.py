# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``WAIT_PREFETCH_STATUS`` request handler path.

The prefetch-controller tests cover watch_prefetch_result in isolation;
these cover the ``LookupModule`` handler that ``WAIT_PREFETCH_STATUS`` dispatches
to, i.e. the ``wait_prefetch_status -> query_prefetch_status`` path: count
computation, event emission, and exactly-once job consumption. The storage
manager is mocked, so no GPU or native bitmap is needed.
"""

# Standard
from concurrent.futures import Future
from unittest import mock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    MemoryLayoutDesc,
    PrefetchHandle,
    PrefetchResult,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    L2AdaptersConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import l1_owners_of
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lookup import (
    LookupModule,
    _PrefetchJob,
    _WaitDeadlines,
)


def _result(found):
    """A finished result whose hits all came from L1, or None."""
    if found is None:
        return None
    return PrefetchResult(
        hit_cells=found,
        l1_hit_cells=[row.copy() for row in found],
        l2_hit_cells=[Bitmap(len(row)) for row in found],
    )


def _make_ctx(wait_result=True, found=None):
    storage_manager = mock.Mock()
    waited: Future[bool] = Future()
    waited.set_result(wait_result)
    storage_manager.watch_prefetch_status.return_value = waited
    storage_manager.query_prefetch_status.return_value = _result(found)
    ctx = mock.Mock()
    ctx.storage_manager = storage_manager
    ctx.event_bus = mock.Mock()
    ctx.chunk_size = 256
    return ctx


def _make_module(ctx):
    # Bypass __init__ (which wires up otel metrics needing a full context); the
    # handler methods only touch _ctx, _prefetch_jobs, _prefetch_job_lock, and
    # _wait_deadlines.
    module = object.__new__(LookupModule)
    module._ctx = ctx
    module._prefetch_jobs = {}
    module._prefetch_job_lock = threading.Lock()
    module._wait_deadlines = _WaitDeadlines()
    return module


def test_wait_prefetch_status_returns_count_and_consumes_job():
    # 8 keys = 4 chunks with world_size=2, 1 object group: one 4-chunk row
    # per kv rank. All bits set -> fold_unfold_grouped returns hit_length=4.
    num_chunks, world_size = 4, 2
    found = [Bitmap(num_chunks, num_chunks) for _ in range(world_size)]
    handle = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req",
        total_requested_keys=num_chunks * world_size,
        submit_time=0.0,
        sliding_windows=(-1,) * world_size,
    )
    ctx = _make_ctx(wait_result=True, found=found)
    module = _make_module(ctx)
    module._prefetch_jobs["req"] = _PrefetchJob(
        handle=handle,
        row_windows=(-1,) * world_size,
        request_id="req",
        requested_tokens=512,
        chunk_size=256,
    )

    reply = module.wait_prefetch_status("req", timeout=1.0)
    assert reply.result(timeout=0) == 4 * ctx.chunk_size
    ctx.storage_manager.watch_prefetch_status.assert_called_once_with(handle)
    ctx.event_bus.publish.assert_called_once()
    # Exactly-once: the job is removed after a non-None result.
    assert "req" not in module._prefetch_jobs
    # The hit length is recorded on the session so free_lookup_locks can
    # later reconstruct which keys the prefetch read-locked.
    ctx.session_manager.get_or_create.assert_called_once_with("req")
    session = ctx.session_manager.get_or_create.return_value
    session.record_prefetch_result.assert_called_once_with(4, (0,), {})


def test_wait_prefetch_status_timeout_returns_none_and_keeps_job():
    ctx = _make_ctx(wait_result=False)
    module = _make_module(ctx)
    job = _PrefetchJob(
        handle=mock.sentinel.handle,
        row_windows=(-1,),
        request_id="req",
        requested_tokens=0,
        chunk_size=256,
    )
    module._prefetch_jobs["req"] = job

    assert module.wait_prefetch_status("req", timeout=0.5).result(timeout=0) is None
    ctx.storage_manager.query_prefetch_status.assert_not_called()
    # Job is kept so a later wait/query can still resolve it.
    assert module._prefetch_jobs["req"] is job


def test_wait_prefetch_status_unknown_request_returns_zero():
    module = _make_module(_make_ctx())
    assert module.wait_prefetch_status("missing", timeout=1.0).result(timeout=0) == 0


def _add_job(module, request_id="req"):
    module._prefetch_jobs[request_id] = _PrefetchJob(
        handle=mock.sentinel.handle,
        row_windows=(-1,),
        request_id=request_id,
        requested_tokens=0,
        chunk_size=256,
    )


def _make_pending_ctx():
    """A context whose prefetch result is published when the test says."""
    ctx = _make_ctx()
    published: Future[bool] = Future()
    ctx.storage_manager.watch_prefetch_status.return_value = published
    return ctx, published


def test_wait_prefetch_status_failed_answer_fails_the_reply():
    ctx = _make_ctx(wait_result=True)
    ctx.storage_manager.query_prefetch_status.side_effect = RuntimeError("boom")
    module = _make_module(ctx)
    _add_job(module)

    with pytest.raises(RuntimeError, match="boom"):
        module.wait_prefetch_status("req", timeout=1.0).result(timeout=0)


def test_wait_prefetch_status_cancelled_reply_keeps_job():
    ctx, published = _make_pending_ctx()
    module = _make_module(ctx)
    _add_job(module)

    assert module.wait_prefetch_status("req", timeout=1.0).ready.cancel()
    assert published.cancelled()
    ctx.storage_manager.query_prefetch_status.assert_not_called()
    assert "req" in module._prefetch_jobs


def test_a_shorter_wait_added_later_times_out_at_its_own_deadline():
    """A wait added after a longer one still answers None on time."""
    ctx = _make_ctx()
    ctx.storage_manager.watch_prefetch_status.side_effect = lambda handle: Future()
    module = _make_module(ctx)
    _add_job(module, "slow")
    _add_job(module, "fast")

    slow = module.wait_prefetch_status("slow", timeout=30.0)
    # Let the deadline thread go to sleep on the 30 s deadline first.
    time.sleep(0.05)
    fast = module.wait_prefetch_status("fast", timeout=0.05)
    assert fast.result(timeout=5.0) is None
    assert not slow.ready.done()
    module.close()


def test_end_session_drops_the_job_a_pending_wait_would_consume():
    ctx, published = _make_pending_ctx()
    ctx.session_manager.remove.return_value = None
    module = _make_module(ctx)
    _add_job(module)

    reply = module.wait_prefetch_status("req", timeout=1.0)
    module.end_session("req")
    published.set_result(True)
    assert reply.result(timeout=0) == 0
    ctx.session_manager.get_or_create.assert_not_called()


def test_wait_prefetch_status_against_real_prefetch_controller():
    """wait_prefetch_status resolves in-flight, already-published, early-exit,
    and timed-out-then-published prefetches against the real controller."""
    chunk_size = 256
    num_chunks = 4
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([16384])],
        dtypes=[torch.uint8],
    )
    ctx = MPCacheServerContext(
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=16 * 1024 * 1024,
                    use_lazy=False,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[
                    MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=0.0004)
                ]
            ),
        ),
        chunk_size=chunk_size,
    )
    ctx.layout_desc_registry.register(
        "m",
        1,
        layout,
        chunk_size=chunk_size,
        hash_algorithm=ctx.hash_algorithm_name,
    )
    module = LookupModule(ctx)
    sm = ctx.storage_manager
    adapter = sm.l2_adapters()[0][1]
    assert isinstance(adapter, MockL2Adapter)
    tokens = list(range(num_chunks * chunk_size))

    def make_key(request_id: str, start: int = 0, worker_id: int | None = None):
        return IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=worker_id,
            token_ids=tokens,
            start=start,
            end=len(tokens),
            request_id=request_id,
        )

    worker_key = make_key("seed", worker_id=0)
    hashes = ctx.token_hasher.compute_packed_chunk_hashes(
        worker_key.token_bytes, end=worker_key.end
    )
    obj_keys = ipc_key_to_object_keys(
        worker_key, hashes, [0], DEFAULT_SHARD_REPLICATION_DESC
    )[0]

    try:
        reserved = sm.reserve_write(obj_keys, layout)
        sm.finish_write(list(reserved), l1_owners_of(reserved))
        deadline = time.monotonic() + 5.0
        l1 = sm.l1_managers()[0]
        while not all(
            adapter.debug_has_key(k) and l1.is_key_evictable(k) for k in obj_keys
        ):
            assert time.monotonic() < deadline
            time.sleep(0.01)

        # 1. Load in flight: evict L1 so the prefetch loads from L2.
        assert sm.delete_l1_keys(obj_keys) == (num_chunks, 0)
        in_flight_key = make_key("in-flight")
        module.lookup(in_flight_key, tp_size=1)
        deferred = module.wait_prefetch_status("in-flight", timeout=5.0)
        assert not deferred.ready.done()
        assert deferred.result(timeout=5.0) == len(tokens)
        module.free_lookup_locks(in_flight_key, tp_size=1)
        module.end_session("in-flight")

        # 2. Already published: keys are still in L1 (and the -1 handle path
        # when all chunks are engine-covered).
        published_key = make_key("published")
        module.lookup(published_key, tp_size=1)
        # L1 hits still go through the prefetch loop; let it publish first.
        assert sm.wait_prefetch_status(
            module._prefetch_jobs["published"].handle, timeout=5.0
        )
        ready_deferred = module.wait_prefetch_status("published", timeout=5.0)
        assert ready_deferred.ready.done()
        assert ready_deferred.result(timeout=1.0) == len(tokens)
        module.free_lookup_locks(published_key, tp_size=1)
        module.end_session("published")

        covered_key = make_key("covered", start=len(tokens))
        module.lookup(covered_key, tp_size=1)
        assert module.wait_prefetch_status("covered", timeout=0.1).result(1.0) == len(
            tokens
        )
        module.end_session("covered")

        # 3. Timed out, then published and consumed by the next wait.
        assert sm.delete_l1_keys(obj_keys) == (num_chunks, 0)
        timeout_key = make_key("timeout-then-hit")
        module.lookup(timeout_key, tp_size=1)
        assert (
            module.wait_prefetch_status("timeout-then-hit", timeout=0.02).result(2.0)
            is None
        )
        assert module.wait_prefetch_status("timeout-then-hit", timeout=5.0).result(
            5.0
        ) == len(tokens)
        assert (
            module.wait_prefetch_status("timeout-then-hit", timeout=0.1).result(1.0)
            == 0
        )
        module.free_lookup_locks(timeout_key, tp_size=1)
        module.end_session("timeout-then-hit")
    finally:
        module.close()
        ctx.close()
