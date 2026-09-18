# SPDX-License-Identifier: Apache-2.0
"""MP LookupModule integration for the L2 load-deadline fallback.

The deadline fallback reaches this layer as an ordinary non-``None`` result
bitmap, so ``query_prefetch_status`` / ``wait_prefetch_status`` must return a
normal final chunk count and remove the ``_PrefetchJob`` exactly once — the same
contract as a normal completion, distinct from the caller-side wait timeout
(which returns ``None`` and keeps the job). The storage manager is mocked, so no
GPU or native bitmap allocation is needed.
"""

# Standard
from unittest import mock
import threading

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import PrefetchHandle
from lmcache.v1.multiprocess.modules.lookup import LookupModule, _PrefetchJob


def _make_ctx(wait_result=True, found=None):
    storage_manager = mock.Mock()
    storage_manager.wait_prefetch_status.return_value = wait_result
    storage_manager.query_prefetch_status.return_value = found
    ctx = mock.Mock()
    ctx.storage_manager = storage_manager
    ctx.event_bus = mock.Mock()
    ctx.chunk_size = 256
    return ctx


def _make_module(ctx):
    module = object.__new__(LookupModule)
    module._ctx = ctx
    module._prefetch_jobs = {}
    module._prefetch_job_lock = threading.Lock()
    return module


def _handle(num_keys):
    return PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=num_keys,
        submit_time=0.0,
    )


def _register(module, handle, world_size=2):
    module._prefetch_jobs["req"] = _PrefetchJob(
        handle=handle,
        world_size=world_size,
        request_id="req",
        requested_tokens=world_size * 256,
    )


def test_deadline_fallback_returns_partial_count_and_removes_job():
    # 8 keys = 4 chunks (world_size=2, 1 group). A deadline fallback that
    # retained only the first 2 chunks sets the leading 4 bits.
    num_keys = 8
    found = Bitmap(num_keys)
    for i in range(4):
        found.set(i)
    ctx = _make_ctx(wait_result=True, found=found)
    module = _make_module(ctx)
    _register(module, _handle(num_keys))

    # Query returns the retained chunk count (2), not None, and pops the job.
    assert module.query_prefetch_status("req") == 2
    assert "req" not in module._prefetch_jobs
    ctx.event_bus.publish.assert_called()  # MP_LOOKUP_PREFETCH_END emitted


def test_deadline_fallback_via_wait_returns_count_and_removes_job():
    num_keys = 8
    found = Bitmap(num_keys)
    for i in range(4):
        found.set(i)
    ctx = _make_ctx(wait_result=True, found=found)
    module = _make_module(ctx)
    _register(module, _handle(num_keys))

    # wait_prefetch_status sees the published fallback (storage wait True) and
    # resolves to the normal chunk count, removing the job exactly once.
    assert module.wait_prefetch_status("req", timeout=1.0) == 2
    ctx.storage_manager.wait_prefetch_status.assert_called_once()
    assert "req" not in module._prefetch_jobs


def test_zero_retained_fallback_returns_zero_and_removes_job():
    # A deadline that retained nothing (e.g. timeout in lookup with no L1 hit)
    # still resolves to a normal final status of 0 chunks and removes the job,
    # so the engine recomputes everything -- it does not hang.
    num_keys = 8
    ctx = _make_ctx(wait_result=True, found=Bitmap(num_keys))
    module = _make_module(ctx)
    _register(module, _handle(num_keys))

    assert module.query_prefetch_status("req") == 0
    assert "req" not in module._prefetch_jobs
