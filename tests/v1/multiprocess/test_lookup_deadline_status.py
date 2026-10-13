# SPDX-License-Identifier: Apache-2.0
"""Lookup status reports deadline fallback hits as a final chunk count."""

# Standard
from unittest import mock
import threading

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import PrefetchHandle, PrefetchResult
from lmcache.v1.multiprocess.modules.lookup import LookupModule, _PrefetchJob


def _make_ctx(wait_result=True, found=None):
    storage_manager = mock.Mock()
    storage_manager.wait_prefetch_status.return_value = wait_result
    storage_manager.query_prefetch_status.return_value = (
        PrefetchResult(
            hit_cells=found,
            l1_hit_cells=[Bitmap(len(row)) for row in found],
            l2_hit_cells=[row.copy() for row in found],
        )
        if found is not None
        else None
    )
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
        total_requested_keys=num_keys,
        submit_time=0.0,
        sliding_windows=(-1, -1),
    )


def _register(module, handle, world_size=2):
    module._prefetch_jobs["req"] = _PrefetchJob(
        handle=handle,
        row_windows=(-1,) * world_size,
        request_id="req",
        requested_tokens=world_size * 256,
    )


@pytest.mark.parametrize("via_wait, retained", [(False, 2), (True, 2), (False, 0)])
def test_deadline_fallback_returns_count_and_removes_job(via_wait, retained):
    ctx = _make_ctx(found=[Bitmap(4, retained) for _ in range(2)])
    module = _make_module(ctx)
    _register(module, _handle(8))

    if via_wait:
        assert module.wait_prefetch_status("req", timeout=1.0) == retained
        ctx.storage_manager.wait_prefetch_status.assert_called_once()
    else:
        assert module.query_prefetch_status("req") == retained
    assert "req" not in module._prefetch_jobs
    ctx.event_bus.publish.assert_called()
