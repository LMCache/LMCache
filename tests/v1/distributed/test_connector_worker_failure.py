# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #4342.

When a native connector's worker threads die during connection setup (e.g.
Redis unreachable when a worker thread first calls create_connection()),
every submitted task must still receive a failed completion instead of
hanging forever:
- in-flight queued tiles are drained with failure completions
- later submissions fail synchronously on the submit path (not queued)

"""

# Standard
import socket
import time

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache.lmcache_redis import LMCacheRedisClient

    _EXTENSION_AVAILABLE = True
except ImportError:
    _EXTENSION_AVAILABLE = False

if _EXTENSION_AVAILABLE:
    # First Party
    from lmcache.v1.distributed.api import (
        MemoryResidency,
        MemorySegment,
        MemoryTags,
        ObjectKey,
    )
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MemoryObjMetadata,
        TensorMemoryObj,
    )

    _MEMORY_TAGS = MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE)

_COMPLETION_TIMEOUT = 10.0

pytestmark = pytest.mark.skipif(
    not _EXTENSION_AVAILABLE, reason="C++ extension not available"
)


def _make_dead_socket() -> socket.socket:
    """Bind to a free port but do not listen — connect(2) gets ECONNREFUSED
    and the port stays claimed so nothing else can grab it."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    return s


def _make_key(chunk_id: int) -> "ObjectKey":
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_4342",
        kv_rank=0,
    )


def _make_obj(size: int = 64) -> "TensorMemoryObj":
    raw_data = torch.zeros(size, dtype=torch.float32)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _submit_and_wait_failure(adapter, keys, objs) -> None:
    """Submit a store task and assert it completes with a failure."""
    task_id = adapter.submit_store_task(keys, objs)

    # poll the completed map — do not poll the native connector's eventfd
    # (the adapter's demux thread owns it) and do not call
    # drain_completions() (it would race the demux thread); completions
    # surface through the adapter's completed map
    deadline = time.monotonic() + _COMPLETION_TIMEOUT
    completed = {}
    while time.monotonic() < deadline:
        completed = adapter.pop_completed_store_tasks()
        if task_id in completed:
            break
        time.sleep(0.01)

    assert task_id in completed, (
        f"task {task_id} got no completion within {_COMPLETION_TIMEOUT}s; "
        "workers died without reporting failure"
    )
    assert not completed[task_id].is_successful(), (
        f"task {task_id} unexpectedly succeeded against a dead endpoint"
    )


def test_dead_endpoint_reports_failure():
    """Workers die at create_connection(); submitted tiles still receive a
    failure completion instead of hanging — via the exit drain when the
    submission wins the race, otherwise via the submit-path fast fail."""
    dead = _make_dead_socket()
    try:
        port = dead.getsockname()[1]
        client = LMCacheRedisClient("127.0.0.1", port, 4)
        adapter = NativeConnectorL2Adapter(client, _MEMORY_TAGS)
        try:
            keys = [_make_key(0)] * 3
            objs = [_make_obj() for _ in range(3)]
            _submit_and_wait_failure(adapter, keys, objs)
        finally:
            adapter.close()
    finally:
        dead.close()


def test_resubmission_after_worker_death_also_fails():
    """Once the last worker has left, the queue is marked drained and the
    second submission fails synchronously on the submit path instead of
    being enqueued to nobody (enqueue_request fast-fail path)."""
    dead = _make_dead_socket()
    try:
        port = dead.getsockname()[1]
        client = LMCacheRedisClient("127.0.0.1", port, 4)
        adapter = NativeConnectorL2Adapter(client, _MEMORY_TAGS)
        try:
            keys = [_make_key(1)] * 3
            objs = [_make_obj() for _ in range(3)]
            _submit_and_wait_failure(adapter, keys, objs)
            _submit_and_wait_failure(adapter, keys, objs)
        finally:
            adapter.close()
    finally:
        dead.close()
