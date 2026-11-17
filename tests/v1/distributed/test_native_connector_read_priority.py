# SPDX-License-Identifier: Apache-2.0
"""
Tests for read-over-write scheduling in the native connector base
(``csrc/storage_backends/connector_base.h``).

Uses the real C++ FS connector, since the scheduling lives in the shared
C++ ``ConnectorBase``. Skipped when the extension is not built.
"""

# Standard
from pathlib import Path
import select
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    _object_key_to_string,
)

lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")

# Large enough that the write batch cannot finish in the few microseconds
# between submitting it and submitting the read behind it.
NUM_WRITE_KEYS = 48
WRITE_OBJ_BYTES = 2 * 1024 * 1024
COMPLETION_TIMEOUT_S = 60.0


def _key_string(chunk_id: int) -> str:
    """Build a serialized key string accepted by the FS connector."""
    return _object_key_to_string(
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
            model_name="test_model",
            kv_rank=0,
        )
    )


def _wait_for_completions(
    client: "lmcache_fs.LMCacheFSClient", count: int
) -> list[tuple[int, bool]]:
    """
    Drain completions until ``count`` have arrived.

    Args:
        client: The native FS client to drain.
        count: Number of completions to wait for.

    Returns:
        ``(future_id, ok)`` pairs in the order the connector completed them.

    Raises:
        TimeoutError: If the completions do not arrive in time.
    """
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + COMPLETION_TIMEOUT_S
    completed: list[tuple[int, bool]] = []
    while len(completed) < count:
        if time.monotonic() > deadline:
            raise TimeoutError(f"only {len(completed)} of {count} completions arrived")
        poller.poll(100)
        for future_id, ok, _error, _results in client.drain_completions():
            completed.append((int(future_id), bool(ok)))
    return completed


@pytest.fixture
def single_worker_client(tmp_path: Path):
    """An FS client with one worker, so scheduling order is observable."""
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    yield client
    client.close()


def _submit_large_write(client: "lmcache_fs.LMCacheFSClient") -> int:
    """Submit one large multi-key write batch and return its future id."""
    payload = bytearray(WRITE_OBJ_BYTES)
    keys = [_key_string(1000 + i) for i in range(NUM_WRITE_KEYS)]
    views = [memoryview(payload) for _ in keys]
    return int(client.submit_batch_set(keys, views))


def test_load_completes_before_earlier_store(single_worker_client):
    """A GET submitted behind a large SET batch must not wait for it."""
    client = single_worker_client
    read_key = _key_string(1)
    data = bytearray(b"\x07" * 4096)
    seed_id = int(client.submit_batch_set([read_key], [memoryview(data)]))
    assert _wait_for_completions(client, 1) == [(seed_id, True)]

    write_id = _submit_large_write(client)
    dst = bytearray(len(data))
    read_id = int(client.submit_batch_get([read_key], [memoryview(dst)]))

    completed = _wait_for_completions(client, 2)

    assert completed == [(read_id, True), (write_id, True)]
    assert dst == data


def test_lookup_completes_before_earlier_store(single_worker_client):
    """An EXISTS submitted behind a large SET batch must not wait for it."""
    client = single_worker_client

    write_id = _submit_large_write(client)
    lookup_id = int(client.submit_batch_exists([_key_string(1)]))

    completed = _wait_for_completions(client, 2)

    assert completed == [(lookup_id, True), (write_id, True)]


def test_preempted_store_writes_every_key(single_worker_client):
    """A SET batch that yielded to reads still stores all of its keys."""
    client = single_worker_client

    write_id = _submit_large_write(client)
    lookup_ids = [int(client.submit_batch_exists([_key_string(1)])) for _ in range(8)]
    completed = _wait_for_completions(client, 1 + len(lookup_ids))
    assert (write_id, True) in completed

    keys = [_key_string(1000 + i) for i in range(NUM_WRITE_KEYS)]
    exists_id = int(client.submit_batch_exists(keys))
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + COMPLETION_TIMEOUT_S
    found: list[bool] = []
    while not found and time.monotonic() < deadline:
        poller.poll(100)
        for future_id, _ok, _error, results in client.drain_completions():
            if int(future_id) == exists_id:
                found = [bool(r) for r in results]

    assert found == [True] * NUM_WRITE_KEYS
