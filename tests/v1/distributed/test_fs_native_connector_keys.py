# SPDX-License-Identifier: Apache-2.0
"""Regression tests for fs_native ObjectKey filename handling."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

lmcache_fs = pytest.importorskip(
    "lmcache.lmcache_fs",
    reason="lmcache_fs native extension required for fs_native connector tests",
)


def create_memory_obj(size: int = 16, fill_value: float = 1.0) -> TensorMemoryObj:
    """Create a CPU-backed memory object for fs_native store tests.

    Args:
        size: Number of float32 elements to allocate.
        fill_value: Value used to initialize the tensor contents.

    Returns:
        A TensorMemoryObj whose byte buffer can be passed to an L2 adapter.
    """
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


@contextmanager
def fs_native_adapter(base_path: Path) -> Iterator[NativeConnectorL2Adapter]:
    """Create a NativeConnectorL2Adapter backed by LMCacheFSClient.

    Args:
        base_path: Directory where the fs_native connector stores data files.

    Yields:
        A NativeConnectorL2Adapter wrapping the native filesystem connector.
    """
    client = lmcache_fs.LMCacheFSClient(str(base_path), 1, "", False, 0)
    adapter = NativeConnectorL2Adapter(client)
    try:
        yield adapter
    finally:
        adapter.close()


def store_key_and_require_success(
    adapter: NativeConnectorL2Adapter,
    key: ObjectKey,
) -> None:
    """Store one ObjectKey and fail the test if the store does not complete.

    Args:
        adapter: Adapter used to submit and observe the store task.
        key: ObjectKey to store through the adapter's public interface.

    Raises:
        AssertionError: If the store event is not signaled or the task fails.
    """
    task_id = adapter.submit_store_task([key], [create_memory_obj()])
    poll = select.poll()
    poll.register(adapter.get_store_event_fd(), select.POLLIN)
    events = poll.poll(5000)
    if not events:
        raise AssertionError("fs_native store task did not signal completion")

    try:
        consume_fd(adapter.get_store_event_fd())
    except BlockingIOError:
        pass

    completed = adapter.pop_completed_store_tasks()
    result = completed.get(task_id)
    if result is None:
        raise AssertionError("fs_native store task was not completed")
    if not result.is_successful():
        raise AssertionError("fs_native store task failed")


def test_fs_native_stores_unsalted_object_group_filename(tmp_path: Path) -> None:
    """Unsalted fs_native stores include object_group_id in the filename."""
    key = ObjectKey(
        chunk_hash=b"\x00\x01\x02\x03",
        model_name="llama",
        kv_rank=255,
        object_group_id=5,
    )

    with fs_native_adapter(tmp_path) as adapter:
        store_key_and_require_success(adapter, key)

    assert (tmp_path / "llama@0x000000ff@5@00010203.data").exists()


def test_fs_native_stores_salted_object_group_filename(tmp_path: Path) -> None:
    """Salted fs_native stores include object_group_id before cache_salt."""
    key = ObjectKey(
        chunk_hash=b"\x00\x01\x02\x03",
        model_name="llama",
        kv_rank=255,
        object_group_id=5,
        cache_salt="alice",
    )

    with fs_native_adapter(tmp_path) as adapter:
        store_key_and_require_success(adapter, key)

    assert (tmp_path / "llama@0x000000ff@5@00010203@alice.data").exists()
