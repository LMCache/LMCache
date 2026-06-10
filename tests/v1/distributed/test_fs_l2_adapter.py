# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the filesystem L2 adapter."""

# Standard
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path
from typing import Any, TypeAlias
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters import fs_l2_adapter as fsmod
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

WriteBuffer: TypeAlias = bytearray | memoryview[int] | bytes


def create_object_key(chunk_id: int, model_name: str = "test/model") -> ObjectKey:
    """Create a stable ObjectKey for FS adapter tests."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 16, fill_value: float = 1.0) -> TensorMemoryObj:
    """Create a CPU TensorMemoryObj with deterministic contents."""
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


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    """Wait for and consume one adapter event notification."""
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


@pytest.fixture
def adapter(tmp_path: Path) -> Generator[FSL2Adapter, None, None]:
    """Create a filesystem adapter rooted in the pytest temp directory."""
    config = FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=False)
    fs_adapter = FSL2Adapter(config)
    yield fs_adapter
    fs_adapter.close()


def test_failed_store_rolls_back_new_files_but_keeps_existing(
    adapter: FSL2Adapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial store failure must not publish a partial batch."""
    existing_key = create_object_key(0)
    existing_objects: list[MemoryObj] = [create_memory_obj()]
    store_tid = adapter.submit_store_task([existing_key], existing_objects)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[store_tid].is_successful()

    original_write: Callable[[Any, WriteBuffer], Awaitable[int]] = (
        fsmod._async_write_full
    )
    write_count = 0

    async def fail_second_write(f: Any, buf: WriteBuffer) -> int:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("injected write failure")
        return await original_write(f, buf)

    monkeypatch.setattr(fsmod, "_async_write_full", fail_second_write)

    keys = [existing_key, create_object_key(1), create_object_key(2)]
    objects: list[MemoryObj] = [
        create_memory_obj(fill_value=float(i)) for i in range(3)
    ]
    failed_tid = adapter.submit_store_task(keys, objects)
    assert wait_for_event_fd(adapter.get_store_event_fd())

    result = adapter.pop_completed_store_tasks()[failed_tid]
    assert not result.is_successful()
    assert result.bytes_transferred() == 0

    lookup_tid = adapter.submit_lookup_and_lock_task(keys)
    assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
    assert bitmap is not None
    assert bitmap.test(0) is True
    assert bitmap.test(1) is False
    assert bitmap.test(2) is False
