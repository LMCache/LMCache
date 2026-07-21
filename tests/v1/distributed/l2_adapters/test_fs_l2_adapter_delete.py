# SPDX-License-Identifier: Apache-2.0
"""Tests for the FS L2 adapter's delete path and byte accounting.

Stores objects through the adapter's own store task, then exercises
``delete``: data files are unlinked, ``on_l2_keys_stored`` /
``on_l2_keys_deleted`` fire with the affected keys, ``get_usage`` tracks
the byte totals, and deleting missing keys is a no-op.
"""

# Standard
from pathlib import Path
from typing import cast
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

OBJ_SIZE = 4096


class _RecordingListener(L2AdapterListener):
    """Record every stored / accessed / deleted notification."""

    def __init__(self) -> None:
        self.stored: list[tuple[ObjectKey, int]] = []
        self.accessed: list[ObjectKey] = []
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.extend(zip(keys, sizes, strict=True))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.extend(keys)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)


class _BytesObj:
    """Minimal store payload exposing the ``byte_array`` the adapter reads."""

    def __init__(self, payload: bytes) -> None:
        self.byte_array = payload


def _key(index: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes([index]) * 8,
        model_name="fs-delete-model",
        kv_rank=0,
        cache_salt=cache_salt,
    )


@pytest.fixture()
def adapter(tmp_path: Path):
    a = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    listener = _RecordingListener()
    a.register_listener(listener)
    yield a, listener, tmp_path
    a.close()


def _store(adapter: FSL2Adapter, keys: list[ObjectKey]) -> None:
    """Store one OBJ_SIZE payload per key and wait for task completion."""
    payloads = cast("list[MemoryObj]", [_BytesObj(b"\xab" * OBJ_SIZE)] * len(keys))
    task_id = adapter.submit_store_task(keys, payloads)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        done = adapter.pop_completed_store_tasks()
        if task_id in done:
            if int(done[task_id]) < 0:
                raise AssertionError("store task failed")
            return
        time.sleep(0.05)
    raise TimeoutError("store task did not complete")


def _data_files(base: Path) -> list[Path]:
    return sorted(base.glob("*.data"))


def test_delete_removes_files_and_notifies(adapter) -> None:
    """``delete`` unlinks stored files, fires the listeners, and updates
    ``get_usage`` totals; keys never stored are skipped."""
    a, listener, base = adapter
    keys = [_key(i) for i in range(3)]
    _store(a, keys)

    assert len(_data_files(base)) == 3
    assert sorted(s for _, s in listener.stored) == [OBJ_SIZE] * 3
    assert a.get_usage().total_bytes_used == 3 * OBJ_SIZE

    a.delete([keys[0], keys[1], _key(9)])

    assert len(_data_files(base)) == 1
    assert listener.deleted == [keys[0], keys[1]]
    assert a.get_usage().total_bytes_used == OBJ_SIZE


def test_delete_is_idempotent(adapter) -> None:
    """A second delete of the same keys removes nothing and fires no
    listener."""
    a, listener, base = adapter
    keys = [_key(i) for i in range(2)]
    _store(a, keys)

    a.delete(keys)
    assert _data_files(base) == []
    assert listener.deleted == keys

    a.delete(keys)
    assert listener.deleted == keys
    assert a.get_usage().total_bytes_used == 0


def test_delete_respects_cache_salt(adapter) -> None:
    """Keys differing only in ``cache_salt`` map to different files; deleting
    one salt leaves the other's data on disk."""
    a, listener, base = adapter
    plain = _key(1)
    salted = _key(1, cache_salt="alice")
    _store(a, [plain, salted])
    assert len(_data_files(base)) == 2

    a.delete([salted])

    assert len(_data_files(base)) == 1
    assert listener.deleted == [salted]
    usage = a.get_usage()
    assert usage.total_bytes_used == OBJ_SIZE
    assert "alice" not in usage.bytes_by_cache_salt
