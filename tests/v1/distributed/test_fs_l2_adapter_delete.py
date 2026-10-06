# SPDX-License-Identifier: Apache-2.0
"""Tests for FSL2Adapter delete and listener notifications.

``delete`` unlinks each key's backing file and fires
``on_l2_keys_deleted``; store/load fire ``on_l2_keys_stored`` /
``on_l2_keys_accessed``. Together these drive the base class's byte
accounting and feed the coordinator's cache-event stream.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import cast
import asyncio
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters import fs_l2_adapter as fs_adapter_module
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

pytestmark = pytest.mark.no_shared_allocator


class _RecordingListener:
    """Captures listener callbacks (duck-typed L2AdapterListener)."""

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


class _Buf:
    """Minimal MemoryObj stand-in: just the ``byte_array`` the FS
    adapter's store/load paths read and write."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._data)


# (adapter, listener) pair produced by the ``adapter`` fixture.
AdapterFixture = tuple[FSL2Adapter, _RecordingListener]


def _key(h: bytes = b"\xde\xad\xbe\xef", salt: str = "alice") -> ObjectKey:
    return ObjectKey(
        chunk_hash=h,
        model_name="llama",
        kv_rank=42,
        cache_salt=salt,
    )


@pytest.fixture
def adapter(
    tmp_path: Path,
) -> Iterator[AdapterFixture]:
    adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    listener = _RecordingListener()
    adp.register_listener(listener)  # type: ignore[arg-type]
    try:
        yield adp, listener
    finally:
        adp.close()


def _bufs(payloads: list[bytes]) -> list[MemoryObj]:
    """Wrap raw payloads as MemoryObj-shaped buffers (see ``_Buf``)."""
    return cast("list[MemoryObj]", [_Buf(p) for p in payloads])


def _lookup_and_wait(adp: FSL2Adapter, keys: list[ObjectKey]) -> list[bool]:
    """Submit a lookup and poll until its hit bitmap is available."""
    task_id = adp.submit_lookup_and_lock_task(
        keys, {0: MemoryLayoutDesc(shapes=[], dtypes=[])}
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        bitmap = adp.query_lookup_and_lock_result(task_id)
        if bitmap is not None:
            return [bitmap.test(i) for i in range(len(keys))]
        time.sleep(0.01)
    raise AssertionError("lookup task did not complete within 5s")


def _store_and_wait(
    adp: FSL2Adapter, keys: list[ObjectKey], payloads: list[bytes]
) -> None:
    """Submit a store and poll until its result is available."""
    task_id = adp.submit_store_task(keys, _bufs(payloads))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adp.pop_completed_store_tasks()
        if task_id in completed:
            # L2StoreResult is an int: >= 0 success, -1 failure.
            assert int(completed[task_id]) >= 0
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


class TestDelete:
    def test_delete_removes_keys_and_notifies(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        k1, k2 = _key(b"\x01" * 4), _key(b"\x02" * 4)
        _store_and_wait(adp, [k1, k2], [b"x" * 100, b"y" * 40])
        assert dict(listener.stored) == {k1: 100, k2: 40}
        assert adp.get_usage().total_bytes_used == 140

        adp.delete([k1])

        assert listener.deleted == [k1]
        # Byte accounting shrank by exactly k1's stat'd size.
        assert adp.get_usage().total_bytes_used == 40
        # k1 is gone, k2 survives.
        assert _lookup_and_wait(adp, [k1, k2]) == [False, True]

    def test_delete_missing_key_is_noop(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        adp.delete([_key(b"\xff" * 4)])
        assert listener.deleted == []
        assert adp.get_usage().total_bytes_used == 0

    def test_double_delete_is_idempotent(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"z" * 10])
        adp.delete([k])
        adp.delete([k])
        assert listener.deleted == [k]
        assert adp.get_usage().total_bytes_used == 0

    def test_large_batch_deletes_all_and_notifies(
        self, adapter: AdapterFixture
    ) -> None:
        """A batch far wider than _DELETE_CONCURRENCY is fully deleted
        with per-key sizes intact — exercises the bounded concurrent
        fan-out (order-preserving gather + semaphore)."""
        adp, listener = adapter
        n = 300
        keys = [_key(i.to_bytes(4, "big")) for i in range(n)]
        payloads = [b"x" * (10 + i % 7) for i in range(n)]
        _store_and_wait(adp, keys, payloads)

        adp.delete(keys)

        assert sorted(listener.deleted, key=str) == sorted(keys, key=str)
        assert adp.get_usage().total_bytes_used == 0
        assert _lookup_and_wait(adp, keys[:8]) == [False] * 8

    def test_delete_after_close_does_not_raise(self, adapter: AdapterFixture) -> None:
        """delete() on a closed adapter is a logged no-op, never a
        crash: run_coroutine_threadsafe raises RuntimeError once the
        background loop is stopped, and delete is best-effort."""
        adp, listener = adapter
        adp.close()
        adp.delete([_key()])
        assert listener.deleted == []

    def test_empty_key_list_is_noop(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        adp.delete([])
        assert listener.deleted == []

    def test_load_after_delete_is_miss(self, adapter: AdapterFixture) -> None:
        """The documented race outcome: a load for a deleted key
        degrades to a miss, never an error."""
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"w" * 16])
        adp.delete([k])

        task_id = adp.submit_load_task([k], _bufs([b"\x00" * 16]))
        deadline = time.monotonic() + 5.0
        bitmap = None
        while time.monotonic() < deadline:
            bitmap = adp.query_load_result(task_id)
            if bitmap is not None:
                break
            time.sleep(0.01)
        assert bitmap is not None, "load task did not complete within 5s"
        assert bitmap.popcount() == 0
        assert listener.accessed == []

    def test_delete_waits_for_delayed_unlink_before_replacement(
        self,
        adapter: AdapterFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return only after an old unlink can no longer remove a replacement."""
        adp, _listener = adapter
        key = _key(b"\x03" * 4)
        _store_and_wait(adp, [key], [b"old-object"])

        unlink_entered = threading.Event()
        release_unlink = threading.Event()
        original_unlink = fs_adapter_module.aiofiles.os.unlink

        async def delayed_unlink(path: str | Path) -> None:
            unlink_entered.set()
            await asyncio.to_thread(release_unlink.wait)
            await original_unlink(path)

        monkeypatch.setattr(fs_adapter_module.aiofiles.os, "unlink", delayed_unlink)
        monkeypatch.setattr(fs_adapter_module, "_DELETE_COMPLETION_WARNING_S", 0.01)

        cleanup_errors: list[BaseException] = []

        def cleanup() -> None:
            try:
                adp.delete([key])
            except BaseException as error:  # pragma: no cover - surfaced below
                cleanup_errors.append(error)

        worker = threading.Thread(target=cleanup)
        worker.start()
        try:
            assert unlink_entered.wait(timeout=5.0)
            time.sleep(0.05)
            assert worker.is_alive(), (
                "delete returned while physical unlink was pending"
            )
        finally:
            release_unlink.set()
            worker.join(timeout=5.0)

        assert not worker.is_alive()
        assert cleanup_errors == []
        _store_and_wait(adp, [key], [b"replacement-object"])
        assert _lookup_and_wait(adp, [key]) == [True]

    def test_close_waits_for_physical_unlink(
        self,
        adapter: AdapterFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Closing the loop must not outlive an executor's pending unlink."""
        adp, listener = adapter
        key = _key(b"\x05" * 4)
        _store_and_wait(adp, [key], [b"old-object"])
        entered = threading.Event()
        release = threading.Event()
        terminal = threading.Event()
        original_unlink = fs_adapter_module.os.unlink

        def blocking_unlink(path: str | Path) -> None:
            entered.set()
            assert release.wait(timeout=5.0)
            original_unlink(path)
            terminal.set()

        async def executor_unlink(path: str | Path) -> None:
            await asyncio.to_thread(blocking_unlink, path)

        monkeypatch.setattr(fs_adapter_module.aiofiles.os, "unlink", executor_unlink)
        monkeypatch.setattr(fs_adapter_module, "_DELETE_COMPLETION_WARNING_S", 0.01)
        deleting = threading.Thread(target=adp.delete, args=([key],))
        closing = threading.Thread(target=adp.close)
        deleting.start()
        try:
            assert entered.wait(timeout=5.0)
            closing.start()
            time.sleep(0.05)
            assert deleting.is_alive(), "close released a pending deletion"
            assert closing.is_alive(), "close left physical unlink running"
            assert not terminal.is_set()
        finally:
            release.set()
            deleting.join(timeout=5.0)
            if closing.ident is not None:
                closing.join(timeout=5.0)
        assert not deleting.is_alive()
        assert not closing.is_alive()
        assert terminal.is_set()
        assert listener.deleted == [key]
        adp.delete([key])

    def test_terminal_unlink_failure_is_not_reported_deleted(
        self,
        adapter: AdapterFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A terminal failure is distinct from a still-pending unlink.

        Once the filesystem operation has raised, ``delete`` may return, but
        it must not notify listeners or decrement byte accounting. The intact
        object remains addressable rather than becoming an untracked deletion.
        """
        adp, listener = adapter
        key = _key(b"\x04" * 4)
        payload = b"survives-terminal-delete-failure"
        _store_and_wait(adp, [key], [payload])

        async def failing_unlink(_path: str | Path) -> None:
            raise OSError("injected terminal unlink failure")

        monkeypatch.setattr(fs_adapter_module.aiofiles.os, "unlink", failing_unlink)

        adp.delete([key])

        assert listener.deleted == []
        assert adp.get_usage().total_bytes_used == len(payload)
        assert _lookup_and_wait(adp, [key]) == [True]


class TestStoreLoadNotifications:
    def test_restore_of_existing_key_does_not_renotify(
        self, adapter: AdapterFixture
    ) -> None:
        """A store that skips (file already on disk) fires no stored
        event, so byte accounting is never double-counted."""
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"a" * 8])
        _store_and_wait(adp, [k], [b"a" * 8])
        assert listener.stored == [(k, 8)]
        assert adp.get_usage().total_bytes_used == 8

    def test_load_notifies_accessed(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"b" * 12])

        task_id = adp.submit_load_task([k], _bufs([b"\x00" * 12]))
        deadline = time.monotonic() + 5.0
        bitmap = None
        while time.monotonic() < deadline:
            bitmap = adp.query_load_result(task_id)
            if bitmap is not None:
                break
            time.sleep(0.01)
        assert bitmap is not None and bitmap.popcount() == 1
        assert listener.accessed == [k]
