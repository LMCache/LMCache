# SPDX-License-Identifier: Apache-2.0
"""Tests for the native C++ filesystem connector."""

# Standard
from pathlib import Path
import errno
import os
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.key_codec import encode_fs_filename, encode_storage_key
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

try:
    # First Party
    from lmcache.lmcache_fs import LMCacheFSClient
except ImportError:
    LMCacheFSClient = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    LMCacheFSClient is None,
    reason="native FS extension is not built",
)

_PIPE_FILL_BYTES = 512 * 1024


def _make_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _make_memory_obj(num_floats: int) -> TensorMemoryObj:
    metadata = MemoryObjMetadata(
        shape=torch.Size([num_floats]),
        dtype=torch.float32,
        address=0,
        phy_size=num_floats * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(
        torch.zeros(num_floats, dtype=torch.float32), metadata, parent_allocator=None
    )


class _DeleteRecorder(L2AdapterListener):
    def __init__(self) -> None:
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        pass

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)


class _CompletionTracker:
    """Drains a native client's single completion stream and indexes by future_id."""

    def __init__(self, client: LMCacheFSClient) -> None:
        self._client = client
        self._seen: dict[int, tuple[bool, str, list[bool] | None]] = {}

    def poll_now(self) -> None:
        for fid, ok, err, per_key in self._client.drain_completions():
            self._seen[int(fid)] = (ok, err, per_key)

    def has_completed(self, future_id: int) -> bool:
        self.poll_now()
        return future_id in self._seen

    def wait(
        self, future_id: int, timeout: float = 5.0
    ) -> tuple[bool, str, list[bool] | None]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll_now()
            if future_id in self._seen:
                return self._seen.pop(future_id)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                select.select([self._client.event_fd()], [], [], min(remaining, 0.05))
        raise AssertionError(f"Timed out waiting for future {future_id}")


def _drain_fifo_and_close(rfd: int) -> None:
    try:
        while os.read(rfd, 65536):
            pass
    finally:
        os.close(rfd)


def _feed_fifo_and_close(wfd: int, payload: bytes) -> None:
    try:
        view = memoryview(payload)
        sent = 0
        while sent < len(view):
            sent += os.write(wfd, view[sent:])
    finally:
        os.close(wfd)


def test_lookup_and_get_bypass_blocked_stores(tmp_path: Path) -> None:
    """Lookups and reads finish on the read pool while every shared worker is
    parked inside a slow write."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    client = LMCacheFSClient(str(tmp_path), 2, "tmp", False, 0, 4, 0)
    tracker = _CompletionTracker(client)
    try:
        present_obj = _make_key(1)
        absent_obj = _make_key(2)
        present_key = encode_storage_key(present_obj)
        absent_key = encode_storage_key(absent_obj)
        expected = bytearray(i % 251 for i in range(8192))
        ok, err, _ = tracker.wait(
            client.submit_batch_set([present_key], [memoryview(expected)])
        )
        assert ok, err

        slow_objs = [_make_key(10), _make_key(11)]
        slow_keys = [encode_storage_key(k) for k in slow_objs]
        fifo_paths = [tmp_dir / encode_fs_filename(k) for k in slow_objs]
        for fifo in fifo_paths:
            os.mkfifo(fifo)

        write_bufs = [bytearray(_PIPE_FILL_BYTES) for _ in slow_keys]
        store_fid = client.submit_batch_set(
            slow_keys, [memoryview(b) for b in write_bufs]
        )
        # Blocking O_RDONLY open returns once each worker enters O_WRONLY open;
        # leaving the 512 KiB pipes unread keeps both workers parked in write().
        rfds = [os.open(fifo, os.O_RDONLY) for fifo in fifo_paths]
        try:
            ex_ok, ex_err, ex_bits = tracker.wait(
                client.submit_batch_exists([present_key, absent_key]), timeout=2.0
            )
            assert ex_ok, ex_err
            assert ex_bits == [True, False]

            dst_present = bytearray(len(expected))
            dst_absent = bytearray(len(expected))
            get_ok, get_err, get_bits = tracker.wait(
                client.submit_batch_get(
                    [present_key, absent_key],
                    [memoryview(dst_present), memoryview(dst_absent)],
                ),
                timeout=2.0,
            )
            assert get_ok, get_err
            assert get_bits == [True, False]
            assert dst_present == expected
            assert not tracker.has_completed(store_fid)
        finally:
            for rfd in rfds:
                _drain_fifo_and_close(rfd)

        st_ok, st_err, _ = tracker.wait(store_fid)
        assert st_ok, st_err
    finally:
        client.close()


def test_concurrent_get_batches_do_not_block_on_num_workers(tmp_path: Path) -> None:
    """A second GET batch completes on idle readers while an earlier GET batch
    is still reading, even with num_workers=1."""
    client = LMCacheFSClient(str(tmp_path), 1, "", False, 0, 4, 0)
    tracker = _CompletionTracker(client)
    try:
        fast_key = encode_storage_key(_make_key(1))
        slow_obj = _make_key(2)
        slow_key = encode_storage_key(slow_obj)
        payload = bytes((i * 7) % 251 for i in range(4096))
        src = bytearray(payload)
        ok, err, _ = tracker.wait(
            client.submit_batch_set([fast_key], [memoryview(src)])
        )
        assert ok, err

        slow_fifo = tmp_path / encode_fs_filename(slow_obj)
        os.mkfifo(slow_fifo)

        slow_dst = bytearray(len(payload))
        slow_fid = client.submit_batch_get([slow_key], [memoryview(slow_dst)])
        wfd = os.open(slow_fifo, os.O_WRONLY)
        try:
            fast_dst = bytearray(len(payload))
            fast_ok, fast_err, fast_bits = tracker.wait(
                client.submit_batch_get([fast_key], [memoryview(fast_dst)]),
                timeout=2.0,
            )
            assert fast_ok, fast_err
            assert fast_bits == [True]
            assert bytes(fast_dst) == payload
            assert not tracker.has_completed(slow_fid)
        finally:
            _feed_fifo_and_close(wfd, payload)

        slow_ok, slow_err, slow_bits = tracker.wait(slow_fid)
        assert slow_ok, slow_err
        assert slow_bits == [True]
        assert bytes(slow_dst) == payload
    finally:
        client.close()


def test_exists_runs_ahead_of_budget_throttled_reads(tmp_path: Path) -> None:
    """Lookups complete on the remaining reader when in-flight reads hold the
    byte budget and a later read is waiting for budget room."""
    size = 4096
    client = LMCacheFSClient(str(tmp_path), 1, "", False, 0, 3, size)
    tracker = _CompletionTracker(client)
    try:
        assert client.read_budget_bytes() == size
        present_key = encode_storage_key(_make_key(1))
        missing_key = encode_storage_key(_make_key(99))
        payload = bytes((i * 3) % 251 for i in range(size))
        src = bytearray(payload)
        ok, err, _ = tracker.wait(
            client.submit_batch_set([present_key], [memoryview(src)])
        )
        assert ok, err

        slow_objs = [_make_key(10), _make_key(11)]
        slow_keys = [encode_storage_key(k) for k in slow_objs]
        fifos = [tmp_path / encode_fs_filename(k) for k in slow_objs]
        for fifo in fifos:
            os.mkfifo(fifo)

        gated_obj = _make_key(12)
        gated_key = encode_storage_key(gated_obj)
        gated_fifo = tmp_path / encode_fs_filename(gated_obj)
        os.mkfifo(gated_fifo)

        slow_dsts = [bytearray(size) for _ in slow_keys]
        slow_fid = client.submit_batch_get(
            slow_keys, [memoryview(b) for b in slow_dsts]
        )
        wfds = [os.open(fifo, os.O_WRONLY) for fifo in fifos]
        gated_wfd = -1
        try:
            # If the byte budget were not enforced, this read would take the
            # third reader and park on gated_fifo, starving exists below.
            gated_dst = bytearray(size)
            gated_fid = client.submit_batch_get([gated_key], [memoryview(gated_dst)])

            ex_ok, ex_err, ex_bits = tracker.wait(
                client.submit_batch_exists([present_key, missing_key]), timeout=2.0
            )
            assert ex_ok, ex_err
            assert ex_bits == [True, False]
            # The reader freed by the first lookup must still leave the gated
            # read waiting, or no reader would be free for this second lookup.
            ex2_ok, ex2_err, ex2_bits = tracker.wait(
                client.submit_batch_exists([present_key]), timeout=2.0
            )
            assert ex2_ok, ex2_err
            assert ex2_bits == [True]
            assert not tracker.has_completed(gated_fid)
        finally:
            for wfd in wfds:
                _feed_fifo_and_close(wfd, payload)
            gated_wfd = os.open(gated_fifo, os.O_WRONLY)
            _feed_fifo_and_close(gated_wfd, payload)

        assert tracker.wait(slow_fid)[2] == [True, True]
        assert tracker.wait(gated_fid)[2] == [True]
        assert bytes(gated_dst) == payload
    finally:
        client.close()


def test_finished_reads_return_their_budget(tmp_path: Path) -> None:
    """Bytes of finished reads go back to the budget, so later reads can fill
    the whole budget again."""
    size = 4096
    client = LMCacheFSClient(str(tmp_path), 1, "", False, 0, 4, 3 * size)
    tracker = _CompletionTracker(client)
    try:
        payload = bytes((i * 5) % 251 for i in range(size))
        warm_keys = [encode_storage_key(_make_key(i)) for i in range(3)]
        srcs = [bytearray(payload) for _ in warm_keys]
        set_fid = client.submit_batch_set(warm_keys, [memoryview(b) for b in srcs])
        assert tracker.wait(set_fid)[0]
        dsts = [bytearray(size) for _ in warm_keys]
        get_fid = client.submit_batch_get(warm_keys, [memoryview(b) for b in dsts])
        assert tracker.wait(get_fid)[2] == [True] * 3

        slow_objs = [_make_key(10 + i) for i in range(3)]
        fifos = [tmp_path / encode_fs_filename(k) for k in slow_objs]
        for fifo in fifos:
            os.mkfifo(fifo)
        slow_dsts = [bytearray(size) for _ in slow_objs]
        slow_fid = client.submit_batch_get(
            [encode_storage_key(k) for k in slow_objs],
            [memoryview(b) for b in slow_dsts],
        )
        # A non-blocking writer open succeeds only once a reader holds the
        # FIFO, i.e. once that read has been dispatched.
        wfds: dict[int, int] = {}
        deadline = time.monotonic() + 2.0
        while len(wfds) < len(fifos) and time.monotonic() < deadline:
            for i, fifo in enumerate(fifos):
                if i in wfds:
                    continue
                try:
                    wfds[i] = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                except OSError as e:
                    if e.errno != errno.ENXIO:
                        raise
                else:
                    os.set_blocking(wfds[i], True)
            time.sleep(0.01)
        try:
            assert len(wfds) == len(fifos)
        finally:
            for wfd in wfds.values():
                _feed_fifo_and_close(wfd, payload)
            for i, fifo in enumerate(fifos):
                if i not in wfds:
                    _feed_fifo_and_close(os.open(fifo, os.O_WRONLY), payload)

        assert tracker.wait(slow_fid)[2] == [True] * 3
    finally:
        client.close()


@pytest.mark.parametrize("use_odirect", [True, False])
def test_disk_block_size_reports_odirect_alignment_unit(
    tmp_path: Path, use_odirect: bool
) -> None:
    """With O_DIRECT on, the client reports the filesystem block size it
    aligns transfers to; with O_DIRECT off it reports 0."""
    client = LMCacheFSClient(str(tmp_path), 1, "", use_odirect, 0, 0, 0)
    try:
        expected = os.statvfs(tmp_path).f_bsize if use_odirect else 0
        assert client.disk_block_size() == expected
    finally:
        client.close()


@pytest.mark.parametrize("num_read_workers", [0, 4])
def test_lookup_error_fails_batch_while_read_error_zeroes_key(
    tmp_path: Path, num_read_workers: int
) -> None:
    """A lookup that cannot stat its file fails the whole batch, while a read
    that cannot open its file reports only that key as missing, with or
    without the read pool."""
    client = LMCacheFSClient(str(tmp_path), 1, "", False, 0, num_read_workers, 0)
    tracker = _CompletionTracker(client)
    try:
        good_key = encode_storage_key(_make_key(1))
        src = bytearray(b"ok-payload" * 100)
        assert tracker.wait(client.submit_batch_set([good_key], [memoryview(src)]))[0]
        loop_obj = _make_key(2)
        loop_path = tmp_path / encode_fs_filename(loop_obj)
        os.symlink(loop_path.name, loop_path)  # stat() fails with ELOOP
        loop_key = encode_storage_key(loop_obj)

        ok, err, _ = tracker.wait(client.submit_batch_exists([good_key, loop_key]))
        assert not ok
        assert err

        dst_good = bytearray(len(src))
        dst_loop = bytearray(len(src))
        ok, err, bits = tracker.wait(
            client.submit_batch_get(
                [good_key, loop_key], [memoryview(dst_good), memoryview(dst_loop)]
            )
        )
        assert ok, err
        assert bits == [True, False]
        assert dst_good == src
    finally:
        client.close()


def test_delete_tells_missing_key_from_failed_delete(tmp_path: Path) -> None:
    """A delete reports a key that is already gone as deleted, and a key it
    could not remove as not deleted."""
    client = LMCacheFSClient(str(tmp_path), 1, "", False, 0, 0, 0)
    tracker = _CompletionTracker(client)
    try:
        stored_key = encode_storage_key(_make_key(1))
        src = bytearray(b"payload" * 100)
        assert tracker.wait(client.submit_batch_set([stored_key], [memoryview(src)]))[0]
        missing_key = encode_storage_key(_make_key(2))
        blocked_obj = _make_key(3)
        blocked_path = tmp_path / encode_fs_filename(blocked_obj)
        (blocked_path / "child").mkdir(parents=True)  # remove() fails: ENOTEMPTY

        ok, err, bits = tracker.wait(
            client.submit_batch_delete(
                [stored_key, missing_key, encode_storage_key(blocked_obj)]
            )
        )
        assert ok, err
        assert bits == [True, True, False]
        assert blocked_path.exists()
    finally:
        client.close()


def test_adapter_drops_key_deleted_by_peer_and_keeps_failed_delete(
    tmp_path: Path,
) -> None:
    """An adapter stops accounting for a key that another adapter on the same
    directory already deleted, and keeps a key whose delete failed."""
    config = FSNativeL2AdapterConfig(str(tmp_path), num_workers=1, max_capacity_gb=1)
    adapter = create_l2_adapter(config)
    peer = create_l2_adapter(config)
    recorder = _DeleteRecorder()
    adapter.register_listener(recorder)
    try:
        gone_key = _make_key(1)
        stuck_key = _make_key(2)
        adapter.submit_store_task(
            [gone_key, stuck_key], [_make_memory_obj(256), _make_memory_obj(512)]
        )
        assert select.select([adapter.get_store_event_fd()], [], [], 5.0)[0]
        assert adapter.get_usage().total_bytes_used == (256 + 512) * 4
        peer.delete([gone_key])
        stuck_path = tmp_path / encode_fs_filename(stuck_key)
        stuck_path.unlink()
        (stuck_path / "child").mkdir(parents=True)  # remove() fails: ENOTEMPTY

        adapter.delete([gone_key, stuck_key])

        assert recorder.deleted == [gone_key]
        assert adapter.get_usage().total_bytes_used == 512 * 4
    finally:
        peer.close()
        adapter.close()
