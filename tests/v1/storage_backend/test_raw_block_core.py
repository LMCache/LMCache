# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from typing import Any
from unittest.mock import Mock, call
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytest.importorskip("lmcache_rust_raw_block_io")


def test_raw_block_core_store_load_and_exists(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        keys = [make_object_key(i) for i in range(3)]
        specs = [encode_object_key(key) for key in keys]
        payloads = [
            bytes([1]) * 1024,
            bytes([2]) * 2048,
            bytes([3]) * 3072,
        ]
        objects = [make_memory_obj(payload) for payload in payloads]

        put_result = core.put_many(specs, objects)

        assert put_result.results == [True, True, True]
        assert put_result.stored_keys == [spec.encoded for spec in specs]
        assert core.exists_many([spec.encoded for spec in specs]) == [
            True,
            True,
            True,
        ]

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)

        assert load_result == [True, True, True]
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_duplicate_put_keeps_original_payload(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(11))
        original = b"original"
        duplicate = b"mutated!"

        first_result = core.put_many([spec], [make_memory_obj(original)])
        duplicate_result = core.put_many([spec], [make_memory_obj(duplicate)])

        assert first_result.results == [True]
        assert first_result.stored_keys == [spec.encoded]
        assert duplicate_result.results == [True]
        assert duplicate_result.stored_keys == []

        loaded = make_empty_memory_obj(len(original))
        assert core.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == original
    finally:
        core.close()


def test_raw_block_core_delete_and_missing_load(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        existing = encode_object_key(make_object_key(21))
        missing = encode_object_key(make_object_key(22))

        put_result = core.put_many([existing], [make_memory_obj(b"delete-me")])
        assert put_result.results == [True]
        assert core.contains_key(existing.encoded) is True

        assert core.delete_many([existing.encoded, missing.encoded]) == [True, False]
        assert core.exists_many([existing.encoded, missing.encoded]) == [False, False]

        loaded = make_empty_memory_obj(len(b"delete-me"))
        assert core.load_many_into([existing.encoded], [loaded]) == [False]
    finally:
        core.close()


def test_raw_block_core_recovers_checkpoint_from_temp_file(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    spec = encode_object_key(make_object_key(31))
    payload = b"recoverable-raw-block-payload"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([spec], [make_memory_obj(payload)])
        assert put_result.results == [True]
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(spec.encoded) is True
        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        recovered.close()


def test_raw_block_core_drops_checkpoint_entry_with_stale_slot_header(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    spec = encode_object_key(make_object_key(41))
    payload = b"stale-slot-header-payload"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([spec], [make_memory_obj(payload)])
        assert put_result.results == [True]
        offset = core.entry_offset(spec.encoded)
        assert offset is not None
        core.checkpoint_now()
    finally:
        core.close()

    with path.open("r+b") as f:
        f.seek(offset)
        f.write(b"STALEHDR")

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(spec.encoded) is False
        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([spec.encoded], [loaded]) == [False]
    finally:
        recovered.close()


def test_raw_block_core_uses_bounded_posix_recovery_read_tasks(monkeypatch):
    specs = [encode_object_key(make_object_key(i)) for i in range(50, 60)]
    core = object.__new__(RawBlockCore)
    core._lock = threading.Lock()
    core._index = {
        spec.encoded: type("Entry", (), {"offset": 4096 * (i + 1), "size": 7})()
        for i, spec in enumerate(specs)
    }
    core._lock_refcnt = {}
    core._meta_dirty_total = 0
    core.io_engine = "posix"
    core._recovery_read_threads = 3
    core.key_namespace = "object"
    core._read_slot_header = lambda offset: (
        specs[(offset // 4096) - 1].slot_identity,
        7,
    )

    max_worker_calls: list[int] = []
    mapped_item_counts: list[int] = []

    class RecordingThreadPoolExecutor:
        def __init__(self, *, max_workers, thread_name_prefix):
            max_worker_calls.append(max_workers)
            self._max_workers = max_workers
            self._thread_name_prefix = thread_name_prefix

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            mapped_items = list(items)
            mapped_item_counts.append(len(mapped_items))
            return map(fn, mapped_items)

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.raw_block.core.ThreadPoolExecutor",
        RecordingThreadPoolExecutor,
    )

    core._validate_loaded_entries()

    # 10 entries with 3 workers → 3 work ranges dispatched
    assert max_worker_calls == [3]
    assert mapped_item_counts == [3]
    assert list(core._index) == [spec.encoded for spec in specs]


def test_raw_block_core_iouring_recovery_does_not_use_posix_threads(monkeypatch):
    specs = [encode_object_key(make_object_key(i)) for i in range(51, 61)]
    core = object.__new__(RawBlockCore)
    core._lock = threading.Lock()
    core._index = {
        spec.encoded: type("Entry", (), {"offset": 4096 * (i + 1), "size": 7})()
        for i, spec in enumerate(specs)
    }
    core._lock_refcnt = {}
    core._meta_dirty_total = 0
    core.io_engine = "io_uring"
    core.use_uring_cmd = False
    core._recovery_read_threads = 8
    core.key_namespace = "object"
    core._read_slot_header = lambda offset: (
        specs[(offset // 4096) - 1].slot_identity,
        7,
    )
    # io_uring dispatch reads headers via the batched path; stub it to the
    # single-read helper so this test exercises dispatch without real io_uring.
    core._read_slot_headers_batched = lambda offsets: [
        core._read_slot_header(off) for off in offsets
    ]

    def fail_thread_pool_executor(*args, **kwargs):
        raise AssertionError("io_uring recovery must not use POSIX read threads")

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.raw_block.core.ThreadPoolExecutor",
        fail_thread_pool_executor,
    )

    core._validate_loaded_entries()

    assert list(core._index) == [spec.encoded for spec in specs]


def _make_recovery_core(
    specs: list[Any],
    *,
    io_engine: str = "io_uring",
    use_uring_cmd: bool = False,
) -> RawBlockCore:
    """Build a minimal RawBlockCore for recovery dispatch tests."""
    core = object.__new__(RawBlockCore)
    core._lock = threading.Lock()
    core._index = {
        spec.encoded: type("Entry", (), {"offset": 4096 * (i + 1), "size": 64})()
        for i, spec in enumerate(specs)
    }
    core._lock_refcnt = {}
    core._meta_dirty_total = 0
    core.io_engine = io_engine
    core.use_uring_cmd = use_uring_cmd
    core._recovery_read_threads = 1
    core.key_namespace = "object"
    return core


def _make_iouring_header_core() -> RawBlockCore:
    """Build a minimal RawBlockCore for batched slot-header reads."""
    core = object.__new__(RawBlockCore)
    core._lock = threading.Lock()
    core._inflight_io_count = 0
    core._last_io_ts = 0.0
    core.block_align = 4096
    core.header_bytes = 4096
    core.iouring_queue_depth = 256
    return core


def _slot_header(core: RawBlockCore, identity: int, payload_len: int) -> bytes:
    header = bytearray(core.header_bytes)
    header[0:8] = b"LMCBLK01"
    header[8:16] = int(identity).to_bytes(8, "little")
    header[16:24] = int(payload_len).to_bytes(8, "little")
    return bytes(header)


def test_read_slot_headers_batched_reads_and_decodes_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # White-box: single-batch decode and the batch_id -> wait_iouring wiring have
    # no public projection, so this asserts the io_uring read path directly.
    core = _make_iouring_header_core()
    offsets = [4096, 8192]
    expected = [(0xCAFE, 128), (0xBEEF, 256)]
    raw_dev = Mock()

    def batched_read(
        batch_offsets: list[int],
        buffers: list[memoryview],
        total_lens: list[int],
    ) -> int:
        assert batch_offsets == offsets
        assert total_lens == [core.header_bytes, core.header_bytes]
        for buffer, header in zip(
            buffers,
            [_slot_header(core, identity, size) for identity, size in expected],
            strict=True,
        ):
            buffer[:] = header
        return 77

    raw_dev.batched_read.side_effect = batched_read
    monkeypatch.setattr(core, "_rawdev", lambda: raw_dev)

    assert core._read_slot_headers_batched(offsets) == expected
    raw_dev.wait_iouring.assert_called_once_with(77)


def test_read_slot_headers_batched_splits_by_iouring_queue_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # White-box: queue_depth batching has no public projection; the only
    # observable is the batched_read call pattern, so assert it directly.
    core = _make_iouring_header_core()
    core.iouring_queue_depth = 2
    offsets = [4096, 8192, 12288, 16384, 20480]
    seen_batches: list[list[int]] = []
    raw_dev = Mock()

    def batched_read(
        batch_offsets: list[int],
        buffers: list[memoryview],
        total_lens: list[int],
    ) -> int:
        seen_batches.append(list(batch_offsets))
        for offset, buffer in zip(batch_offsets, buffers, strict=True):
            buffer[:] = _slot_header(core, offset, 64)
        return len(seen_batches)

    raw_dev.batched_read.side_effect = batched_read
    monkeypatch.setattr(core, "_rawdev", lambda: raw_dev)

    assert core._read_slot_headers_batched(offsets) == [
        (4096, 64),
        (8192, 64),
        (12288, 64),
        (16384, 64),
        (20480, 64),
    ]
    assert seen_batches == [[4096, 8192], [12288, 16384], [20480]]
    assert raw_dev.wait_iouring.call_count == 3


def test_read_slot_headers_batched_falls_back_per_slot_on_batch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # White-box: per-slot fallback isolation has no public projection; the only
    # observable is the per-slot re-read pattern, so assert it directly.
    core = _make_iouring_header_core()
    raw_dev = Mock()
    raw_dev.batched_read.side_effect = RuntimeError("read failed")
    monkeypatch.setattr(core, "_rawdev", lambda: raw_dev)
    read_mock = Mock(side_effect=[(1, 64), None, (3, 64)])
    monkeypatch.setattr(core, "_read_slot_header", read_mock)

    assert core._read_slot_headers_batched([4096, 8192, 12288]) == [
        (1, 64),
        None,
        (3, 64),
    ]
    assert read_mock.call_args_list == [
        call(4096),
        call(8192),
        call(12288),
    ]


def test_validate_loaded_entries_iouring_multi_entry_uses_batched_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [encode_object_key(make_object_key(i)) for i in range(3)]
    core = _make_recovery_core(specs)
    expected_offsets = [4096, 8192, 12288]
    batched_mock = Mock(return_value=[(spec.slot_identity, 64) for spec in specs])
    monkeypatch.setattr(core, "_read_slot_headers_batched", batched_mock)
    read_mock = Mock()
    monkeypatch.setattr(core, "_read_slot_header", read_mock)

    core._validate_loaded_entries()

    batched_mock.assert_called_once_with(expected_offsets)
    read_mock.assert_not_called()
    assert list(core._index) == [spec.encoded for spec in specs]


def test_validate_loaded_entries_uring_cmd_keeps_sequential_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [encode_object_key(make_object_key(i)) for i in range(3)]
    core = _make_recovery_core(specs, use_uring_cmd=True)
    batched_mock = Mock()
    monkeypatch.setattr(core, "_read_slot_headers_batched", batched_mock)
    read_mock = Mock(side_effect=[(spec.slot_identity, 64) for spec in specs])
    monkeypatch.setattr(core, "_read_slot_header", read_mock)

    core._validate_loaded_entries()

    batched_mock.assert_not_called()
    assert read_mock.call_args_list == [
        call(4096),
        call(8192),
        call(12288),
    ]
    assert list(core._index) == [spec.encoded for spec in specs]
