# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
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
    core._recovery_read_threads = 8
    core.key_namespace = "object"
    core._read_slot_header = lambda offset: (
        specs[(offset // 4096) - 1].slot_identity,
        7,
    )

    def fail_thread_pool_executor(*args, **kwargs):
        raise AssertionError("io_uring recovery must not use POSIX read threads")

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.raw_block.core.ThreadPoolExecutor",
        fail_thread_pool_executor,
    )

    core._validate_loaded_entries()

    assert list(core._index) == [spec.encoded for spec in specs]
