# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from unittest.mock import patch
import importlib.util

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

requires_rust_raw_block_io = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)


@requires_rust_raw_block_io
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


@requires_rust_raw_block_io
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


@requires_rust_raw_block_io
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


@dataclass
class _BatchedWriteCall:
    """Recorded arguments of one fake ``batched_write`` invocation."""

    offsets: list[int]
    buffer_byte_lens: list[int]
    total_lens: list[int]
    payload_lens: list[int]


@dataclass
class _FakeRawDevice:
    """In-memory raw device that records io_uring batched submissions.

    The fake mirrors the subset of the Rust raw-device interface that
    ``RawBlockCore`` uses for both the io_uring and posix write/read paths.
    Stored bytes are keyed by device offset so round-trip reads return what
    was written.
    """

    size: int
    store: dict[int, bytes] = field(default_factory=dict)
    batched_write_calls: list[_BatchedWriteCall] = field(default_factory=list)
    wait_iouring_count: int = 0
    pwrite_count: int = 0
    write_uring_count: int = 0
    fail_batched_write: bool = False
    fail_after_write_entries: int | None = None

    def size_bytes(self) -> int:
        return self.size

    def batched_write(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        total_lens: Sequence[int],
        payload_lens: Sequence[int] | None = None,
    ) -> int:
        resolved_payload_lens = (
            [int(p) for p in payload_lens]
            if payload_lens is not None
            else [int(total) for total in total_lens]
        )
        self.batched_write_calls.append(
            _BatchedWriteCall(
                offsets=[int(off) for off in offsets],
                buffer_byte_lens=[len(bytes(buf)) for buf in buffers],
                total_lens=[int(total) for total in total_lens],
                payload_lens=resolved_payload_lens,
            )
        )
        if self.fail_batched_write:
            raise RuntimeError("injected batched_write failure")
        for i, (off, buf, total, payload) in enumerate(
            zip(offsets, buffers, total_lens, resolved_payload_lens, strict=True)
        ):
            if (
                self.fail_after_write_entries is not None
                and i >= self.fail_after_write_entries
            ):
                raise RuntimeError("injected partial batched_write failure")
            # Mirror the Rust bounce: store only the valid payload, zero-padded
            # up to total so round-trip reads observe the padded transfer.
            self.store[int(off)] = bytes(buf)[: int(payload)].ljust(int(total), b"\x00")
        return len(self.batched_write_calls)

    def wait_iouring(self, batch_id: int) -> None:
        self.wait_iouring_count += 1

    def batched_read(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        total_lens: Sequence[int],
    ) -> int:
        for off, buf, total in zip(offsets, buffers, total_lens, strict=True):
            self._copy_into(int(off), buf, int(total))
        return -1

    def pwrite_from_buffer(
        self, offset: int, buf: Any, payload_len: int, total_len: int
    ) -> None:
        self.pwrite_count += 1
        self.store[int(offset)] = bytes(buf)[: int(total_len)]

    def write_uring(
        self, offset: int, buf: Any, payload_len: int, total_len: int
    ) -> None:
        self.write_uring_count += 1
        self.store[int(offset)] = bytes(buf)[: int(total_len)]

    def pread_into(
        self, offset: int, buf: Any, payload_len: int, total_len: int
    ) -> None:
        self._copy_into(int(offset), buf, int(total_len))

    def read_uring(
        self, offset: int, buf: Any, payload_len: int, total_len: int
    ) -> None:
        self._copy_into(int(offset), buf, int(total_len))

    def close(self) -> None:
        pass

    def _copy_into(self, offset: int, buf: Any, total_len: int) -> None:
        data = self.store.get(offset, b"")
        n = min(len(data), total_len, len(buf))
        if n:
            buf[:n] = data[:n]


def _make_core_with_fake(path, fake, io_engine, capacity_bytes=None, use_odirect=False):
    """Build a RawBlockCore wired to a fake raw device for a given engine.

    When ``capacity_bytes`` is given it overrides the config capacity so a
    test can constrain the number of allocatable slots. When ``use_odirect``
    is set, writes are padded to ``block_align`` so payload_len < total_len.
    """
    overrides = dict(
        io_engine=io_engine, load_checkpoint_on_init=False, use_odirect=use_odirect
    )
    if capacity_bytes is not None:
        overrides["capacity_bytes"] = capacity_bytes
    config = replace(make_raw_block_core_config(path), **overrides)
    with patch.object(RawBlockCore, "_rawdev", return_value=fake):
        core = RawBlockCore(config, key_namespace="object")
    core.set_raw_device_for_testing(fake)
    return core


def _available_slots(status: dict) -> int:
    """Return slots still allocatable from the free list plus the high-water tail."""
    return status["free_slot_count"] + (status["max_slots"] - status["next_slot"])


def test_raw_block_core_io_uring_put_many_single_submit(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(10)]
        objects = [make_memory_obj(bytes([i + 1]) * 1024) for i in range(10)]

        put_result = core.put_many(specs, objects)

        assert put_result.results == [True] * 10
        assert put_result.stored_keys == [spec.encoded for spec in specs]

        assert len(fake.batched_write_calls) == 1
        call = fake.batched_write_calls[0]
        assert len(call.offsets) == 20
        assert len(call.buffer_byte_lens) == 20
        assert len(call.total_lens) == 20
        assert fake.wait_iouring_count == 1
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_chunks_large_batches(
    tmp_path,
):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(70)]
        keys = specs[:65] + [specs[0]] + specs[65:]
        payloads = [bytes([i + 1]) * 1024 for i in range(len(keys))]
        objects = [make_memory_obj(payload) for payload in payloads]

        put_result = core.put_many(keys, objects)

        assert put_result.results == [True] * len(keys)
        assert put_result.stored_keys == [spec.encoded for spec in specs]
        assert [len(call.offsets) for call in fake.batched_write_calls] == [128, 12]
        assert fake.wait_iouring_count == 2

        loaded = make_empty_memory_obj(len(payloads[0]))
        assert core.load_many_into([specs[0].encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payloads[0]
    finally:
        core.close()


def test_raw_block_core_io_uring_header_buffer_length_guard(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(4)]
        objects = [make_memory_obj(bytes([i + 1]) * 2048) for i in range(4)]

        core.put_many(specs, objects)

        call = fake.batched_write_calls[0]
        for buf_len, total_len in zip(
            call.buffer_byte_lens, call.total_lens, strict=True
        ):
            assert buf_len >= total_len
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_round_trip(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(10)]
        payloads = [bytes([i + 1]) * (1024 + i * 16) for i in range(10)]
        objects = [make_memory_obj(payload) for payload in payloads]

        assert core.put_many(specs, objects).results == [True] * 10

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)

        assert load_result == [True] * 10
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_io_uring_padded_odirect_uses_batched_write(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring", use_odirect=True)

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(4)]
        # Non-block-aligned payloads force O_DIRECT padding: payload_len < total_len.
        payloads = [bytes([i + 1]) * (1000 + i * 37) for i in range(4)]
        objects = [make_memory_obj(payload) for payload in payloads]

        assert core.put_many(specs, objects).results == [True] * 4

        # Padded O_DIRECT writes go through batched_write, not the per-entry
        # write_uring fallback.
        assert len(fake.batched_write_calls) == 1
        assert fake.write_uring_count == 0
        call = fake.batched_write_calls[0]
        # payload_lens are forwarded and at least one payload entry is padded.
        assert call.payload_lens != call.total_lens
        for payload_len, total_len in zip(
            call.payload_lens, call.total_lens, strict=True
        ):
            assert payload_len <= total_len

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)
        assert load_result == [True] * 4
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_io_uring_non_padded_forwards_payload_lens(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(3)]
        objects = [make_memory_obj(bytes([i + 1]) * 1024) for i in range(3)]

        assert core.put_many(specs, objects).results == [True] * 3

        assert len(fake.batched_write_calls) == 1
        assert fake.write_uring_count == 0
        call = fake.batched_write_calls[0]
        # No padding required, so payload_lens equals total_lens but is still
        # forwarded explicitly.
        assert call.payload_lens == call.total_lens
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_skips_already_indexed(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        first_specs = [encode_object_key(make_object_key(i)) for i in range(5)]
        first_objs = [make_memory_obj(bytes([i + 1]) * 1024) for i in range(5)]
        assert core.put_many(first_specs, first_objs).results == [True] * 5

        new_specs = [encode_object_key(make_object_key(i)) for i in range(5, 10)]
        new_objs = [make_memory_obj(bytes([i + 1]) * 1024) for i in range(5, 10)]
        combined_specs = first_specs + new_specs
        combined_objs = first_objs + new_objs

        result = core.put_many(combined_specs, combined_objs)

        assert result.results == [True] * 10
        assert result.stored_keys == [spec.encoded for spec in new_specs]
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_all_or_nothing_rollback(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        before = _available_slots(core.report_status())

        specs = [encode_object_key(make_object_key(i)) for i in range(5)]
        objects = [make_memory_obj(bytes([i + 1]) * 1024) for i in range(5)]

        fake.fail_batched_write = True
        result = core.put_many(specs, objects)

        assert result.results == [False] * 5
        assert result.stored_keys == []

        status = core.report_status()
        assert status["inflight_key_count"] == 0
        assert status["indexed_key_count"] == 0
        assert _available_slots(status) == before
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_partial_slot_exhaustion(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    # Capacity leaves room for exactly 3 data slots after the metadata region,
    # so the last keys of a 5-key batch find no free slot.
    capacity = RAW_BLOCK_CI_META_TOTAL_BYTES + 3 * RAW_BLOCK_CI_SLOT_BYTES
    core = _make_core_with_fake(
        path, fake, io_engine="io_uring", capacity_bytes=capacity
    )

    try:
        assert core.report_status()["max_slots"] == 3

        specs = [encode_object_key(make_object_key(i)) for i in range(5)]
        objects = [make_memory_obj(bytes([i + 1]) * 1024) for i in range(5)]

        result = core.put_many(specs, objects)

        # First three keys allocate a slot and commit; the slot-starved tail
        # fails individually while the batch for the rest still succeeds.
        assert result.results == [True, True, True, False, False]
        assert result.stored_keys == [spec.encoded for spec in specs[:3]]

        # Only the committed keys were written: one batched_write of 2*3 entries.
        assert len(fake.batched_write_calls) == 1
        assert len(fake.batched_write_calls[0].offsets) == 6

        status = core.report_status()
        assert status["indexed_key_count"] == 3
        assert status["inflight_key_count"] == 0
        assert _available_slots(status) == 0

        loaded = [make_empty_memory_obj(1024) for _ in range(3)]
        load_result = core.load_many_into([spec.encoded for spec in specs[:3]], loaded)
        assert load_result == [True] * 3
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_duplicate_keys_in_batch(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        spec = encode_object_key(make_object_key(51))
        original = b"original-batch-payload"
        duplicate = b"duplicate-must-not-overwrite"

        result = core.put_many(
            [spec, spec],
            [make_memory_obj(original), make_memory_obj(duplicate)],
        )

        assert result.results == [True, True]
        assert result.stored_keys == [spec.encoded]
        assert len(fake.batched_write_calls) == 1
        assert len(fake.batched_write_calls[0].offsets) == 2

        loaded = make_empty_memory_obj(len(original))
        assert core.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == original
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_oversize_key_isolated(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        before = _available_slots(core.report_status())
        specs = [encode_object_key(make_object_key(i)) for i in range(60, 65)]
        payloads = [
            b"a" * 1024,
            b"b" * 2048,
            b"c" * RAW_BLOCK_CI_SLOT_BYTES,
            b"d" * 3072,
            b"e" * 4096,
        ]
        objects = [make_memory_obj(payload) for payload in payloads]

        result = core.put_many(specs, objects)

        assert result.results == [True, True, False, True, True]
        expected_stored = [spec.encoded for i, spec in enumerate(specs) if i != 2]
        assert result.stored_keys == expected_stored
        assert len(fake.batched_write_calls) == 1
        assert len(fake.batched_write_calls[0].offsets) == 8

        status = core.report_status()
        assert status["inflight_key_count"] == 0
        assert status["indexed_key_count"] == 4
        assert _available_slots(status) == before - 4

        loaded_payloads = [payload for i, payload in enumerate(payloads) if i != 2]
        loaded = [make_empty_memory_obj(len(payload)) for payload in loaded_payloads]
        assert core.load_many_into(expected_stored, loaded) == [True] * 4
        assert [memory_obj_bytes(obj) for obj in loaded] == loaded_payloads
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_partial_completion_rolls_back(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024, fail_after_write_entries=3)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    try:
        before = _available_slots(core.report_status())
        specs = [encode_object_key(make_object_key(i)) for i in range(70, 74)]
        objects = [make_memory_obj(bytes([i]) * 1024) for i in range(4)]

        result = core.put_many(specs, objects)

        assert result.results == [False] * 4
        assert result.stored_keys == []
        assert len(fake.batched_write_calls) == 1
        assert len(fake.store) == 3

        status = core.report_status()
        assert status["inflight_key_count"] == 0
        assert status["indexed_key_count"] == 0
        assert _available_slots(status) == before
        assert core.exists_many([spec.encoded for spec in specs]) == [False] * 4
    finally:
        core.close()


def test_raw_block_core_posix_put_many_uses_sequential_pwrite(tmp_path):
    path = make_raw_block_file(tmp_path)
    fake = _FakeRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="posix")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(3)]
        payloads = [bytes([i + 1]) * 1024 for i in range(3)]
        objects = [make_memory_obj(payload) for payload in payloads]

        put_result = core.put_many(specs, objects)

        assert put_result.results == [True] * 3
        assert fake.batched_write_calls == []
        assert fake.pwrite_count == 6

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)
        assert load_result == [True] * 3
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


@requires_rust_raw_block_io
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
