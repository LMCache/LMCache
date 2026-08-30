# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import patch
import ctypes
import dataclasses
import importlib.util
import stat
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockCoreConfig,
    encode_object_key,
    normalize_raw_block_placement_ids,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)
import lmcache.v1.storage_backend.raw_block.core as raw_block_core

Buffer = bytes | bytearray | memoryview

requires_rust_raw_block_io = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)


def test_normalize_raw_block_placement_ids_rejects_out_of_range() -> None:
    assert normalize_raw_block_placement_ids([65535], 1) == [65535]

    with pytest.raises(ValueError, match="range 1..=65535"):
        normalize_raw_block_placement_ids([65536], 1)


class _RecordingUringCmdRawDevice:
    def __init__(self) -> None:
        self.offsets: list[int] = []
        self.buffers: list[memoryview] = []
        self.lengths: list[int] = []
        self.read_buffers: list[memoryview] = []
        self.read_data = b""
        self.read_cursor = 0
        self.waited_batch_id: int | None = None

    def batched_write(
        self,
        offsets: list[int],
        buffers: list[memoryview],
        lengths: list[int],
        placement_ids: list[int | None] | None = None,
    ) -> int:
        del placement_ids
        self.offsets = offsets
        self.buffers = buffers
        self.lengths = lengths
        return 17

    def wait_iouring(self, batch_id: int) -> None:
        self.waited_batch_id = batch_id

    def read_uring(
        self,
        offset: int,
        target: memoryview,
        payload_len: int,
        total_len: int,
    ) -> None:
        del offset, payload_len
        self.read_buffers.append(target)
        end = self.read_cursor + total_len
        target[:total_len] = self.read_data[self.read_cursor : end]
        self.read_cursor = end


def _buffer_address(buf: memoryview) -> int:
    return ctypes.addressof((ctypes.c_byte * 1).from_buffer(buf))


def test_raw_block_core_uring_cmd_write_padding_uses_aligned_chunks(monkeypatch):
    core = RawBlockCore.__new__(RawBlockCore)
    core.block_align = 4096
    core.max_data_transfer_size = 4096
    raw_dev = _RecordingUringCmdRawDevice()
    monkeypatch.setattr(core, "_rawdev", lambda: raw_dev)

    payload = bytes([3]) * 5000

    core._write_uring_cmd_buffers(
        offsets=[4096],
        buffers=[bytearray(payload)],
        payload_lens=[len(payload)],
        total_lens=[8192],
    )

    assert raw_dev.offsets == [4096, 8192]
    assert raw_dev.lengths == [4096, 4096]
    assert raw_dev.waited_batch_id == 17
    assert all(_buffer_address(buf) % core.block_align == 0 for buf in raw_dev.buffers)
    assert b"".join(bytes(buf) for buf in raw_dev.buffers) == payload + bytes(3192)


def test_raw_block_core_uring_cmd_read_copyback_uses_aligned_chunks(monkeypatch):
    core = RawBlockCore.__new__(RawBlockCore)
    core.block_align = 4096
    core.max_data_transfer_size = 4096
    raw_dev = _RecordingUringCmdRawDevice()
    monkeypatch.setattr(core, "_rawdev", lambda: raw_dev)

    payload = bytes([5]) * 5000
    raw_dev.read_data = payload + bytes(3192)
    dst = bytearray(len(payload))

    core._read_uring_cmd_buffers(
        offsets=[4096],
        buffers=[dst],
        payload_lens=[len(payload)],
        total_lens=[8192],
    )

    assert dst == payload
    assert all(
        _buffer_address(buf) % core.block_align == 0 for buf in raw_dev.read_buffers
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


@requires_rust_raw_block_io
@pytest.mark.parametrize(
    ("field_name", "mismatched_value"),
    [
        ("block_align", 8192),
        ("header_bytes", 8192),
    ],
)
def test_raw_block_core_rejects_checkpoint_layout_mismatch(
    tmp_path,
    field_name,
    mismatched_value,
):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        state = {
            "version": 1,
            "device_path": str(path),
            "capacity_bytes": core.capacity_bytes,
            "block_align": core.block_align,
            "header_bytes": core.header_bytes,
            "slot_bytes": core.slot_bytes,
            "meta_total_bytes": core.meta_total_bytes,
            "meta_magic": core.meta_magic_text,
            "meta_version": core.meta_version,
            "data_base_offset": core.data_base_offset(),
            "next_slot": 0,
            "free_slots": [],
            "entries": {},
        }
        state[field_name] = mismatched_value

        assert core.apply_loaded_state(state) is False
    finally:
        core.close()


@dataclass
class _BatchedWriteCall:
    """Recorded arguments of one fake ``batched_write`` invocation."""

    offsets: list[int]
    buffer_byte_lens: list[int]
    total_lens: list[int]
    placement_ids: list[int | None]


@dataclass
class _RecordingRawDevice:
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
        buffers: Sequence[Buffer],
        total_lens: Sequence[int],
        placement_ids: Sequence[int | None] | None = None,
    ) -> int:
        self.batched_write_calls.append(
            _BatchedWriteCall(
                offsets=[int(off) for off in offsets],
                buffer_byte_lens=[len(bytes(buf)) for buf in buffers],
                total_lens=[int(total) for total in total_lens],
                placement_ids=list(placement_ids or [None] * len(offsets)),
            )
        )
        if self.fail_batched_write:
            raise RuntimeError("injected batched_write failure")
        for i, (off, buf, total) in enumerate(
            zip(offsets, buffers, total_lens, strict=True)
        ):
            if (
                self.fail_after_write_entries is not None
                and i >= self.fail_after_write_entries
            ):
                raise RuntimeError("injected partial batched_write failure")
            self.store[int(off)] = bytes(buf)[: int(total)]
        return len(self.batched_write_calls)

    def wait_iouring(self, batch_id: int) -> None:
        self.wait_iouring_count += 1

    def batched_read(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Buffer],
        total_lens: Sequence[int],
    ) -> int:
        for off, buf, total in zip(offsets, buffers, total_lens, strict=True):
            self._copy_into(int(off), buf, int(total))
        return -1

    def pwrite_from_buffer(
        self, offset: int, buf: Buffer, payload_len: int, total_len: int
    ) -> None:
        self.pwrite_count += 1
        self.store[int(offset)] = bytes(buf)[: int(total_len)]

    def write_uring(
        self,
        offset: int,
        buf: Buffer,
        payload_len: int,
        total_len: int,
        placement_id: int | None = None,
    ) -> None:
        self.write_uring_count += 1
        self.store[int(offset)] = bytes(buf)[: int(total_len)]

    def pread_into(
        self, offset: int, buf: Buffer, payload_len: int, total_len: int
    ) -> None:
        self._copy_into(int(offset), buf, int(total_len))

    def close(self) -> None:
        pass

    def _copy_into(self, offset: int, buf: Buffer, total_len: int) -> None:
        data = self.store.get(offset, b"")
        view = memoryview(buf).cast("B")
        n = min(len(data), total_len, len(view))
        if n:
            view[:n] = data[:n]


def _make_core_with_fake(
    path: Path,
    fake: _RecordingRawDevice,
    io_engine: str,
    capacity_bytes: int | None = None,
) -> RawBlockCore:
    """Build a RawBlockCore wired to a fake raw device for a given engine.

    When ``capacity_bytes`` is given it overrides the config capacity so a
    test can constrain the number of allocatable slots.
    """
    config = replace(
        make_raw_block_core_config(path),
        io_engine=io_engine,
        load_checkpoint_on_init=False,
    )
    if capacity_bytes is not None:
        config = replace(config, capacity_bytes=capacity_bytes)
    with patch.object(RawBlockCore, "_rawdev", return_value=fake):
        core = RawBlockCore(config, key_namespace="object")
    core.set_raw_device_for_testing(fake)
    return core


def _available_slots(status: Mapping[str, int]) -> int:
    """Return slots still allocatable from the free list plus the high-water tail."""
    return status["free_slot_count"] + (status["max_slots"] - status["next_slot"])


def test_raw_block_core_io_uring_put_many_single_submit(tmp_path: Path) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_header_buffer_length_guard(tmp_path: Path) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_put_many_round_trip(tmp_path: Path) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_put_many_skips_already_indexed(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_put_many_all_or_nothing_rollback(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_put_many_partial_slot_exhaustion(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_put_many_duplicate_keys_in_batch(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_put_many_oversize_key_isolated(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


def test_raw_block_core_io_uring_put_many_prep_failure_leaves_no_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A buffer-preparation failure must queue nothing for the failed key.

    The failed key's slot is returned to the free list, so leaving its header
    in the shared submission would write into a slot the allocator considers
    free.
    """
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
    core = _make_core_with_fake(path, fake, io_engine="io_uring")

    real_prepare = RawBlockCore._prepare_write_payload
    prepare_calls: list[int] = []

    def failing_prepare(
        self: RawBlockCore, memory_obj: MemoryObj
    ) -> tuple[object, int, int]:
        prepare_calls.append(len(memory_obj.byte_array))
        if len(prepare_calls) == 2:
            raise RuntimeError("injected preparation failure")
        return real_prepare(self, memory_obj)

    monkeypatch.setattr(RawBlockCore, "_prepare_write_payload", failing_prepare)

    try:
        before = _available_slots(core.report_status())
        specs = [encode_object_key(make_object_key(i)) for i in range(80, 83)]
        objects = [make_memory_obj(bytes([i + 1]) * 1024) for i in range(3)]

        result = core.put_many(specs, objects)

        assert result.results == [True, False, True]
        assert result.stored_keys == [specs[0].encoded, specs[2].encoded]

        # Only the two surviving keys may reach the device, each contributing a
        # header and a payload entry.
        assert len(fake.batched_write_calls) == 1
        submitted = fake.batched_write_calls[0].offsets
        assert len(submitted) == 4

        # Nothing may be written into the failed key's reclaimed slot.
        failed_slot_offset = RAW_BLOCK_CI_META_TOTAL_BYTES + RAW_BLOCK_CI_SLOT_BYTES
        failed_slot_end = failed_slot_offset + RAW_BLOCK_CI_SLOT_BYTES
        assert not [
            offset
            for offset in submitted
            if failed_slot_offset <= offset < failed_slot_end
        ]

        status = core.report_status()
        assert status["inflight_key_count"] == 0
        assert status["indexed_key_count"] == 2
        assert _available_slots(status) == before - 2

        # The reclaimed slot stays usable for a later key.
        reuse_spec = encode_object_key(make_object_key(90))
        assert core.put_many([reuse_spec], [make_memory_obj(b"z" * 1024)]).results == [
            True
        ]
        assert fake.batched_write_calls[1].offsets[0] == failed_slot_offset
    finally:
        core.close()


def test_raw_block_core_io_uring_put_many_partial_completion_rolls_back(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024, fail_after_write_entries=3)
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


def test_raw_block_core_posix_put_many_uses_sequential_pwrite(tmp_path: Path) -> None:
    path = make_raw_block_file(tmp_path)
    fake = _RecordingRawDevice(size=128 * 1024 * 1024)
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


@requires_rust_raw_block_io
def test_raw_block_core_rebuilds_missing_free_slots_from_checkpoint(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    existing = encode_object_key(make_object_key(41))
    recovered = encode_object_key(make_object_key(42))
    existing_payload = b"already-committed"
    recovered_payload = b"recovered-hole"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([existing], [make_memory_obj(existing_payload)])
        assert put_result.results == [True]

        committed_offset = core.entry_offset(existing.encoded)
        assert committed_offset == core.data_base_offset()

        applied = core.apply_loaded_state(
            {
                "version": 1,
                "device_path": str(path),
                "capacity_bytes": core.capacity_bytes,
                "block_align": core.block_align,
                "header_bytes": core.header_bytes,
                "slot_bytes": core.slot_bytes,
                "meta_total_bytes": core.meta_total_bytes,
                "meta_magic": core.meta_magic_text,
                "meta_version": core.meta_version,
                "data_base_offset": core.data_base_offset(),
                # Simulates a checkpoint taken after slot 1 was reserved but
                # before its key was committed into the metadata index.
                "next_slot": 2,
                # Older checkpoints include free_slots; keep this empty to
                # verify recovery ignores stale/missing free-list data and
                # reconstructs reusable slots from entries plus next_slot.
                "free_slots": [],
                "entries": {
                    existing.encoded: {
                        "offset": committed_offset,
                        "size": len(existing_payload),
                        "shape": [len(existing_payload)],
                        "dtype": "uint8",
                        "fmt": "BINARY",
                        "cached_positions": None,
                    }
                },
            }
        )

        assert applied is True
        status = core.report_status()
        assert status["next_slot"] == 2
        assert status["free_slot_count"] == 1

        put_recovered = core.put_many([recovered], [make_memory_obj(recovered_payload)])

        assert put_recovered.results == [True]
        assert core.entry_offset(recovered.encoded) == (
            core.data_base_offset() + core.slot_bytes
        )
        assert core.report_status()["next_slot"] == 2
    finally:
        core.close()


class _FakeRawDevice:
    def __init__(self, size_bytes: int = RAW_BLOCK_CI_CAPACITY_BYTES) -> None:
        self._size_bytes = int(size_bytes)
        self.batched_write_calls: list[
            tuple[list[int], list[int], list[int | None] | None]
        ] = []
        self.write_uring_calls: list[tuple[int, int, int, int | None]] = []

    def size_bytes(self) -> int:
        return self._size_bytes

    def pread_into(self, offset, out, payload_len, total_len=None):
        del offset, total_len
        out[:payload_len] = b"\x00" * payload_len

    def pwrite_from_buffer(self, offset, data, payload_len=None, total_len=None):
        del offset, data, payload_len, total_len

    def batched_write(
        self,
        offsets: list[int],
        buffers: list[bytearray],
        total_lens: list[int],
        placement_ids: list[int | None] | None = None,
    ) -> int:
        del buffers
        self.batched_write_calls.append((offsets, total_lens, placement_ids))
        return 123

    def wait_iouring(self, batch_id: int) -> None:
        assert batch_id == 123

    def write_uring(
        self,
        offset: int,
        data: bytearray,
        payload_len: int,
        total_len: int,
        placement_id: int | None = None,
    ) -> None:
        del data
        self.write_uring_calls.append((offset, payload_len, total_len, placement_id))

    def close(self) -> None:
        return None


def _make_fake_io_uring_core(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_uring_cmd: bool = False,
    max_data_transfer_size: int = 0,
    meta_checkpoint_placement_id: int | None = None,
    fdp_slot_affinity_enabled: bool = False,
) -> tuple[RawBlockCore, _FakeRawDevice]:
    raw_devices: list[_FakeRawDevice] = []

    def create_fake_device(path: str, **kwargs):
        del path, kwargs
        raw_device = _FakeRawDevice()
        raw_devices.append(raw_device)
        return raw_device

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(RawBlockDevice=create_fake_device),
    )
    if use_uring_cmd:
        monkeypatch.setattr(
            raw_block_core.os,
            "stat",
            lambda path: types.SimpleNamespace(st_mode=stat.S_IFCHR),
        )

    device_path = tmp_path / "ng0n1"
    core = RawBlockCore(
        RawBlockCoreConfig(
            device_path=str(device_path),
            capacity_bytes=RAW_BLOCK_CI_CAPACITY_BYTES,
            block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
            header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
            slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
            use_odirect=False,
            enable_zero_copy=False,
            meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=60,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            load_checkpoint_on_init=True,
            meta_verify_on_load=True,
            io_engine="io_uring",
            iouring_queue_depth=8,
            use_uring_cmd=use_uring_cmd,
            max_data_transfer_size=max_data_transfer_size,
            meta_checkpoint_placement_id=meta_checkpoint_placement_id,
            fdp_slot_affinity_enabled=fdp_slot_affinity_enabled,
        ),
        key_namespace="object",
    )
    return core, raw_devices[0]


def test_raw_block_core_checkpoint_uses_metadata_placement_id(tmp_path, monkeypatch):
    core, raw_device = _make_fake_io_uring_core(
        tmp_path,
        monkeypatch,
        use_uring_cmd=True,
        max_data_transfer_size=RAW_BLOCK_CI_BLOCK_ALIGN,
        meta_checkpoint_placement_id=7,
    )
    spec = encode_object_key(make_object_key(505))

    try:
        assert core.put_many([spec], [make_memory_obj(b"checkpoint")]).results == [True]
        core.checkpoint_now()

        checkpoint_calls = raw_device.batched_write_calls[1:]
        assert checkpoint_calls
        assert checkpoint_calls[-1][2] == [7, 7]
    finally:
        core.close()


def test_raw_block_core_checkpoint_placement_requires_uring_cmd(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="meta_checkpoint_placement_id requires"):
        _make_fake_io_uring_core(
            tmp_path,
            monkeypatch,
            meta_checkpoint_placement_id=7,
        )


def test_raw_block_core_checkpoint_rejects_zero_metadata_placement_id(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="placement identifier 0"):
        _make_fake_io_uring_core(
            tmp_path,
            monkeypatch,
            meta_checkpoint_placement_id=0,
        )


def test_raw_block_core_put_many_preserves_none_and_positive_placement(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(tmp_path, monkeypatch)
    try:
        specs = [encode_object_key(make_object_key(500 + i)) for i in range(2)]
        put_result = core.put_many(
            specs,
            [make_memory_obj(b"a"), make_memory_obj(b"b")],
            placement_ids=[None, 1],
        )

        assert put_result.results == [True, True]
        # The io_uring path submits the whole batch at once, so both keys share
        # one submission and each key's header/payload entries carry that key's
        # placement identifier.
        assert [call[2] for call in raw_device.batched_write_calls] == [
            [None, None, 1, 1],
        ]
    finally:
        core.close()


def test_raw_block_core_put_many_rejects_zero_placement_before_io(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(tmp_path, monkeypatch)
    spec = encode_object_key(make_object_key(502))

    try:
        with pytest.raises(ValueError, match="placement identifier 0"):
            core.put_many([spec], [make_memory_obj(b"data")], placement_ids=[0])

        assert raw_device.batched_write_calls == []
    finally:
        core.close()


def test_raw_block_core_put_many_sets_same_placement_for_header_and_payload(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(tmp_path, monkeypatch)
    spec = encode_object_key(make_object_key(503))

    try:
        assert core.put_many(
            [spec], [make_memory_obj(b"data")], placement_ids=[1]
        ).results == [True]
        assert [call[2] for call in raw_device.batched_write_calls] == [[1, 1]]
    finally:
        core.close()


def test_raw_block_core_put_many_chunks_uring_cmd_with_placement_ids(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(
        tmp_path,
        monkeypatch,
        use_uring_cmd=True,
        max_data_transfer_size=RAW_BLOCK_CI_BLOCK_ALIGN,
    )
    spec = encode_object_key(make_object_key(504))

    try:
        assert core.put_many(
            [spec],
            [make_memory_obj(b"x" * (RAW_BLOCK_CI_BLOCK_ALIGN * 2))],
            placement_ids=[7],
        ).results == [True]

        assert len(raw_device.batched_write_calls) == 1
        _, total_lens, placement_ids = raw_device.batched_write_calls[0]
        assert total_lens == [
            RAW_BLOCK_CI_BLOCK_ALIGN,
            RAW_BLOCK_CI_BLOCK_ALIGN,
            RAW_BLOCK_CI_BLOCK_ALIGN,
        ]
        assert placement_ids == [7, 7, 7]
    finally:
        core.close()


def test_raw_block_core_reuses_free_slot_with_matching_placement_id(
    tmp_path, monkeypatch
):
    core, _ = _make_fake_io_uring_core(
        tmp_path,
        monkeypatch,
        fdp_slot_affinity_enabled=True,
    )
    first, second, replacement = [
        encode_object_key(make_object_key(600 + i)) for i in range(3)
    ]

    try:
        assert core.put_many(
            [first, second],
            [make_memory_obj(b"a"), make_memory_obj(b"b")],
            placement_ids=[1, 7],
        ).results == [True, True]
        first_offset = core.entry_offset(first.encoded)
        second_offset = core.entry_offset(second.encoded)
        assert first_offset is not None
        assert second_offset is not None
        assert first_offset != second_offset

        assert core.delete_many([first.encoded, second.encoded]) == [True, True]
        assert core.put_many(
            [replacement],
            [make_memory_obj(b"c")],
            placement_ids=[1],
        ).results == [True]

        assert core.entry_offset(replacement.encoded) == first_offset
        status = core.report_status()
        assert status["fdp_slot_affinity_hit_count"] == 1
        assert status["fdp_slot_affinity_fallback_count"] == 0
    finally:
        core.close()


def test_raw_block_core_falls_back_and_rebinds_slot_placement_id(tmp_path, monkeypatch):
    core, _ = _make_fake_io_uring_core(
        tmp_path,
        monkeypatch,
        fdp_slot_affinity_enabled=True,
    )
    first, second, fallback, replacement = [
        encode_object_key(make_object_key(610 + i)) for i in range(4)
    ]

    try:
        assert core.put_many(
            [first, second],
            [make_memory_obj(b"a"), make_memory_obj(b"b")],
            placement_ids=[1, 7],
        ).results == [True, True]
        second_offset = core.entry_offset(second.encoded)
        assert second_offset is not None

        assert core.delete_many([first.encoded, second.encoded]) == [True, True]
        assert core.put_many(
            [fallback],
            [make_memory_obj(b"c")],
            placement_ids=[9],
        ).results == [True]
        assert core.entry_offset(fallback.encoded) == second_offset

        assert core.delete_many([fallback.encoded]) == [True]
        assert core.put_many(
            [replacement],
            [make_memory_obj(b"d")],
            placement_ids=[9],
        ).results == [True]
        assert core.entry_offset(replacement.encoded) == second_offset

        status = core.report_status()
        assert status["fdp_slot_affinity_hit_count"] == 1
        assert status["fdp_slot_affinity_fallback_count"] == 1
    finally:
        core.close()


def test_raw_block_core_slot_affinity_none_preserves_global_lifo(tmp_path, monkeypatch):
    core, _ = _make_fake_io_uring_core(tmp_path, monkeypatch)
    first, second, replacement = [
        encode_object_key(make_object_key(620 + i)) for i in range(3)
    ]

    try:
        assert core.put_many(
            [first, second],
            [make_memory_obj(b"a"), make_memory_obj(b"b")],
            placement_ids=[1, 7],
        ).results == [True, True]
        second_offset = core.entry_offset(second.encoded)
        assert second_offset is not None

        assert core.delete_many([first.encoded, second.encoded]) == [True, True]
        assert core.put_many(
            [replacement],
            [make_memory_obj(b"c")],
            placement_ids=[1],
        ).results == [True]

        assert core.entry_offset(replacement.encoded) == second_offset
        status = core.report_status()
        assert status["fdp_slot_affinity_enabled"] is False
        assert status["fdp_slot_affinity_hit_count"] == 0
        assert status["fdp_slot_affinity_fallback_count"] == 0
    finally:
        core.close()


def test_raw_block_core_omitted_placement_clears_previous_affinity(
    tmp_path, monkeypatch
):
    core, _ = _make_fake_io_uring_core(
        tmp_path,
        monkeypatch,
        fdp_slot_affinity_enabled=True,
    )
    first, second, unplaced, replacement = [
        encode_object_key(make_object_key(630 + i)) for i in range(4)
    ]

    try:
        assert core.put_many(
            [first, second],
            [make_memory_obj(b"a"), make_memory_obj(b"b")],
            placement_ids=[7, 1],
        ).results == [True, True]
        second_offset = core.entry_offset(second.encoded)
        assert second_offset is not None

        assert core.delete_many([first.encoded, second.encoded]) == [True, True]
        assert core.put_many([unplaced], [make_memory_obj(b"c")]).results == [True]
        assert core.entry_offset(unplaced.encoded) == second_offset
        assert core.delete_many([unplaced.encoded]) == [True]

        assert core.put_many(
            [replacement],
            [make_memory_obj(b"d")],
            placement_ids=[1],
        ).results == [True]
        assert core.entry_offset(replacement.encoded) == second_offset

        status = core.report_status()
        assert status["fdp_slot_affinity_hit_count"] == 0
        assert status["fdp_slot_affinity_fallback_count"] == 1
    finally:
        core.close()


@requires_rust_raw_block_io
def test_raw_block_core_does_not_restore_slot_affinity_from_checkpoint(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = dataclasses.replace(
        make_raw_block_core_config(path),
        fdp_slot_affinity_enabled=True,
    )
    original = encode_object_key(make_object_key(640))
    replacement = encode_object_key(make_object_key(641))

    core = RawBlockCore(config, key_namespace="object")
    try:
        assert core.put_many(
            [original],
            [make_memory_obj(b"before-restart")],
            placement_ids=[7],
        ).results == [True]
        original_offset = core.entry_offset(original.encoded)
        assert original_offset is not None
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.delete_many([original.encoded]) == [True]
        assert recovered.put_many(
            [replacement],
            [make_memory_obj(b"after-restart")],
            placement_ids=[7],
        ).results == [True]
        assert recovered.entry_offset(replacement.encoded) == original_offset

        status = recovered.report_status()
        assert status["fdp_slot_affinity_hit_count"] == 0
        assert status["fdp_slot_affinity_fallback_count"] == 1
    finally:
        recovered.close()
