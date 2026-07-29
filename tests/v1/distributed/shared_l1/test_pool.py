# SPDX-License-Identifier: Apache-2.0
"""Functional tests for the minimal shared-L1 pool state."""

from __future__ import annotations

# Standard
import hashlib
import mmap
import multiprocessing
from dataclasses import fields, replace
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.shared_l1 import (
    InMemorySharedL1Pool,
    InvalidReservationError,
    ObjectAlreadyExistsError,
    ObjectBusyError,
    ReadReservation,
    RegionContractMismatchError,
    SharedMemoryRegion,
    SharedObjectHandle,
    SharedRegionContract,
    StaleHandleError,
)

_REGION_ID = "test-shared-region"
_CAPACITY = 256 * 1024
_ALIGNMENT = 4096
_LAYOUT_ID = "test-layout-v1"


def _deterministic_payload(length: int, seed: int) -> bytes:
    return bytes((index * 37 + seed) % 256 for index in range(length))


class _PoolManager(BaseManager):
    pass


_PoolManager.register(
    "create_pool",
    InMemorySharedL1Pool,
)


def _spawn_writer(
    pool: Any,
    contract: SharedRegionContract,
    region_path: str,
    mapping_offset: int,
    object_key: str,
    payload_length: int,
    payload_seed: int,
    ready: Any,
    publish: Any,
    result_queue: Any,
) -> None:
    payload = _deterministic_payload(payload_length, payload_seed)
    reservation = pool.reserve_write(object_key, payload_length)
    with SharedMemoryRegion(
        region_path,
        contract,
        mapping_offset,
    ) as region:
        region.write(reservation.handle, payload)
        region.flush()
    result_queue.put(reservation.handle)
    ready.set()
    if not publish.wait(timeout=20):
        raise TimeoutError("writer was not allowed to publish")
    pool.finish_write(reservation)


def _spawn_reader(
    pool: Any,
    contract: SharedRegionContract,
    region_path: str,
    mapping_offset: int,
    object_key: str,
    expected_handle: SharedObjectHandle,
    payload_seed: int,
    result_queue: Any,
) -> None:
    reservation = pool.reserve_read(object_key, expected_handle)
    if reservation is None:
        raise RuntimeError("published object was unavailable")
    with SharedMemoryRegion(
        region_path,
        contract,
        mapping_offset,
    ) as region:
        with region.read_view(reservation.handle) as observed:
            expected = _deterministic_payload(
                expected_handle.length,
                payload_seed,
            )
            matched = observed == expected
            digest = hashlib.sha256(observed).hexdigest()
            observed_length = len(observed)
            read_only = observed.readonly
    result_queue.put(
        (
            matched,
            digest,
            observed_length,
            read_only,
        )
    )
    pool.finish_read(reservation)


def _spawn_allocator(
    pool: Any,
    object_key: str,
    length: int,
    result_queue: Any,
) -> None:
    reservation = pool.reserve_write(object_key, length)
    pool.finish_write(reservation)
    result_queue.put(reservation.handle)


def _spawn_same_key_writer(
    pool: Any,
    object_key: str,
    length: int,
    result_queue: Any,
) -> None:
    try:
        reservation = pool.reserve_write(object_key, length)
    except ObjectAlreadyExistsError:
        result_queue.put(("already-exists", None))
        return
    pool.finish_write(reservation)
    result_queue.put(("published", reservation.handle))


def _spawn_pinned_reader(
    pool: Any,
    object_key: str,
    handle: SharedObjectHandle,
    ready: Any,
    release: Any,
) -> None:
    reservation = pool.reserve_read(object_key, handle)
    if reservation is None:
        raise RuntimeError("published object was unavailable")
    ready.set()
    if not release.wait(timeout=20):
        raise TimeoutError("reader pin was not released")
    pool.finish_read(reservation)


def _join_process(process: multiprocessing.Process) -> None:
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("spawned process timed out")
    assert process.exitcode == 0


def _create_region(path: Path, mapping_offset: int = 0) -> None:
    with path.open("wb") as region_file:
        region_file.truncate(mapping_offset + _CAPACITY)


def test_spawned_writer_reader_share_payload_only(tmp_path: Path) -> None:
    region_path = tmp_path / "payload.bin"
    mapping_offset = mmap.PAGESIZE
    _create_region(region_path, mapping_offset)
    payload_length = 32 * 1024
    payload_seed = 11
    context = multiprocessing.get_context("spawn")

    with _PoolManager(ctx=context) as manager:
        pool = manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        contract = pool.region_contract()
        ready = context.Event()
        publish = context.Event()
        writer_results = context.Queue()
        writer = context.Process(
            target=_spawn_writer,
            args=(
                pool,
                contract,
                str(region_path),
                mapping_offset,
                "session/turn",
                payload_length,
                payload_seed,
                ready,
                publish,
                writer_results,
            ),
        )
        writer.start()
        assert ready.wait(timeout=20)
        handle = writer_results.get(timeout=20)
        assert isinstance(handle, SharedObjectHandle)

        # Payload bytes have been written, but WRITING metadata is not readable.
        assert pool.reserve_read("session/turn", handle) is None
        publish.set()
        _join_process(writer)

        reader_results = context.Queue()
        reader = context.Process(
            target=_spawn_reader,
            args=(
                pool,
                contract,
                str(region_path),
                mapping_offset,
                "session/turn",
                handle,
                payload_seed,
                reader_results,
            ),
        )
        reader.start()
        matched, digest, observed_length, read_only = reader_results.get(timeout=20)
        assert matched is True
        assert (
            digest
            == hashlib.sha256(
                _deterministic_payload(payload_length, payload_seed)
            ).hexdigest()
        )
        assert observed_length == payload_length
        assert read_only is True
        _join_process(reader)

        snapshot = pool.snapshot()
        assert snapshot["region_id"] == _REGION_ID
        assert snapshot["capacity"] == _CAPACITY
        assert snapshot["alignment"] == _ALIGNMENT
        assert snapshot["layout_id"] == _LAYOUT_ID
        assert snapshot["generation_epoch"] == contract.generation_epoch
        assert snapshot["next_offset"] == payload_length
        assert snapshot["next_generation"] == handle.generation + 1
        assert snapshot["objects"] == {
            "session/turn": {
                "handle": handle,
                "state": "VALID",
                "active_readers": 0,
            }
        }
        assert "token" not in repr(snapshot)

    # The mapped pool contains payload bytes only, with no metadata superblock.
    assert region_path.read_bytes()[:mapping_offset] == bytes(mapping_offset)
    assert [field.name for field in fields(handle)] == [
        "region_id",
        "offset",
        "length",
        "generation",
    ]
    assert list(tmp_path.iterdir()) == [region_path]


def test_concurrent_allocations_do_not_overlap() -> None:
    context = multiprocessing.get_context("spawn")
    with _PoolManager(ctx=context) as manager:
        pool = manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        result_queue = context.Queue()
        lengths = [101, 4097, 777, 8193, 63, 2048, 5000, 91]
        processes = [
            context.Process(
                target=_spawn_allocator,
                args=(pool, f"object-{index}", length, result_queue),
            )
            for index, length in enumerate(lengths)
        ]

        for process in processes:
            process.start()
        handles = [result_queue.get(timeout=30) for _ in processes]
        for process in processes:
            _join_process(process)

    ordered = sorted(handles, key=lambda handle: handle.offset)
    assert all(handle.offset % _ALIGNMENT == 0 for handle in ordered)
    assert all(
        left.offset + left.length <= right.offset
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    assert len({handle.generation for handle in ordered}) == len(ordered)


def test_concurrent_same_key_writers_have_one_winner() -> None:
    context = multiprocessing.get_context("spawn")
    with _PoolManager(ctx=context) as manager:
        pool = manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_spawn_same_key_writer,
                args=(pool, "same-key", 2048, result_queue),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=20) for _ in processes]
        for process in processes:
            _join_process(process)

        published = [value for outcome, value in results if outcome == "published"]
        losers = [outcome for outcome, _ in results if outcome == "already-exists"]
        assert len(published) == 1
        assert len(losers) == 1
        winner = published[0]
        snapshot = pool.snapshot()
        assert snapshot["next_offset"] == winner.length
        assert snapshot["objects"] == {
            "same-key": {
                "handle": winner,
                "state": "VALID",
                "active_readers": 0,
            }
        }


def test_handles_and_tokens_reject_wrong_ownership() -> None:
    context = multiprocessing.get_context("spawn")
    with _PoolManager(ctx=context) as manager:
        pool = manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        write = pool.reserve_write("object", 1024)
        writing_snapshot = pool.snapshot()
        assert writing_snapshot["objects"]["object"]["state"] == "WRITING"
        assert write.token not in repr(writing_snapshot)
        with pytest.raises(InvalidReservationError):
            pool.finish_write(replace(write, token="not-the-owner"))
        pool.finish_write(write)

        with pytest.raises(ObjectAlreadyExistsError):
            pool.reserve_write("object", 1024)
        with pytest.raises(StaleHandleError):
            pool.reserve_read(
                "object",
                replace(write.handle, region_id="different-region"),
            )
        with pytest.raises(StaleHandleError):
            pool.reserve_read(
                "object",
                replace(write.handle, generation=write.handle.generation + 1),
            )

        read = pool.reserve_read("object", write.handle)
        assert isinstance(read, ReadReservation)
        reading_snapshot = pool.snapshot()
        assert reading_snapshot["objects"]["object"]["active_readers"] == 1
        assert read.token not in repr(reading_snapshot)
        with pytest.raises(InvalidReservationError):
            pool.finish_read(replace(read, token="not-the-owner"))
        pool.abort_read(read)


def test_delete_rejects_writers_and_reader_pins_without_reuse() -> None:
    context = multiprocessing.get_context("spawn")
    with _PoolManager(ctx=context) as manager:
        pool = manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        assert pool.delete("missing") is False

        write = pool.reserve_write("object", 1024)
        with pytest.raises(ObjectBusyError, match="WRITING"):
            pool.delete("object", write.handle)
        pool.finish_write(write)

        with pytest.raises(StaleHandleError):
            pool.delete(
                "object",
                replace(write.handle, generation=write.handle.generation + 1),
            )
        ready = context.Event()
        release = context.Event()
        reader = context.Process(
            target=_spawn_pinned_reader,
            args=(
                pool,
                "object",
                write.handle,
                ready,
                release,
            ),
        )
        reader.start()
        try:
            assert ready.wait(timeout=20)
            assert pool.snapshot()["objects"]["object"]["active_readers"] == 1
            with pytest.raises(ObjectBusyError, match="pinned"):
                pool.delete("object", write.handle)
        finally:
            release.set()
            _join_process(reader)

        assert pool.delete("object", write.handle) is True
        assert pool.delete("object", write.handle) is False
        replacement = pool.reserve_write("object", 1024)
        assert replacement.handle.offset >= write.handle.offset + _ALIGNMENT
        assert replacement.handle.generation != write.handle.generation
        with pytest.raises(StaleHandleError):
            pool.delete("object", write.handle)
        pool.abort_write(replacement)


def test_aborted_write_extent_is_not_reused() -> None:
    context = multiprocessing.get_context("spawn")
    with _PoolManager(ctx=context) as manager:
        pool = manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        abandoned = pool.reserve_write("retry", 1)
        pool.abort_write(abandoned)
        retry = pool.reserve_write("retry", 1)

        assert retry.handle.offset >= abandoned.handle.offset + _ALIGNMENT
        assert retry.handle.generation > abandoned.handle.generation
        pool.finish_write(retry)


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("region_id", "different-region"),
        ("capacity", _CAPACITY // 2),
        ("alignment", _ALIGNMENT * 2),
        ("layout_id", "different-layout"),
        ("generation_epoch", 1),
    ],
)
def test_mapping_rejects_region_contract_mismatch(
    tmp_path: Path,
    field_name: str,
    wrong_value: str | int,
) -> None:
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        _REGION_ID,
        _CAPACITY,
        _ALIGNMENT,
        _LAYOUT_ID,
    )
    advertised = pool.region_contract()
    local_expected = replace(advertised, **{field_name: wrong_value})

    with pytest.raises(RegionContractMismatchError):
        SharedMemoryRegion(
            region_path,
            advertised,
            expected_contract=local_expected,
        )


def test_region_contract_validation_and_matching_map(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="power of two"):
        InMemorySharedL1Pool(
            _REGION_ID,
            _CAPACITY,
            alignment=3,
            layout_id=_LAYOUT_ID,
        )
    with pytest.raises(ValueError, match="layout_id"):
        InMemorySharedL1Pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            layout_id="",
        )

    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        _REGION_ID,
        _CAPACITY,
        _ALIGNMENT,
        _LAYOUT_ID,
    )
    contract = pool.region_contract()
    assert contract == SharedRegionContract(
        _REGION_ID,
        _CAPACITY,
        _ALIGNMENT,
        _LAYOUT_ID,
        contract.generation_epoch,
    )
    with SharedMemoryRegion(
        region_path,
        contract,
        expected_contract=contract,
    ):
        pass


def test_restart_epoch_rejects_old_generation(tmp_path: Path) -> None:
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    context = multiprocessing.get_context("spawn")
    with _PoolManager(ctx=context) as first_manager:
        first = first_manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        first_write = first.reserve_write("object", 1024)
        first.finish_write(first_write)
        old_handle = first_write.handle
        old_contract = first.region_contract()

    with _PoolManager(ctx=context) as second_manager:
        second = second_manager.create_pool(
            _REGION_ID,
            _CAPACITY,
            _ALIGNMENT,
            _LAYOUT_ID,
        )
        second_write = second.reserve_write("object", 1024)
        second.finish_write(second_write)
        new_handle = second_write.handle
        new_contract = second.region_contract()

        assert new_handle.offset == old_handle.offset
        assert new_handle.length == old_handle.length
        assert new_handle.generation != old_handle.generation
        assert new_contract.generation_epoch != old_contract.generation_epoch
        with pytest.raises(StaleHandleError):
            second.reserve_read("object", old_handle)
        with SharedMemoryRegion(region_path, new_contract) as region:
            with pytest.raises(StaleHandleError, match="epoch"):
                with region.read_view(old_handle):
                    pass
