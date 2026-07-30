# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the coordinator child's monotonic metadata state."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.shared_l1.pool import (
    InvalidReservationError,
    OutOfSpaceError,
    SharedL1Pool,
    SharedObjectHandle,
    SharedRegionContract,
    StaleHandleError,
)

_REGION = "test-region"
_LAYOUT = "layout-v1"
_CAPACITY = 64 * 1024
_ALIGNMENT = 4096


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([8, 8])], [torch.float16])


def _pool(capacity: int = _CAPACITY) -> SharedL1Pool:
    return SharedL1Pool(_REGION, capacity, _ALIGNMENT, _LAYOUT)


def _objects(pool: SharedL1Pool) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], pool.snapshot()["objects"])


def test_batched_write_read_state_machine() -> None:
    pool = _pool()
    layout = _layout()
    writes = pool.reserve_writes([("key-a", layout), ("key-b", layout)])
    assert all(write is not None for write in writes)
    granted = [write for write in writes if write is not None]

    # Payload bytes may already exist, but WRITING metadata is never readable.
    assert pool.reserve_reads(["key-a", "missing"]) == [None, None]
    pool.finish_writes(granted)

    reads = pool.reserve_reads(["key-a", "missing", "key-b"])
    assert reads[1] is None
    active = [read for read in reads if read is not None]
    assert [read.handle for read in active] == [
        granted[0].handle,
        granted[1].handle,
    ]
    assert all(read.layout == layout for read in active)
    assert [_objects(pool)[key]["active_readers"] for key in ("key-a", "key-b")] == [
        1,
        1,
    ]
    pool.finish_reads(active)
    assert [_objects(pool)[key]["active_readers"] for key in ("key-a", "key-b")] == [
        0,
        0,
    ]

    # Duplicate immutable objects do not allocate a second physical copy.
    before = pool.snapshot()["next_offset"]
    assert pool.reserve_writes([("key-a", layout)]) == [None]
    assert pool.snapshot()["next_offset"] == before


def test_batch_validation_and_capacity_fail_without_partial_commit() -> None:
    pool = _pool(capacity=2 * _ALIGNMENT)
    layout = _layout()
    writes = pool.reserve_writes([("key-a", layout), ("key-b", layout)])
    granted = [write for write in writes if write is not None]
    stale = replace(
        granted[1],
        handle=replace(
            granted[1].handle,
            generation=granted[1].handle.generation + 1,
        ),
    )
    with pytest.raises(StaleHandleError):
        pool.finish_writes([granted[0], stale])
    assert {item["state"] for item in _objects(pool).values()} == {"WRITING"}

    pool.abort_writes(granted)
    assert pool.snapshot()["objects"] == {}
    used_after_abort = pool.snapshot()["next_offset"]
    with pytest.raises(OutOfSpaceError):
        pool.reserve_writes([("too-large-a", layout), ("too-large-b", layout)])
    assert pool.snapshot()["next_offset"] == used_after_abort


def test_concurrent_batches_never_overlap_and_same_key_has_one_winner() -> None:
    pool = _pool(capacity=512 * 1024)
    layout = _layout()

    def reserve(key: str):
        return pool.reserve_writes([(key, layout)])[0]

    with ThreadPoolExecutor(max_workers=16) as executor:
        distinct = list(executor.map(reserve, [f"key-{index}" for index in range(16)]))
        duplicates = list(executor.map(reserve, ["same-key"] * 8))

    granted = [write for write in distinct + duplicates if write is not None]
    assert len(granted) == 17
    ordered = sorted((write.handle for write in granted), key=lambda item: item.offset)
    assert all(handle.offset % _ALIGNMENT == 0 for handle in ordered)
    assert all(
        left.offset + left.length <= right.offset
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    assert len({handle.generation for handle in ordered}) == len(ordered)
    pool.finish_writes(granted)
    assert _objects(pool)["same-key"]["state"] == "VALID"


def test_reservation_tokens_and_restart_epoch_fail_closed() -> None:
    pool = _pool()
    layout = _layout()
    write = pool.reserve_writes([("key", layout)])[0]
    assert write is not None
    with pytest.raises(InvalidReservationError):
        pool.finish_writes([replace(write, token="wrong")])
    pool.finish_writes([write])

    read = pool.reserve_reads(["key"])[0]
    assert read is not None
    with pytest.raises(InvalidReservationError):
        pool.finish_reads([replace(read, token="wrong")])
    pool.abort_reads([read])

    replacement = _pool()
    assert (
        replacement.region_contract().generation_epoch
        != pool.region_contract().generation_epoch
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"region_id": ""},
        {"capacity": 0},
        {"alignment": 3},
        {"layout_id": ""},
        {"generation_epoch": 0},
    ],
)
def test_region_contract_rejects_invalid_geometry(kwargs: dict[str, object]) -> None:
    values = {
        "region_id": _REGION,
        "capacity": _CAPACITY,
        "alignment": _ALIGNMENT,
        "layout_id": _LAYOUT,
        "generation_epoch": 1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        SharedRegionContract(**values)  # type: ignore[arg-type]


def test_handle_is_pool_relative_and_generation_bounded() -> None:
    handle = SharedObjectHandle(_REGION, 0, 1, 1)
    assert (handle.region_id, handle.offset, handle.length, handle.generation) == (
        _REGION,
        0,
        1,
        1,
    )
    with pytest.raises(ValueError):
        SharedObjectHandle(_REGION, 0, 1, 1 << 64)
