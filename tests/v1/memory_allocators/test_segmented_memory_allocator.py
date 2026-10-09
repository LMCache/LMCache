# SPDX-License-Identifier: Apache-2.0
"""Tests for SegmentedMemoryAllocator over fake non-contiguous segments."""

# Standard
import ctypes
import os
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.segmented_memory_allocator import (
    SegmentedMemoryAllocator,
    plan_segments,
)

PAGE = os.sysconf("SC_PAGE_SIZE")
MIB = 1 << 20
GIB = 1 << 30

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


class FakeSegments:
    """Segment source returning anonymous mappings separated by guard pages,
    so consecutive segments are never virtually contiguous."""

    def __init__(
        self,
        fail_calls: set[int] | None = None,
        misalign: bool = False,
        align: int = PAGE,
    ):
        self.fail_calls = fail_calls or set()
        self.misalign = misalign
        self.align = align
        self.calls = 0
        self.live: dict[int, tuple[int, int]] = {}
        self.allocated: list[int] = []
        self.freed: list[int] = []
        self.lock = threading.Lock()

    def alloc(self, size: int) -> int:
        with self.lock:
            call = self.calls
            self.calls += 1
        if call in self.fail_calls:
            raise RuntimeError("cudaHostAlloc failed: 2")
        span = size + 2 * PAGE + self.align
        base = _libc.mmap(None, span, 3, 0x22, -1, 0)
        assert base not in (None, ctypes.c_void_p(-1).value)
        ptr = -(-(base + PAGE) // self.align) * self.align
        ptr += PAGE // 2 if self.misalign else 0
        with self.lock:
            self.live[ptr] = (base, span)
            self.allocated.append(ptr)
        return ptr

    def free(self, ptr: int) -> None:
        with self.lock:
            base, span = self.live.pop(ptr)
            self.freed.append(ptr)
        _libc.munmap(base, span)


def _make(
    fake: FakeSegments,
    init: int,
    final: int,
    segment: int,
    align: int = PAGE,
) -> SegmentedMemoryAllocator:
    return SegmentedMemoryAllocator(
        init, final, segment, fake.alloc, fake.free, align_bytes=align
    )


def _wait_capacity(alloc: SegmentedMemoryAllocator, total: int) -> None:
    deadline = time.time() + 10
    while alloc.get_memory_usage()[1] < total and time.time() < deadline:
        time.sleep(0.01)
    assert alloc.get_memory_usage()[1] == total


def _owner(segments: list[tuple[int, int]], ptr: int, size: int) -> int:
    owners = [
        i for i, (base, length) in enumerate(segments) if base <= ptr < base + length
    ]
    assert len(owners) == 1
    base, length = segments[owners[0]]
    assert ptr + size <= base + length, "object crosses a segment boundary"
    return owners[0]


def test_plan_segments_customer_configs():
    seg = (512 << 30) - 16384
    assert plan_segments(1 * GIB, 1000 * GIB, seg, 16384) == (
        [1 * GIB],
        [seg, 1000 * GIB - 1 * GIB - seg],
    )
    assert plan_segments(600 * GIB, 1000 * GIB, seg, 16384) == (
        [seg, 600 * GIB - seg],
        [400 * GIB],
    )
    assert plan_segments(1000 * GIB, 1000 * GIB, seg, 16384) == (
        [seg, 1000 * GIB - seg],
        [],
    )
    initial, expansion = plan_segments(1000 * GIB, 1000 * GIB, seg, 16384)
    assert all(0 < s < 512 << 30 for s in initial + expansion)


def test_objects_stay_inside_one_noncontiguous_segment():
    fake = FakeSegments()
    alloc = _make(fake, 3 * MIB, 10 * MIB, 3 * MIB)
    _wait_capacity(alloc, 10 * MIB)
    segments = alloc.segments()
    assert [size for _, size in segments] == [3 * MIB, 3 * MIB, 3 * MIB, 1 * MIB]
    assert all(
        segments[i][0] + segments[i][1] != segments[i + 1][0]
        for i in range(len(segments) - 1)
    )

    # 40 objects of 256 KiB fill 10 MiB exactly.
    objs = alloc.batched_allocate(torch.Size([256 * 1024]), torch.uint8, 40)
    assert objs is not None and len(objs) == 40
    assert len({o.meta.address for o in objs}) == 40
    hits = [0] * len(segments)
    for i, obj in enumerate(objs):
        hits[_owner(segments, obj.data_ptr, obj.get_size())] += 1
        obj.tensor.fill_(i % 251)
    assert hits == [12, 12, 12, 4]
    for i, obj in enumerate(objs):
        assert bool((obj.tensor == i % 251).all())
    assert alloc.get_memory_usage() == (10 * MIB, 10 * MIB)
    assert alloc.allocate(torch.Size([256 * 1024]), torch.uint8) is None

    alloc.batched_free(objs)
    assert alloc.get_memory_usage() == (0, 10 * MIB)
    assert alloc.memcheck()
    alloc.close()


def test_batch_is_all_or_nothing_and_may_span_segments():
    fake = FakeSegments()
    alloc = _make(fake, 2 * MIB, 4 * MIB, 2 * MIB)
    _wait_capacity(alloc, 4 * MIB)
    one = alloc.allocate(torch.Size([MIB]), torch.uint8)
    assert one is not None
    assert alloc.batched_allocate(torch.Size([MIB]), torch.uint8, 4) is None
    assert alloc.get_memory_usage() == (MIB, 4 * MIB)

    batch = alloc.batched_allocate(torch.Size([MIB]), torch.uint8, 3)
    assert batch is not None
    owners = {_owner(alloc.segments(), o.data_ptr, o.get_size()) for o in batch}
    assert owners == {0, 1}
    alloc.batched_free(batch + [one])
    assert alloc.memcheck()
    alloc.close()


def test_free_space_never_merges_across_a_boundary():
    fake = FakeSegments()
    alloc = _make(fake, 2 * MIB, 4 * MIB, 2 * MIB)
    _wait_capacity(alloc, 4 * MIB)
    objs = alloc.batched_allocate(torch.Size([MIB]), torch.uint8, 4)
    assert objs is not None
    # Freeing the two objects around the boundary in one batch coalesces them
    # logically; the 2 MiB hole must still not be usable as one block.
    alloc.batched_free([objs[1], objs[2]])
    assert alloc.memcheck()
    assert alloc.allocate(torch.Size([2 * MIB]), torch.uint8) is None
    alloc.batched_free([objs[0], objs[3]])
    big = alloc.allocate(torch.Size([2 * MIB]), torch.uint8)
    assert big is not None
    _owner(alloc.segments(), big.data_ptr, big.get_size())
    alloc.free(big)
    alloc.close()


def test_fragmentation_returns_none_until_space_is_freed():
    fake = FakeSegments()
    alloc = _make(fake, 2 * MIB, 2 * MIB, 2 * MIB)
    objs = alloc.batched_allocate(torch.Size([512 * 1024]), torch.uint8, 4)
    assert objs is not None
    alloc.batched_free([objs[0], objs[2]])
    assert alloc.allocate(torch.Size([MIB]), torch.uint8) is None
    alloc.free(objs[1])
    assert alloc.allocate(torch.Size([MIB + 512 * 1024]), torch.uint8) is not None
    alloc.close()


def test_object_larger_than_any_segment_fails_explicitly():
    fake = FakeSegments()
    alloc = _make(fake, 2 * MIB, 4 * MIB, 2 * MIB)
    with pytest.raises(ValueError, match="cannot fit in one"):
        alloc.allocate(torch.Size([2 * MIB + 1]), torch.uint8)
    with pytest.raises(ValueError, match="cannot fit in one"):
        alloc.batched_allocate(torch.Size([3 * MIB]), torch.uint8, 1)
    alloc.close()


def test_alignment_of_objects():
    align = 4 * PAGE
    fake = FakeSegments(align=align)
    alloc = _make(fake, 4 * MIB, 4 * MIB, 4 * MIB, align=align)
    objs = alloc.batched_allocate(torch.Size([1000]), torch.uint8, 5)
    assert objs is not None
    assert all(o.meta.address % align == 0 for o in objs)
    alloc.batched_free(objs)
    alloc.close()


def test_misaligned_segment_is_rejected_and_freed():
    fake = FakeSegments(misalign=True)
    with pytest.raises(RuntimeError, match="not aligned"):
        _make(fake, MIB, MIB, MIB)
    assert sorted(fake.freed) == sorted(fake.allocated) and not fake.live


def test_initial_partial_failure_frees_allocated_segments():
    fake = FakeSegments(fail_calls={1})
    with pytest.raises(RuntimeError, match="cudaHostAlloc failed"):
        _make(fake, 3 * MIB, 3 * MIB, 2 * MIB)
    assert len(fake.allocated) == 1
    assert fake.freed == fake.allocated and not fake.live


def test_initial_capacity_and_per_segment_publish():
    gate = threading.Event()
    fake = FakeSegments()

    def gated_alloc(size: int) -> int:
        if fake.calls >= 1:
            assert gate.wait(10)
        return fake.alloc(size)

    alloc = SegmentedMemoryAllocator(
        MIB, 3 * MIB, MIB, gated_alloc, fake.free, align_bytes=PAGE
    )
    time.sleep(0.2)
    # Background segment is being allocated but not yet published.
    assert alloc.get_memory_usage() == (0, MIB)
    assert alloc.memory_region_count() == 1
    with_obj = alloc.allocate(torch.Size([MIB]), torch.uint8)
    assert with_obj is not None
    assert alloc.allocate(torch.Size([MIB]), torch.uint8) is None
    gate.set()
    _wait_capacity(alloc, 3 * MIB)
    assert alloc.memory_region_count() == 3
    assert alloc.get_memory_usage() == (MIB, 3 * MIB)
    alloc.free(with_obj)
    alloc.close()
    assert sorted(fake.freed) == sorted(fake.allocated)


def test_expansion_failure_keeps_capacity_and_retries_bounded(monkeypatch):
    monkeypatch.setattr(
        SegmentedMemoryAllocator, "EXPANSION_RETRY_DELAYS_S", (0.01, 0.01)
    )
    # Call 0: initial; 1: first background segment OK; 2-4: second fails 3x.
    fake = FakeSegments(fail_calls={2, 3, 4})
    alloc = _make(fake, MIB, 4 * MIB, MIB)
    obj = alloc.allocate(torch.Size([MIB]), torch.uint8)
    assert obj is not None
    obj.tensor.fill_(7)
    deadline = time.time() + 5
    while fake.calls < 5 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    assert fake.calls == 5  # 1 initial + 1 ok + 1 attempt + 2 retries, then stop
    assert alloc.get_memory_usage() == (MIB, 2 * MIB)
    assert bool((obj.tensor == 7).all())
    alloc.free(obj)
    alloc.close()
    assert sorted(fake.freed) == sorted(fake.allocated) and not fake.live


def test_close_is_idempotent_and_frees_each_segment_once():
    fake = FakeSegments()
    alloc = _make(fake, MIB, 3 * MIB, MIB)
    _wait_capacity(alloc, 3 * MIB)
    alloc.close()
    alloc.close()
    assert len(fake.freed) == 3 and sorted(fake.freed) == sorted(fake.allocated)
    assert alloc.get_memory_usage() == (0, 0)
    assert alloc.allocate(torch.Size([PAGE]), torch.uint8) is None


def test_close_waits_for_inflight_expansion():
    gate = threading.Event()
    started = threading.Event()
    fake = FakeSegments()

    def gated_alloc(size: int) -> int:
        if fake.calls >= 1:
            started.set()
            assert gate.wait(10)
        return fake.alloc(size)

    alloc = SegmentedMemoryAllocator(
        MIB, 3 * MIB, MIB, gated_alloc, fake.free, align_bytes=PAGE
    )
    assert started.wait(5)
    closer = threading.Thread(target=alloc.close)
    closer.start()
    time.sleep(0.1)
    assert closer.is_alive()
    gate.set()
    closer.join(10)
    assert not closer.is_alive()
    # The in-flight segment finished and was freed; no further segment started.
    assert len(fake.allocated) == 2
    assert sorted(fake.freed) == sorted(fake.allocated) and not fake.live
