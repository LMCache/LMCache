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
)

PAGE = os.sysconf("SC_PAGE_SIZE")
MIB = 1 << 20

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
        self.live: dict[int, tuple[int, int, int]] = {}
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
            self.live[ptr] = (base, span, size)
            self.allocated.append(ptr)
        return ptr

    def free(self, ptr: int) -> None:
        with self.lock:
            base, span, _ = self.live.pop(ptr)
            self.freed.append(ptr)
        _libc.munmap(base, span)

    def sizes(self) -> list[int]:
        return [self.live[ptr][2] for ptr in self.allocated if ptr in self.live]

    def owner(self, ptr: int, size: int) -> int:
        """Index of the live segment holding [ptr, ptr + size); asserts one."""
        owners = [
            i
            for i, seg in enumerate(self.allocated)
            if seg in self.live and seg <= ptr < seg + self.live[seg][2]
        ]
        assert len(owners) == 1
        seg = self.allocated[owners[0]]
        assert ptr + size <= seg + self.live[seg][2], "object crosses a segment"
        return owners[0]


def _make(
    fake: FakeSegments, init: int, final: int, segment: int, align: int = PAGE
) -> SegmentedMemoryAllocator:
    return SegmentedMemoryAllocator(
        init, final, segment, fake.alloc, fake.free, align_bytes=align
    )


def _wait_capacity(alloc: SegmentedMemoryAllocator, total: int) -> None:
    deadline = time.time() + 10
    while alloc.get_memory_usage()[1] < total and time.time() < deadline:
        time.sleep(0.01)
    assert alloc.get_memory_usage()[1] == total


def test_objects_stay_inside_one_noncontiguous_segment():
    align = 4 * PAGE
    fake = FakeSegments(align=align)
    alloc = _make(fake, 5 * MIB, 10 * MIB, 3 * MIB, align=align)
    _wait_capacity(alloc, 10 * MIB)
    # Initial and background capacity are each split into segment-size pieces.
    assert fake.sizes() == [3 * MIB, 2 * MIB, 3 * MIB, 2 * MIB]
    assert alloc.memory_region_count() == 4

    objs = alloc.batched_allocate(torch.Size([256 * 1024]), torch.uint8, 40)
    assert objs is not None and len(objs) == 40
    assert len({o.meta.address for o in objs}) == 40
    hits = [0] * 4
    for i, obj in enumerate(objs):
        assert obj.meta.address % align == 0 and obj.data_ptr % align == 0
        hits[fake.owner(obj.data_ptr, obj.get_size())] += 1
        obj.tensor.fill_(i % 251)
    assert hits == [12, 8, 12, 8]
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
    assert {fake.owner(o.data_ptr, o.get_size()) for o in batch} == {0, 1}
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
    fake.owner(big.data_ptr, big.get_size())
    alloc.free(big)
    alloc.close()


def test_misaligned_segment_is_rejected_and_freed():
    fake = FakeSegments(misalign=True)
    with pytest.raises(RuntimeError, match="not aligned"):
        _make(fake, MIB, MIB, MIB)
    assert fake.freed == fake.allocated and not fake.live


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
    # A background segment is being allocated but not yet published.
    assert alloc.get_memory_usage() == (0, MIB)
    assert alloc.memory_region_count() == 1
    obj = alloc.allocate(torch.Size([MIB]), torch.uint8)
    assert obj is not None
    assert alloc.allocate(torch.Size([MIB]), torch.uint8) is None
    gate.set()
    _wait_capacity(alloc, 3 * MIB)
    assert alloc.memory_region_count() == 3
    assert alloc.get_memory_usage() == (MIB, 3 * MIB)
    alloc.free(obj)
    alloc.close()
    assert sorted(fake.freed) == sorted(fake.allocated)


def test_expansion_failure_keeps_published_capacity():
    # Call 0: initial; 1: first background segment; 2: fails, expansion stops.
    fake = FakeSegments(fail_calls={2})
    alloc = _make(fake, MIB, 4 * MIB, MIB)
    obj = alloc.allocate(torch.Size([MIB]), torch.uint8)
    assert obj is not None
    obj.tensor.fill_(7)
    deadline = time.time() + 5
    while fake.calls < 3 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    assert fake.calls == 3
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
