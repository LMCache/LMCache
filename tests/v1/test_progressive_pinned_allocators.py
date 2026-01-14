# SPDX-License-Identifier: Apache-2.0
"""Unified tests for progressive pinned allocators (free-list and paged)."""

# Standard
import time
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryFormat,
    ProgressivePinnedMemoryAllocator,
    ProgressivePinnedPagedMemoryAllocator,
)


@dataclass(frozen=True)
class AllocCase:
    name: str


CASES = [
    AllocCase(name="free_list"),
    AllocCase(name="paged"),
]


def _page_shape() -> torch.Size:
    # Used for paged allocator page size & for common test allocation shapes.
    return torch.Size([2, 128, 64])


def _pool_bytes_mb_aligned(
    size_mb: int, page_shape: torch.Size, dtype: torch.dtype
) -> int:
    """Make pool bytes compatible with paged allocator (multiple of page bytes)."""
    page_bytes = page_shape.numel() * dtype.itemsize
    total_bytes = size_mb * 1024 * 1024
    total_bytes = (total_bytes // page_bytes) * page_bytes
    return max(total_bytes, page_bytes * 4)


def _create_allocator(
    case: AllocCase,
    *,
    size_mb: int,
    initial: float,
    trigger: float,
    step: float,
    dtype: torch.dtype,
) -> object:
    if case.name == "free_list":
        return ProgressivePinnedMemoryAllocator(
            size=size_mb * 1024 * 1024,
            initial_ratio=initial,
            expand_trigger_ratio=trigger,
            step_ratio=step,
        )
    if case.name == "paged":
        shape = _page_shape()
        return ProgressivePinnedPagedMemoryAllocator(
            size=_pool_bytes_mb_aligned(size_mb, shape, dtype),
            shapes=[shape],
            dtypes=[dtype],
            fmt=MemoryFormat.KV_2LTD,
            initial_ratio=initial,
            expand_trigger_ratio=trigger,
            step_ratio=step,
        )
    raise ValueError(f"Unknown case: {case.name}")


def _verify_mem_obj(mem_obj, shape: torch.Size, dtype: torch.dtype | None):
    assert mem_obj and mem_obj.is_valid()
    assert mem_obj.get_shape() == shape
    if dtype is not None:
        assert mem_obj.get_dtype() == dtype


def _wait_for_growth(
    case: AllocCase,
    alloc,
    *,
    initial_reg: int,
    initial_pages: int | None,
    timeout_s: float = 1.0,
) -> None:
    """Poll for any sign of growth from the background expander."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        reg = alloc._registered_size  # noqa: SLF001
        if reg > initial_reg:
            return
        if case.name == "paged" and initial_pages is not None:
            if len(alloc.free_blocks) > initial_pages:
                return
        time.sleep(0.05)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_basic_allocation(case: AllocCase):
    dtype = torch.float16 if case.name == "paged" else torch.float32
    alloc = _create_allocator(
        case, size_mb=16, initial=0.5, trigger=0.9, step=0.2, dtype=dtype
    )
    try:
        shape = _page_shape() if case.name == "paged" else torch.Size([512, 256])
        mem = alloc.allocate(shape, dtype, MemoryFormat.KV_2LTD)
        _verify_mem_obj(mem, shape, dtype)
        assert mem.tensor is not None and mem.tensor.shape == shape
        mem.ref_count_down()
    finally:
        alloc.close()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_async_expansion_trigger(case: AllocCase):
    dtype = torch.float16 if case.name == "paged" else torch.float32
    alloc = _create_allocator(
        case, size_mb=64, initial=0.1, trigger=0.05, step=0.2, dtype=dtype
    )
    try:
        initial_reg = alloc._registered_size  # noqa: SLF001
        assert initial_reg > 0
        # For paged allocator, free_blocks will drop as we allocate pages, so
        # assertions should compare against the post-allocation baseline.

        shape = _page_shape() if case.name == "paged" else torch.Size([512, 512])

        # Trigger expansion:
        # - For the free-list allocator, one large allocation is usually enough.
        # - For the paged allocator, allocate multiple pages to exceed the
        #   expand_trigger_ratio threshold.
        n_alloc = 32 if case.name == "paged" else 1
        mems = []
        for _ in range(n_alloc):
            m = alloc.allocate(shape, dtype, MemoryFormat.KV_2LTD)
            assert m is not None
            mems.append(m)

        pages_after_alloc = len(alloc.free_blocks) if case.name == "paged" else None

        _wait_for_growth(
            case,
            alloc,
            initial_reg=initial_reg,
            initial_pages=pages_after_alloc,
        )
        grown_reg = alloc._registered_size  # noqa: SLF001
        assert grown_reg >= initial_reg
        if case.name == "paged" and pages_after_alloc is not None:
            # At minimum, expansion should not reduce free pages further.
            assert len(alloc.free_blocks) >= pages_after_alloc
            # And expansion should have started adding pages back
            # (or registered size grew).
            assert grown_reg > initial_reg or len(alloc.free_blocks) > pages_after_alloc

        for m in mems:
            m.ref_count_down()
    finally:
        alloc.close()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_multiple_allocations(case: AllocCase):
    dtype = torch.float16 if case.name == "paged" else torch.float32
    alloc = _create_allocator(
        case, size_mb=64, initial=0.1, trigger=0.3, step=0.2, dtype=dtype
    )
    try:
        shape = _page_shape() if case.name == "paged" else torch.Size([256, 512])
        mems = [alloc.allocate(shape, dtype, MemoryFormat.KV_2LTD) for _ in range(20)]
        mems = [m for m in mems if m is not None]
        assert len(mems) > 0
        for m in mems:
            _verify_mem_obj(m, shape, dtype)

        time.sleep(0.2)
        for _ in range(10):
            m = alloc.allocate(shape, dtype, MemoryFormat.KV_2LTD)
            if m is not None:
                mems.append(m)

        for m in mems:
            m.ref_count_down()
    finally:
        alloc.close()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_batched_allocation(case: AllocCase):
    dtype = torch.float16 if case.name == "paged" else torch.float32
    alloc = _create_allocator(
        case, size_mb=64, initial=0.5, trigger=0.9, step=0.2, dtype=dtype
    )
    try:
        shape = _page_shape() if case.name == "paged" else torch.Size([128, 256])
        batch_size = 8
        mems = alloc.batched_allocate(shape, dtype, batch_size, MemoryFormat.KV_2LTD)
        assert mems is not None and len(mems) == batch_size
        for m in mems:
            _verify_mem_obj(m, shape, dtype)
            m.ref_count_down()
    finally:
        alloc.close()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_free_and_reuse(case: AllocCase):
    dtype = torch.float16 if case.name == "paged" else torch.float32
    alloc = _create_allocator(
        case, size_mb=16, initial=0.5, trigger=0.9, step=0.2, dtype=dtype
    )
    try:
        shape = _page_shape() if case.name == "paged" else torch.Size([256, 256])
        m1 = alloc.allocate(shape, dtype, MemoryFormat.KV_2LTD)
        assert m1 is not None
        m1.ref_count_down()

        m2 = alloc.allocate(shape, dtype, MemoryFormat.KV_2LTD)
        assert m2 is not None
        m2.ref_count_down()
    finally:
        alloc.close()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_binary_buffer_passthrough(case: AllocCase):
    dtype = torch.float16 if case.name == "paged" else torch.float32
    alloc = _create_allocator(
        case, size_mb=8, initial=0.5, trigger=0.9, step=0.2, dtype=dtype
    )
    try:
        mem = alloc.allocate(torch.Size([1024]), [], MemoryFormat.BINARY_BUFFER)
        assert mem is not None
        assert mem.get_memory_format() == MemoryFormat.BINARY_BUFFER
        mem.ref_count_down()
    finally:
        alloc.close()
