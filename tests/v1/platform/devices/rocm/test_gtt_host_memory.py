# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm GTT host memory segment source."""

# Standard
import ctypes
import os
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rocm import gtt_host_memory
from lmcache.v1.platform.devices.rocm.gtt_host_memory import (
    GTT_MAX_ALLOCATION_BYTES,
    GttEnvironmentError,
    alloc_gtt_segment,
    check_gtt_environment,
    default_gtt_segment_size,
    free_gtt_segment,
    validate_gtt_segment_size,
)

PAGE = os.sysconf("SC_PAGE_SIZE")
GIB = 1 << 30


class _SpyOps:
    """Stands in for device_ops; returns an anonymous (non-GTT) mapping."""

    def __init__(self) -> None:
        self.allocs: list[int] = []
        self.frees: list[int] = []
        self._bufs: dict[int, ctypes.Array] = {}

    def alloc_pinned_ptr(self, size: int, flags: int) -> int:
        buf = (ctypes.c_uint8 * size)()
        ptr = ctypes.addressof(buf)
        self._bufs[ptr] = buf
        self.allocs.append(size)
        return ptr

    def free_pinned_ptr(self, ptr: int) -> None:
        self.frees.append(ptr)
        self._bufs.pop(ptr)


def test_environment_must_be_zero(monkeypatch):
    monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)
    with pytest.raises(GttEnvironmentError, match="HSA_USERPTR_FOR_PAGED_MEM"):
        check_gtt_environment()
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "1")
    with pytest.raises(GttEnvironmentError, match='value: "0"'):
        check_gtt_environment()
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    check_gtt_environment()


def test_default_segment_size_is_largest_aligned_below_512_gib():
    assert default_gtt_segment_size(16384) == GTT_MAX_ALLOCATION_BYTES - 16384
    assert default_gtt_segment_size(4096) == GTT_MAX_ALLOCATION_BYTES - max(4096, PAGE)
    assert default_gtt_segment_size(1 << 30) == GTT_MAX_ALLOCATION_BYTES - GIB


@pytest.mark.parametrize(
    "size",
    [0, -PAGE, GTT_MAX_ALLOCATION_BYTES, GTT_MAX_ALLOCATION_BYTES + 16384, 1000],
)
def test_invalid_segment_sizes(size):
    with pytest.raises(ValueError, match="512 GiB"):
        validate_gtt_segment_size(size, 16384)


def test_valid_segment_sizes():
    validate_gtt_segment_size(GTT_MAX_ALLOCATION_BYTES - 16384, 16384)
    validate_gtt_segment_size(256 * GIB, 16384)
    with pytest.raises(ValueError):
        validate_gtt_segment_size(GTT_MAX_ALLOCATION_BYTES - 4096, 16384)


def test_alloc_checks_environment_before_allocating(monkeypatch):
    spy = _SpyOps()
    monkeypatch.setattr(gtt_host_memory, "device_ops", spy)
    monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)
    with pytest.raises(GttEnvironmentError):
        alloc_gtt_segment(GIB)
    assert spy.allocs == []


def test_non_gtt_mapping_is_freed_and_rejected(monkeypatch):
    spy = _SpyOps()
    monkeypatch.setattr(gtt_host_memory, "device_ops", spy)
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    with pytest.raises(GttEnvironmentError, match="instead of a /dev/dri/render"):
        alloc_gtt_segment(1 << 20)
    assert spy.allocs == [1 << 20] and len(spy.frees) == 1


def test_allocation_error_keeps_hip_error(monkeypatch):
    def failing_alloc(size: int, flags: int) -> int:
        raise RuntimeError("cudaHostAlloc failed: 2")

    monkeypatch.setattr(
        gtt_host_memory,
        "device_ops",
        types.SimpleNamespace(alloc_pinned_ptr=failing_alloc),
    )
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    with pytest.raises(gtt_host_memory.GttAllocationError, match="failed: 2"):
        alloc_gtt_segment(GIB)


@pytest.mark.skipif(
    os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") != "0"
    or getattr(torch.version, "hip", None) is None
    or not torch.cuda.is_available(),
    reason="needs ROCm and HSA_USERPTR_FOR_PAGED_MEM=0 at process start",
)
def test_real_gtt_segment_is_a_render_node_mapping():
    size = 64 << 20
    ptr = alloc_gtt_segment(size)
    try:
        with open("/proc/self/maps") as maps:
            line = next(
                line
                for line in maps
                if int(line.split()[0].split("-")[0], 16)
                <= ptr
                < int(line.split()[0].split("-")[1], 16)
            )
        assert "/dev/dri/render" in line
        ctypes.memset(ptr, 0x5A, size)
        assert ctypes.string_at(ptr + size - 16, 16) == b"\x5a" * 16
    finally:
        free_gtt_segment(ptr)
