# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm GTT host memory segment source."""

# Standard
import ctypes
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rocm import gtt_host_memory
from lmcache.v1.platform.devices.rocm.gtt_host_memory import (
    alloc_gtt_segment,
    check_gtt_environment,
    free_gtt_segment,
)


class _AnonymousOps:
    """Stands in for device_ops; returns anonymous (non-GTT) memory."""

    def __init__(self) -> None:
        self.freed: list[int] = []
        self._bufs: dict[int, ctypes.Array] = {}

    def alloc_pinned_ptr(self, size: int, flags: int) -> int:
        buf = (ctypes.c_uint8 * size)()
        self._bufs[ctypes.addressof(buf)] = buf
        return ctypes.addressof(buf)

    def free_pinned_ptr(self, ptr: int) -> None:
        self.freed.append(ptr)
        del self._bufs[ptr]


def test_environment_must_be_zero(monkeypatch):
    monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)
    with pytest.raises(RuntimeError, match="HSA_USERPTR_FOR_PAGED_MEM=0"):
        check_gtt_environment()
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "1")
    with pytest.raises(RuntimeError, match="found '1'"):
        check_gtt_environment()
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    check_gtt_environment()


def test_non_gtt_memory_is_freed_and_rejected(monkeypatch):
    ops = _AnonymousOps()
    monkeypatch.setattr(gtt_host_memory, "device_ops", ops)
    with pytest.raises(RuntimeError, match="instead of GTT memory"):
        alloc_gtt_segment(1 << 20)
    assert len(ops.freed) == 1


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
