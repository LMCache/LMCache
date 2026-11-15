# SPDX-License-Identifier: Apache-2.0
"""Regression tests for lazy MUSA copies at pin-registration boundaries."""

# Standard
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.gpu_connector import gpu_ops
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.devices.musa import device_ops as musa_device_ops
from lmcache.v1.platform.devices.musa.device_ops import MusaDeviceOps
from tests.v1.platform.devices.musa.staging_copy_utils import FakeMemoryObj


def _install_fake_musa_pointer_views(
    monkeypatch: pytest.MonkeyPatch,
    device_buffer: torch.Tensor,
) -> list[tuple[int, int]]:
    """Map fake MUSA pointers to slices of one CPU tensor for unit tests."""
    calls: list[tuple[int, int]] = []
    device_base = device_buffer.data_ptr()
    monkeypatch.setattr(
        musa_device_ops,
        "_current_musa_device",
        lambda: torch.device("cpu"),
    )

    def _construct(
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        assert dtype is torch.uint8
        size = shape[0]
        calls.append((ptr, size))
        start = ptr - device_base
        return device_buffer[start : start + size]

    monkeypatch.setattr(
        musa_device_ops,
        "construct_musa_tensor_from_data_pointer",
        _construct,
    )
    return calls


def test_lazy_gpu_ops_h2d_uses_musa_boundary_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public lazy H2D path delegates boundary splitting to MUSA ops."""
    chunk_size = 16
    start = chunk_size - 2
    size = 2 * chunk_size + 5
    host = torch.arange(4 * chunk_size, dtype=torch.uint8)
    device = torch.zeros(4 * chunk_size, dtype=torch.uint8)
    calls = _install_fake_musa_pointer_views(monkeypatch, device)
    memory_tensor = host[start : start + size]
    memory_obj = FakeMemoryObj(
        memory_tensor.data_ptr(),
        memory_tensor.nbytes,
        start,
        tensor=memory_tensor,
    )
    monkeypatch.setattr(gpu_ops, "device_ops", MusaDeviceOps())
    monkeypatch.setattr(LazyMemoryAllocator, "PIN_CHUNK_SIZE", chunk_size)

    gpu_ops.lmcache_memcpy_async_h2d(
        cast(MemoryObj, memory_obj),
        device[:size],
    )

    assert [segment_size for _, segment_size in calls] == [2, 16, 16, 3]
    assert torch.equal(device[:size], host[start : start + size])


def test_lazy_gpu_ops_d2h_uses_musa_boundary_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public lazy D2H path delegates boundary splitting to MUSA ops."""
    chunk_size = 16
    start = chunk_size - 3
    size = chunk_size + 8
    host = torch.zeros(4 * chunk_size, dtype=torch.uint8)
    device = torch.arange(4 * chunk_size, dtype=torch.uint8)
    calls = _install_fake_musa_pointer_views(monkeypatch, device)
    memory_tensor = host[start : start + size]
    memory_obj = FakeMemoryObj(
        memory_tensor.data_ptr(),
        memory_tensor.nbytes,
        start,
        tensor=memory_tensor,
    )
    monkeypatch.setattr(gpu_ops, "device_ops", MusaDeviceOps())
    monkeypatch.setattr(LazyMemoryAllocator, "PIN_CHUNK_SIZE", chunk_size)

    gpu_ops.lmcache_memcpy_async_d2h(
        device[:size],
        cast(MemoryObj, memory_obj),
    )

    assert [segment_size for _, segment_size in calls] == [3, 16, 5]
    assert torch.equal(host[start : start + size], device[:size])


def test_musa_device_ops_preserves_tensor_copy_mode() -> None:
    """MUSA ops retain the base operation's tensor-mode contract."""
    source = torch.arange(16, dtype=torch.uint8)
    destination = torch.zeros_like(source)

    MusaDeviceOps().lmcache_memcpy_async(
        destination,
        source,
        source.nbytes,
        TransferDirection.H2D,
        0,
        16,
    )

    assert torch.equal(destination, source)
