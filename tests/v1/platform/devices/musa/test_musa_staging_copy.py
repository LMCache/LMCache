# SPDX-License-Identifier: Apache-2.0
"""Regression tests for lazy MUSA copies at pin-registration boundaries."""

# Standard
from types import SimpleNamespace
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
from lmcache.v1.storage_backend.naive_serde import cachegen_encoder
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


def test_cachegen_uses_boundary_safe_copy_for_lazy_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CacheGen stages lazy inputs independently of the accelerator name."""
    source = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    memory_obj = FakeMemoryObj(
        source.data_ptr(),
        source.nbytes,
        host_offset=14,
        tensor=source,
    )
    copy_calls: list[tuple[object, torch.Tensor]] = []
    encoded: dict[str, torch.Tensor] = {}
    original_empty_like = torch.empty_like

    def _cpu_empty_like(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_empty_like(tensor, *args, **kwargs)

    def _copy_to_device(obj: object, destination: torch.Tensor) -> None:
        copy_calls.append((obj, destination))
        destination.copy_(source)

    def _encode(tensor: torch.Tensor, *_args: object) -> SimpleNamespace:
        encoded["tensor"] = tensor.clone()
        return SimpleNamespace(to_bytes=lambda: b"encoded")

    monkeypatch.setattr(cachegen_encoder, "torch_device_type", "cuda")
    monkeypatch.setattr(cachegen_encoder.torch, "empty_like", _cpu_empty_like)
    monkeypatch.setattr(
        cachegen_encoder,
        "lmcache_memcpy_async_h2d",
        _copy_to_device,
    )
    monkeypatch.setattr(cachegen_encoder, "encode_function", _encode)
    monkeypatch.setattr(cachegen_encoder.torch_dev, "current_device", lambda: None)

    serializer = cachegen_encoder.CacheGenSerializer.__new__(
        cachegen_encoder.CacheGenSerializer
    )
    serializer.cachegen_config = cast(
        cachegen_encoder.CacheGenConfig,
        SimpleNamespace(),
    )
    serializer.key_bins = torch.zeros(1)
    serializer.value_bins = torch.zeros(1)
    serializer.kv_shape = torch.Size([2, 1, 3, 2, 2])

    result = serializer.serialize(cast(MemoryObj, memory_obj))

    assert len(copy_calls) == 1
    assert copy_calls[0][0] is memory_obj
    assert torch.equal(
        encoded["tensor"],
        source.view(2, 1, 3, 2, 2).permute(1, 0, 2, 3, 4),
    )
    assert result.byte_array == b"encoded"
