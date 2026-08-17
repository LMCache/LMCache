# SPDX-License-Identifier: Apache-2.0
"""Tests for MUSA lazy-allocator staging copies."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import gpu_ops
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.musa import device_ops


def test_musa_lazy_staging_roundtrip_uses_device_tensor_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy H2D/D2H copies reconstruct the device pointer as a MUSA tensor."""
    host_source = torch.arange(16, dtype=torch.uint8)
    gpu_buffer = torch.zeros_like(host_source)
    host_destination = torch.zeros_like(host_source)
    lazy_allocator = object.__new__(LazyMemoryAllocator)

    def make_memory_obj(tensor: torch.Tensor) -> MemoryObj:
        return cast(
            MemoryObj,
            SimpleNamespace(
                raw_tensor=tensor,
                get_size=lambda: tensor.nbytes,
                parent=lambda: lazy_allocator,
                data_ptr=tensor.data_ptr(),
                meta=SimpleNamespace(address=0),
            ),
        )

    source_obj = make_memory_obj(host_source)
    destination_obj = make_memory_obj(host_destination)
    reconstructed_pointers: list[int] = []

    def construct_tensor(
        pointer: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        **kwargs: object,
    ) -> torch.Tensor:
        reconstructed_pointers.append(pointer)
        assert shape == (host_source.nbytes,)
        assert dtype == torch.uint8
        assert kwargs["nbytes"] == host_source.nbytes
        return gpu_buffer

    monkeypatch.setattr(
        device_ops,
        "construct_musa_tensor_from_data_pointer",
        construct_tensor,
    )
    monkeypatch.setattr(
        device_ops,
        "_current_musa_device",
        lambda: torch.device("cpu"),
        raising=False,
    )
    monkeypatch.setattr(gpu_ops, "device_ops", device_ops.MusaDeviceOps())

    gpu_ops.lmcache_memcpy_async_h2d(source_obj, gpu_buffer)
    assert torch.equal(gpu_buffer, host_source)

    gpu_ops.lmcache_memcpy_async_d2h(gpu_buffer, destination_obj)

    assert torch.equal(host_destination, host_source)
    assert reconstructed_pointers == [gpu_buffer.data_ptr(), gpu_buffer.data_ptr()]
