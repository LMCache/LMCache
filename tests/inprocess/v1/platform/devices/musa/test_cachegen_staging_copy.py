# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for CacheGen staging on MUSA-compatible memory."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.storage_backend.naive_serde import cachegen_encoder
from lmcache.v1.memory_management import MemoryObj
from tests.v1.platform.devices.musa.staging_copy_utils import FakeMemoryObj


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
