# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable tests for CUDA pointer-backed tensor construction."""

# Standard
from typing import Protocol, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.cuda import device_ops


class _CudaArrayInterface(Protocol):
    __cuda_array_interface__: dict[str, object]


class _FakeTensor:
    """Record view operations without requiring a CUDA runtime."""

    def __init__(self) -> None:
        self.views: list[object] = []

    def view(self, *shape_or_dtype: object) -> "_FakeTensor":
        self.views.append(shape_or_dtype)
        return self


def test_constructs_cuda_array_interface_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preferred path creates a zero-copy tensor with the requested shape."""
    fake_tensor = _FakeTensor()
    captured: dict[str, object] = {}

    def fake_as_tensor(value: object, *, device: torch.device) -> _FakeTensor:
        captured["array_interface"] = cast(
            _CudaArrayInterface, value
        ).__cuda_array_interface__
        captured["device"] = device
        return fake_tensor

    monkeypatch.setattr(device_ops.torch, "as_tensor", fake_as_tensor)
    device = torch.device("cuda:1")

    result = device_ops.CudaDeviceOps().tensor_from_ptr(
        0x1000,
        (2, 3),
        torch.float16,
        device,
    )

    assert result is fake_tensor
    assert fake_tensor.views == [(2, 3)]
    assert captured == {
        "array_interface": {
            "data": (0x1000, False),
            "shape": (6,),
            "typestr": "<f2",
            "version": 3,
        },
        "device": device,
    }


def test_bfloat16_view_restores_requested_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bfloat16 uses the int16 array descriptor and restores its torch dtype."""
    fake_tensor = _FakeTensor()

    monkeypatch.setattr(
        device_ops.torch,
        "as_tensor",
        lambda _value, *, device: fake_tensor,
    )

    result = device_ops.CudaDeviceOps().tensor_from_ptr(
        0x1000,
        (4,),
        torch.bfloat16,
        torch.device("cuda"),
    )

    assert result is fake_tensor
    assert fake_tensor.views == [(torch.bfloat16,), (4,)]


def test_reports_unavailable_copy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure is explicit when neither CUDA construction path is available."""

    def unavailable_array_interface(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("CUDA array interface unavailable")

    monkeypatch.setattr(
        device_ops.torch,
        "as_tensor",
        unavailable_array_interface,
    )
    monkeypatch.setattr(device_ops, "_get_copy_lib", lambda: None)

    with pytest.raises(RuntimeError, match="libcudart/libamdhip"):
        device_ops.CudaDeviceOps().tensor_from_ptr(
            0x1000,
            (4,),
            torch.float16,
            torch.device("cuda"),
        )
