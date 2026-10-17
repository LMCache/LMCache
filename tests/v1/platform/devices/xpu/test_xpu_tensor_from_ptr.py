# SPDX-License-Identifier: Apache-2.0
"""Tests for the XPU-local pointer-to-Tensor adapter."""

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.xpu import tensor_from_ptr


def test_contiguous_strides() -> None:
    """The helper reports standard dense element strides."""
    assert tensor_from_ptr.contiguous_row_major_strides((2, 3, 4)) == (12, 4, 1)


def test_storage_nbytes_covers_strided_view() -> None:
    """The storage must span the last addressable element, not just the extent."""
    # Rows are 4 elements apart but only 3 are read, so the last element sits
    # at 1 * 4 + 2 = 6 -> 7 elements -> 28 bytes for float32.
    assert tensor_from_ptr._storage_nbytes((2, 3), torch.float32, (4, 1), 0) == 28
    assert tensor_from_ptr._storage_nbytes((0, 3), torch.float32, (4, 1), 0) == 0


def test_constructs_non_owning_view_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch receives the pointer and byte size, and the view metadata is applied."""
    fake_device = torch.device("xpu", 0)
    storage = object()
    captured: dict[str, Any] = {}

    def construct_storage(ptr: int, device: object, nbytes: int) -> object:
        captured["storage_args"] = (ptr, device, nbytes)
        return storage

    class FakeTensor:
        def set_(self, *args: object) -> None:
            captured["set_args"] = args

    def fake_empty(*args: object, **kwargs: object) -> FakeTensor:
        captured["empty_kwargs"] = kwargs
        return FakeTensor()

    monkeypatch.setattr(
        torch._C,
        "_construct_storage_from_data_pointer",
        construct_storage,
        raising=False,
    )
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(tensor_from_ptr, "_CONSTRUCT_XPU_TENSOR", None)

    result = tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
        0xABCD,
        (2, 3),
        torch.float32,
        fake_device,
        stride=(4, 1),
    )

    assert isinstance(result, FakeTensor)
    assert captured["storage_args"] == (0xABCD, fake_device, 28)
    assert captured["empty_kwargs"] == {
        "dtype": torch.float32,
        "device": fake_device,
    }
    assert captured["set_args"] == (storage, 0, (2, 3), (4, 1))


def test_wraps_level_zero_pointer_above_int64_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high USM address reaches torch as its two's-complement form.

    Level Zero hands out device allocations near the top of the address
    space, so a real ``Tensor.data_ptr()`` on Intel XPU looks like
    ``0xffffe38f1c000000``.  ``_construct_storage_from_data_pointer`` takes an
    ``int64_t``, so pybind rejects the unsigned value outright; the adapter
    must wrap it first.  A low CUDA-style pointer must survive untouched.
    """
    fake_device = torch.device("xpu", 0)
    captured: dict[str, Any] = {}

    def construct_storage(ptr: int, device: object, nbytes: int) -> object:
        if not -(2**63) <= ptr < 2**63:
            raise TypeError("incompatible function arguments")
        captured["ptr"] = ptr
        return object()

    class FakeTensor:
        def set_(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        torch._C,
        "_construct_storage_from_data_pointer",
        construct_storage,
        raising=False,
    )
    monkeypatch.setattr(torch, "empty", lambda *a, **k: FakeTensor())
    monkeypatch.setattr(tensor_from_ptr, "_CONSTRUCT_XPU_TENSOR", None)

    high_pointer = 0xFFFFE38F1C000000
    tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
        high_pointer, (2,), torch.float32, fake_device
    )
    assert captured["ptr"] == high_pointer - 2**64
    assert captured["ptr"] & 0xFFFFFFFFFFFFFFFF == high_pointer

    tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
        0x1B4D0580, (2,), torch.float32, fake_device
    )
    assert captured["ptr"] == 0x1B4D0580


def test_prefers_xpu_constructor_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch's XPU constructor receives the metadata dict and the storage.

    ``torch.empty`` must not be reached: the ``set_`` fallback is only for
    torch builds without the constructor.
    """
    fake_device = torch.device("xpu", 0)
    storage = object()
    sentinel = object()
    captured: dict[str, Any] = {}

    def construct_storage(ptr: int, device: object, nbytes: int) -> object:
        captured["storage_args"] = (ptr, device, nbytes)
        return storage

    def construct_tensor(metadata: dict[str, Any], storage_arg: object) -> object:
        captured["ctor_args"] = (metadata, storage_arg)
        return sentinel

    def fail_empty(*args: object, **kwargs: object) -> None:
        raise AssertionError("set_ fallback used although the constructor exists")

    monkeypatch.setattr(
        torch._C,
        "_construct_storage_from_data_pointer",
        construct_storage,
        raising=False,
    )
    monkeypatch.setattr(tensor_from_ptr, "_CONSTRUCT_XPU_TENSOR", construct_tensor)
    monkeypatch.setattr(torch, "empty", fail_empty)

    high_pointer = 0xFFFFE38F1C000000
    result = tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
        high_pointer,
        (2, 3),
        torch.float32,
        fake_device,
        stride=(4, 1),
        storage_offset=1,
    )

    assert result is sentinel
    assert captured["storage_args"] == (high_pointer - 2**64, fake_device, 32)
    assert captured["ctor_args"] == (
        {
            "size": (2, 3),
            "stride": (4, 1),
            "dtype": torch.float32,
            "storage_offset": 1,
        },
        storage,
    )


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="requires an Intel XPU device",
)
@pytest.mark.parametrize(
    "path",
    [
        "set_fallback",
        pytest.param(
            "xpu_constructor",
            marks=pytest.mark.skipif(
                tensor_from_ptr._CONSTRUCT_XPU_TENSOR is None,
                reason="torch lacks _construct_XPU_Tensor_From_Storage_And_Metadata",
            ),
        ),
    ],
)
def test_real_xpu_view_aliases_allocation(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Both paths yield a strided view that aliases the live USM allocation.

    Uses a real Level Zero pointer, so it also exercises the high-address
    two's-complement wrap against the real torch binding.
    """
    if path == "set_fallback":
        monkeypatch.setattr(tensor_from_ptr, "_CONSTRUCT_XPU_TENSOR", None)
    base = torch.arange(16, dtype=torch.float32, device="xpu")

    view = tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
        base.data_ptr(),
        (2, 3),
        torch.float32,
        base.device,
        stride=(4, 1),
        storage_offset=1,
    )

    assert view.device == base.device
    assert view.shape == (2, 3)
    assert view.stride() == (4, 1)
    assert view.data_ptr() == base.data_ptr() + 4
    assert torch.equal(view.cpu(), base.cpu()[1:9].view(2, 4)[:, :3])
    view.fill_(-1.0)
    torch.xpu.synchronize()
    assert base[1].item() == -1.0 and base[7].item() == -1.0
    assert base[4].item() == 4.0  # stride gap untouched


def test_resolves_current_device_when_index_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unindexed device is pinned to the current ordinal.

    The storage and the tensor aliasing it must agree on the index, so
    ``xpu`` must not reach ``_construct_storage_from_data_pointer`` unindexed.
    """
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(is_available=lambda: True, current_device=lambda: 3),
    )
    assert tensor_from_ptr._normalize_device("xpu") == torch.device("xpu", 3)
    assert tensor_from_ptr._normalize_device(1) == torch.device("xpu", 1)


def test_rejects_invalid_pointer_dtype_and_device() -> None:
    """Invalid process-local pointers, dtypes, and non-XPU devices fail early."""
    with pytest.raises(ValueError, match="non-zero"):
        tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
            0, (1,), torch.float32, "xpu:0"
        )
    with pytest.raises(TypeError, match="torch.dtype"):
        tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
            1, (1,), cast(torch.dtype, "float32"), "xpu:0"
        )
    with pytest.raises(ValueError, match="XPU"):
        tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
            1, (1,), torch.float32, "cpu"
        )
    with pytest.raises(ValueError, match="stride"):
        tensor_from_ptr.construct_xpu_tensor_from_data_pointer(
            1, (2, 3), torch.float32, "xpu:0", stride=(1,)
        )


def test_device_ops_delegates_to_the_xpu_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``XpuDeviceOps`` must override the base, which rejects raw pointers.

    The LMCache-driven transfer path hands the block-transfer fallback raw
    pointers, so without this override XPU raises instead of transferring.
    """
    # First Party
    from lmcache.v1.platform.devices.xpu import device_ops as xpu_device_ops

    sentinel = object()
    seen: dict[str, Any] = {}

    def fake_construct(
        ptr: int, shape: tuple[int, ...], dtype: torch.dtype, device: object
    ) -> object:
        seen["args"] = (ptr, shape, dtype, device)
        return sentinel

    monkeypatch.setattr(
        xpu_device_ops, "construct_xpu_tensor_from_data_pointer", fake_construct
    )

    ops = xpu_device_ops.XpuDeviceOps()
    device = torch.device("xpu", 0)
    assert ops.tensor_from_ptr(0x1234, (2, 2), torch.float16, device) is sentinel
    assert seen["args"] == (0x1234, (2, 2), torch.float16, device)
