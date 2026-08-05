# SPDX-License-Identifier: Apache-2.0
"""Tests for the MUSA-local pointer-to-Tensor adapter."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.musa import tensor_from_ptr


def test_contiguous_strides() -> None:
    """The helper reports standard dense element strides."""
    assert tensor_from_ptr.contiguous_row_major_strides((2, 3, 4)) == (12, 4, 1)


def test_constructs_non_owning_view_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TorchMUSA receives the pointer, byte size, and view metadata."""
    fake_device = SimpleNamespace(type="musa", index=0)
    storage = object()
    expected = object()
    captured: dict[str, object] = {}

    def construct_storage(ptr: int, device: object, nbytes: int) -> object:
        captured["storage_args"] = (ptr, device, nbytes)
        return storage

    def construct_tensor(metadata: dict[str, object], storage_arg: object) -> object:
        captured["tensor_args"] = (metadata, storage_arg)
        return expected

    monkeypatch.setattr(
        tensor_from_ptr,
        "_normalize_device",
        lambda _device: fake_device,  # type: ignore[return-value]
    )
    monkeypatch.setattr(
        torch._C,
        "_construct_storage_from_data_pointer",
        construct_storage,
        raising=False,
    )
    monkeypatch.setattr(
        tensor_from_ptr,
        "_musac_module",
        lambda: SimpleNamespace(
            _construct_MUSA_Tensor_From_Storage_And_Metadata=construct_tensor
        ),
    )

    result = tensor_from_ptr.construct_musa_tensor_from_data_pointer(
        0xABCD,
        (2, 3),
        torch.float32,
        fake_device,
        stride=(4, 1),
    )

    assert result is expected
    assert captured["storage_args"] == (0xABCD, fake_device, 28)
    metadata, storage_arg = cast(
        tuple[dict[str, object], object], captured["tensor_args"]
    )
    assert storage_arg is storage
    assert metadata == {
        "size": (2, 3),
        "stride": (4, 1),
        "dtype": torch.float32,
        "device": fake_device,
        "storage_offset": 0,
    }


def test_rejects_invalid_pointer_and_device() -> None:
    """Invalid process-local pointers and non-MUSA devices fail early."""
    with pytest.raises(ValueError, match="non-zero"):
        tensor_from_ptr.construct_musa_tensor_from_data_pointer(
            0,
            (1,),
            torch.float32,
            "cpu",
        )
    with pytest.raises(ValueError, match="MUSA"):
        tensor_from_ptr.construct_musa_tensor_from_data_pointer(
            1,
            (1,),
            torch.float32,
            "cpu",
        )
