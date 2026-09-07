# SPDX-License-Identifier: Apache-2.0
"""Tests for MUSA-local reconstruction of generic pointer operands."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform.musa import device_ops
from lmcache.v1.platform.ops_types import PageBufferShapeDesc


def test_pointer_operands_are_reconstructed_inside_musa_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paged and staging pointers become views before native dispatch."""
    fake_device = SimpleNamespace(type="musa", index=0)
    calls: list[tuple[int, tuple[int, ...], torch.dtype, tuple[int, ...] | None]] = []

    def construct(
        ptr: int,
        shape: tuple[int, ...],
        _dtype: torch.dtype,
        _device: object,
        *,
        stride: tuple[int, ...] | None = None,
        **_kwargs: object,
    ) -> torch.Tensor:
        calls.append((ptr, shape, _dtype, stride))
        return torch.empty(shape, dtype=_dtype)

    monkeypatch.setattr(
        device_ops,
        "_as_device",
        lambda _device, _paged, _objects: fake_device,  # type: ignore[return-value]
    )
    monkeypatch.setattr(
        device_ops,
        "construct_musa_tensor_from_data_pointer",
        construct,
    )
    monkeypatch.setattr(
        "lmcache.v1.platform.musa.native_kv_transfer"
        ".try_native_multi_layer_block_kv_transfer",
        lambda **_kwargs: True,
    )

    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = 2
    shape_desc.nb = 3
    shape_desc.bs = 4
    shape_desc.nh = 1
    shape_desc.hs = 8
    shape_desc.element_size = 2
    shape_desc.dtype = torch.bfloat16
    shape_desc.block_stride_elems = 40

    device_ops.MusaDeviceOps().multi_layer_block_kv_transfer(
        torch.tensor([101, 202], dtype=torch.int64),
        [303],
        torch.tensor([0], dtype=torch.int64),
        fake_device,  # type: ignore[arg-type]
        TransferDirection.D2H,
        shape_desc,
        4,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )

    assert calls == [
        (101, (3, 4, 8), torch.bfloat16, (40, 8, 1)),
        (202, (3, 4, 8), torch.bfloat16, (40, 8, 1)),
        (303, (2, 4, 8), torch.bfloat16, None),
    ]


def test_pointer_transfer_rejects_ambiguous_two_byte_dtype() -> None:
    """Pointer-only transfer fails closed without an exact 2-byte dtype."""
    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = 1
    shape_desc.nb = 1
    shape_desc.bs = 1
    shape_desc.nh = 1
    shape_desc.hs = 1
    shape_desc.element_size = 2

    with pytest.raises(ValueError, match="exact shape_desc.dtype"):
        device_ops.MusaDeviceOps().multi_layer_block_kv_transfer(
            torch.tensor([101], dtype=torch.int64),
            [202],
            torch.tensor([0], dtype=torch.int64),
            torch.device("cpu"),
            TransferDirection.D2H,
            shape_desc,
            1,
            EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )
