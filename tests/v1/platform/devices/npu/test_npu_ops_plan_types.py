# SPDX-License-Identifier: Apache-2.0
"""``lmcache.npu_ops`` consumes the plan types registered by ``lmcache_native``.

The server builds shape descriptors through
:func:`~lmcache.v1.common.utils.make_page_buffer_shape_desc` and kernel-group
specs through :mod:`lmcache.v1.common.ops_types`. ``lmcache.npu_ops``
registers neither type, so the contract is cross-module and can only be
checked against the compiled extension: a descriptor from the production
factory, plane geometry included, must drive a real multi-plane transfer,
both through the direct entry point and through the plan executor. The torch
baseline serves as the oracle.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.ops_types import KernelGroupSpec, PageBufferShapeDesc
from lmcache.v1.common.utils import make_page_buffer_shape_desc
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.devices.npu.device_ops import NpuDeviceOps
import lmcache.lmcache_native as lmcache_native

pytestmark = pytest.mark.npu

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU hardware is required",
)

F = lmcache_native.EngineKVFormat
D2H = lmcache_native.TransferDirection.D2H
H2D = lmcache_native.TransferDirection.H2D

NL, NB, BS = 2, 4, 4
BLOCK_IDS = [2, 0]
CHUNK = BS * len(BLOCK_IDS)

#: Per format: the per-plane ``[NB, BS, NH, HS]`` engine shapes, the dtype,
#: and the LMCache object shape.
_Layout = tuple[tuple[tuple[int, ...], ...], torch.dtype, tuple[int, ...]]
_LAYOUTS: dict[lmcache_native.EngineKVFormat, _Layout] = {
    # K slab, then V slab.
    F.NL_X_TWO_X_NB_BS_NH_HS: (
        ((NB, BS, 2, 8), (NB, BS, 2, 8)),
        torch.float16,
        (2, NL, CHUNK, 16),
    ),
    # One packed row per token: latent, then rope.
    F.NL_X_NP_X_NB_BS_ONE_HS: (
        ((NB, BS, 1, 128), (NB, BS, 1, 64)),
        torch.bfloat16,
        (NL, CHUNK, 192),
    ),
}
_LAYOUT_IDS = ["kv_tuple", "mla_tuple"]


def _fill(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, salt: int
) -> torch.Tensor:
    values = torch.arange(torch.tensor(shape).prod(), dtype=torch.int32) + salt
    return values.remainder(1009).to(dtype).to(device).view(shape)


def _build(fmt: lmcache_native.EngineKVFormat, device: torch.device) -> dict[str, Any]:
    """Build engine planes, their pointer table and the factory descriptor."""
    shapes, dtype, _ = _LAYOUTS[fmt]
    layers = [
        tuple(
            _fill(shape, dtype, device, 137 * (2 * layer + plane))
            for plane, shape in enumerate(shapes)
        )
        for layer in range(NL)
    ]
    desc = make_page_buffer_shape_desc(
        layers,
        fmt,
        layer_idx=0,
        num_layers_in_group=NL,
        num_blocks=NB,
        block_size=BS,
    )
    table = torch.tensor(
        [int(plane.data_ptr()) for layer in layers for plane in layer],
        dtype=torch.int64,
        device=device,
    )
    return {"layers": layers, "desc": desc, "table": table, "dtype": dtype}


def _object(fmt: lmcache_native.EngineKVFormat, device: torch.device) -> torch.Tensor:
    _, dtype, shape = _LAYOUTS[fmt]
    return torch.zeros(shape, dtype=dtype, device=device)


def _native_ops() -> NpuDeviceOps:
    """Bind the compiled module directly; the plugin hook has its own tests."""
    # First Party
    import lmcache.npu_ops as npu_ops

    ops = NpuDeviceOps()
    ops.bind_native(npu_ops)
    return ops


def _transfer_natively(
    ops: NpuDeviceOps,
    path: str,
    built: dict[str, Any],
    obj: torch.Tensor,
    block_ids: torch.Tensor,
    device: torch.device,
    direction: lmcache_native.TransferDirection,
    fmt: lmcache_native.EngineKVFormat,
) -> None:
    if path == "direct":
        ops.multi_layer_block_kv_transfer(
            built["table"],
            [int(obj.data_ptr())],
            block_ids,
            device,
            direction,
            built["desc"],
            CHUNK,
            fmt,
            0,
        )
    else:
        spec = KernelGroupSpec(
            paged_buffer_ptrs=int(built["table"].data_ptr()),
            lmcache_objects_ptrs=[int(obj.data_ptr())],
            shape_desc=built["desc"],
            lmcache_chunk_size=CHUNK,
            engine_kv_format=int(fmt),
            block_ids_base=int(block_ids.data_ptr()),
            block_ids_capacity=block_ids.numel(),
        )
        launch = ops.LaunchVar(0, 0, block_ids.numel(), 1, 0)
        ops.execute_object_group_transfer(
            int(direction), device, 1 << 26, [spec], [ops.BatchStep([], [launch])]
        )
    torch.npu.synchronize()


@requires_npu
def test_binding_keeps_shared_plan_types() -> None:
    """``lmcache.npu_ops`` must not shadow the shared types with its own.

    The server creates descriptors and kernel-group specs through the shared
    classes; a native copy of either would make every transfer fail to cast.
    """
    ops = _native_ops()
    assert ops.PageBufferShapeDesc is PageBufferShapeDesc
    assert ops.KernelGroupSpec is KernelGroupSpec
    assert type(ops.BatchStep([], [])).__module__ == "lmcache.npu_ops"


@requires_npu
@pytest.mark.parametrize("path", ["direct", "plan"])
@pytest.mark.parametrize("direction", [D2H, H2D], ids=["d2h", "h2d"])
@pytest.mark.parametrize("fmt", list(_LAYOUTS), ids=_LAYOUT_IDS)
def test_factory_desc_moves_the_baseline_bytes(
    path: str,
    direction: lmcache_native.TransferDirection,
    fmt: lmcache_native.EngineKVFormat,
) -> None:
    """A factory-built descriptor drives the same bytes as the torch baseline."""
    device = torch.device("npu:0")
    ops = _native_ops()
    native, reference = _build(fmt, device), _build(fmt, device)

    desc = native["desc"]
    assert isinstance(desc, ops.PageBufferShapeDesc)
    shapes, dtype, _ = _LAYOUTS[fmt]
    assert desc.num_planes == len(shapes)
    assert tuple(desc.plane_slot_bytes) == tuple(
        shape[2] * shape[3] * dtype.itemsize for shape in shapes
    )
    assert tuple(desc.plane_block_stride_bytes) == tuple(
        BS * shape[2] * shape[3] * dtype.itemsize for shape in shapes
    )

    block_ids = torch.tensor(BLOCK_IDS, dtype=torch.int64, device=device)
    if direction == H2D:
        # A valid source object comes from a D2H pass over a third layout.
        source = _object(fmt, device)
        torch_ops.multi_layer_block_kv_transfer(
            _build(fmt, device)["layers"],
            [source],
            block_ids,
            device,
            D2H,
            desc,
            CHUNK,
            fmt,
            0,
        )
        native_obj, reference_obj = source.clone(), source.clone()
        for planes in native["layers"] + reference["layers"]:
            for plane in planes:
                plane.zero_()
    else:
        native_obj, reference_obj = _object(fmt, device), _object(fmt, device)

    _transfer_natively(ops, path, native, native_obj, block_ids, device, direction, fmt)
    torch_ops.multi_layer_block_kv_transfer(
        reference["layers"],
        [reference_obj],
        block_ids,
        device,
        direction,
        reference["desc"],
        CHUNK,
        fmt,
        0,
    )

    if direction == D2H:
        assert torch.any(native_obj.cpu() != 0)
        assert torch.equal(native_obj.cpu(), reference_obj.cpu())
    else:
        for native_planes, reference_planes in zip(
            native["layers"], reference["layers"], strict=True
        ):
            for got, expected in zip(native_planes, reference_planes, strict=True):
                assert torch.any(got.cpu() != 0)
                assert torch.equal(got.cpu(), expected.cpu())
