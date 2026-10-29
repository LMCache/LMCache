# SPDX-License-Identifier: Apache-2.0
"""Compiled pointer kernels must honor canonical K/V region strides."""

# Standard
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache import lmcache_native
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
from lmcache.v1.gpu_connector.utils import (
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

pytestmark = [pytest.mark.cuda, pytest.mark.no_shared_allocator]


@pytest.fixture
def region_kernel() -> Callable[..., None]:
    """Do not substitute tensor fallback ops for the advertised native kernel."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    native = pytest.importorskip("lmcache.cuda_ops")
    if not getattr(native, "supports_strided_kv_regions", lambda: False)():
        pytest.skip("CUDA extension does not advertise strided K/V regions")
    return native.multi_layer_block_kv_transfer


@pytest.mark.parametrize("kv_axis", [0, 1], ids=["kv-first", "blocks-first"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("skip_blocks", [0, 1])
def test_native_region_pointer_gather_scatter(
    region_kernel: Callable[..., None],
    kv_axis: int,
    dtype: torch.dtype,
    skip_blocks: int,
) -> None:
    """Exercise the daemon's grouped address operands against the real kernel.

    Two layers can share one descriptor; a third requires another descriptor.
    B=2 cannot identify the K/V axis by shape. Padding, nonzero storage origin,
    nonunit D stride and skipped prefix blocks all survive exact writeback.
    """
    device = torch.device("cuda")
    native_layers = []
    expected_layers = []
    regions: list[DiscoverableKVCache] = []
    for layer, width in enumerate((12, 12, 16)):
        size = 2 * 2 * 3 * 4 * width
        backing = torch.full((size + 17,), -123, device=device, dtype=dtype)
        native = backing[7 : 7 + size].view(2, 2, 3, 1, 4, width)
        planes = []
        for component in range(2):
            plane = native.select(kv_axis, component).squeeze(2)[..., :10:2]
            plane = plane.permute(0, 2, 1, 3)
            plane.copy_(
                torch.arange(120, device=device).reshape(2, 4, 3, 5)
                + layer * 200
                + component * 50
            )
            planes.append(plane)
        native_layers.append(native)
        expected_layers.append(native.clone())
        regions.append(cast(DiscoverableKVCache, planes))

    normalized, formats = normalize_and_discover_per_layer_formats(
        regions,
        (),
        EngineType.VLLM,
        {"spec_transfer_regions": True, "kv_list_layout": "unified"},
    )
    manager = KVLayerGroupsManager(
        normalized,
        formats,
        lmcache_tokens_per_chunk=8,
        allow_strided_regions=True,
    )
    assert sorted(len(group.layer_indices) for group in manager.kernel_groups) == [1, 2]
    block_order = [1, 0]
    block_ids = torch.tensor(block_order, device=device, dtype=torch.int64)
    for group_index, group in enumerate(manager.kernel_groups):
        desc = manager.get_shape_desc(group_index)
        assert tuple(desc.region_k_strides)[-1] == 2
        assert tuple(desc.region_v_strides)[-1] == 2
        pointers = torch.tensor(
            get_group_data_ptrs(
                normalized, formats[group.layer_indices[0]], group.layer_indices
            ),
            device=device,
            dtype=torch.int64,
        )
        chunk = torch.full(
            (2, len(group.layer_indices), 8, 15),
            -777,
            dtype=dtype,
            pin_memory=True,
        )
        expected_chunk = torch.full_like(chunk, -777)
        for local_layer, layer in enumerate(group.layer_indices):
            planes = regions[layer]
            assert isinstance(planes, list)
            for component, plane in enumerate(planes):
                assert isinstance(plane, torch.Tensor)
                expected_chunk[component, local_layer, skip_blocks * 4 :] = (
                    plane[block_order[skip_blocks:]].reshape(-1, 15).cpu()
                )
        fmt = formats[group.layer_indices[0]]
        region_kernel(
            pointers,
            [chunk.data_ptr()],
            block_ids,
            device,
            lmcache_native.TransferDirection.D2H,
            desc,
            8,
            fmt,
            skip_blocks,
        )
        torch.cuda.synchronize()
        assert torch.equal(chunk, expected_chunk)
        chunk[0].add_(16)
        chunk[1].add_(32)
        region_kernel(
            pointers,
            [chunk.data_ptr()],
            block_ids,
            device,
            lmcache_native.TransferDirection.H2D,
            desc,
            8,
            fmt,
            skip_blocks,
        )
        torch.cuda.synchronize()
        for layer in group.layer_indices:
            for component, increment in enumerate((16, 32)):
                plane = expected_layers[layer].select(kv_axis, component).squeeze(2)
                # Index each block separately: advanced indexing followed by
                # add_ would update a temporary and weaken this assertion.
                for block in block_order[skip_blocks:]:
                    plane[block, ..., :10:2].add_(increment)
    for native, expected in zip(native_layers, expected_layers, strict=True):
        assert torch.equal(native, expected)


def _consume_cuda_handles(conn: "Connection", payload: bytes) -> None:
    """A real daemon process opens native CUDA handles before recipe restore."""
    # Standard
    from typing import Any
    import traceback

    # First Party
    from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
    from lmcache.v1.multiprocess.custom_types import get_customized_decoder
    from lmcache.v1.platform.devices.cuda.cache_context import GPUCacheContext
    import lmcache.cuda_ops as native

    context = None
    try:
        wire = get_customized_decoder(Any).decode(payload)
        context = GPUCacheContext(
            wire["wrappers"], lmcache_tokens_per_chunk=4, layout_hints=wire["hints"]
        )
        assert hasattr(native, "execute_object_group_transfer")
        groups = context.kv_layer_groups_manager.kernel_groups
        assert len(context.kv_layer_groups_manager.object_groups) == 1
        raw = torch.empty(
            context.get_temp_object_group_buffer(0, 0).numel(),
            dtype=torch.uint8,
            pin_memory=True,
        )
        obj = TensorMemoryObj(
            raw,
            MemoryObjMetadata(
                shape=raw.shape,
                dtype=raw.dtype,
                address=0,
                phy_size=raw.numel(),
                ref_count=1,
            ),
            parent_allocator=None,
        )
        block_ids = [
            torch.tensor([1], device=context.device, dtype=torch.int64) for _ in groups
        ]
        # No host block IDs: the public server entry point selects the native
        # object-group executor, including its pointer-based staging kernels.
        context.transfer_object_group(
            block_ids,
            [obj],
            object_group_id=0,
            batch_size=1,
            skip_first_n_tokens=0,
            direction=lmcache_native.TransferDirection.D2H,
            transfer_key="region-handle-gather",
        )
        torch.cuda.synchronize()
        result = {}
        offset = 0
        for index, group in enumerate(groups):
            shape, dtype = context.get_kernel_group_shape_dtype(4, index)
            nbytes = shape.numel() * dtype.itemsize
            chunk = raw[offset : offset + nbytes].view(dtype).view(shape)
            for local, layer in enumerate(group.layer_indices):
                result[layer] = chunk[:, local].tolist()
            chunk[0].add_(100)
            chunk[1].add_(200)
            offset += nbytes
        assert offset == raw.numel()
        context.transfer_object_group(
            block_ids,
            [obj],
            object_group_id=0,
            batch_size=1,
            skip_first_n_tokens=0,
            direction=lmcache_native.TransferDirection.H2D,
            transfer_key="region-handle-scatter",
        )
        torch.cuda.synchronize()
        conn.send(("ok", result))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if context is not None:
            context.close()
        conn.close()


@pytest.mark.parametrize("kv_axis", [0, 1], ids=["kv-first", "blocks-first"])
@pytest.mark.parametrize("owner_step", [1, 2], ids=["dense-owner", "sliced-owner"])
@pytest.mark.parametrize("kind", ["torch", "raw"])
def test_cuda_lmcache_driven_native_handle_round_trip(
    region_kernel: Callable[..., None], kv_axis: int, owner_step: int, kind: str
) -> None:
    """Worker register -> wrapper wire -> GPU context -> native pointer copy.

    Sliced owners require the region-owner wrapper to preserve shape/stride.
    Ordinary CUDA IPC retains the original storage origin, while raw CUDA IPC
    rebases its imported tensor; both must update the same producer cache.
    """
    # Standard
    import multiprocessing

    # First Party
    from lmcache.v1.platform.ipc_policy import is_isolated_ipc, set_isolated_ipc
    from tests.v1.platform.test_transfer_regions import _caches, _planes, _register

    del region_kernel  # The fixture verifies the native capability before spawn.
    previous = is_isolated_ipc()
    set_isolated_ipc(kind == "raw")
    context = None
    process = None
    parent_conn, child_conn = multiprocessing.get_context("spawn").Pipe()
    try:
        native, regions = _caches(kv_axis, device="cuda", owner_step=owner_step)
        expected_native = {name: tensor.clone() for name, tensor in native.items()}
        expected_chunks = {}
        for layer, (name, planes) in enumerate(regions.items()):
            expected_chunks[layer] = (
                torch.stack([plane[1].reshape(4, 15) for plane in _planes(planes)])
                .cpu()
                .tolist()
            )
            for component, increment in enumerate((100, 200)):
                plane = expected_native[name].select(kv_axis, component).squeeze(2)
                plane[1, ..., :5].add_(increment)
        context, payload = _register(native, regions)
        torch.cuda.synchronize()
        process = multiprocessing.get_context("spawn").Process(
            target=_consume_cuda_handles, args=(child_conn, payload)
        )
        process.start()
        child_conn.close()
        assert parent_conn.poll(60), "CUDA daemon did not finish the handle transfer"
        status, result = parent_conn.recv()
        assert status == "ok", result
        assert result == expected_chunks
        process.join(10)
        assert process.exitcode == 0
        torch.cuda.synchronize()
        for name, tensor in native.items():
            assert torch.equal(tensor, expected_native[name]), name
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(10)
        if context is not None:
            context.close()
        parent_conn.close()
        child_conn.close()
        set_isolated_ipc(previous)
