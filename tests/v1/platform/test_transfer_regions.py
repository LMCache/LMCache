# SPDX-License-Identifier: Apache-2.0
"""Handle IPC must preserve producer-owned K/V views through real pointer copies."""

# Future
from __future__ import annotations

# Standard
from multiprocessing.connection import Connection
from typing import Any, cast
from unittest.mock import MagicMock
import multiprocessing
import traceback

# Third Party
import pytest
import torch

# First Party
from lmcache import lmcache_native
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    LMCacheDrivenTransferContext,
    create_transfer_context,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.cpu.cache_context import CPUCacheContext
from lmcache.v1.platform.devices.cpu.device_ops import CpuDeviceOps
from lmcache.v1.platform.transfer_regions import (
    describe_transfer_regions,
    restore_transfer_regions,
)

pytestmark = pytest.mark.no_shared_allocator


def _caches(
    kv_axis: int, *, device: str = "cpu", owner_step: int = 1
) -> tuple[dict[str, torch.Tensor], dict[str, DiscoverableKVCache]]:
    """B=2 deliberately makes shape-based K/V-axis guessing ambiguous.

    Each contiguous native tensor has a nonzero storage origin. Its K/V
    planes have inner padding, and layer 1 has a different block stride.
    """
    native: dict[str, torch.Tensor] = {}
    regions: dict[str, DiscoverableKVCache] = {}
    for layer in range(2):
        width = 7 + layer
        shape = (2, 2, 3, 1, 4, width)
        size = 2 * 2 * 3 * 4 * width
        backing = torch.full((size * owner_step + 17,), -123.0, device=device)
        tensor = backing[7 : 7 + size * owner_step : owner_step].view(shape)
        planes = []
        for component in range(2):
            plane = tensor.select(kv_axis, component).squeeze(2)
            plane = plane[..., :5].permute(0, 2, 1, 3)
            values = torch.arange(2 * 4 * 3 * 5, device=device).view(2, 4, 3, 5)
            plane.copy_(values + layer * 10000 + component * 1000)
            planes.append(plane)
        native[f"layer_{layer}"] = tensor
        regions[f"layer_{layer}"] = cast(DiscoverableKVCache, planes)
    return native, regions


def _planes(value: DiscoverableKVCache) -> list[torch.Tensor]:
    assert isinstance(value, list)
    assert all(isinstance(plane, torch.Tensor) for plane in value)
    return cast(list[torch.Tensor], value)


def _completed(value: Any) -> MessagingFuture:
    future: MessagingFuture[Any] = MessagingFuture()
    future.set_result(value)
    return future


def _register(
    native: dict[str, torch.Tensor], regions: dict[str, DiscoverableKVCache]
) -> tuple[LMCacheDrivenTransferContext, bytes]:
    """Only the request transport is mocked; worker wrapping remains real."""
    client = MagicMock()
    client.register_kv_cache.return_value = _completed(True)
    context = create_transfer_context(
        regions, instance_id=73, req_client=client, mode="lmcache_driven"
    )
    assert isinstance(context, LMCacheDrivenTransferContext)
    context.register(
        regions,
        "region-ipc-test",
        1,
        1,
        10.0,
        layout_hints={"spec_transfer_regions": True, "kv_list_layout": "unified"},
        native_kv_caches=native,
    )
    # CPU wrapping migrates the native tensor into SHM. The worker must
    # refresh its views instead of retaining pre-migration storage aliases.
    for name, planes in regions.items():
        assert isinstance(planes, list)
        for plane in planes:
            assert isinstance(plane, torch.Tensor)
            assert (
                plane.untyped_storage().data_ptr()
                == native[name].untyped_storage().data_ptr()
            )
    client.register_kv_cache.assert_called_once()
    client.register_kv_cache_engine_driven_context.assert_not_called()
    args = client.register_kv_cache.call_args.args
    wrappers, hints = args[1], args[5]
    assert len(wrappers) == len(native)
    assert all(isinstance(wrapper, DeviceIPCWrapper) for wrapper in wrappers)
    # Use the actual tagged data-only IPC serializer, then msgpack the same
    # primitive hints carried by REGISTER_KV_CACHE. No tensor is pickled.
    payload = get_customized_encoder(Any).encode({"wrappers": wrappers, "hints": hints})
    return context, payload


def _consume_in_server(conn: Connection, payload: bytes) -> None:
    """Spawn target: import handles, build groups, then copy via raw pointers."""
    context = None
    try:
        wire = get_customized_decoder(Any).decode(payload)
        wrappers = wire["wrappers"]
        assert all(isinstance(w, DeviceIPCWrapper) for w in wrappers)
        context = CPUCacheContext(
            wrappers, lmcache_tokens_per_chunk=4, layout_hints=wire["hints"]
        )
        ops = CpuDeviceOps()
        gathered = {}
        groups = context.kv_layer_groups_manager.kernel_groups
        # Different native layer strides must not share one kernel descriptor.
        assert len(groups) == 2
        for index, group in enumerate(groups):
            pointers = context.get_kernel_group_kv_pointers(index)
            assert pointers.ndim == 1 and pointers.dtype == torch.int64
            assert pointers.numel() == 2 * len(group.layer_indices)
            chunk = torch.empty((2, len(group.layer_indices), 4, 15))
            common = (
                pointers,
                [chunk.data_ptr()],
                torch.tensor([1], dtype=torch.long),
                torch.device("cpu"),
            )
            desc = context.get_shape_desc(index)
            fmt = context.get_engine_kv_format(index)
            ops.multi_layer_block_kv_transfer(
                *common, lmcache_native.TransferDirection.D2H, desc, 4, fmt, 0
            )
            for local_index, layer in enumerate(group.layer_indices):
                gathered[layer] = chunk[:, local_index].tolist()
            chunk[0].add_(100)
            chunk[1].add_(200)
            ops.multi_layer_block_kv_transfer(
                *common, lmcache_native.TransferDirection.H2D, desc, 4, fmt, 0
            )
        conn.send(("ok", gathered))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if context is not None:
            context.close()
        conn.close()


@pytest.mark.parametrize("kv_axis", [0, 1], ids=["kv-first", "blocks-first"])
def test_lmcache_driven_regions_cross_process_pointer_round_trip(kv_axis: int) -> None:
    """Register real native handles; server writes must reach the engine owner.

    Check every native element, including the untouched block and padded
    head tails: a copied temporary, swapped K/V pointer, lost offset, or
    tight-stride reconstruction must fail this test.
    """
    native, regions = _caches(kv_axis)
    expected_native = {name: tensor.clone() for name, tensor in native.items()}
    expected_chunks = {}
    for layer, (name, planes) in enumerate(regions.items()):
        assert isinstance(planes, list)
        expected_chunks[layer] = torch.stack(
            [plane[1].reshape(4, 15) for plane in _planes(planes)]
        ).tolist()
        for component, increment in enumerate((100, 200)):
            target = expected_native[name].select(kv_axis, component).squeeze(2)
            target[1, ..., :5].add_(increment)

    context, payload = _register(native, regions)
    parent_conn, child_conn = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=_consume_in_server, args=(child_conn, payload)
    )
    try:
        process.start()
        child_conn.close()
        assert parent_conn.poll(60), "server child did not finish the IPC transfer"
        status, result = parent_conn.recv()
        assert status == "ok", result
        assert result == expected_chunks
        process.join(10)
        assert process.exitcode == 0
        for name, tensor in native.items():
            assert torch.equal(tensor, expected_native[name]), name
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent_conn.close()
        child_conn.close()
        context.close()


@pytest.mark.parametrize("receiver_origin", [0, 11])
def test_recipe_offsets_are_relative_to_imported_native_view(
    receiver_origin: int,
) -> None:
    """Raw/VMM imports can reset storage origin; ordinary IPC can retain it."""
    native, regions = _caches(1)
    hints: LayoutHints = {
        "spec_transfer_regions": True,
        "kv_list_layout": "unified",
        "transfer_region_views": describe_transfer_regions(native, regions),
    }
    imported = []
    for tensor in native.values():
        storage = torch.full((tensor.numel() + receiver_origin + 3,), -999.0)
        view = storage[receiver_origin : receiver_origin + tensor.numel()].view_as(
            tensor
        )
        view.copy_(tensor)
        imported.append(view)
    restored = restore_transfer_regions(imported, hints)
    for index, (name, planes) in enumerate(regions.items()):
        assert isinstance(planes, list)
        rebuilt = restored[index]
        assert isinstance(rebuilt, list)
        for component, (expected, actual) in enumerate(
            zip(planes, rebuilt, strict=True)
        ):
            assert isinstance(expected, torch.Tensor)
            assert isinstance(actual, torch.Tensor)
            assert torch.equal(actual, expected)
            actual.add_(17)
            original = imported[index].select(1, component).squeeze(2)[..., :5]
            assert torch.equal(original.permute(0, 2, 1, 3), expected + 17)


def test_absent_region_metadata_preserves_legacy_tensors() -> None:
    """Unchanged engines keep their original registration and tensor identity."""
    tensor = torch.zeros(2, 3, 4, 5, 6)
    assert restore_transfer_regions([tensor], None)[0] is tensor
    assert restore_transfer_regions([tensor], {})[0] is tensor


@pytest.mark.parametrize(
    "case,match",
    [
        ("version", "version or layer count"),
        ("layer-count", "version or layer count"),
        ("negative-offset", "relative offset"),
        ("outside-native-view", "exceeds.*native view"),
        ("overlap", "non-overlapping"),
        ("wrong-rank", "shape, stride"),
        ("bool-stride", "shape, stride"),
        ("mismatched-planes", "matching K/V"),
    ],
)
def test_import_rejects_invalid_view_recipes(case: str, match: str) -> None:
    """Reject hostile wire geometry before constructing pointer-visible views."""
    native, regions = _caches(1)
    recipes = describe_transfer_regions(native, regions)
    layer = recipes["layers"][0]
    assert layer is not None
    view = layer[0]
    if case == "version":
        cast(dict[str, Any], recipes)["version"] = 99
    elif case == "layer-count":
        recipes["layers"].pop()
    elif case == "negative-offset":
        view["offset"] = -1
    elif case == "outside-native-view":
        # The backing storage has a suffix; bounds must be against the
        # registered native view rather than the whole allocation.
        view["offset"] = next(iter(native.values())).numel() - 1
    elif case == "overlap":
        view["stride"] = (1, 1, 1, 1)
    elif case == "wrong-rank":
        view["shape"] = (2, 4, 3)  # type: ignore[typeddict-item]
        layer[1]["shape"] = view["shape"]
    elif case == "bool-stride":
        view["stride"] = (True, 21, 7, 1)
    else:
        layer[1]["shape"] = (2, 4, 3, 4)
    hints: LayoutHints = {"transfer_region_views": recipes}
    with pytest.raises(ValueError, match=match):
        restore_transfer_regions(list(native.values()), hints)


def test_export_rejects_detached_planes() -> None:
    """A valid-shaped copy cannot serve as an engine-owned transfer region."""
    native, regions = _caches(1)
    planes = regions["layer_0"]
    assert isinstance(planes, list)
    plane = planes[0]
    assert isinstance(plane, torch.Tensor)
    planes[0] = plane.clone()
    with pytest.raises(ValueError, match="alias native cache storage"):
        describe_transfer_regions(native, regions)


def test_spec_region_marker_requires_recipes_on_handle_receiver() -> None:
    """Fail early instead of silently guessing a registered six-dimensional pool."""
    native, _ = _caches(1)
    with pytest.raises(ValueError, match="require IPC view recipes"):
        restore_transfer_regions(list(native.values()), {"spec_transfer_regions": True})
