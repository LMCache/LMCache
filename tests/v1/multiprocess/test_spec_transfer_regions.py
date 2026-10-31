# SPDX-License-Identifier: Apache-2.0
"""Spec regions through real worker registration, wire store, and retrieve."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    apply_spec_transfer_regions,
    create_engine_group_infos_from_vllm,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import expand_engine_block_ids
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.tensor_wire import (
    decode_grouped_tensors,
    decode_tensors,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    create_transfer_context,
)

T = TypeVar("T")


# The producer is the external boundary; transfer code below is not mocked.
@dataclass
class _ProducerSpec:
    kv_axis: int
    block_size: int = 4

    def get_kv_cache_transfer_regions(
        self, native: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            name: native.select(self.kv_axis, index).squeeze(2)
            for index, name in enumerate(("key", "value"))
        }


@dataclass
class _PerLayerSpecs:
    kv_cache_specs: dict[str, _ProducerSpec]
    block_size: int = 4


@dataclass
class _Group:
    layer_names: list[str]
    kv_cache_spec: _ProducerSpec | _PerLayerSpecs


@dataclass
class _Config:
    kv_cache_groups: list[_Group]


def _completed(value: T) -> MessagingFuture[T]:
    future: MessagingFuture[T] = MessagingFuture()
    future.set_result(value)
    return future


def _native(values: torch.Tensor, kv_axis: int) -> dict[str, torch.Tensor]:
    return {
        f"layer_{i}": (
            layer if kv_axis == 1 else layer.permute(1, 0, 2, 3, 4).contiguous()
        ).unsqueeze(3)
        for i, layer in enumerate(values)
    }


def _export(
    caches: Mapping[str, torch.Tensor], kv_axis: int
) -> tuple[dict[str, DiscoverableKVCache], LayoutHints]:
    config = _Config([_Group(list(caches), _ProducerSpec(kv_axis))])
    hints: LayoutHints = {}
    return apply_spec_transfer_regions(config, caches, hints), hints


@pytest.mark.parametrize(
    "source_axis,destination_axis", [(1, 1), (1, 0), (0, 1), (0, 0)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("skip_tokens", [0, 4])
def test_spec_regions_store_and_restore_original_native_storage(
    source_axis: int, destination_axis: int, dtype: torch.dtype, skip_tokens: int
) -> None:
    """B=2 must not swap K/V; restore must reach the engine's original tensors."""
    values = torch.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=dtype).view(2, 2, 2, 3, 4, 5)
    source_native = _native(values, source_axis)
    destination_native = _native(torch.full_like(values, -1), destination_axis)
    source, source_hints = _export(source_native, source_axis)
    destination, destination_hints = _export(destination_native, destination_axis)
    for name, planes in destination.items():
        assert isinstance(planes, list)
        for plane in planes:
            assert isinstance(plane, torch.Tensor)
            assert (
                plane.untyped_storage().data_ptr()
                == destination_native[name].untyped_storage().data_ptr()
            )

    client = MagicMock()
    client.register_kv_cache_engine_driven_context.return_value = _completed(
        RegisterEngineDrivenContextResponse()
    )
    client.prepare_store.return_value = _completed(PrepareStoreResponse())
    client.commit_store.return_value = _completed(True)
    client.commit_retrieve.return_value = _completed(True)
    contexts = [EngineDrivenTransferContext(i, client) for i in (1, 2)]
    for context, caches, hints in zip(
        contexts, (source, destination), (source_hints, destination_hints), strict=True
    ):
        context.register(caches, "model", 1, 2, 1.0, layout_hints=hints)
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, list(range(8)), 0, 8, "req")
    assert contexts[0].submit_store("req", key, source, [[1, 0]], None, 2).result()
    payload = client.commit_store.call_args.args[2]
    chunks = decode_tensors(
        payload,
        expected_shape=torch.Size((2, 2, 8, 15)),
        expected_dtype=dtype,
        expected_count=1,
    )
    expected = values[:, [1, 0]].permute(2, 0, 1, 4, 3, 5).reshape(2, 2, 8, 15)
    torch.testing.assert_close(chunks[0], expected)
    client.prepare_retrieve.return_value = _completed(
        PrepareRetrieveResponse(success=True, data=payload)
    )
    assert (
        contexts[1]
        .submit_retrieve(
            "req", key, destination, [[0, 1]], None, 2, skip_first_n_tokens=skip_tokens
        )
        .result()
    )
    for layer_index, native in enumerate(destination_native.values()):
        restored = native.squeeze(3)
        if destination_axis == 0:
            restored = restored.permute(1, 0, 2, 3, 4)
        torch.testing.assert_close(restored[1], values[layer_index, 0])
        torch.testing.assert_close(
            restored[0],
            values[layer_index, 1]
            if skip_tokens == 0
            else torch.full_like(restored[0], -1),
        )
    for context in contexts:
        context.close()


def test_specs_without_region_export_keep_native_tensors() -> None:
    """Released vLLM specs and absent configs retain the legacy path."""
    native = {"layer": torch.zeros(2, 2, 3, 1, 4, 5)}
    hints: LayoutHints = {}
    assert apply_spec_transfer_regions(None, native, hints)["layer"] is native["layer"]
    assert hints == {}


def test_handle_transfer_requires_native_owners_for_component_views() -> None:
    """Handle mode accepts regions, but cannot export detached plane handles."""
    values = torch.zeros(1, 2, 2, 3, 4, 5)
    caches, hints = _export(_native(values, 1), 1)
    client = MagicMock()
    context = create_transfer_context(
        caches, instance_id=1, req_client=client, mode="lmcache_driven"
    )
    try:
        with pytest.raises(ValueError, match="native"):
            context.register(caches, "model", 1, 1, 1.0, layout_hints=hints)
        client.register_kv_cache.assert_not_called()
    finally:
        context.close()


def test_region_copies_are_rejected_before_registration() -> None:
    """Exporting new storage would detach retrieve writes from attention's cache."""

    @dataclass
    class CopyingSpec(_ProducerSpec):
        def get_kv_cache_transfer_regions(
            self, native: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            return {
                name: region.clone()
                for name, region in super()
                .get_kv_cache_transfer_regions(native)
                .items()
            }

    native = {"layer": torch.zeros(2, 2, 3, 1, 4, 5)}
    config = _Config([_Group(["layer"], CopyingSpec(1))])
    with pytest.raises(ValueError, match="aliasing BHNC"):
        apply_spec_transfer_regions(config, native, {})


def test_shared_layer_alias_keeps_the_producers_regions() -> None:
    """An alias omitted from allocation specs must not return to shape guessing."""
    tensor = torch.arange(240, dtype=torch.float32).view(2, 2, 3, 1, 4, 5)
    native = {"producer": tensor, "alias": tensor}
    config = _Config([_Group(["producer"], _ProducerSpec(1))])
    hints: LayoutHints = {}
    exported = apply_spec_transfer_regions(config, native, hints)
    assert exported["alias"] is exported["producer"]
    alias = exported["alias"]
    assert isinstance(alias, list)
    value = alias[1]
    assert isinstance(value, torch.Tensor)
    value.fill_(9)
    assert torch.all(tensor[:, 1] == 9)


def test_meta_regions_are_rejected_before_registration() -> None:
    """Meta tensors have no readable storage despite sharing a zero pointer."""
    native = {"layer": torch.empty(2, 2, 3, 1, 4, 5, device="meta")}
    config = _Config([_Group(["layer"], _ProducerSpec(1))])
    with pytest.raises(ValueError, match="materialized"):
        apply_spec_transfer_regions(config, native, {})


def test_per_layer_specs_keep_different_native_orders() -> None:
    """A uniform-type container still resolves the actual spec of each layer."""
    first = torch.arange(240, dtype=torch.float32).view(2, 2, 3, 1, 4, 5)
    second = (first + 1000).clone()
    config = _Config(
        [
            _Group(
                ["full", "window"],
                _PerLayerSpecs({"full": _ProducerSpec(1), "window": _ProducerSpec(0)}),
            )
        ]
    )
    regions = apply_spec_transfer_regions(config, {"full": first, "window": second}, {})
    for name, tensor, axis in (("full", first, 1), ("window", second, 0)):
        planes = regions[name]
        assert isinstance(planes, list)
        for kv in (0, 1):
            plane = planes[kv]
            assert isinstance(plane, torch.Tensor)
            torch.testing.assert_close(
                plane, tensor.select(axis, kv).squeeze(2).permute(0, 2, 1, 3)
            )


def test_pointer_only_block_ops_reject_regions_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine-driven mode must not hand strided regions to pointer-only ops."""
    caches, hints = _export(_native(torch.zeros(1, 2, 2, 3, 4, 5), 1), 1)
    client = MagicMock()
    monkeypatch.setattr(
        worker_transfer, "supports_tensor_block_transfer", lambda _device: False
    )
    context = EngineDrivenTransferContext(1, client)
    try:
        with pytest.raises(NotImplementedError, match="tensor-based block transfer"):
            context.register(caches, "model", 1, 2, 1.0, layout_hints=hints)
        client.register_kv_cache_engine_driven_context.assert_not_called()
    finally:
        context.close()


def test_two_spec_region_groups_preserve_geometry_and_native_writeback() -> None:
    """Distinct block sizes and K/V orders retain group-local page mappings."""
    group_values = [
        torch.arange(240, dtype=torch.float32).view(1, 2, 2, 3, 4, 5),
        (torch.arange(48, dtype=torch.float32) + 1000).view(1, 2, 2, 2, 2, 3),
    ]
    layer_names = ["wide", "compact"]
    native_caches: list[dict[str, torch.Tensor]] = []
    exported_caches: list[dict[str, DiscoverableKVCache]] = []
    exported_hints: list[LayoutHints] = []
    group_infos = []
    for is_destination, axes in ((False, (1, 0)), (True, (0, 1))):
        native = {
            name: _native(
                torch.full_like(values, -1) if is_destination else values, axis
            )["layer_0"]
            for name, values, axis in zip(layer_names, group_values, axes, strict=True)
        }
        config = _Config(
            [
                _Group([name], _ProducerSpec(axis, block_size=block_size))
                for name, axis, block_size in zip(
                    layer_names, axes, (4, 2), strict=True
                )
            ]
        )
        hints: LayoutHints = {}
        caches = apply_spec_transfer_regions(config, native, hints)
        group_infos.append(create_engine_group_infos_from_vllm(config, caches, hints))
        native_caches.append(native)
        exported_caches.append(caches)
        exported_hints.append(hints)

    client = MagicMock()
    client.register_kv_cache_engine_driven_context.return_value = _completed(
        RegisterEngineDrivenContextResponse()
    )
    client.prepare_store.return_value = _completed(PrepareStoreResponse())
    client.commit_store.return_value = _completed(True)
    client.commit_retrieve.return_value = _completed(True)
    contexts = [EngineDrivenTransferContext(i, client) for i in (1, 2)]
    try:
        for context, caches, hints, groups in zip(
            contexts, exported_caches, exported_hints, group_infos, strict=True
        ):
            context.register(
                caches,
                "model",
                1,
                1,
                1.0,
                layout_hints=hints,
                engine_group_infos=groups,
                lmcache_chunk_tokens=4,
            )
        registration = client.register_kv_cache_engine_driven_context.call_args.args[0]
        assert [group.tokens_per_block for group in registration.group_layouts] == [
            4,
            2,
        ]
        assert [group.hidden_dim_size for group in registration.group_layouts] == [
            15,
            6,
        ]
        key = IPCCacheServerKey.from_token_ids(
            "model", 1, 0, list(range(4)), 0, 4, "req"
        )
        assert (
            contexts[0]
            .submit_store("req", key, exported_caches[0], [[1], [1, 0]], None, 1)
            .result()
        )
        payload = client.commit_store.call_args.args[2]
        decoded = decode_grouped_tensors(
            payload,
            expected=[
                (torch.Size((2, 1, 4, 15)), torch.float32, 1),
                (torch.Size((2, 1, 4, 6)), torch.float32, 1),
            ],
        )
        for values, blocks, hidden_size, (indices, chunks) in zip(
            group_values, ([1], [1, 0]), (15, 6), decoded, strict=True
        ):
            assert indices == [0]
            expected = (
                values[0, blocks].permute(1, 0, 3, 2, 4).reshape(2, 1, 4, hidden_size)
            )
            torch.testing.assert_close(chunks[0], expected)
        client.prepare_retrieve.return_value = _completed(
            PrepareRetrieveResponse(success=True, data=payload)
        )
        assert (
            contexts[1]
            .submit_retrieve("req", key, exported_caches[1], [[0], [0, 1]], None, 1)
            .result()
        )
        for name, values, axis in zip(layer_names, group_values, (0, 1), strict=True):
            restored = native_caches[1][name].squeeze(3)
            if axis == 0:
                restored = restored.permute(1, 0, 2, 3, 4)
            torch.testing.assert_close(restored[0], values[0, 1])
            torch.testing.assert_close(
                restored[1],
                torch.full_like(values[0, 0], -1) if name == "wide" else values[0, 0],
            )
    finally:
        for context in contexts:
            context.close()


def test_one_engine_group_splits_region_strides_and_reuses_its_block_ids() -> None:
    """Physical copy groups share their original engine block address space.

    Both layers have identical logical sizes, including the ambiguous B=2, but
    their native K/V axis positions give different per-block plane strides.
    The producer grouping and request expansion must agree on that split.
    """
    values = torch.arange(2 * 2 * 2 * 3 * 4 * 5, dtype=torch.float32).view(
        2, 2, 2, 3, 4, 5
    )
    native = {
        "kv_first": _native(values[:1], 0)["layer_0"],
        "blocks_first": _native(values[1:], 1)["layer_0"],
    }
    config = _Config(
        [
            _Group(
                list(native),
                _PerLayerSpecs(
                    {
                        "kv_first": _ProducerSpec(0),
                        "blocks_first": _ProducerSpec(1),
                    }
                ),
            )
        ]
    )
    hints: LayoutHints = {}
    caches = apply_spec_transfer_regions(config, native, hints)
    infos = create_engine_group_infos_from_vllm(config, caches, hints)

    assert len(config.kv_cache_groups) == 1
    assert [info.engine_group_id for info in infos] == [0, 0]
    assert [info.layer_indices for info in infos] == [(0,), (1,)]
    assert [info.tokens_per_block for info in infos] == [4, 4]
    engine_block_ids = [[1, 0]]
    expanded = expand_engine_block_ids(infos, engine_block_ids)
    assert expanded == [[1, 0], [1, 0]]
    # Independent lists prevent a group's trailing-window edits changing peers.
    expanded[0].pop()
    assert expanded[1] == engine_block_ids[0] == [1, 0]
