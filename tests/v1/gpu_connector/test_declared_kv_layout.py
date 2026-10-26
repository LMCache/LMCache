# SPDX-License-Identifier: Apache-2.0
"""Producer axes survive normalization, IPC and per-layer format selection."""

# Standard
from dataclasses import dataclass
from enum import IntEnum
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemGPUConnectorV3,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.integration.vllm.kv_cache_layout import encode_tensor_layouts
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.declared import (
    resolve_declared_tensor,
    select_layout_hints,
    tensor_axes,
)
from lmcache.v1.gpu_connector.kv_format.detection import detect_format
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
from lmcache.v1.gpu_connector.utils import (
    get_num_blocks,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.tensor_wire import decode_grouped_tensors
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.kv_wrap import wrap_kv_caches
import lmcache.lmcache_native as lmcache_native

F = lmcache_native.EngineKVFormat
BKH = ["BLOCK", "KV", "HEAD", "SINGLETON", "STATE", "HEAD_DIM"]
KBH = ["KV", "BLOCK", "HEAD", "SINGLETON", "STATE", "HEAD_DIM"]


def _hints(*axes: list[str] | None) -> LayoutHints:
    # Deliberately contradict the dense tensor declaration: it must win.
    return {"kv_layout": "NHD", "tensor_layouts": {"version": 1, "axes": list(axes)}}


@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize(
    "axes,expected",
    [(BKH, F.NL_X_NB_TWO_NH_BS_HS), (KBH, F.NL_X_TWO_NB_NH_BS_HS)],
)
def test_rbln_two_blocks_is_not_guessed(
    heads: int, axes: list[str], expected: lmcache_native.EngineKVFormat
) -> None:
    tensor = torch.arange(2 * 2 * heads * 5 * 8).view(2, 2, heads, 1, 5, 8)
    fmt, views = detect_format([tensor], EngineType.VLLM, _hints(axes))
    assert fmt == expected
    assert get_num_blocks(views, fmt) == 2
    assert isinstance(views[0], torch.Tensor)
    assert views[0].data_ptr() == tensor.data_ptr()
    assert torch.equal(views[0], tensor.squeeze(3))


@pytest.mark.parametrize(
    "physical_axes,shape,expected",
    [
        (
            ["KV", "BLOCK", "HEAD", "STATE", "HEAD_DIM"],
            (2, 7, 3, 5, 8),
            F.NL_X_TWO_NB_NH_BS_HS,
        ),
        (
            ["KV", "BLOCK", "STATE", "HEAD", "HEAD_DIM"],
            (2, 7, 5, 3, 8),
            F.NL_X_TWO_NB_BS_NH_HS,
        ),
        (
            ["BLOCK", "KV", "HEAD", "STATE", "HEAD_DIM"],
            (7, 2, 3, 5, 8),
            F.NL_X_NB_TWO_NH_BS_HS,
        ),
        (
            ["BLOCK", "KV", "STATE", "HEAD", "HEAD_DIM"],
            (7, 2, 5, 3, 8),
            F.NL_X_NB_TWO_BS_NH_HS,
        ),
    ],
)
def test_semantics_follow_stride_permutation(
    physical_axes: list[str],
    shape: tuple[int, ...],
    expected: lmcache_native.EngineKVFormat,
) -> None:
    storage = torch.arange(torch.tensor(shape).prod().item() + 9)
    original = storage[9:].view(shape)
    tensor = original.transpose(2, 3)
    axes = physical_axes.copy()
    axes[2], axes[3] = axes[3], axes[2]
    fmt, normalized = resolve_declared_tensor(tensor, axes)
    assert fmt == expected
    assert (
        normalized.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    )
    assert normalized.storage_offset() == 9
    assert torch.equal(normalized, original)


def test_equal_shape_layers_keep_different_formats_and_excluded_alias() -> None:
    kv = [torch.zeros(2, 2, 3, 1, 5, 8) for _ in range(3)]
    normalized, formats = normalize_and_discover_per_layer_formats(
        kv, [[0, 1]], EngineType.VLLM, _hints(BKH, KBH, None)
    )
    assert formats[:2] == [F.NL_X_NB_TWO_NH_BS_HS, F.NL_X_TWO_NB_NH_BS_HS]
    assert normalized[2] is kv[2]
    assert isinstance(normalized[0], torch.Tensor)
    assert isinstance(normalized[1], torch.Tensor)
    assert normalized[0].data_ptr() == kv[0].data_ptr()
    assert normalized[1].data_ptr() == kv[1].data_ptr()


def test_partial_declarations_keep_legacy_path() -> None:
    caches = [torch.zeros(2, 7, 5, 3, 8), torch.zeros(2, 7, 5, 3, 8)]
    axes = ["KV", "BLOCK", "HEAD", "STATE", "HEAD_DIM"]
    _, formats = normalize_and_discover_per_layer_formats(
        caches, (), EngineType.VLLM, _hints(axes, None)
    )
    assert formats == [F.NL_X_TWO_NB_NH_BS_HS, F.NL_X_TWO_NB_BS_NH_HS]


@pytest.mark.parametrize("heads", [1, 3])
def test_content_and_dim0_padding(heads: int) -> None:
    backing = torch.arange(7 * (heads * 5 * 8 + 16))
    tensor = backing.as_strided((7, heads, 5, 8), (heads * 5 * 8 + 16, 40, 8, 1))
    fmt, normalized = resolve_declared_tensor(
        tensor, ["BLOCK", "HEAD", "STATE", "CONTENT"]
    )
    assert fmt == F.NL_X_NB_NH_BS_CS
    assert normalized.stride(0) == tensor.stride(0)
    assert normalized.data_ptr() == tensor.data_ptr()


def test_mla_has_no_explicit_kv_selector() -> None:
    tensor = torch.zeros(7, 1, 5, 8)
    fmt, normalized = resolve_declared_tensor(
        tensor, ["BLOCK", "HEAD", "STATE", "CONTENT"]
    )
    assert fmt == F.NL_X_NB_NH_BS_CS
    assert normalized.shape == tensor.shape


@pytest.mark.parametrize(
    "shape,axes,message",
    [
        ((2, 7), ["KV"], "rank"),
        ((1, 7), ["KV", "BLOCK"], "size 2"),
        ((2, 7), ["SINGLETON", "BLOCK"], "size 1"),
        ((2, 7), ["BLOCK", "BLOCK"], "Invalid"),
        ((2, 7), ["FUTURE_AXIS", "BLOCK"], "Invalid"),
        ((7, 8), ["BLOCK", "CONTENT"], "Unsupported declared"),
    ],
)
def test_invalid_metadata_never_falls_back(
    shape: tuple[int, ...], axes: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        detect_format([torch.zeros(shape)], EngineType.VLLM, _hints(axes))


def test_unsupported_inner_padding_rejected() -> None:
    tensor = torch.zeros(7, 3, 5, 16)[..., ::2]
    with pytest.raises(ValueError, match="not a supported"):
        resolve_declared_tensor(tensor, ["BLOCK", "HEAD", "STATE", "CONTENT"])


def test_envelope_version_count_and_subset() -> None:
    with pytest.raises(ValueError, match="version"):
        tensor_axes({"tensor_layouts": {"version": 2, "axes": [BKH]}}, 1)
    with pytest.raises(ValueError, match="layer count"):
        tensor_axes(_hints(BKH), 2)
    assert select_layout_hints(_hints(BKH, None, KBH), [2, 0]) == _hints(KBH, BKH)


class Axis(IntEnum):
    BLOCK = 42
    KV = 43
    HEAD = 44
    SINGLETON = 45
    STATE = 46
    HEAD_DIM = 47


@dataclass(frozen=True)
class Layout:
    axes: tuple[Axis, ...]


def test_producer_bridge_uses_names_and_registration_order() -> None:
    caches = {"unspecified": torch.zeros(1), "rbln": torch.zeros(2, 2, 3, 1, 5, 8)}
    layout = Layout(tuple(Axis[name] for name in BKH))
    encoded = encode_tensor_layouts(caches, {"rbln": layout})
    assert encoded == {"version": 1, "axes": [None, BKH]}
    # This is the representation that travels in existing REGISTER_KV_CACHE hints.
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(encoded)) == encoded
    with pytest.raises(ValueError, match="unregistered"):
        encode_tensor_layouts(caches, {"unknown": layout})


def test_cpu_ipc_preserves_declared_view_and_writes_through() -> None:
    backing = torch.arange(9 + 2 * 2 * 3 * 1 * 5 * 8)
    tensor = backing[9:].view(2, 2, 3, 1, 5, 8).transpose(4, 5)
    axes = BKH.copy()
    axes[4], axes[5] = axes[5], axes[4]
    wrappers = wrap_kv_caches({"rbln": tensor}, preserve_views=["rbln"])
    wrapper = wrappers[0]
    decoded = DeviceIPCWrapper.Deserialize(DeviceIPCWrapper.Serialize(wrapper))
    received = decoded.to_tensor()
    assert received.shape == tensor.shape
    assert received.stride() == tensor.stride()
    assert received.storage_offset() == tensor.storage_offset()
    hints = msgspec.msgpack.decode(msgspec.msgpack.encode(_hints(axes)))
    fmt, normalized = detect_format([received], EngineType.VLLM, hints)
    assert fmt == F.NL_X_NB_TWO_NH_BS_HS
    assert isinstance(normalized[0], torch.Tensor)
    normalized[0][0, 1, 2, 3, 4] = -19
    assert tensor[0, 1, 2, 0, 4, 3] == -19


@pytest.mark.parametrize("axes", [BKH, KBH])
def test_engine_driven_gather_scatter_obeys_declared_kv_axis(axes: list[str]) -> None:
    source = torch.arange(2 * 2 * 3 * 5 * 8, dtype=torch.float32).view(2, 2, 3, 1, 5, 8)
    destination = torch.full_like(source, -1)
    chunks = gather_paged_kv_to_cpu(
        {"layer": source}, [1], 1, layout_hints=_hints(axes)
    )
    scatter_cpu_to_paged_kv(
        {"layer": destination}, [0], chunks, 1, layout_hints=_hints(axes)
    )
    block_axis = axes.index("BLOCK")
    assert torch.equal(destination.select(block_axis, 0), source.select(block_axis, 1))
    assert torch.all(destination.select(block_axis, 1) == -1)


def test_same_logical_axes_with_different_physical_strides() -> None:
    nhd = torch.zeros(7, 5, 3, 8)
    hnd = torch.zeros(7, 3, 5, 8).transpose(1, 2)
    axes = ["BLOCK", "STATE", "HEAD", "CONTENT"]
    _, formats = normalize_and_discover_per_layer_formats(
        [nhd, hnd], (), EngineType.VLLM, _hints(axes, axes)
    )
    assert formats == [F.NL_X_NB_BS_NH_CS, F.NL_X_NB_NH_BS_CS]


class ViewMetadataWrapper(DeviceIPCWrapper):
    supports_view_metadata = True


@pytest.mark.parametrize("offset_origin", [0, 9])
def test_ipc_retains_the_wrapper_offset_origin(
    monkeypatch: pytest.MonkeyPatch, offset_origin: int
) -> None:
    # Allocation handles count data_ptr displacement separately from this offset.
    tensor = torch.arange(489)[9:].view(2, 2, 3, 1, 5, 8)
    wrapper = ViewMetadataWrapper()
    wrapper.storage_offset = offset_origin
    monkeypatch.setattr(
        "lmcache.v1.platform.kv_wrap.wrap_one_kv_cache", lambda _: wrapper
    )
    result = wrap_kv_caches({"layer": tensor}, preserve_views=["layer"])[0]
    assert result.storage_offset == offset_origin
    assert result.shape == tuple(tensor.shape)
    assert result.stride == tensor.stride()


def test_tensor_handle_without_view_metadata_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.platform.kv_wrap.wrap_one_kv_cache", lambda _: DeviceIPCWrapper()
    )
    with pytest.raises(NotImplementedError, match="engine-driven"):
        wrap_kv_caches({"layer": torch.zeros(2, 3)}, preserve_views=["layer"])


@pytest.mark.parametrize(
    "connector_type",
    [
        VLLMPagedMemGPUConnectorV2,
        VLLMPagedMemGPUConnectorV3,
        VLLMBufferLayerwiseGPUConnector,
        VLLMPagedMemLayerwiseGPUConnector,
    ],
)
def test_inprocess_repeated_requests_keep_views_and_axes_aligned(
    connector_type: type[GPUConnectorInterface],
) -> None:
    # No device allocation: exercise the public shared pointer-input lifecycle.
    connector = object.__new__(connector_type)
    connector.layout_hints = {"kv_layout": "NHD"}
    physical = torch.arange(480).view(2, 2, 3, 1, 5, 8)
    registered = physical.transpose(4, 5)
    axes = BKH.copy()
    axes[4], axes[5] = axes[5], axes[4]
    connector.set_tensor_layouts(_hints(axes)["tensor_layouts"])
    for _ in range(2):
        connector.initialize_kvcaches_ptr(kvcaches=[registered])
        fmt, views = detect_format(
            connector.kvcaches, EngineType.VLLM, connector.layout_hints
        )
        assert fmt == F.NL_X_NB_TWO_NH_BS_HS
        assert torch.equal(views[0], physical.squeeze(3))
        assert isinstance(views[0], torch.Tensor)
        assert views[0].data_ptr() == physical.data_ptr()


def test_engine_driven_grouped_registration_and_store_selects_matching_axes() -> None:
    caches = {
        name: torch.arange(480, dtype=torch.float32).view(2, 2, 3, 1, 5, 8) + bias
        for name, bias in [("blocks_first", 0), ("kv_first", 1000)]
    }
    client = MagicMock()
    client.register_kv_cache_engine_driven_context.return_value.result.return_value = (
        RegisterEngineDrivenContextResponse()
    )
    client.commit_store.return_value.result.return_value = True
    context = EngineDrivenTransferContext(7, client)
    context.register(
        caches,
        "model",
        1,
        1,
        1.0,
        layout_hints=_hints(BKH, KBH),
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(0,), tokens_per_block=5),
            EngineGroupInfo(engine_group_id=1, layer_indices=(1,), tokens_per_block=5),
        ],
        lmcache_chunk_tokens=5,
    )
    key = IPCCacheServerKey("model", 1, 0, b"\x00" * 20, 0, 5, "request")
    result = context.submit_store("request", key, caches, [[1], [1]], None, 1)
    assert result.result() is True
    payload = client.commit_store.call_args.args[2]
    groups = decode_grouped_tensors(
        payload, expected=[(torch.Size([2, 1, 5, 24]), torch.float32, 1)] * 2
    )
    for (name, tensor), axes, (indices, chunks) in zip(
        caches.items(), [BKH, KBH], groups, strict=True
    ):
        expected = tensor.select(axes.index("BLOCK"), 1).squeeze(2)
        expected = expected.permute(0, 2, 1, 3).reshape(2, 1, 5, 24)
        assert indices == [0], name
        assert torch.equal(chunks[0], expected), name
    context.unregister()
