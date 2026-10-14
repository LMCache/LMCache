# SPDX-License-Identifier: Apache-2.0
"""Transfer dispatch follows paged KV devices rather than the global backend."""

# Standard
from dataclasses import dataclass
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
from lmcache.v1.multiprocess.transfer_context import base, worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache
import lmcache.lmcache_native as lmcache_native

_NHD: LayoutHints = {"kv_layout": "NHD"}
_BLOCK_SIZE = 4


@dataclass
class _BlockTransferCall:
    """Operands one ``multi_layer_block_kv_transfer`` call received."""

    paged: DiscoverableKVCache
    chunks: list[int] | list[torch.Tensor]
    block_ids: torch.Tensor | list[int]
    direction: lmcache_native.TransferDirection
    skip_prefix_n_blocks: int


class _RecordingOps(DeviceOps):
    """Ops that record block transfers and output allocations."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[_BlockTransferCall] = []
        self.allocations: list[tuple[tuple[int, ...], torch.dtype]] = []

    def allocate_block_transfer_output(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        self.allocations.append((shape, dtype))
        return torch.empty(shape, dtype=dtype)

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: DiscoverableKVCache,
        lmcache_objects_ptrs: list[int] | list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        self.calls.append(
            _BlockTransferCall(
                paged=paged_buffer_ptrs_tensor,
                chunks=lmcache_objects_ptrs,
                block_ids=block_ids,
                direction=direction,
                skip_prefix_n_blocks=skip_prefix_n_blocks,
            )
        )


def _patch_resolver(monkeypatch: pytest.MonkeyPatch, ops: DeviceOps) -> list[str]:
    """Route every ops lookup in gather/scatter to ``ops``.

    Returns:
        The device types gather/scatter asked the resolver for, in order.
    """
    requested: list[str] = []

    def _resolve(device_type: str) -> DeviceOps:
        requested.append(device_type)
        return ops

    monkeypatch.setattr(base, "resolve_device_ops", _resolve)
    return requested


def _make_kv(num_blocks: int) -> dict[str, torch.Tensor]:
    """Two NHD layers of ``[2, num_blocks, block_size, heads, head_size]``."""
    return {
        f"layer.{i}": torch.zeros(2, num_blocks, _BLOCK_SIZE, 2, 8) for i in range(2)
    }


def _assert_roundtrip(
    device: str, dtype: torch.dtype, output_kind: str = "allocated"
) -> None:
    """Check packing, scatter placement, and caller-owned output preservation."""
    source = {
        f"layer.{i}": (
            torch.arange(2 * 6 * 4 * 2 * 8).reshape(2, 6, 4, 2, 8) + i * 1024
        ).to(device=device, dtype=dtype)
        for i in range(2)
    }
    expected = torch.stack(
        [tensor[:, [1, 3]].reshape(2, 8, 16).cpu() for tensor in source.values()],
        dim=1,
    )
    out = None
    if output_kind != "allocated":
        buffer = torch.empty_like(expected)
        if output_kind == "shared":
            buffer.share_memory_()
        out = [buffer]

    chunks = gather_paged_kv_to_cpu(
        source, [1, 3], 2, layout_hints={"kv_layout": "NHD"}, out=out
    )
    if device == "cuda":
        torch.cuda.synchronize()
    assert len(chunks) == 1
    assert torch.equal(chunks[0], expected)
    if out is not None:
        assert chunks[0] is out[0]

    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(
        destination, [2, 4], chunks, 2, layout_hints={"kv_layout": "NHD"}
    )
    if device == "cuda":
        torch.cuda.synchronize()
    for name, tensor in destination.items():
        assert torch.equal(tensor[:, [2, 4]], source[name][:, [1, 3]])
        assert torch.count_nonzero(tensor[:, [0, 1, 3, 5]]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_kind", ["allocated", "provided", "shared"])
def test_cpu_kv_ignores_process_global_ops(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, output_kind: str
) -> None:
    """CPU KV must round-trip even when the global transfer op rejects it.

    Args:
        monkeypatch: Pytest fixture for replacing the unrelated global backend.
        dtype: KV element type to preserve through the CPU fallback.
        output_kind: Allocate chunks or write into caller-owned ordinary/SHM data.
    """
    global_ops = MagicMock()
    global_ops.multi_layer_block_kv_transfer.side_effect = AssertionError(
        "CPU KV must not dispatch through process-global ops"
    )
    monkeypatch.setattr(lmcache, "device_ops", global_ops)
    _assert_roundtrip("cpu", dtype, output_kind)
    global_ops.multi_layer_block_kv_transfer.assert_not_called()
    assert lmcache.device_ops is global_ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("gpu_first", [False, True])
@pytest.mark.parametrize("output_kind", ["allocated", "shared"])
def test_cpu_and_cuda_kv_can_share_one_process(
    gpu_first: bool, output_kind: str
) -> None:
    """CPU/CUDA alternation must preserve both backends' argument conventions.

    Args:
        gpu_first: Exercise native-first as well as CPU-first capability caching.
        output_kind: Allocate chunks or use unpinned shared-memory output buffers.
    """
    native = pytest.importorskip("lmcache.cuda_ops")
    if not hasattr(native, "multi_layer_block_kv_transfer"):
        pytest.skip("requires the native CUDA block-transfer kernel")
    devices = ("cuda", "cpu", "cuda") if gpu_first else ("cpu", "cuda", "cpu")
    for device in devices:
        _assert_roundtrip(device, torch.float16, output_kind)


def test_gather_scatter_pass_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ops resolved for the KV device receive tensors, never addresses."""
    ops = _RecordingOps()
    requested = _patch_resolver(monkeypatch, ops)
    kv = _make_kv(num_blocks=6)
    out = [torch.empty(2, 2, 2 * _BLOCK_SIZE, 16) for _ in range(2)]

    assert gather_paged_kv_to_cpu(kv, [1, 3, 0, 5], 2, _NHD, out=out) is out
    scatter_cpu_to_paged_kv(
        kv, [2, 4, 0, 1], out, 2, skip_first_n_tokens=_BLOCK_SIZE, layout_hints=_NHD
    )

    assert requested == ["cpu", "cpu"]
    gather_call, scatter_call = ops.calls
    layer_ptrs = [tensor.data_ptr() for tensor in kv.values()]
    for call in (gather_call, scatter_call):
        assert isinstance(call.paged, list)
        layers = [layer for layer in call.paged if isinstance(layer, torch.Tensor)]
        assert [layer.data_ptr() for layer in layers] == layer_ptrs
        assert call.chunks is out
    assert gather_call.block_ids == [1, 3, 0, 5]
    assert gather_call.direction == lmcache_native.TransferDirection.D2H
    assert gather_call.skip_prefix_n_blocks == 0
    assert scatter_call.block_ids == [2, 4, 0, 1]
    assert scatter_call.direction == lmcache_native.TransferDirection.H2D
    assert scatter_call.skip_prefix_n_blocks == 1
    assert ops.allocations == []


def test_gather_without_out_uses_backend_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gather asks the backend for its output buffers instead of choosing them."""
    ops = _RecordingOps()
    _patch_resolver(monkeypatch, ops)

    chunks = gather_paged_kv_to_cpu(_make_kv(num_blocks=6), [1, 3, 0, 5], 2, _NHD)

    assert ops.allocations == [((2, 2, 2 * _BLOCK_SIZE, 16), torch.float32)] * 2
    (call,) = ops.calls
    assert call.chunks == chunks


def test_cpu_context_never_synchronizes_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU STORE/RETRIEVE preserves KV without touching a global accelerator."""
    global_sync = MagicMock(side_effect=AssertionError("unexpected accelerator sync"))
    monkeypatch.setattr(worker_transfer.torch_dev, "synchronize", global_sync)
    monkeypatch.setattr(torch.cuda, "synchronize", global_sync)
    transport = MagicMock()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **kw: transport
    )
    kv = {"layer_0": torch.arange(512).reshape(2, 4, 4, 2, 8).float()}
    expected = kv["layer_0"].clone()
    ctx = worker_transfer.EngineDrivenTransferContext(1, MagicMock())
    ctx.register(kv, "test", 1, 4, 1.0, layout_hints={"kv_layout": "NHD"})
    transport.prepare_store.return_value = None
    transport.commit_store.return_value = True
    try:
        assert ctx.submit_store("req", "key", kv, [[0, 1, 2, 3]], None, 4).result()
        chunks = transport.commit_store.call_args.args[2]
        transport.prepare_retrieve.return_value = chunks
        kv["layer_0"].zero_()
        assert ctx.submit_retrieve("req", "key", kv, [[0, 1, 2, 3]], None, 4).result()
        assert torch.equal(kv["layer_0"], expected)
        global_sync.assert_not_called()
    finally:
        ctx.close()
