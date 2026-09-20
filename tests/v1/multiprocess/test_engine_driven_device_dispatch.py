# SPDX-License-Identifier: Apache-2.0
"""Transfer dispatch follows paged KV devices rather than the global backend."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
import lmcache


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
    if (
        lmcache.device_ops.multi_layer_block_kv_transfer
        is not native.multi_layer_block_kv_transfer
    ):
        pytest.skip("requires the native CUDA transfer backend")
    devices = ("cuda", "cpu", "cuda") if gpu_first else ("cpu", "cuda", "cpu")
    for device in devices:
        _assert_roundtrip(device, torch.float16, output_kind)
