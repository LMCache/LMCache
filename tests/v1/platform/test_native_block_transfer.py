# SPDX-License-Identifier: Apache-2.0
"""``transfer_with_address_kernel`` lowers tensor operands to an address kernel.

CPU tests pass a recording kernel and check the operands each launch gets.
CUDA tests run the compiled ``lmcache.cuda_ops`` kernel end to end.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
from lmcache.v1.gpu_connector.utils import (
    make_page_buffer_shape_desc,
    normalize_kv_and_discover_format,
)
from lmcache.v1.platform import native_block_transfer
from lmcache.v1.platform.native_block_transfer import transfer_with_address_kernel
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

D2H = lmcache_native.TransferDirection.D2H
H2D = lmcache_native.TransferDirection.H2D
CPU = torch.device("cpu")
_BLOCK_SIZE = 4
_BLOCKS_PER_CHUNK = 2
_CHUNK_TOKENS = _BLOCK_SIZE * _BLOCKS_PER_CHUNK


@dataclass
class _Launch:
    """Operands one kernel launch received."""

    paged: torch.Tensor
    chunks: list[int]
    block_ids: torch.Tensor
    direction: lmcache_native.TransferDirection
    skip: int


class _RecordingKernel:
    """Stand-in for a compiled address-taking block-transfer kernel."""

    def __init__(self) -> None:
        self.launches: list[_Launch] = []

    def __call__(
        self,
        paged: torch.Tensor,
        chunks: list[int],
        block_ids: torch.Tensor,
        device: torch.device,
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip: int,
    ) -> None:
        self.launches.append(_Launch(paged, list(chunks), block_ids, direction, skip))


def _paged_kv(
    num_blocks: int, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> tuple[DiscoverableKVCache, lmcache_native.EngineKVFormat, PageBufferShapeDesc]:
    """Two NHD layers ``[2, num_blocks, 4, 2, 8]`` with their format and shape."""
    layers = [
        (
            torch.arange(2 * num_blocks * _BLOCK_SIZE * 16).reshape(
                2, num_blocks, _BLOCK_SIZE, 2, 8
            )
            + 1000 * i
        ).to(device=device, dtype=dtype)
        for i in range(2)
    ]
    fmt, normalized = normalize_kv_and_discover_format(
        layers, EngineType.VLLM, layout_hints={"kv_layout": "NHD"}
    )
    shape_desc = make_page_buffer_shape_desc(
        normalized,
        fmt,
        layer_idx=0,
        num_layers_in_group=2,
        num_blocks=num_blocks,
        block_size=_BLOCK_SIZE,
    )
    return normalized, fmt, shape_desc


def _layers(paged: DiscoverableKVCache) -> list[torch.Tensor]:
    """Return the per-layer tensors of an NHD ``paged`` KV list."""
    assert isinstance(paged, list)
    return [layer for layer in paged if isinstance(layer, torch.Tensor)]


def _expected_chunk(paged: DiscoverableKVCache, blocks: list[int]) -> torch.Tensor:
    """Canonical ``[2, L, T, H*D]`` chunk holding ``blocks`` of every layer."""
    return torch.stack(
        [
            layer.cpu()[:, blocks].reshape(2, len(blocks) * _BLOCK_SIZE, 16)
            for layer in _layers(paged)
        ],
        dim=1,
    )


def _host_chunk(address: int, pinned: bool = True) -> MagicMock:
    """Host chunk stand-in; real pinning needs an accelerator."""
    chunk = MagicMock(spec=torch.Tensor)
    chunk.is_pinned.return_value = pinned
    chunk.data_ptr.return_value = address
    return chunk


# -- CPU: recording kernel -------------------------------------------------


@pytest.mark.parametrize("direction", [D2H, H2D])
def test_tensor_operands_lower_to_kernel_launches(
    direction: lmcache_native.TransferDirection,
) -> None:
    """Five chunks become a 4-chunk and a 1-chunk launch with kernel operands."""
    kernel = _RecordingKernel()
    paged, fmt, desc = _paged_kv(num_blocks=10)
    addresses = [0x1000 * (i + 1) for i in range(5)]
    chunks = cast(list[torch.Tensor], [_host_chunk(a) for a in addresses])

    transfer_with_address_kernel(
        kernel,
        paged,
        chunks,
        list(range(10)),
        CPU,
        direction,
        desc,
        _CHUNK_TOKENS,
        fmt,
        1,
    )

    layer_addresses = [layer.data_ptr() for layer in _layers(paged)]
    assert [launch.chunks for launch in kernel.launches] == [
        addresses[:4],
        addresses[4:],
    ]
    assert [launch.block_ids.tolist() for launch in kernel.launches] == [
        list(range(8)),
        [8, 9],
    ]
    # The skip applies to the first launch only, as before the move.
    assert [launch.skip for launch in kernel.launches] == [1, 0]
    for launch in kernel.launches:
        assert launch.paged.dtype == torch.int64
        assert launch.paged.tolist() == layer_addresses
        assert launch.block_ids.dtype == torch.int64
        assert launch.direction == direction


def test_pinned_chunks_keep_the_async_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Already pinned chunks are passed as-is and nothing synchronizes."""
    kernel = _RecordingKernel()
    syncs: list[torch.device] = []
    monkeypatch.setattr(native_block_transfer, "synchronize_device", syncs.append)
    paged, fmt, desc = _paged_kv(num_blocks=2)

    transfer_with_address_kernel(
        kernel,
        paged,
        cast(list[torch.Tensor], [_host_chunk(0x3000)]),
        [0, 1],
        CPU,
        H2D,
        desc,
        _CHUNK_TOKENS,
        fmt,
        0,
    )

    assert [launch.chunks for launch in kernel.launches] == [[0x3000]]
    assert syncs == []


def test_unpinned_h2d_chunks_sync_after_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunks pinned into temporaries are synchronized before returning.

    The kernel reads the temporaries through raw addresses that torch's stream
    tracking cannot see, so returning without a sync would let the caching
    host allocator recycle them while the copy is still in flight.
    """
    kernel = _RecordingKernel()
    syncs: list[int] = []
    monkeypatch.setattr(
        native_block_transfer,
        "synchronize_device",
        lambda _device: syncs.append(len(kernel.launches)),
    )
    unpinned = _host_chunk(0, pinned=False)
    unpinned.pin_memory.return_value = _host_chunk(0x2000)
    paged, fmt, desc = _paged_kv(num_blocks=2)

    transfer_with_address_kernel(
        kernel,
        paged,
        cast(list[torch.Tensor], [unpinned]),
        [0, 1],
        CPU,
        H2D,
        desc,
        _CHUNK_TOKENS,
        fmt,
        0,
    )

    assert [launch.chunks for launch in kernel.launches] == [[0x2000]]
    assert syncs == [1], "expected exactly one sync, after the launch"


# -- CUDA: the compiled kernel ---------------------------------------------


@pytest.fixture
def cuda_kernel() -> object:
    """The compiled CUDA block kernel, skipping without CUDA or the extension."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    native = pytest.importorskip("lmcache.cuda_ops")
    return native.multi_layer_block_kv_transfer


@pytest.mark.cuda
def test_cuda_tensor_round_trip_across_launches(cuda_kernel: object) -> None:
    """Five chunks (two launches) gather and scatter back byte-exact."""
    kernel = cast(Callable[..., None], cuda_kernel)
    device = torch.device("cuda")
    paged, fmt, desc = _paged_kv(num_blocks=10, device="cuda", dtype=torch.float16)
    chunks = [
        torch.empty(2, 2, _CHUNK_TOKENS, 16, dtype=torch.float16, pin_memory=True)
        for _ in range(5)
    ]
    block_ids = list(range(10))

    transfer_with_address_kernel(
        kernel, paged, chunks, block_ids, device, D2H, desc, _CHUNK_TOKENS, fmt, 0
    )
    torch.cuda.synchronize()
    for i, chunk in enumerate(chunks):
        assert torch.equal(chunk, _expected_chunk(paged, [2 * i, 2 * i + 1]))

    restored, _, _ = _paged_kv(num_blocks=10, device="cuda", dtype=torch.float16)
    for layer in _layers(restored):
        layer.zero_()
    transfer_with_address_kernel(
        kernel, restored, chunks, block_ids, device, H2D, desc, _CHUNK_TOKENS, fmt, 0
    )
    torch.cuda.synchronize()
    for source, target in zip(_layers(paged), _layers(restored), strict=True):
        assert torch.equal(source, target)


@pytest.mark.cuda
def test_cuda_unpinned_output_is_staged_in_place(cuda_kernel: object) -> None:
    """An unpinned output is filled in place through pinned staging."""
    kernel = cast(Callable[..., None], cuda_kernel)
    paged, fmt, desc = _paged_kv(num_blocks=6, device="cuda", dtype=torch.float16)
    out = torch.zeros(2, 2, _CHUNK_TOKENS, 16, dtype=torch.float16).share_memory_()
    assert not out.is_pinned()

    transfer_with_address_kernel(
        kernel,
        paged,
        [out],
        [1, 3],
        torch.device("cuda"),
        D2H,
        desc,
        _CHUNK_TOKENS,
        fmt,
        0,
    )

    # Staging synchronizes before copying back, so no extra sync is needed.
    assert torch.equal(out, _expected_chunk(paged, [1, 3]))
