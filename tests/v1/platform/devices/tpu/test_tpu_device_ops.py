# SPDX-License-Identifier: Apache-2.0
"""Tests for the TorchTPU block transfer entry point.

Chunks must be byte-identical to the ones the shared torch path writes for
``NL_X_NB_BS_NHP_P_HS``: LMCache's single-plane ``[L, T, NHP * P * HS]``
layout. CPU tensors stand in for TPU ones; three blocks per chunk make every
full chunk pad its index to four and leave a partial trailing chunk.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.tpu.device_ops import TpuDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 8
BLOCK_SIZE = 4
NHP, P, HS = 3, 2, 8
BLOCKS_PER_CHUNK = 3
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.bfloat16
PACKED = EngineKVFormat.NL_X_NB_BS_NHP_P_HS
# Out of order and non-consecutive, as vLLM hands them out.
SCATTERED_BLOCKS = [5, 1, 6, 2, 0, 7, 3, 4]


def _paged_layers(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer KV as vllm-torchtpu allocates it: ``[NB, BS, NHP, P, HS]``."""
    generator = torch.Generator().manual_seed(11)
    shape = (NUM_BLOCKS, BLOCK_SIZE, NHP, P, HS * DTYPE.itemsize)
    return [
        (
            torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
            if fill_random
            else torch.zeros(shape, dtype=torch.uint8)
        ).view(DTYPE)
        for _ in range(NUM_LAYERS)
    ]


def _chunks() -> list[torch.Tensor]:
    """Single-plane chunks, as the MP transfer context allocates them."""
    return [
        torch.zeros((NUM_LAYERS, CHUNK_TOKENS, NHP * P * HS), dtype=DTYPE)
        for _ in range(-(-NUM_BLOCKS // BLOCKS_PER_CHUNK))
    ]


def _shape_desc() -> PageBufferShapeDesc:
    desc = PageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NHP * P // 2
    desc.hs = 2 * HS
    desc.element_size = DTYPE.itemsize
    desc.dtype = DTYPE
    return desc


def _transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    skip_prefix_n_blocks: int = 0,
    engine_kv_format: EngineKVFormat = PACKED,
    chunk_tokens: int = CHUNK_TOKENS,
) -> None:
    TpuDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        SCATTERED_BLOCKS,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        chunk_tokens,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


def _canonical(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    skip_prefix_n_blocks: int = 0,
) -> None:
    multi_layer_block_kv_transfer(
        layers,
        chunks,
        SCATTERED_BLOCKS,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        PACKED,
        skip_prefix_n_blocks,
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


# -- Parity with the shared torch path -------------------------------------


def test_store_matches_the_canonical_torch_path() -> None:
    layers = _paged_layers()
    ours, theirs = _chunks(), _chunks()
    _transfer(layers, ours, TransferDirection.D2H)
    _canonical(layers, theirs, TransferDirection.D2H)
    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(_bytes(got), _bytes(expected))


def test_retrieve_matches_the_canonical_torch_path() -> None:
    chunks = _chunks()
    _canonical(_paged_layers(), chunks, TransferDirection.D2H)
    ours, theirs = _paged_layers(False), _paged_layers(False)
    _transfer(ours, chunks, TransferDirection.H2D, skip_prefix_n_blocks=4)
    _canonical(theirs, chunks, TransferDirection.H2D, skip_prefix_n_blocks=4)
    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(_bytes(got), _bytes(expected))


# -- Round trip ------------------------------------------------------------


def test_round_trip_restores_the_paged_cache() -> None:
    src, dst, chunks = _paged_layers(), _paged_layers(False), _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(_bytes(got), _bytes(expected))


def test_prefix_skip_spans_chunk_boundaries() -> None:
    """A skip of four covers the whole first chunk and one block of the next."""
    src, dst, chunks = _paged_layers(), _paged_layers(False), _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=4)
    skipped, written = SCATTERED_BLOCKS[:4], SCATTERED_BLOCKS[4:]
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(_bytes(got[skipped])) == 0
        assert torch.equal(_bytes(got[written]), _bytes(expected[written]))


# -- Guards ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt",
    [fmt for fmt in EngineKVFormat.__members__.values() if fmt != PACKED],
    ids=lambda f: f.name,
)
def test_other_formats_are_refused(fmt: EngineKVFormat) -> None:
    with pytest.raises(ValueError, match="NL_X_NB_BS_NHP_P_HS"):
        _transfer(
            _paged_layers(), _chunks(), TransferDirection.D2H, engine_kv_format=fmt
        )


def test_non_5d_layers_are_refused() -> None:
    """The same bytes viewed as ``NL_X_NB_BS_NH_CS``'s 4-D tensor."""
    layers = [t.flatten(2, 3) for t in _paged_layers()]
    with pytest.raises(ValueError, match="5-D"):
        _transfer(layers, _chunks(), TransferDirection.D2H)


def test_pointer_operands_are_refused() -> None:
    with pytest.raises(ValueError, match="tensor operands"):
        TpuDeviceOps().multi_layer_block_kv_transfer(
            torch.tensor([0, 1], dtype=torch.int64),
            [0, 1],
            SCATTERED_BLOCKS,
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            CHUNK_TOKENS,
            PACKED,
            0,
        )


def test_chunk_size_must_be_a_block_multiple() -> None:
    with pytest.raises(ValueError, match="multiple of shape_desc.bs"):
        _transfer(
            _paged_layers(),
            _chunks(),
            TransferDirection.D2H,
            chunk_tokens=BLOCK_SIZE + 1,
        )


def test_unknown_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="direction"):
        _transfer(_paged_layers(), _chunks(), TransferDirection(7))
