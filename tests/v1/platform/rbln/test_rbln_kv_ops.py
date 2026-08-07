# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN head-major block transfer.

RBLN stores heads before block tokens, so upstream's token-major staging would
cost a head<->token permute on every store and restore. ``RblnDeviceOps``
overrides ``multi_layer_block_kv_transfer`` to fill the same buffer head-major
instead.

The load-bearing test is :func:`test_chunk_is_head_major_not_token_major`: a
round trip alone passes under either layout, because the same code writes and
reads the chunk. Only a positive check against the head-major expectation --
with the token-major reading asserted *false* -- proves the override is
actually in effect.

No RBLN hardware is needed: the kernels are torch-only, so CPU tensors are
enough to pin the layout contract. That is also the limit of what these cover
-- the device dispatch and the transfer cost are not exercised here.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
)
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.rbln.kv_ops import head_major_view

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.float32


def _paged_layers(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer HND KV in the native 6-D layout the detector reports."""
    torch.manual_seed(11)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _chunks() -> list[torch.Tensor]:
    """Staging chunks sized token-major, as upstream allocates them."""
    return [
        torch.zeros((2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


def _shape_desc() -> PageBufferShapeDesc:
    """Descriptor matching the paged layers above."""
    desc = PageBufferShapeDesc()
    desc.kv_size = 2
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NUM_HEADS
    desc.hs = HEAD_SIZE
    desc.element_size = DTYPE.itemsize
    return desc


def _transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    skip_prefix_n_blocks: int = 0,
    engine_kv_format: EngineKVFormat = EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
) -> None:
    """Run the RBLN block transfer over every block."""
    RblnDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_chunk_is_head_major_not_token_major() -> None:
    """The whole point: heads lead tokens in the staged chunk."""
    layers = _paged_layers()
    chunks = _chunks()
    _transfer(layers, chunks, TransferDirection.D2H)

    head_major = chunks[0].view(2, NUM_LAYERS, NUM_HEADS, CHUNK_TOKENS, HEAD_SIZE)
    assert all(
        torch.equal(
            head_major[kv, li, hi, ti],
            layers[li][kv, ti // BLOCK_SIZE, hi, 0, ti % BLOCK_SIZE],
        )
        for kv in (0, 1)
        for li in range(NUM_LAYERS)
        for hi in range(NUM_HEADS)
        for ti in range(CHUNK_TOKENS)
    )

    # Control: reading the same bytes token-major must NOT line up, otherwise
    # the override silently did nothing.
    assert not all(
        torch.equal(
            chunks[0][kv, li, ti],
            layers[li][kv, ti // BLOCK_SIZE, :, 0, ti % BLOCK_SIZE, :].reshape(
                NUM_HEADS * HEAD_SIZE
            ),
        )
        for kv in (0, 1)
        for li in range(NUM_LAYERS)
        for ti in range(CHUNK_TOKENS)
    )


def test_round_trip_restores_the_paged_cache() -> None:
    """Gather then scatter reproduces the source exactly."""
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_prefix_skip_leaves_leading_blocks_untouched() -> None:
    """A whole-block prefix skip is neither read nor written."""
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=1)

    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[:, 0]) == 0
        assert torch.equal(got[:, 1:], expected[:, 1:])


def test_head_major_view_requires_contiguity() -> None:
    """A non-contiguous buffer would silently address the wrong bytes."""
    buf = torch.zeros(2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE)
    with pytest.raises(ValueError, match="contiguous"):
        head_major_view(
            buf.transpose(2, 3), NUM_LAYERS, NUM_HEADS, CHUNK_TOKENS, HEAD_SIZE
        )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_unsupported_format_is_refused() -> None:
    """Only the HND layout the detector produces is validated."""
    with pytest.raises(ValueError, match="NL_X_TWO_NB_NH_ONE_BS_HS"):
        _transfer(
            _paged_layers(),
            _chunks(),
            TransferDirection.D2H,
            engine_kv_format=EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )


def test_pointer_operands_are_refused() -> None:
    """RBLN has no compiled block-transfer extension, so pointers can't occur."""
    with pytest.raises(ValueError, match="tensor operands"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            torch.tensor([0, 1], dtype=torch.int64),
            [0, 1],
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            CHUNK_TOKENS,
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
            0,
        )


def test_chunk_size_must_be_a_block_multiple() -> None:
    """A ragged chunk size would mis-slice the block list."""
    with pytest.raises(ValueError, match="multiple of shape_desc.bs"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            _paged_layers(),
            _chunks(),
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            BLOCK_SIZE + 1,
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
            0,
        )
