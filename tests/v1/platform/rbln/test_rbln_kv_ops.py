# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN block transfer.

``RblnDeviceOps`` overrides ``multi_layer_block_kv_transfer`` for the torch op
sequence RBLN is tuned for, **not** for the chunk layout: chunks stay in
LMCache's canonical token-major ``[2, L, T, H*D]``.

The load-bearing test is :func:`test_chunk_matches_the_canonical_torch_path`:
a round trip alone passes under any self-consistent layout, because the same
code writes and reads the chunk. Only byte equality against the shared torch
path proves an RBLN chunk is interchangeable with one written by another
device -- the property cross-device sharing and PD disaggregation rely on.

No RBLN hardware is needed: the kernels are torch-only, so CPU tensors are
enough to pin the layout contract. That is also the limit of what these cover
-- the device dispatch and the transfer cost are not exercised here.

The MLA tests pin the same properties for the ``NL_X_NB_BS_HS`` layout
(``[NB, BS, HS]``, single latent plane), which ``RblnDeviceOps`` validates and
routes to the shared torch path.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.rbln.kv_layout import squeeze_singleton_axis
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

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


def test_chunk_is_canonical_token_major() -> None:
    """Each chunk row is one token's heads laid end to end."""
    layers = _paged_layers()
    chunks = _chunks()
    _transfer(layers, chunks, TransferDirection.D2H)

    assert all(
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


def test_chunk_matches_the_canonical_torch_path() -> None:
    """RBLN chunks are byte-identical to the shared torch HND path's.

    The RBLN layout is the 5-D HND format plus a singleton axis, so the shared
    path fed the squeezed tensors must produce exactly the same bytes. This is
    what makes a chunk written on RBLN readable by another device.
    """
    layers = _paged_layers()
    ours = _chunks()
    theirs = _chunks()

    _transfer(layers, ours, TransferDirection.D2H)
    multi_layer_block_kv_transfer(
        squeeze_singleton_axis(layers),
        theirs,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        TransferDirection.D2H,
        _shape_desc(),
        CHUNK_TOKENS,
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        0,
    )

    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(got, expected)


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


def test_trailing_partial_chunk_is_handled() -> None:
    """A chunk holding fewer blocks than it is sized for round-trips."""
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    partial = NUM_BLOCKS - 1

    for direction, layers in (
        (TransferDirection.D2H, src),
        (TransferDirection.H2D, dst),
    ):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            layers,
            chunks,
            list(range(partial)),
            torch.device("cpu"),
            direction,
            _shape_desc(),
            CHUNK_TOKENS,
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
            0,
        )

    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got[:, :partial], expected[:, :partial])
        assert torch.count_nonzero(got[:, partial:]) == 0


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


# ---------------------------------------------------------------------------
# MLA (NL_X_NB_BS_HS): validated in RblnDeviceOps, moved by the shared path
# ---------------------------------------------------------------------------

MLA_HEAD_SIZE = 16


def _mla_layers(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer MLA KV in the native 3-D layout the detector reports."""
    torch.manual_seed(23)
    shape = (NUM_BLOCKS, BLOCK_SIZE, MLA_HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _mla_chunks() -> list[torch.Tensor]:
    """Single-plane staging chunks, as upstream allocates them for MLA."""
    return [
        torch.zeros((NUM_LAYERS, CHUNK_TOKENS, MLA_HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


def _mla_shape_desc() -> PageBufferShapeDesc:
    """Descriptor matching the MLA layers above."""
    desc = PageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = 1
    desc.hs = MLA_HEAD_SIZE
    desc.element_size = DTYPE.itemsize
    return desc


def _mla_transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    skip_prefix_n_blocks: int = 0,
    block_ids: "list[int] | None" = None,
) -> None:
    """Run the RBLN block transfer over MLA layers."""
    RblnDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)) if block_ids is None else block_ids,
        torch.device("cpu"),
        direction,
        _mla_shape_desc(),
        CHUNK_TOKENS,
        EngineKVFormat.NL_X_NB_BS_HS,
        skip_prefix_n_blocks,
    )


def test_mla_chunk_is_token_major() -> None:
    """Each MLA chunk row is one token's latent vector."""
    layers = _mla_layers()
    chunks = _mla_chunks()
    _mla_transfer(layers, chunks, TransferDirection.D2H)

    assert all(
        torch.equal(
            chunks[0][li, ti],
            layers[li][ti // BLOCK_SIZE, ti % BLOCK_SIZE],
        )
        for li in range(NUM_LAYERS)
        for ti in range(CHUNK_TOKENS)
    )


def test_mla_chunk_matches_the_canonical_torch_path() -> None:
    """RBLN MLA chunks are byte-identical to the shared torch path's.

    Today this holds by construction (the MLA branch delegates), but the test
    is what keeps it true if the branch ever grows an RBLN-specific sequence.
    """
    layers = _mla_layers()
    ours = _mla_chunks()
    theirs = _mla_chunks()

    _mla_transfer(layers, ours, TransferDirection.D2H)
    multi_layer_block_kv_transfer(
        layers,
        theirs,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        TransferDirection.D2H,
        _mla_shape_desc(),
        CHUNK_TOKENS,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )

    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(got, expected)


def test_mla_round_trip_restores_the_paged_cache() -> None:
    """MLA gather then scatter reproduces the source exactly."""
    src = _mla_layers()
    dst = _mla_layers(fill_random=False)
    chunks = _mla_chunks()
    _mla_transfer(src, chunks, TransferDirection.D2H)
    _mla_transfer(dst, chunks, TransferDirection.H2D)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_mla_prefix_skip_leaves_leading_blocks_untouched() -> None:
    """A whole-block prefix skip is neither read nor written for MLA."""
    src = _mla_layers()
    dst = _mla_layers(fill_random=False)
    chunks = _mla_chunks()
    _mla_transfer(src, chunks, TransferDirection.D2H)
    _mla_transfer(dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=1)

    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[0]) == 0
        assert torch.equal(got[1:], expected[1:])


def test_mla_trailing_partial_chunk_is_handled() -> None:
    """An MLA chunk holding fewer blocks than it is sized for round-trips."""
    src = _mla_layers()
    dst = _mla_layers(fill_random=False)
    chunks = _mla_chunks()
    partial = NUM_BLOCKS - 1

    for direction, layers in (
        (TransferDirection.D2H, src),
        (TransferDirection.H2D, dst),
    ):
        _mla_transfer(layers, chunks, direction, block_ids=list(range(partial)))

    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got[:partial], expected[:partial])
        assert torch.count_nonzero(got[partial:]) == 0


def test_mla_rejects_wrong_rank_layers() -> None:
    """The 6-D attention layout under the MLA format fails loudly."""
    with pytest.raises(ValueError, match=r"\[NB, BS, HS\]"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            _paged_layers(),
            _mla_chunks(),
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _mla_shape_desc(),
            CHUNK_TOKENS,
            EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )
