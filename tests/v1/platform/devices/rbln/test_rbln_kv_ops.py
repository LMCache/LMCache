# SPDX-License-Identifier: Apache-2.0
"""RBLN block transfer checks that need no NPU.

``RblnDeviceOps`` overrides ``multi_layer_block_kv_transfer`` for the torch op
sequence RBLN is tuned for, **not** for the chunk layout: chunks stay in
LMCache's canonical token-major ``[2, L, T, H*D]``.

Both layouts move through device staging (and, for HND, device programs),
so their data paths are tested on the NPU (``test_rbln_device_transfer.py``).
Here, with no ``torch.rbln`` at all, are what must hold before any device
work: the layout and operand checks that refuse a transfer or a warm-up.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rbln import warm_up_kv_transfer
from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.devices.rbln.kv_layout import squeeze_singleton_axis
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 64
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def no_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ``torch.rbln``: none of these paths may touch a device."""
    monkeypatch.setattr(torch, "rbln", None, raising=False)


def _paged_layers(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer HND KV in the native 6-D layout the detector reports."""
    torch.manual_seed(11)
    shape = (NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
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
    engine_kv_format: EngineKVFormat = EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS,
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
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine_kv_format",
    [
        EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # NHD
        EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,  # HND, K/V packed in content
        EngineKVFormat.NB_NL_TWO_NH_BS_HS,  # cross-layer HND
        EngineKVFormat.NL_X_NBBS_ONE_HS,  # MLA, blocks fused with tokens
        EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS,  # MLA plane tuples
    ],
    ids=lambda fmt: fmt.name,
)
def test_unsupported_layouts_are_refused(engine_kv_format: EngineKVFormat) -> None:
    """A layout the op sequences cannot address fails on its layout facts."""
    with pytest.raises(ValueError, match="supports only per-layer"):
        _transfer(
            _paged_layers(),
            _chunks(),
            TransferDirection.D2H,
            engine_kv_format=engine_kv_format,
        )


def test_hnd_without_the_singleton_axis_is_refused() -> None:
    """A 5-D HND cache passes the layout check but fails the strict squeeze."""
    with pytest.raises(ValueError, match=r"\[NB, 2, NH, 1, BS, HS\]"):
        _transfer(
            squeeze_singleton_axis(_paged_layers()),
            _chunks(),
            TransferDirection.D2H,
            engine_kv_format=EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
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
            EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS,
            0,
        )


def test_warm_up_refuses_a_layout_rbln_cannot_transfer() -> None:
    """Caches without RBLN's singleton axis are refused before the device.

    Detection reads them as plain HND or as NHD depending on the host, so
    the refusal is the squeeze's or the layout check's; either is the error
    a transfer of them raises.
    """
    caches = {
        f"layers.{i}": layer
        for i, layer in enumerate(squeeze_singleton_axis(_paged_layers()))
    }
    with pytest.raises(ValueError, match="^RBLN "):
        warm_up_kv_transfer(caches)


@pytest.mark.parametrize("chunk_tokens", [0, -BLOCK_SIZE, BLOCK_SIZE + 1])
def test_chunk_size_must_be_a_block_multiple(chunk_tokens: int) -> None:
    """A ragged chunk size would mis-slice the block list."""
    with pytest.raises(ValueError, match="multiple of shape_desc.bs"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            _paged_layers(),
            _chunks(),
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            chunk_tokens,
            EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS,
            0,
        )


# ---------------------------------------------------------------------------
# K/V-first HND (NL_X_TWO_NB_NH_ONE_BS_HS): vLLM-RBLN's rbln_triton_ops
# kernels allocate [2, NB, NH, 1, BS, HS].
# ---------------------------------------------------------------------------


def test_kv_first_without_the_singleton_axis_is_refused() -> None:
    """A 5-D two-major HND cache fails the strict K/V-first squeeze."""
    squeezed = [t.transpose(0, 1).squeeze(3).contiguous() for t in _paged_layers()]
    with pytest.raises(ValueError, match=r"\[2, NB, NH, 1, BS, HS\]"):
        _transfer(
            squeezed,
            _chunks(),
            TransferDirection.D2H,
            engine_kv_format=EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        )


# ---------------------------------------------------------------------------
# MLA (NL_X_NB_BS_HS): validated in RblnDeviceOps, moved by the shared path
# ---------------------------------------------------------------------------

MLA_HEAD_SIZE = 16
MLA_DTYPE = torch.float32


def _mla_layers(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer MLA KV in the native 3-D layout the detector reports."""
    torch.manual_seed(23)
    shape = (NUM_BLOCKS, BLOCK_SIZE, MLA_HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=MLA_DTYPE) for _ in range(NUM_LAYERS)]


def _mla_chunks() -> list[torch.Tensor]:
    """Single-plane staging chunks, as upstream allocates them for MLA."""
    return [
        torch.zeros((NUM_LAYERS, CHUNK_TOKENS, MLA_HEAD_SIZE), dtype=MLA_DTYPE)
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
    desc.element_size = MLA_DTYPE.itemsize
    return desc


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


def test_mla_blocked_scale_layout_is_refused() -> None:
    """An MLA-shaped cache whose blocks store values before scales is refused.

    ``NL_X_NB_BSV_BSS`` shares the MLA ``[NB, BS, HS]`` shape, so a whole-block
    copy would silently write blocked bytes into a token-major chunk.
    """
    with pytest.raises(ValueError, match="supports only per-layer"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            _mla_layers(),
            _mla_chunks(),
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _mla_shape_desc(),
            CHUNK_TOKENS,
            EngineKVFormat.NL_X_NB_BSV_BSS,
            0,
        )
