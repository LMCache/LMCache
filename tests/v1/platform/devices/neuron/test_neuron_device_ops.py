# SPDX-License-Identifier: Apache-2.0
"""Tests for the Neuron block transfer.

Chunks must be byte-identical to the shared torch path's, so they stay
interchangeable with chunks written by any other device. CPU tensors stand in
for Neuron ones; the device-only constraints are covered by the ``neuron``
marked test.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.neuron.device_ops import NeuronDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
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
HND = EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
# Out of order and non-consecutive, as vLLM hands them out.
SCATTERED_BLOCKS = [5, 1, 6, 2, 0, 7, 3, 4]


def _paged_layers(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer KV as vllm-neuron allocates it: ``[2, NB, NH, BS, HS]``."""
    torch.manual_seed(11)
    shape = (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _chunks() -> list[torch.Tensor]:
    """Token-major chunks, as the MP transfer context allocates them."""
    return [
        torch.zeros((2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


def _shape_desc() -> PageBufferShapeDesc:
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
    block_ids: list[int] = SCATTERED_BLOCKS,
    skip_prefix_n_blocks: int = 0,
    engine_kv_format: EngineKVFormat = HND,
) -> None:
    NeuronDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


def _canonical(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    block_ids: list[int] = SCATTERED_BLOCKS,
    skip_prefix_n_blocks: int = 0,
) -> None:
    multi_layer_block_kv_transfer(
        layers,
        chunks,
        block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        HND,
        skip_prefix_n_blocks,
    )


# -- Parity with the shared torch path -------------------------------------


def test_store_matches_the_canonical_torch_path() -> None:
    layers = _paged_layers()
    ours, theirs = _chunks(), _chunks()
    _transfer(layers, ours, TransferDirection.D2H)
    _canonical(layers, theirs, TransferDirection.D2H)
    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(got, expected)


def test_retrieve_matches_the_canonical_torch_path() -> None:
    chunks = _chunks()
    _canonical(_paged_layers(), chunks, TransferDirection.D2H)
    ours, theirs = _paged_layers(False), _paged_layers(False)
    _transfer(ours, chunks, TransferDirection.H2D, skip_prefix_n_blocks=3)
    _canonical(theirs, chunks, TransferDirection.H2D, skip_prefix_n_blocks=3)
    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(got, expected)


# -- Round trip ------------------------------------------------------------


def test_round_trip_restores_the_paged_cache() -> None:
    src, dst, chunks = _paged_layers(), _paged_layers(False), _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_prefix_skip_leaves_leading_blocks_untouched() -> None:
    src, dst, chunks = _paged_layers(), _paged_layers(False), _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=3)
    skipped, written = SCATTERED_BLOCKS[:3], SCATTERED_BLOCKS[3:]
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[:, skipped]) == 0
        assert torch.equal(got[:, written], expected[:, written])


def test_trailing_partial_chunk_is_handled() -> None:
    src, dst, chunks = _paged_layers(), _paged_layers(False), _chunks()
    blocks = SCATTERED_BLOCKS[:-1]
    _transfer(src, chunks, TransferDirection.D2H, block_ids=blocks)
    _transfer(dst, chunks, TransferDirection.H2D, block_ids=blocks)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got[:, blocks], expected[:, blocks])
        assert torch.count_nonzero(got[:, SCATTERED_BLOCKS[-1]]) == 0


# -- Neuron constraints ----------------------------------------------------


def test_never_indexes_the_paged_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """``index_select``/``index_copy_`` cost time proportional to the whole
    cache on Neuron, so the transfer must not use them."""

    def _banned(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("indexed the paged cache")

    for name in ("index_select", "index_copy_"):
        monkeypatch.setattr(torch.Tensor, name, _banned)
    monkeypatch.setattr(torch, "index_select", _banned)

    src, dst, chunks = _paged_layers(), _paged_layers(False), _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D)


# -- Guards ----------------------------------------------------------------


def test_unsupported_format_is_refused() -> None:
    with pytest.raises(ValueError, match="NL_X_TWO_NB_NH_BS_HS"):
        _transfer(
            _paged_layers(),
            _chunks(),
            TransferDirection.D2H,
            engine_kv_format=EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )


def test_pointer_operands_are_refused() -> None:
    with pytest.raises(ValueError, match="tensor operands"):
        NeuronDeviceOps().multi_layer_block_kv_transfer(
            torch.tensor([0, 1], dtype=torch.int64),
            [0, 1],
            SCATTERED_BLOCKS,
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            CHUNK_TOKENS,
            HND,
            0,
        )


def test_chunk_size_must_be_a_block_multiple() -> None:
    with pytest.raises(ValueError, match="multiple of shape_desc.bs"):
        NeuronDeviceOps().multi_layer_block_kv_transfer(
            _paged_layers(),
            _chunks(),
            SCATTERED_BLOCKS,
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            BLOCK_SIZE + 1,
            HND,
            0,
        )


# -- Hardware --------------------------------------------------------------


@pytest.mark.neuron
def test_round_trip_on_device() -> None:
    """Round trip through real Neuron memory, against the host original."""
    torch_neuron = getattr(torch, "neuron", None)
    if torch_neuron is None or not torch_neuron.is_available():
        pytest.skip("no Neuron device")
    host = _paged_layers()
    device = [t.to("neuron:0") for t in host]
    cleared = [torch.zeros_like(t) for t in device]
    chunks = _chunks()
    _transfer(device, chunks, TransferDirection.D2H)
    _transfer(cleared, chunks, TransferDirection.H2D)
    for got, expected in zip(cleared, host, strict=True):
        assert torch.equal(got.cpu(), expected)
