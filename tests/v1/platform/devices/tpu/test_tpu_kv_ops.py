# SPDX-License-Identifier: Apache-2.0
"""CPU round trips for the TorchTPU block transfer kernels.

CPU tensors stand in for TPU ones, so these pin which bytes move -- including
the power-of-two index padding -- but not the device dispatch or its compile
behaviour, which the ``tpu`` marked test in ``test_tpu_backend.py`` covers.
Caches hold random bytes and are compared as bytes, so every bit pattern of
the dtype (fp8 NaN encodings included) must survive.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.tpu.kv_ops import (
    gather_blocks_to_chunk,
    scatter_chunk_to_blocks,
)

NUM_LAYERS = 2
NUM_BLOCKS = 16
BLOCK_SIZE = 4
FP8 = getattr(torch, "float8_e4m3fn", torch.uint8)
# (dtype, NHP, P, HS) as vllm-torchtpu allocates them: bf16 packs two K/V rows,
# fp8 four (here one KV head plus a padding head).
LAYOUTS = [
    pytest.param((torch.bfloat16, 3, 2, 128), id="bf16"),
    pytest.param((FP8, 1, 4, 128), id="fp8"),
]
# Out of order and non-consecutive, as vLLM hands them out.
SCATTERED_BLOCKS = [9, 2, 14, 5, 0, 11, 7]


def _random_layers(dtype: torch.dtype, nhp: int, p: int, hs: int) -> list[torch.Tensor]:
    """Per-layer ``[NB, BS, NHP, P, HS]`` caches filled with random bytes."""
    generator = torch.Generator().manual_seed(7)
    shape = (NUM_BLOCKS, BLOCK_SIZE, nhp, p, hs * dtype.itemsize)
    return [
        torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator).view(dtype)
        for _ in range(NUM_LAYERS)
    ]


def _chunk(num_blocks: int, layer: torch.Tensor) -> torch.Tensor:
    """A zeroed host chunk ``[L, T, NHP * P * HS]`` for ``num_blocks`` blocks."""
    _nb, _bs, nhp, p, hs = layer.shape
    return torch.zeros(
        (NUM_LAYERS, num_blocks * BLOCK_SIZE, nhp * p * hs), dtype=layer.dtype
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("num_blocks", [1, 3, 5, 7])
def test_round_trip_restores_the_blocks(
    layout: tuple[torch.dtype, int, int, int], num_blocks: int
) -> None:
    src = _random_layers(*layout)
    dst = [torch.zeros_like(t) for t in src]
    blocks = SCATTERED_BLOCKS[:num_blocks]
    chunk = _chunk(num_blocks, src[0])

    gather_blocks_to_chunk(src, blocks, chunk)
    scatter_chunk_to_blocks(dst, blocks, chunk)

    untouched = [b for b in range(NUM_BLOCKS) if b not in blocks]
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(_bytes(got[blocks]), _bytes(expected[blocks]))
        assert torch.count_nonzero(_bytes(got[untouched])) == 0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_chunk_rows_are_token_major(
    layout: tuple[torch.dtype, int, int, int],
) -> None:
    """Each chunk row is one token's packed K/V rows, in block-id order."""
    src = _random_layers(*layout)
    blocks = SCATTERED_BLOCKS[:3]
    chunk = _chunk(len(blocks), src[0])

    gather_blocks_to_chunk(src, blocks, chunk)

    for layer_idx, layer in enumerate(src):
        for token in range(len(blocks) * BLOCK_SIZE):
            expected = layer[blocks[token // BLOCK_SIZE], token % BLOCK_SIZE]
            assert torch.equal(
                _bytes(chunk[layer_idx, token]), _bytes(expected.reshape(-1))
            )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_gather_writes_only_the_leading_tokens(
    layout: tuple[torch.dtype, int, int, int],
) -> None:
    """Index padding never reaches the chunk: three blocks pad to four, and
    the fourth block's tokens stay as they were."""
    src = _random_layers(*layout)
    chunk = _chunk(4, src[0])

    gather_blocks_to_chunk(src, SCATTERED_BLOCKS[:3], chunk)

    assert torch.count_nonzero(_bytes(chunk[:, 3 * BLOCK_SIZE :])) == 0


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("skip", [2, 5])
def test_prefix_skip_leaves_leading_blocks_untouched(
    layout: tuple[torch.dtype, int, int, int], skip: int
) -> None:
    src = _random_layers(*layout)
    dst = [torch.zeros_like(t) for t in src]
    blocks = SCATTERED_BLOCKS
    chunk = _chunk(len(blocks), src[0])

    gather_blocks_to_chunk(src, blocks, chunk)
    scatter_chunk_to_blocks(dst, blocks, chunk, skip_prefix_n_blocks=skip)

    skipped, written = blocks[:skip], blocks[skip:]
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(_bytes(got[skipped])) == 0
        assert torch.equal(_bytes(got[written]), _bytes(expected[written]))


def test_skipping_every_block_writes_nothing() -> None:
    src = _random_layers(torch.bfloat16, 3, 2, 128)
    dst = [torch.zeros_like(t) for t in src]
    chunk = _chunk(3, src[0])
    gather_blocks_to_chunk(src, SCATTERED_BLOCKS[:3], chunk)

    scatter_chunk_to_blocks(dst, SCATTERED_BLOCKS[:3], chunk, skip_prefix_n_blocks=3)

    assert all(torch.count_nonzero(_bytes(t)) == 0 for t in dst)
