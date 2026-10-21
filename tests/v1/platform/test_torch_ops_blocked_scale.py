# SPDX-License-Identifier: Apache-2.0
"""NL_X_NB_BSV_BSS through the torch fallback block transfer.

A blocked-scale page stores, per block, all tokens' values and then all
tokens' 4-byte scales, while LMCache chunk rows are token-major
``[vals | scale]``. The torch fallback must repack between the two exactly as
the CUDA kernels do (``tests/v1/multiprocess/test_blocked_scale_format.py``),
rather than copying the page bytes as if they were token-major.
"""

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

# Third Party
import pytest

_FMT = EngineKVFormat.NL_X_NB_BSV_BSS
_BS = 4  # tokens per block
_NB = 6
_NL = 2
# (value bytes, scale bytes) per token: the DSA indexer row and DeepSeek-V4's
# packed main-KV row. Module globals are rebound per parametrization.
_RECORDS = {132: (128, 4), 584: (576, 8)}
_VALS = 128
_SCALE = 4
_ROW = _VALS + _SCALE


@pytest.fixture(params=sorted(_RECORDS), ids=lambda r: f"row{r}")
def record(request):
    global _VALS, _SCALE, _ROW
    _VALS, _SCALE = _RECORDS[request.param]
    _ROW = _VALS + _SCALE
    return request.param


def _blocked_write(page: torch.Tensor, slot: int, row: torch.Tensor) -> None:
    """Write one token's row into the blocked page layout (the ground truth)."""
    block, offset = divmod(slot, _BS)
    flat = page[block].view(-1)
    flat[offset * _VALS : (offset + 1) * _VALS] = row[:_VALS]
    scale_base = _BS * _VALS
    flat[scale_base + offset * _SCALE : scale_base + (offset + 1) * _SCALE] = row[
        _VALS:
    ]


def _pages() -> list[torch.Tensor]:
    return [torch.zeros(_NB, _BS, _ROW, dtype=torch.uint8) for _ in range(_NL)]


def _rows(n_tokens: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 255, (_NL, n_tokens, _ROW), dtype=torch.uint8)


def _shape_desc() -> PageBufferShapeDesc:
    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = _NL
    shape_desc.nb = _NB
    shape_desc.bs = _BS
    shape_desc.nh = 1
    shape_desc.hs = _ROW
    shape_desc.element_size = 1
    shape_desc.kv_size = 1
    return shape_desc


def _block_transfer(
    pages: list[torch.Tensor],
    chunks: list[torch.Tensor],
    block_ids: list[int],
    direction: TransferDirection,
) -> None:
    torch_ops.multi_layer_block_kv_transfer(
        pages,
        chunks,
        block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        chunks[0].shape[1],
        _FMT,
        0,
    )


def test_block_transfer_gathers_token_major_rows(record) -> None:
    """D2H chunk rows are canonical; H2D writes them back to blocked slots."""
    block_ids = [4, 1, 3]
    slots = [b * _BS + t for b in block_ids for t in range(_BS)]
    rows = _rows(len(slots))
    src = _pages()
    for layer in range(_NL):
        for i, slot in enumerate(slots):
            _blocked_write(src[layer], slot, rows[layer, i])

    chunks = [torch.zeros(_NL, len(slots), _ROW, dtype=torch.uint8)]
    _block_transfer(src, chunks, block_ids, TransferDirection.D2H)
    assert torch.equal(chunks[0], rows)

    dst = _pages()
    _block_transfer(dst, chunks, block_ids, TransferDirection.H2D)
    for layer in range(_NL):
        assert torch.equal(dst[layer], src[layer])
