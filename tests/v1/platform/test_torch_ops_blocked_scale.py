# SPDX-License-Identifier: Apache-2.0
"""NL_X_NB_BSV_BSS through the torch fallback block transfer.

A blocked-scale page stores, per block, all tokens' values and then all
tokens' 4-byte scales, while LMCache chunk rows are token-major
``[vals | scale]``. The torch fallback must repack between the two exactly as
the CUDA kernels do (``tests/v1/multiprocess/test_blocked_scale_format.py``),
rather than copying the page bytes as if they were token-major.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

_FMT = EngineKVFormat.NL_X_NB_BSV_BSS
_BS = 4  # tokens per block
_NB = 6
_NL = 2


def _blocked_write(page: torch.Tensor, slot: int, row: torch.Tensor) -> None:
    """Write one token's row into the blocked page layout (the ground truth)."""
    block, offset = divmod(slot, _BS)
    flat = page[block].view(-1)
    vals = row.numel() - 4
    flat[offset * vals : (offset + 1) * vals] = row[:vals]
    scale_base = _BS * vals
    flat[scale_base + offset * 4 : scale_base + (offset + 1) * 4] = row[vals:]


def _pages(
    row: int, padding: int, offset: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    stride = _BS * row + padding
    storage = [
        torch.full((offset + _NB * stride + 8,), 165, dtype=torch.uint8)
        for _ in range(_NL)
    ]
    pages = [
        buffer.as_strided((_NB, _BS, row), (stride, row, 1), offset).zero_()
        for buffer in storage
    ]
    return pages, storage


def _rows(n_tokens: int, row: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 255, (_NL, n_tokens, row), dtype=torch.uint8)


def _shape_desc(row: int) -> PageBufferShapeDesc:
    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = _NL
    shape_desc.nb = _NB
    shape_desc.bs = _BS
    shape_desc.nh = 1
    shape_desc.hs = row
    shape_desc.element_size = 1
    shape_desc.kv_size = 1
    return shape_desc


def _block_transfer(
    pages: list[torch.Tensor],
    chunks: list[torch.Tensor],
    block_ids: list[int],
    direction: TransferDirection,
    skip: int,
) -> None:
    torch_ops.multi_layer_block_kv_transfer(
        pages,
        chunks,
        block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(pages[0].shape[-1]),
        chunks[0].shape[1],
        _FMT,
        skip,
    )


@pytest.mark.parametrize("row", [12, 132])
@pytest.mark.parametrize(("padding", "offset"), [(0, 0), (16, 7)])
@pytest.mark.parametrize("skip", [0, 1, 2])
def test_block_transfer_gathers_token_major_rows(
    row: int, padding: int, offset: int, skip: int
) -> None:
    """D2H chunk rows are canonical; H2D writes them back to blocked slots."""
    block_ids = [4, 1, 3]
    slots = [b * _BS + t for b in block_ids for t in range(_BS)]
    rows = _rows(len(slots), row)
    src, _ = _pages(row, padding, offset)
    for layer in range(_NL):
        for i, slot in enumerate(slots):
            _blocked_write(src[layer], slot, rows[layer, i])

    chunks = [torch.full((_NL, 2 * _BS, row), 165, dtype=torch.uint8) for _ in range(2)]
    _block_transfer(src, chunks, block_ids, TransferDirection.D2H, skip)
    combined = torch.cat(chunks, dim=1)
    assert torch.equal(combined[:, skip * _BS : len(slots)], rows[:, skip * _BS :])
    assert torch.all(combined[:, : skip * _BS] == 165)
    assert torch.all(combined[:, len(slots) :] == 165)

    dst, storage = _pages(row, padding, offset)
    _block_transfer(dst, chunks, block_ids, TransferDirection.H2D, skip)
    stride = _BS * row + padding
    for layer, buffer in enumerate(storage):
        for block in range(_NB):
            if block in block_ids[skip:]:
                assert torch.equal(dst[layer][block], src[layer][block])
            else:
                assert torch.count_nonzero(dst[layer][block]) == 0
        assert torch.all(buffer[:offset] == 165)
        assert torch.all(buffer[offset + _NB * stride :] == 165)
        physical = buffer[offset : offset + _NB * stride].view(_NB, stride)
        assert torch.all(physical[:, _BS * row :] == 165)
