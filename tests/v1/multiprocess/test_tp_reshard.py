# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.tp_reshard import (
    copy_heads,
    is_valid_tp,
    kv_head_range,
    plan_head_reshard,
    reshard_layout_desc,
)

TP_SIZES = [1, 2, 4, 8, 16]
HEAD_COUNTS = [1, 2, 8]
HEAD_ELEMS = 4


def _full_kv(h: int) -> torch.Tensor:
    # [kv_size, layers, tokens, heads * head_elems] with unique values
    return torch.arange(2 * 3 * 5 * h * HEAD_ELEMS, dtype=torch.float32).reshape(
        2, 3, 5, h * HEAD_ELEMS
    )


def _shard(full: torch.Tensor, h: int, tp: int, rank: int) -> torch.Tensor:
    first, n = kv_head_range(h, tp, rank)
    return full[..., first * HEAD_ELEMS : (first + n) * HEAD_ELEMS].clone()


def test_replicated_ranks_share_heads():
    assert [kv_head_range(2, 8, r) for r in range(8)] == [(0, 1)] * 4 + [(1, 1)] * 4
    assert [kv_head_range(8, 2, r) for r in range(2)] == [(0, 4), (4, 4)]


def test_invalid_tp_rejected():
    assert not is_valid_tp(8, 3)
    assert not is_valid_tp(0, 1)
    with pytest.raises(ValueError):
        kv_head_range(8, 3, 0)
    with pytest.raises(ValueError):
        kv_head_range(8, 2, 2)


@pytest.mark.parametrize("h", HEAD_COUNTS)
@pytest.mark.parametrize("src_tp", TP_SIZES)
@pytest.mark.parametrize("dst_tp", TP_SIZES)
def test_reshard_rebuilds_destination_shards(h, src_tp, dst_tp):
    if not (is_valid_tp(h, src_tp) and is_valid_tp(h, dst_tp)):
        pytest.skip("vLLM rejects this TP size")
    full = _full_kv(h)
    src_by_rank = {r: [_shard(full, h, src_tp, r)] for r in range(src_tp)}
    for dst_rank in range(dst_tp):
        expected = _shard(full, h, dst_tp, dst_rank)
        out = torch.full_like(expected, -1.0)
        copies = plan_head_reshard(h, src_tp, dst_tp, dst_rank)
        assert sum(c.num_heads for c in copies) == kv_head_range(h, dst_tp, dst_rank)[1]
        copy_heads([out], src_by_rank, copies, kv_head_range(h, dst_tp, dst_rank)[1])
        assert torch.equal(out, expected)


def test_layout_desc_scales_last_dim():
    desc = MemoryLayoutDesc(
        shapes=[torch.Size((2, 28, 256, 4 * 128))], dtypes=[torch.bfloat16]
    )
    at_tp1 = reshard_layout_desc(desc, num_kv_heads=8, from_tp=2, to_tp=1)
    assert at_tp1.shapes == [torch.Size((2, 28, 256, 8 * 128))]
    assert at_tp1.dtypes == [torch.bfloat16]
    back = reshard_layout_desc(at_tp1, num_kv_heads=8, from_tp=1, to_tp=2)
    assert back.shapes == desc.shapes
