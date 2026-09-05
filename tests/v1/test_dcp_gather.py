# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DCP gather/scatter block (de)interleave.

These cover the correctness-critical, pure-tensor reshaping used by the DCP-aware
CPU offload -- no GPU or torch.distributed required. The distributed save/load
paths (all_gather / broadcast) are exercised end-to-end in the vLLM integration.
"""

# Standard
import itertools

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.dcp_gather import _deinterleave, _interleave


def _gathered(world: int, nlb: int, blk: int) -> torch.Tensor:
    """Rank-major concatenation [1, 1, world*nl, 1] where each token carries its
    GLOBAL position. Rank r, local block j, offset w -> global block (j*world + r),
    global position (j*world + r)*blk + w."""
    nl = nlb * blk
    g = torch.empty(1, 1, world * nl, 1)
    for r in range(world):
        for j in range(nlb):
            for w in range(blk):
                g[0, 0, r * nl + j * blk + w, 0] = (j * world + r) * blk + w
    return g


@pytest.mark.parametrize(
    "world,nlb,blk", list(itertools.product([2, 4], [1, 3], [1, 4]))
)
def test_interleave_produces_global_order(world, nlb, blk):
    n = world * nlb * blk
    full = _interleave(_gathered(world, nlb, blk), world, blk, n)
    # full[..., p, :] must hold global position p.
    expected = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1)
    torch.testing.assert_close(full, expected)


@pytest.mark.parametrize(
    "world,nlb,blk", list(itertools.product([2, 4], [1, 3], [1, 4]))
)
def test_deinterleave_inverts_interleave(world, nlb, blk):
    nl = nlb * blk
    n = world * nl
    gathered = _gathered(world, nlb, blk)
    full = _interleave(gathered, world, blk, n)
    for r in range(world):
        shard = _deinterleave(full, world, r, blk, nl)
        # Must recover exactly rank r's slice of the rank-major gather.
        torch.testing.assert_close(shard, gathered[:, :, r * nl : (r + 1) * nl, :])


def test_deinterleave_selects_strided_blocks():
    """Rank r must own exactly the global blocks b with b % world == r."""
    world, nlb, blk = 3, 2, 4
    nl = nlb * blk
    n = world * nl
    full = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1)
    for r in range(world):
        shard = _deinterleave(full, world, r, blk, nl).flatten().tolist()
        owned = {int(x) // blk for x in shard}
        assert owned == {b for b in range(n // blk) if b % world == r}
