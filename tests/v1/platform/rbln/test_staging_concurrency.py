# SPDX-License-Identifier: Apache-2.0
"""Concurrent gather/scatter must not share a staging buffer.

The multiprocess server runs blocking handlers on a thread pool, so two
transfers can be inside the RBLN kernels at once. When they shared one staging
buffer, each would overwrite the other's staged bytes between the two legs of
the transfer -- silent KV corruption, no exception. This holds for the HND
host buffer and the MLA device buffer alike. CPU tensors are enough to catch
it: the failure is in which bytes land where, not in any device behaviour.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import torch

# First Party
from lmcache.v1.platform.rbln.kv_ops import (
    gather_blocks_to_chunk_hnd,
    gather_blocks_to_chunk_mla,
    scatter_chunk_to_blocks_hnd,
    scatter_chunk_to_blocks_mla,
)

LAYERS, HEADS, BLOCK, HEAD_SIZE, BLOCKS = 4, 2, 8, 4, 8


def _paged(fill):
    return [
        torch.full((2, BLOCKS, HEADS, BLOCK, HEAD_SIZE), float(fill + i))
        for i in range(LAYERS)
    ]


def _chunk(n_blocks):
    return torch.empty(2, LAYERS, n_blocks * BLOCK, HEADS * HEAD_SIZE)


def test_parallel_gathers_do_not_share_staging():
    """Each thread's gather must reflect its own paged tensors, not a neighbour's."""

    def run(fill):
        layers = _paged(fill)
        dst = _chunk(2)
        for _ in range(50):
            gather_blocks_to_chunk_hnd(layers, [0, 1], dst)
        expected = _chunk(2)
        gather_blocks_to_chunk_hnd(layers, [0, 1], expected)
        return torch.equal(dst, expected)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(run, range(8)))


def test_parallel_gather_and_scatter_do_not_share_staging():
    """A gather running next to a scatter must not see the scatter's bytes."""

    def gather(fill):
        layers = _paged(fill)
        dst = _chunk(2)
        ref = _chunk(2)
        gather_blocks_to_chunk_hnd(layers, [0, 1], ref)
        for _ in range(50):
            gather_blocks_to_chunk_hnd(layers, [0, 1], dst)
            if not torch.equal(dst, ref):
                return False
        return True

    def scatter(fill):
        layers = _paged(fill)
        src = _chunk(2)
        src.fill_(float(fill) + 100.0)
        for _ in range(50):
            scatter_chunk_to_blocks_hnd(layers, [2, 3], src)
        return all(
            torch.equal(
                layer[:, 2:4],
                torch.full_like(layer[:, 2:4], float(fill) + 100.0),
            )
            for layer in layers
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(gather, i) for i in range(4)]
        futures += [pool.submit(scatter, i) for i in range(4)]
        assert all(f.result() for f in futures)


def _mla_paged(fill):
    return [
        torch.full((BLOCKS, BLOCK, HEAD_SIZE), float(fill + i)) for i in range(LAYERS)
    ]


def _mla_chunk(n_blocks):
    return torch.empty(LAYERS, n_blocks * BLOCK, HEAD_SIZE)


def test_parallel_mla_gather_and_scatter_do_not_share_staging():
    """MLA stages on the device; that buffer must be per thread as well."""

    def gather(fill):
        layers = _mla_paged(fill)
        dst = _mla_chunk(2)
        ref = _mla_chunk(2)
        gather_blocks_to_chunk_mla(layers, [0, 1], ref)
        for _ in range(50):
            gather_blocks_to_chunk_mla(layers, [0, 1], dst)
            if not torch.equal(dst, ref):
                return False
        return True

    def scatter(fill):
        layers = _mla_paged(fill)
        src = _mla_chunk(2)
        src.fill_(float(fill) + 100.0)
        for _ in range(50):
            scatter_chunk_to_blocks_mla(layers, [2, 3], src)
        return all(
            torch.equal(layer[2:4], torch.full_like(layer[2:4], float(fill) + 100.0))
            for layer in layers
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(gather, i) for i in range(4)]
        futures += [pool.submit(scatter, i) for i in range(4)]
        assert all(f.result() for f in futures)
