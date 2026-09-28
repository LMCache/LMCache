# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the APC-covered-lookup skip (shared foundation).

Covers the correctness-critical pieces that do not require the native compute
backend:

- ``IPCCacheServerKey.covered_chunks`` default and round-trip through
  ``no_worker_id_version``.
- ``resolve_prefetched_obj_keys`` clamping the release range to the covered
  prefix, so a release never touches a chunk the lookup did not lock (risk 1:
  over-release would drop a concurrent prefix-sharing request's read lock).
"""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys

CHUNK_SIZE = 16


def _make_key(n_chunks: int, covered_chunks: int = 0) -> IPCCacheServerKey:
    n_tokens = n_chunks * CHUNK_SIZE
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(n_tokens)),
        start=0,
        end=n_tokens,
        request_id="r",
        num_kv_readers=1,
        covered_chunks=covered_chunks,
    )


def _ctx_with_hashes(n_chunks: int) -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.token_hasher.compute_chunk_hashes.return_value = [
        bytes([i]) * 4 for i in range(n_chunks)
    ]
    return ctx


def test_covered_chunks_default_zero():
    key = _make_key(4)
    assert key.covered_chunks == 0


def test_covered_chunks_survives_no_worker_id_version():
    key = _make_key(4, covered_chunks=2).no_worker_id_version()
    assert key.covered_chunks == 2
    assert key.worker_id is None


def test_covered_chunks_not_part_of_cache_identity():
    # covered_chunks is compare=False, so two otherwise-equal keys are equal.
    a = _make_key(4, covered_chunks=0)
    b = _make_key(4, covered_chunks=3)
    assert a == b


def test_resolve_full_attention_clamps_to_covered():
    # Full attention (window -1): locked range is [covered, hit).
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=2)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(-1,), covered_chunks=2
    )
    # world_size 1, one group => (hit - covered) = 2 keys, for chunks 2 and 3.
    assert len(obj_keys) == 2


def test_resolve_covered_zero_releases_full_prefix():
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=0)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(-1,), covered_chunks=0
    )
    assert len(obj_keys) == 4


def test_resolve_covered_at_hit_releases_nothing():
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=4)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(-1,), covered_chunks=4
    )
    assert obj_keys == []


def test_resolve_sliding_window_clamped_by_covered():
    # Sliding window w=2, hit=4 => window [2,4); covered=3 clamps it to [3,4).
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=3)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(2,), covered_chunks=3
    )
    assert len(obj_keys) == 1
