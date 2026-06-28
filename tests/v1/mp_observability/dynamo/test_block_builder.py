# SPDX-License-Identifier: Apache-2.0

"""Tests for build_blocks (pure block-building logic for Dynamo KV events)."""

# Third Party
import pytest

pytest.importorskip("numba", reason="numba required by TokenHasher")

# First Party
from lmcache.v1.mp_observability.dynamo.block_builder import build_blocks  # noqa: E402
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402

KV_BLOCK_SIZE = 16


@pytest.fixture
def hasher() -> TokenHasher:
    """A real TokenHasher with block-sized chunks (blake3 default)."""
    return TokenHasher(chunk_size=KV_BLOCK_SIZE, hash_algorithm="blake3")


def test_slices_full_blocks_and_drops_tail(hasher: TokenHasher) -> None:
    # 40 tokens, block size 16 -> 2 full blocks (32 tokens), last 8 dropped.
    token_ids = list(range(40))
    _, blocks = build_blocks(token_ids, KV_BLOCK_SIZE, hasher.none_hash, hasher)

    assert len(blocks) == 2
    assert blocks[0][0] == token_ids[0:16]
    assert blocks[1][0] == token_ids[16:32]
    # The trailing 8 tokens never appear in any block.
    flat = [t for block_tokens, _ in blocks for t in block_tokens]
    assert flat == token_ids[0:32]


def test_parent_chain(hasher: TokenHasher) -> None:
    token_ids = list(range(32))

    # Sequence start: prefix is none_hash -> parent is None.
    parent_start, blocks_start = build_blocks(
        token_ids, KV_BLOCK_SIZE, hasher.none_hash, hasher
    )
    assert parent_start is None
    assert len(blocks_start) == 2

    # Non-start: a non-none prefix yields a non-None parent. build_blocks
    # truncates it to a signed-i64; assert the contract (int, in range)
    # without recomputing the hash itself.
    prefix = hasher.hash_tokens(list(range(100, 116)), hasher.none_hash)
    parent_mid, blocks_mid = build_blocks(token_ids, KV_BLOCK_SIZE, prefix, hasher)
    assert parent_mid is not None
    assert isinstance(parent_mid, int)
    assert -(2**63) <= parent_mid < 2**63
    # Block ordering preserved.
    assert [bt for bt, _ in blocks_mid] == [token_ids[0:16], token_ids[16:32]]


def test_prefix_consistency(hasher: TokenHasher) -> None:
    # The first two blocks must hash identically whether the sequence is
    # 32 tokens or 48 tokens, since both share the same 32-token prefix.
    token_ids = list(range(48))
    _, blocks_32 = build_blocks(token_ids[:32], KV_BLOCK_SIZE, hasher.none_hash, hasher)
    _, blocks_48 = build_blocks(token_ids[:48], KV_BLOCK_SIZE, hasher.none_hash, hasher)

    assert len(blocks_32) == 2
    assert len(blocks_48) == 3
    assert blocks_32[0][1] == blocks_48[0][1]
    assert blocks_32[1][1] == blocks_48[1][1]
