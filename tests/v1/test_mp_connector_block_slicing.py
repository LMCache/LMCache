# SPDX-License-Identifier: Apache-2.0
"""Per-group block-id slicing in the LMCache MP connector.

Hybrid models (e.g. ``google/gemma-4-E4B-it``) give each KV cache group its own
vLLM ``block_size`` while the connector's block accounting stays in the
canonical (GCD) ``vllm_block_size`` unit. These tests drive the public
``LMCacheMPRequestMetadata.GetStoreMetadata`` / ``GetRetrieveMetadata`` and
assert that the emitted ``op.block_ids`` give each group the right number of
block IDs: a group whose block size is ``k`` times the canonical size receives
``1/k`` as many IDs for the same token span.
"""

# Standard
from collections.abc import Iterable

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPRequestMetadata


class _FakeTracker:
    """Minimal ``LMCacheMPRequestTracker`` stand-in for the metadata builders.

    Implements only the attributes/methods ``GetStoreMetadata`` and
    ``GetRetrieveMetadata`` read, so the tests exercise the public metadata API
    without constructing a real vLLM ``Request``.
    """

    def __init__(
        self,
        allocated_block_ids: dict[int, list[int]],
        num_block_hashes: int,
        num_scheduled_tokens: int,
        num_vllm_hit_blocks: int = 0,
        num_lmcache_hit_blocks: int = 0,
        num_stored_blocks: int = 0,
    ) -> None:
        self.allocated_block_ids = allocated_block_ids
        self.block_hashes: list[int] = [0] * num_block_hashes
        self.num_scheduled_tokens = num_scheduled_tokens
        self.num_vllm_hit_blocks = num_vllm_hit_blocks
        self.num_lmcache_hit_blocks = num_lmcache_hit_blocks
        self.num_stored_blocks = num_stored_blocks
        self.all_token_ids: list[int] = list(range(num_scheduled_tokens))
        self.request_id = "req-test"
        self.cache_salt = ""

    def num_allocated_blocks(self) -> dict[int, int]:
        return {g: len(ids) for g, ids in self.allocated_block_ids.items()}

    def increase_num_stored_blocks(self, num_new_blocks: int) -> None:
        self.num_stored_blocks += num_new_blocks

    def is_ready_for_retrieving(self) -> bool:
        return True


def _block_id_counts(block_ids: Iterable[list[int]]) -> list[int]:
    return [len(group) for group in block_ids]


class TestGetStoreMetadata:
    def test_uniform_block_sizes(self):
        """All groups share the canonical block size -> equal block-id counts."""
        # chunk = 256 tokens, canonical block size 16 -> 16 canonical blocks.
        tracker = _FakeTracker(
            allocated_block_ids={0: list(range(16)), 1: list(range(100, 116))},
            num_block_hashes=16,
            num_scheduled_tokens=256,
        )
        meta = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            blocks_in_chunk=16,
            vllm_block_size=16,
            group_block_sizes=[16, 16],
        )
        assert meta is not None
        assert _block_id_counts(meta.op.block_ids) == [16, 16]
        assert meta.op.block_ids[1] == list(range(100, 116))

    def test_heterogeneous_block_sizes_gemma4(self):
        """A block_size-32 group gets half the IDs of a block_size-16 group.

        Canonical range [0, 16) spans 256 tokens. The full-attention group
        (block_size 16) needs 16 IDs; the sliding-window group (block_size 32)
        needs 8 IDs covering the same 256 tokens.
        """
        tracker = _FakeTracker(
            allocated_block_ids={
                0: list(range(16)),  # full attention, block_size 16 (k=1)
                1: list(range(8)),  # sliding window, block_size 32 (k=2)
            },
            num_block_hashes=16,
            num_scheduled_tokens=256,
        )
        meta = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            blocks_in_chunk=16,
            vllm_block_size=16,
            group_block_sizes=[16, 32],
        )
        assert meta is not None
        assert _block_id_counts(meta.op.block_ids) == [16, 8]
        assert meta.op.block_ids[0] == list(range(16))
        assert meta.op.block_ids[1] == list(range(8))

    def test_range_misaligned_to_group_block_size_raises(self):
        """A chunk that is not a whole number of a group's blocks is rejected."""
        # chunk = 128 tokens -> 8 canonical blocks; group block_size 48 (k=3)
        # does not divide 8 evenly, so the slice cannot align.
        tracker = _FakeTracker(
            allocated_block_ids={0: list(range(8)), 1: list(range(8))},
            num_block_hashes=8,
            num_scheduled_tokens=128,
        )
        with pytest.raises(ValueError, match="does not align"):
            LMCacheMPRequestMetadata.GetStoreMetadata(
                tracker,
                blocks_in_chunk=8,
                vllm_block_size=16,
                group_block_sizes=[16, 48],
            )


class TestGetRetrieveMetadata:
    def test_heterogeneous_block_sizes_gemma4(self):
        """Retrieve slices per group like store does (full 16 IDs, sliding 8)."""
        tracker = _FakeTracker(
            allocated_block_ids={0: list(range(16)), 1: list(range(8))},
            num_block_hashes=16,
            num_scheduled_tokens=256,
            num_vllm_hit_blocks=0,
            num_lmcache_hit_blocks=16,
        )
        meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
            tracker,
            blocks_in_chunk=16,
            vllm_block_size=16,
            group_block_sizes=[16, 32],
        )
        assert meta is not None
        assert _block_id_counts(meta.op.block_ids) == [16, 8]
