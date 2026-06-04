# SPDX-License-Identifier: Apache-2.0
"""Per-group block-id slicing in the LMCache MP connector.

Hybrid models (e.g. ``google/gemma-4-E4B-it``) give each KV cache group its
own vLLM ``block_size`` while the connector's block accounting stays in the
canonical (GCD) ``vllm_block_size`` unit. These tests verify that
``LMCacheMPRequestMetadata._slice_block_ids`` maps a canonical block range to
the correct per-group block range, so a group whose block size is ``k`` times
the canonical size receives ``1/k`` as many block IDs for the same token span.
"""
# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPRequestMetadata


def _tracker(allocated_block_ids: dict[int, list[int]]) -> SimpleNamespace:
    """Minimal stand-in exposing only what ``_slice_block_ids`` reads."""
    return SimpleNamespace(allocated_block_ids=allocated_block_ids)


class TestSliceBlockIds:
    def test_uniform_block_sizes(self):
        """All groups share the canonical block size -> identical slices."""
        tracker = _tracker({0: list(range(16)), 1: list(range(100, 116))})
        sliced = LMCacheMPRequestMetadata._slice_block_ids(
            tracker,
            group_block_sizes=[16, 16],
            vllm_block_size=16,
            start=0,
            end=16,
        )
        assert sliced == [list(range(16)), list(range(100, 116))]

    def test_heterogeneous_block_sizes_gemma4(self):
        """A 32-token (k=2) group gets half the IDs of a 16-token (k=1) group.

        Canonical range [0, 16) spans 16 * 16 = 256 tokens. The full-attention
        group (block_size=16) needs 16 IDs; the sliding-window group
        (block_size=32) needs 8 IDs covering the same 256 tokens.
        """
        tracker = _tracker(
            {
                0: list(range(16)),  # full attention, block_size 16
                1: list(range(8)),  # sliding window, block_size 32
            }
        )
        sliced = LMCacheMPRequestMetadata._slice_block_ids(
            tracker,
            group_block_sizes=[16, 32],
            vllm_block_size=16,
            start=0,
            end=16,
        )
        assert [len(group) for group in sliced] == [16, 8]
        assert sliced[0] == list(range(16))
        assert sliced[1] == list(range(8))

    def test_nonzero_start_offset(self):
        """Start/end offsets are divided per group by the block factor."""
        tracker = _tracker(
            {
                0: list(range(32)),  # block_size 16, k=1
                1: list(range(16)),  # block_size 32, k=2
            }
        )
        # Canonical [16, 32): skip the first 16 canonical blocks.
        sliced = LMCacheMPRequestMetadata._slice_block_ids(
            tracker,
            group_block_sizes=[16, 32],
            vllm_block_size=16,
            start=16,
            end=32,
        )
        assert sliced[0] == list(range(16, 32))  # [16:32]
        assert sliced[1] == list(range(8, 16))  # [8:16]

    def test_block_size_not_multiple_of_canonical_raises(self):
        tracker = _tracker({0: list(range(16))})
        with pytest.raises(ValueError, match="positive multiple"):
            LMCacheMPRequestMetadata._slice_block_ids(
                tracker,
                group_block_sizes=[24],
                vllm_block_size=16,
                start=0,
                end=16,
            )

    def test_misaligned_range_raises(self):
        tracker = _tracker({0: list(range(16))})
        # end=15 is not divisible by the k=2 factor of a block_size-32 group.
        with pytest.raises(ValueError, match="does not align"):
            LMCacheMPRequestMetadata._slice_block_ids(
                tracker,
                group_block_sizes=[32],
                vllm_block_size=16,
                start=0,
                end=15,
            )
