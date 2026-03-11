# SPDX-License-Identifier: Apache-2.0
"""
Regression test for multi-context hash collision bug.

The old single-slot direct-address table in BlendTokenRangeMatcher silently
overwrote entries when two chunks from different contexts hashed to the same
lower 20 bits.  This caused hit rates to collapse at 3+ stored contexts.

This test registers 10 distinct contexts (mimicking the blend bench workload)
and verifies that ALL contexts remain fully retrievable.
"""

import random

import pytest

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.blend_server_v2 import BlendTokenRangeMatcher


CHUNK_SIZE = 256
WORD_POOL = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
    "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
    "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
    "omega", "anvil", "breeze", "cactus", "dune", "ember", "frost",
]


def _make_token_ids(length: int, seed: int) -> list[int]:
    """Generate a reproducible pseudo-random token sequence.

    Uses word indices as token IDs to mimic the real workload where
    contexts are built from a small word pool.
    """
    rng = random.Random(seed)
    return [rng.randint(100, 30000) for _ in range(length)]


class TestMultiContextCollision:
    """Verify that multiple stored contexts don't clobber each other."""

    @pytest.mark.parametrize("num_contexts", [3, 5, 10])
    def test_all_contexts_retrievable(self, num_contexts: int):
        """Register N contexts (no shared prefix), query each — all get full hits."""
        context_length = CHUNK_SIZE * 10  # 10 chunks per context
        chunks_per_ctx = context_length // CHUNK_SIZE
        matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE)

        contexts = []
        for i in range(num_contexts):
            # Each context is fully unique (no shared prefix)
            ctx_tokens = _make_token_ids(context_length, seed=1000 + i)
            token_hashes = [
                ObjectKey.IntHash2Bytes(i * 1000 + j)
                for j in range(chunks_per_ctx)
            ]
            matcher.on_new_token_hashes(ctx_tokens, token_hashes)
            contexts.append(ctx_tokens)

        # Now query each context and verify full hit rate
        for i, ctx_tokens in enumerate(contexts):
            results = matcher.match_sub_sequence(ctx_tokens)
            assert len(results) >= chunks_per_ctx, (
                f"Context {i}: expected {chunks_per_ctx} chunk hits, "
                f"got {len(results)}. Multi-context collision detected!"
            )

    @pytest.mark.parametrize("num_contexts", [3, 5, 10])
    def test_shared_prefix_unique_body(self, num_contexts: int):
        """Register N contexts with a shared system prompt prefix.

        The shared prefix chunks will collide (same tokens → same poly hash)
        but each context's unique body chunks must all be retrievable.
        """
        context_length = CHUNK_SIZE * 10
        matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE)

        # Shared system prompt — 2 chunks, identical for all
        system_tokens = _make_token_ids(CHUNK_SIZE * 2, seed=0)

        contexts = []
        for i in range(num_contexts):
            ctx_tokens = _make_token_ids(context_length, seed=1000 + i)
            full_tokens = system_tokens + ctx_tokens
            num_full_chunks = len(full_tokens) // CHUNK_SIZE
            token_hashes = [
                ObjectKey.IntHash2Bytes(i * 1000 + j)
                for j in range(num_full_chunks)
            ]
            matcher.on_new_token_hashes(full_tokens, token_hashes)
            contexts.append(full_tokens)

        for i, full_tokens in enumerate(contexts):
            results = matcher.match_sub_sequence(full_tokens)
            # Unique body chunks (10) must all hit; shared prefix (2) may
            # partially collide between contexts but at least 1 should hit
            body_hits = sum(1 for r in results if r.cur_st >= CHUNK_SIZE * 2)
            expected_body = context_length // CHUNK_SIZE
            assert body_hits >= expected_body, (
                f"Context {i}: expected {expected_body} body chunk hits, "
                f"got {body_hits}. Multi-context collision detected!"
            )

    def test_low_entropy_tokens_no_collision(self):
        """Stress test with very low token diversity (worst case for hashing).

        Uses only 30 distinct token IDs across 5 contexts to maximize
        hash collision potential.
        """
        context_length = CHUNK_SIZE * 8
        matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE)

        contexts = []
        for i in range(5):
            rng = random.Random(i)
            # Only 30 distinct values — very collision-prone
            tokens = [rng.choice(range(30)) for _ in range(context_length)]
            num_chunks = context_length // CHUNK_SIZE
            token_hashes = [
                ObjectKey.IntHash2Bytes(i * 100 + j)
                for j in range(num_chunks)
            ]
            matcher.on_new_token_hashes(tokens, token_hashes)
            contexts.append(tokens)

        for i, tokens in enumerate(contexts):
            results = matcher.match_sub_sequence(tokens)
            expected = context_length // CHUNK_SIZE
            # Allow up to 1 chunk miss for hash collisions that can't be avoided
            assert len(results) >= expected - 1, (
                f"Context {i}: expected ~{expected} hits, got {len(results)}"
            )

    def test_interleaved_register_and_query(self):
        """Register context, query it, register another, query both.

        Earlier contexts must remain retrievable after new ones are added.
        """
        matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE)
        context_length = CHUNK_SIZE * 5

        ctx_a = _make_token_ids(context_length, seed=999)
        hashes_a = [ObjectKey.IntHash2Bytes(j) for j in range(5)]
        matcher.on_new_token_hashes(ctx_a, hashes_a)

        # ctx_a should be fully retrievable
        results_a = matcher.match_sub_sequence(ctx_a)
        assert len(results_a) == 5, f"ctx_a: expected 5 hits, got {len(results_a)}"

        # Now register ctx_b
        ctx_b = _make_token_ids(context_length, seed=888)
        hashes_b = [ObjectKey.IntHash2Bytes(100 + j) for j in range(5)]
        matcher.on_new_token_hashes(ctx_b, hashes_b)

        # Both should still be fully retrievable
        results_a2 = matcher.match_sub_sequence(ctx_a)
        results_b = matcher.match_sub_sequence(ctx_b)
        assert len(results_a2) == 5, (
            f"ctx_a after adding ctx_b: expected 5 hits, got {len(results_a2)}"
        )
        assert len(results_b) == 5, (
            f"ctx_b: expected 5 hits, got {len(results_b)}"
        )

    @pytest.mark.parametrize("num_contexts", [20, 50])
    def test_scale_many_contexts(self, num_contexts: int):
        """Scale test: even with many contexts, most chunks should hit."""
        context_length = CHUNK_SIZE * 4  # smaller to keep test fast
        chunks_per_ctx = context_length // CHUNK_SIZE
        matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE)

        contexts = []
        for i in range(num_contexts):
            tokens = _make_token_ids(context_length, seed=2000 + i)
            token_hashes = [
                ObjectKey.IntHash2Bytes(i * 100 + j)
                for j in range(chunks_per_ctx)
            ]
            matcher.on_new_token_hashes(tokens, token_hashes)
            contexts.append(tokens)

        total_expected = 0
        total_hits = 0
        for i, tokens in enumerate(contexts):
            results = matcher.match_sub_sequence(tokens)
            total_expected += chunks_per_ctx
            total_hits += len(results)

        hit_rate = total_hits / total_expected
        assert hit_rate > 0.95, (
            f"Overall hit rate {hit_rate:.1%} for {num_contexts} contexts "
            f"— too low, collision handling may be broken"
        )
