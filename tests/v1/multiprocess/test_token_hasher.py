# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenHasher."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.token_hasher import TokenHasher


@pytest.fixture
def hasher() -> TokenHasher:
    """TokenHasher with small chunk_size for testing."""
    return TokenHasher(chunk_size=4, hash_algorithm="blake3")


class TestTokenHasher:
    def test_init_blake3(self) -> None:
        hasher = TokenHasher(chunk_size=256, hash_algorithm="blake3")
        assert hasher.chunk_size == 256
        assert hasher.none_hash is not None

    def test_hash_tokens_returns_bytes(self, hasher: TokenHasher) -> None:
        h = hasher.hash_tokens([1, 2, 3, 4])
        assert isinstance(h, bytes)

    def test_hash_tokens_deterministic(self, hasher: TokenHasher) -> None:
        h1 = hasher.hash_tokens([1, 2, 3, 4])
        h2 = hasher.hash_tokens([1, 2, 3, 4])
        assert h1 == h2

    def test_hash_tokens_different_input(self, hasher: TokenHasher) -> None:
        h1 = hasher.hash_tokens([1, 2, 3, 4])
        h2 = hasher.hash_tokens([5, 6, 7, 8])
        assert h1 != h2

    def test_hash_tokens_with_prefix(self, hasher: TokenHasher) -> None:
        """Rolling hash: same tokens with different prefix produces different hash."""
        h_no_prefix = hasher.hash_tokens([1, 2, 3, 4])
        h_with_prefix = hasher.hash_tokens([1, 2, 3, 4], prefix_hash=h_no_prefix)
        assert h_no_prefix != h_with_prefix

    def test_compute_chunk_hashes_exact_chunks(self, hasher: TokenHasher) -> None:
        """8 tokens with chunk_size=4 produces 2 hashes."""
        tokens = list(range(8))
        hashes = hasher.compute_chunk_hashes(tokens)
        assert len(hashes) == 2

    def test_compute_chunk_hashes_partial_chunk_discarded(
        self, hasher: TokenHasher
    ) -> None:
        """10 tokens with chunk_size=4 produces 2 hashes (last 2 tokens discarded)."""
        tokens = list(range(10))
        hashes = hasher.compute_chunk_hashes(tokens)
        assert len(hashes) == 2

    def test_compute_chunk_hashes_too_few_tokens(self, hasher: TokenHasher) -> None:
        """3 tokens with chunk_size=4 produces 0 hashes."""
        hashes = hasher.compute_chunk_hashes([1, 2, 3])
        assert len(hashes) == 0

    def test_compute_chunk_hashes_empty(self, hasher: TokenHasher) -> None:
        hashes = hasher.compute_chunk_hashes([])
        assert len(hashes) == 0

    def test_compute_chunk_hashes_rolling(self, hasher: TokenHasher) -> None:
        """Second chunk hash depends on the first (rolling property)."""
        tokens = list(range(8))
        hashes = hasher.compute_chunk_hashes(tokens)
        # Hash of chunk [4,5,6,7] alone (no prefix) should differ
        standalone = hasher.hash_tokens([4, 5, 6, 7])
        assert hashes[1] != standalone

    def test_compute_chunk_hashes_matches_manual(self, hasher: TokenHasher) -> None:
        """compute_chunk_hashes should match manual rolling hash_tokens calls."""
        tokens = list(range(12))  # 3 chunks
        auto_hashes = hasher.compute_chunk_hashes(tokens)

        h0 = hasher.hash_tokens(tokens[0:4])
        h1 = hasher.hash_tokens(tokens[4:8], prefix_hash=h0)
        h2 = hasher.hash_tokens(tokens[8:12], prefix_hash=h1)
        assert auto_hashes == [h0, h1, h2]

    def test_hash_to_bytes_from_bytes(self) -> None:
        val = b"\x01\x02\x03"
        assert TokenHasher.hash_to_bytes(val) is val

    def test_hash_to_bytes_from_int(self) -> None:
        val = 42
        result = TokenHasher.hash_to_bytes(val)
        assert isinstance(result, bytes)
        assert len(result) == 8
