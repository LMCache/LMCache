# SPDX-License-Identifier: Apache-2.0
"""Tests for the packed ``token_ids`` representation.

What makes the change safe is that it is a pure representation change:
blake3 was already fed exactly these bytes, so no chunk hash and no cache
key moves. These tests pin that equivalence at each layer that hashes.
"""

# Standard
import struct

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.multiprocess.token_codec import (
    TOKEN_STRIDE,
    num_packed_tokens,
    pack_token_ids,
    unpack_token_ids,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 4


@pytest.fixture
def hasher() -> TokenHasher:
    """TokenHasher with a small chunk_size so ranges stay readable."""
    return TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")


class TestTokenCodec:
    def test_round_trip(self) -> None:
        tokens = [0, 1, 127, 128_255, 2**32 - 1]
        assert unpack_token_ids(pack_token_ids(tokens)) == tokens

    def test_empty_round_trip(self) -> None:
        assert pack_token_ids([]) == b""
        assert unpack_token_ids(b"") == []

    def test_stride(self) -> None:
        packed = pack_token_ids([1, 2, 3])
        assert len(packed) == 3 * TOKEN_STRIDE
        assert num_packed_tokens(packed) == 3

    def test_big_endian_uint32_layout(self) -> None:
        """The layout blake3 already hashes; changing it would rekey the cache."""
        assert pack_token_ids([1, 258]) == b"\x00\x00\x00\x01\x00\x00\x01\x02"

    def test_ragged_buffer_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a multiple"):
            unpack_token_ids(b"\x00\x00\x00")

    def test_token_too_wide_rejected(self) -> None:
        with pytest.raises(OverflowError):
            pack_token_ids([2**32])


class TestPackedHashingMatchesTokenList:
    """Packed hashing must be bit-identical to the token-list path."""

    def test_whole_sequence_with_a_ragged_tail(self, hasher: TokenHasher) -> None:
        tokens = list(range(400, 430))  # 7 full chunks + 2 spare tokens
        packed = hasher.compute_packed_chunk_hashes(pack_token_ids(tokens))
        assert packed == hasher.compute_chunk_hashes(tokens)
        assert len(packed) == 7

    def test_sub_range(self, hasher: TokenHasher) -> None:
        tokens = list(range(400, 432))
        assert hasher.compute_packed_chunk_hashes(
            pack_token_ids(tokens), start=8, end=24
        ) == hasher.compute_chunk_hashes(tokens, start=8, end=24)

    def test_single_chunk(self, hasher: TokenHasher) -> None:
        chunk = [7, 8, 9, 10]
        assert hasher.hash_packed_chunk(pack_token_ids(chunk)) == hasher.hash_tokens(
            chunk
        )

    def test_digest_matches_the_byte_contract(self, hasher: TokenHasher) -> None:
        """blake3(prefix || big-endian uint32 tokens), recomputed here.

        The other tests in this class compare two entry points that share one
        implementation, so they cannot see the contract itself move. "No
        cached key moves" rests on these exact bytes, so they are pinned
        against an independent computation.
        """
        blake3 = pytest.importorskip("blake3")
        tokens = [400, 401, 402, 403]
        expected = blake3.blake3(
            hasher.none_hash + struct.pack(f">{len(tokens)}I", *tokens)
        ).digest()
        assert hasher.hash_tokens(tokens) == expected
        assert hasher.hash_packed_chunk(pack_token_ids(tokens)) == expected

    def test_rolling_prefix_matches_the_byte_contract(
        self, hasher: TokenHasher
    ) -> None:
        """Each chunk hashes over its predecessor's digest, not its tokens."""
        blake3 = pytest.importorskip("blake3")
        tokens = [400, 401, 402, 403, 500, 501, 502, 503]
        first = blake3.blake3(
            hasher.none_hash + struct.pack(">4I", *tokens[:4])
        ).digest()
        second = blake3.blake3(first + struct.pack(">4I", *tokens[4:])).digest()
        assert hasher.compute_packed_chunk_hashes(pack_token_ids(tokens)) == [
            first,
            second,
        ]

    def test_non_blake3_algorithm_still_agrees(self) -> None:
        """Algorithms that cannot take the buffer unpack, and must still match."""
        sha_hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="sha256")
        tokens = list(range(400, 416))
        assert sha_hasher.compute_packed_chunk_hashes(
            pack_token_ids(tokens)
        ) == sha_hasher.compute_chunk_hashes(tokens)


class TestSessionHashesMatchTokenListPath:
    """The server's per-request hashes are unchanged by the representation."""

    def test_incremental_growth_matches_one_shot(self, hasher: TokenHasher) -> None:
        """A session grown over several steps hashes like one that saw it all."""
        tokens = list(range(500, 532))
        session = Session(request_id="req-growing", hasher=hasher)
        for end in range(CHUNK_SIZE, len(tokens) + 1, CHUNK_SIZE):
            session.set_tokens(pack_token_ids(tokens[:end]))
            session.get_hashes(0, end)
        assert [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0, len(tokens))
        ] == hasher.compute_chunk_hashes(tokens)


class TestKeyValidation:
    def _key(self, **overrides: object) -> IPCCacheServerKey:
        kwargs: dict[str, object] = {
            "model_name": "m",
            "world_size": 1,
            "worker_id": 0,
            "token_bytes": pack_token_ids([1, 2, 3, 4]),
            "start": 0,
            "end": 4,
            "request_id": "r",
        }
        kwargs.update(overrides)
        return IPCCacheServerKey(**kwargs)  # type: ignore[arg-type]

    def test_num_tokens(self) -> None:
        assert self._key().num_tokens == 4

    def test_ragged_token_bytes_rejected(self) -> None:
        """A buffer that is not a whole number of tokens is a protocol bug."""
        with pytest.raises(ValueError, match="whole number"):
            self._key(token_bytes=b"\x00\x00\x00")

    def test_from_token_ids_packs(self) -> None:
        key = IPCCacheServerKey.from_token_ids(
            model_name="m", world_size=1, worker_id=0, token_ids=[1, 2, 3, 4]
        )
        assert key.token_bytes == pack_token_ids([1, 2, 3, 4])
