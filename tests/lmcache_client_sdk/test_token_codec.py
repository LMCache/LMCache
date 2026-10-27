# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the packed token-id wire codec."""

# Third Party
import pytest

# First Party
from lmcache_client_sdk.token_codec import (
    TOKEN_STRIDE,
    num_packed_tokens,
    pack_token_ids,
    unpack_token_ids,
)


def test_pack_unpack_roundtrip() -> None:
    """Check that packing and unpacking preserve a token sequence."""
    tokens = [1, 2, 3, 65536, 0, 4294967295]
    packed = pack_token_ids(tokens)
    assert len(packed) == TOKEN_STRIDE * len(tokens)
    assert unpack_token_ids(packed) == tokens


def test_pack_empty_sequence() -> None:
    """Check that an empty sequence packs to empty bytes."""
    assert pack_token_ids([]) == b""
    assert unpack_token_ids(b"") == []


def test_num_packed_tokens_counts_whole_words() -> None:
    """Check that token counts ignore a ragged trailing byte."""
    packed = pack_token_ids([7, 8, 9])
    assert num_packed_tokens(packed) == 3
    assert num_packed_tokens(packed + b"\x00") == 3


def test_unpack_rejects_ragged_buffer() -> None:
    """Check that a non-whole buffer raises a ValueError."""
    with pytest.raises(ValueError):
        unpack_token_ids(b"\x00\x00\x00")
