# SPDX-License-Identifier: Apache-2.0
"""Compatibility and exact-byte tests for the shared token wire codec."""

# First Party
from lmcache.v1.multiprocess import token_codec as legacy
from lmcache.v1.transport import token_codec


def test_legacy_token_codec_reexports_canonical_objects() -> None:
    assert legacy.pack_token_ids is token_codec.pack_token_ids
    assert legacy.unpack_token_ids is token_codec.unpack_token_ids
    assert legacy.num_packed_tokens is token_codec.num_packed_tokens
    assert legacy.TOKEN_STRIDE == token_codec.TOKEN_STRIDE == 4


def test_big_endian_uint32_wire_bytes_are_frozen() -> None:
    payload = token_codec.pack_token_ids([0, 1, 258, 2**32 - 1])
    assert payload == bytes.fromhex("000000000000000100000102ffffffff")
    assert token_codec.unpack_token_ids(payload) == [0, 1, 258, 2**32 - 1]
