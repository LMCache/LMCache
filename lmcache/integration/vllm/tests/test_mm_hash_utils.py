# SPDX-License-Identifier: Apache-2.0
# Standard
import dataclasses

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_ids,
    hex_hash_to_int,
    hex_hash_to_int16,
)


@dataclasses.dataclass(frozen=True)
class DummyPlaceholderRange:
    offset: int
    length: int


def test_hex_hash_to_int16_accepts_hex_and_non_hex() -> None:
    # Hex behavior preserved (with and without 0x prefix).
    assert hex_hash_to_int16("0000") == 0
    assert hex_hash_to_int16("ffff") == 0xFFFF
    assert hex_hash_to_int16("0xFFFF") == 0xFFFF
    assert hex_hash_to_int16("0x0001") == 1

    # Non-hex identifiers must not raise and must be deterministic.
    s = "chatcmpl-a2a48871c4aad192-image-0"
    v1 = hex_hash_to_int16(s)
    v2 = hex_hash_to_int16(s)
    assert isinstance(v1, int)
    assert 0 <= v1 <= 0xFFFF
    assert v1 == v2


def test_hex_hash_to_int16_hex_variants_whitespace_and_truncation() -> None:
    # Whitespace should be ignored and case should not matter.
    assert hex_hash_to_int16(" FfFf ") == 0xFFFF
    assert hex_hash_to_int16("\n0x00aB\t") == 0x00AB

    # Long hex should be truncated to 16 bits via masking.
    assert hex_hash_to_int16("123456") == 0x3456
    assert hex_hash_to_int16("0x123456") == 0x3456


def test_hex_hash_to_int16_empty_and_invalid_hex_are_safe_and_deterministic() -> None:
    # Empty (or effectively empty) values should not raise.
    for s in ("", "   ", "0x"):
        v1 = hex_hash_to_int16(s)
        v2 = hex_hash_to_int16(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2

    # Invalid "hex-looking" strings must fall back to hashing.
    for s in ("0xGG", "deadbeeg", "0x12xz"):
        v1 = hex_hash_to_int16(s)
        v2 = hex_hash_to_int16(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2


def test_hex_hash_to_int16_non_string_inputs_are_safe() -> None:
    # Be defensive: callers may pass None or other non-string types.
    for val in (None, 0, 12345, 3.14, b"deadbeef"):
        v1 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        v2 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2


def test_apply_mm_hashes_to_token_ids_handles_non_hex_mm_hash() -> None:
    token_ids = torch.arange(0, 10, dtype=torch.long)
    mm_hashes = ["chatcmpl-a2a48871c4aad192-image-0"]
    mm_positions = [DummyPlaceholderRange(offset=2, length=4)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    expected_val = hex_hash_to_int(mm_hashes[0])
    assert out[2:6].tolist() == [expected_val] * 4


def test_apply_mm_hashes_to_token_ids_out_of_bounds_is_safe() -> None:
    token_ids = torch.zeros(5, dtype=torch.long)
    mm_hashes = ["deadbeef"]
    mm_positions = [DummyPlaceholderRange(offset=999, length=10)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    assert out.tolist() == token_ids.tolist()


def test_apply_mm_hashes_to_token_ids_multiple_placeholders_and_length_mismatch() -> (
    None
):
    token_ids = torch.zeros(12, dtype=torch.long)
    mm_hashes = ["deadbeef", "chatcmpl-a2a48871c4aad192-image-0", "EXTRA_HASH_IGNORED"]
    mm_positions = [
        DummyPlaceholderRange(offset=0, length=3),
        DummyPlaceholderRange(offset=5, length=4),
    ]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    v0 = hex_hash_to_int(mm_hashes[0])
    v1 = hex_hash_to_int(mm_hashes[1])

    assert out[0:3].tolist() == [v0] * 3
    assert out[5:9].tolist() == [v1] * 4
    # Other regions remain unchanged.
    assert out[3:5].tolist() == [0, 0]
    assert out[9:12].tolist() == [0, 0, 0]


def test_apply_mm_hashes_keeps_identifiers_that_share_their_low_16_bits_apart() -> None:
    # Two distinct multimodal identifiers whose hex digests agree on the last
    # four hex digits. Truncating to 16 bits maps them onto the same filler
    # token, so two images sharing a prompt would hash to the same chunk key
    # and one request would be served the other's KV cache.
    hash_a = "d34dd8790f1c4b0a9e6b2c5f8a1d3e7b4c9f0a2d6e8b1c3f5a7d9e0b2c4f6a8d"
    hash_b = "7a65eaf5c2b8d4e60193f7a2c5b8e1d40a6f3c9b2e5d8a1f4c7b0e3d6a9f6a8d"
    assert hash_a != hash_b
    assert hex_hash_to_int16(hash_a) == hex_hash_to_int16(hash_b)

    mm_positions = [DummyPlaceholderRange(offset=1, length=3)]
    out_a = apply_mm_hashes_to_token_ids(
        torch.zeros(6, dtype=torch.long), [hash_a], mm_positions
    )
    out_b = apply_mm_hashes_to_token_ids(
        torch.zeros(6, dtype=torch.long), [hash_b], mm_positions
    )
    assert out_a.tolist() != out_b.tolist()


def test_hex_hash_to_int_stays_within_the_signed_64_bit_range() -> None:
    identifiers = [
        "d34dd8790f1c4b0a9e6b2c5f8a1d3e7b4c9f0a2d6e8b1c3f5a7d9e0b2c4f6a8d",
        "chatcmpl-a2a48871c4aad192-image-0",
        "0x" + "f" * 64,
        "",
    ]
    for identifier in identifiers:
        value = hex_hash_to_int(identifier)
        assert 0 <= value < 2**63
        assert value == hex_hash_to_int(identifier)
