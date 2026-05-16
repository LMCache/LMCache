# SPDX-License-Identifier: Apache-2.0
# Standard
import dataclasses

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_ids,
    hex_hash_to_int16,
    hex_hash_to_int64,
)

INT64_MAX = 0x7FFFFFFFFFFFFFFF


@dataclasses.dataclass(frozen=True)
class DummyPlaceholderRange:
    offset: int
    length: int


def test_hex_hash_to_int64_accepts_hex_and_non_hex() -> None:
    # Hex behavior preserved (with and without 0x prefix).
    assert hex_hash_to_int64("0000") == 0
    assert hex_hash_to_int64("ffff") == 0xFFFF
    assert hex_hash_to_int64("0xFFFF") == 0xFFFF
    assert hex_hash_to_int64("0x0001") == 1

    # Non-hex identifiers must not raise and must be deterministic.
    s = "chatcmpl-a2a48871c4aad192-image-0"
    v1 = hex_hash_to_int64(s)
    v2 = hex_hash_to_int64(s)
    assert isinstance(v1, int)
    assert 0 <= v1 <= INT64_MAX
    assert v1 == v2


def test_hex_hash_to_int64_hex_variants_whitespace_and_truncation() -> None:
    # Whitespace should be ignored and case should not matter.
    assert hex_hash_to_int64(" FfFf ") == 0xFFFF
    assert hex_hash_to_int64("\n0x00aB\t") == 0x00AB

    # Long hex should be masked to signed int64 range.
    assert hex_hash_to_int64("123456") == 0x123456
    assert hex_hash_to_int64("0xFFFFFFFFFFFFFFFF") == INT64_MAX


def test_hex_hash_to_int64_empty_and_invalid_hex_are_safe_and_deterministic() -> None:
    # Empty (or effectively empty) values should not raise.
    for s in ("", "   ", "0x"):
        v1 = hex_hash_to_int64(s)
        v2 = hex_hash_to_int64(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= INT64_MAX
        assert v1 == v2

    # Invalid "hex-looking" strings must fall back to hashing.
    for s in ("0xGG", "deadbeeg", "0x12xz"):
        v1 = hex_hash_to_int64(s)
        v2 = hex_hash_to_int64(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= INT64_MAX
        assert v1 == v2


def test_hex_hash_to_int64_non_string_inputs_are_safe() -> None:
    # Be defensive: callers may pass None or other non-string types.
    for val in (None, 0, 12345, 3.14, b"deadbeef"):
        v1 = hex_hash_to_int64(val)  # type: ignore[arg-type]
        v2 = hex_hash_to_int64(val)  # type: ignore[arg-type]
        assert isinstance(v1, int)
        assert 0 <= v1 <= INT64_MAX
        assert v1 == v2


def test_hex_hash_to_int16_deprecated_alias_matches_int64() -> None:
    s = "chatcmpl-a2a48871c4aad192-image-0"
    with pytest.deprecated_call(match="hex_hash_to_int16 is deprecated"):
        legacy_value = hex_hash_to_int16(s)
    assert legacy_value == hex_hash_to_int64(s)


def test_hex_hash_to_int64_different_inputs_do_not_collide_in_working_set() -> None:
    seen: set[int] = set()
    for i in range(10_000):
        h = hex_hash_to_int64(f"chatcmpl-test-{i:05d}-image-0")
        assert h not in seen
        seen.add(h)


def test_apply_mm_hashes_to_token_ids_handles_non_hex_mm_hash() -> None:
    token_ids = torch.arange(0, 10, dtype=torch.long)
    mm_hashes = ["chatcmpl-a2a48871c4aad192-image-0"]
    mm_positions = [DummyPlaceholderRange(offset=2, length=4)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    expected_val = hex_hash_to_int64(mm_hashes[0])
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
    v0 = hex_hash_to_int64(mm_hashes[0])
    v1 = hex_hash_to_int64(mm_hashes[1])

    assert out[0:3].tolist() == [v0] * 3
    assert out[5:9].tolist() == [v1] * 4
    # Other regions remain unchanged.
    assert out[3:5].tolist() == [0, 0]
    assert out[9:12].tolist() == [0, 0, 0]
