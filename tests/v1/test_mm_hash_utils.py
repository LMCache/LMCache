# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal placeholder token substitution.

vLLM emits identical placeholder token IDs for every multimodal item, so
LMCache overwrites placeholder spans with values derived from the full
multimodal identifier before token-based chunk hashing. These tests pin the
properties that make that substitution collision-safe (full-entropy,
position-dependent, prefix-stable) and act as the regression suite for the
historical 16-bit truncation bug (LMCache issue #3301).
"""

# Standard
import dataclasses

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_ids,
    mm_hash_to_token_values,
)


@dataclasses.dataclass(frozen=True)
class DummyPlaceholderRange:
    offset: int
    length: int


def test_mm_hash_to_token_values_deterministic_and_in_range() -> None:
    for identifier in (
        "d41d8cd98f00b204e9800998ecf8427e",
        "0xdeadbeef",
        "chatcmpl-a2a48871c4aad192-image-0",
        "",
    ):
        v1 = mm_hash_to_token_values(identifier, 64)
        v2 = mm_hash_to_token_values(identifier, 64)
        assert v1 == v2
        assert len(v1) == 64
        assert all(0 <= v < 2**31 for v in v1)


def test_mm_hash_to_token_values_position_dependent() -> None:
    # Values differ across positions, so a chunk overlapping any part of the
    # span sees content that encodes the offset within the item.
    values = mm_hash_to_token_values("some-image-hash", 128)
    assert len(set(values)) > 120


def test_mm_hash_to_token_values_prefix_stable() -> None:
    # A truncated span (e.g. the save path cutting at a chunk boundary) must
    # produce a prefix of the full-span substitution, or prefix hashes of
    # partial and full prompts would diverge.
    full = mm_hash_to_token_values("some-image-hash", 300)
    for length in (0, 1, 7, 8, 9, 255, 299):
        assert mm_hash_to_token_values("some-image-hash", length) == full[:length]


def test_mm_hash_to_token_values_distinct_identifiers_disjoint() -> None:
    a = mm_hash_to_token_values("image-hash-a", 32)
    b = mm_hash_to_token_values("image-hash-b", 32)
    assert a != b
    # Full-entropy: not just different somewhere, different (almost)
    # everywhere.
    assert sum(x == y for x, y in zip(a, b, strict=True)) <= 1


def test_mm_hash_to_token_values_rejects_negative_length() -> None:
    with pytest.raises(ValueError):
        mm_hash_to_token_values("some-image-hash", -1)


def test_16bit_truncation_collision_regression() -> None:
    """Regression for issue #3301: identifiers that collide in the old
    16-bit truncation must produce different substitutions now.

    ``0x1234`` and ``0x11234`` both truncated to ``0x1234`` under
    ``int(hex, 16) & 0xFFFF``, so two different images could silently share
    cache keys.
    """
    a = mm_hash_to_token_values("0x1234", 16)
    b = mm_hash_to_token_values("0x11234", 16)
    assert a != b
    assert sum(x == y for x, y in zip(a, b, strict=True)) <= 1


def test_apply_mm_hashes_fills_span_with_derived_values() -> None:
    token_ids = torch.arange(0, 10, dtype=torch.long)
    mm_hashes = ["chatcmpl-a2a48871c4aad192-image-0"]
    mm_positions = [DummyPlaceholderRange(offset=2, length=4)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    expected = list(mm_hash_to_token_values(mm_hashes[0], 4))
    assert out[2:6].tolist() == expected
    # Other regions remain unchanged.
    assert out[0:2].tolist() == [0, 1]
    assert out[6:10].tolist() == [6, 7, 8, 9]


def test_apply_mm_hashes_truncated_span_matches_full_span_prefix() -> None:
    # The save path may pass a prefix of the prompt; the substituted values in
    # the overlapping region must match the full-prompt substitution.
    mm_hashes = ["some-image-hash"]
    mm_positions = [DummyPlaceholderRange(offset=2, length=6)]

    full = apply_mm_hashes_to_token_ids(
        torch.zeros(10, dtype=torch.long), mm_hashes, mm_positions
    )
    partial = apply_mm_hashes_to_token_ids(
        torch.zeros(5, dtype=torch.long), mm_hashes, mm_positions
    )
    assert partial.tolist() == full[:5].tolist()


def test_apply_mm_hashes_out_of_bounds_is_safe() -> None:
    token_ids = torch.zeros(5, dtype=torch.long)
    mm_hashes = ["deadbeef"]
    mm_positions = [DummyPlaceholderRange(offset=999, length=10)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    assert out.tolist() == token_ids.tolist()


def test_apply_mm_hashes_multiple_placeholders_and_length_mismatch() -> None:
    token_ids = torch.zeros(12, dtype=torch.long)
    mm_hashes = ["deadbeef", "chatcmpl-a2a48871c4aad192-image-0", "EXTRA_IGNORED"]
    mm_positions = [
        DummyPlaceholderRange(offset=0, length=3),
        DummyPlaceholderRange(offset=5, length=4),
    ]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    assert out[0:3].tolist() == list(mm_hash_to_token_values(mm_hashes[0], 3))
    assert out[5:9].tolist() == list(mm_hash_to_token_values(mm_hashes[1], 4))
    # Other regions remain unchanged.
    assert out[3:5].tolist() == [0, 0]
    assert out[9:12].tolist() == [0, 0, 0]
