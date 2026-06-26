# SPDX-License-Identifier: Apache-2.0
# Standard
import dataclasses

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_list,
    apply_mm_hashes_to_token_ids,
    hex_hash_to_int16,
    hex_hash_to_int64,
    get_mm_aware_token_ids,
    has_unsafe_mm_metadata,
    mm_token_surrogate_id,
    try_get_mm_aware_token_ids,
)


@dataclasses.dataclass(frozen=True)
class DummyPlaceholderRange:
    offset: int
    length: int


@dataclasses.dataclass(frozen=True)
class DummyMMFeature:
    identifier: str
    mm_position: DummyPlaceholderRange


@dataclasses.dataclass
class DummyRequest:
    all_token_ids: list[int]
    mm_features: list[DummyMMFeature] | None = None
    mm_hashes: list[str] | None = None
    mm_positions: list[DummyPlaceholderRange] | None = None


def test_hex_hash_to_int64_accepts_hex_and_non_hex() -> None:
    assert hex_hash_to_int64("0000") >= 2**62
    assert hex_hash_to_int64("0x0001") >= 2**62
    assert hex_hash_to_int64("0x0001") != 1
    assert hex_hash_to_int64("0x10000") > 0xFFFF
    assert hex_hash_to_int64("0xffffffffffffffff") >= 2**62

    s = "chatcmpl-a2a48871c4aad192-image-0"
    v1 = hex_hash_to_int64(s)
    v2 = hex_hash_to_int64(s)
    assert isinstance(v1, int)
    assert 2**62 <= v1 < 2**63
    assert v1 == v2


def test_hex_hash_to_int16_accepts_hex_and_non_hex() -> None:
    # Hex strings are still accepted, but no longer map directly into the text
    # token namespace.
    assert hex_hash_to_int16("0000") == (hex_hash_to_int64("0000") & 0xFFFF)
    assert hex_hash_to_int16("ffff") == (hex_hash_to_int64("ffff") & 0xFFFF)
    assert hex_hash_to_int16("0xFFFF") == (hex_hash_to_int64("0xFFFF") & 0xFFFF)
    assert hex_hash_to_int16("0x0001") != 1

    # Non-hex identifiers must not raise and must be deterministic.
    s = "chatcmpl-a2a48871c4aad192-image-0"
    v1 = hex_hash_to_int16(s)
    v2 = hex_hash_to_int16(s)
    assert isinstance(v1, int)
    assert 0 <= v1 <= 0xFFFF
    assert v1 == v2
    assert v1 == (hex_hash_to_int64(s) & 0xFFFF)


def test_hex_hash_to_int16_normalizes_whitespace() -> None:
    assert hex_hash_to_int16(" FfFf ") == hex_hash_to_int16("FfFf")
    assert hex_hash_to_int16("\n0x00aB\t") == hex_hash_to_int16("0x00aB")


def test_hex_hash_to_int16_rejects_empty_identifiers() -> None:
    for s in ("", "   "):
        with pytest.raises(ValueError):
            hex_hash_to_int16(s)

    # Invalid "hex-looking" strings must fall back to hashing.
    for s in ("0xGG", "deadbeeg", "0x12xz"):
        v1 = hex_hash_to_int16(s)
        v2 = hex_hash_to_int16(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2


def test_hex_hash_to_int16_non_string_inputs_are_safe() -> None:
    with pytest.raises(ValueError):
        hex_hash_to_int16(None)

    # Be defensive: callers may pass other non-string identifier types.
    for val in (0, 12345, 3.14, b"deadbeef"):
        v1 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        v2 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2
        assert v1 == (hex_hash_to_int64(val) & 0xFFFF)


def test_apply_mm_hashes_to_token_ids_handles_non_hex_mm_hash() -> None:
    token_ids = torch.arange(0, 10, dtype=torch.long)
    mm_hashes = ["chatcmpl-a2a48871c4aad192-image-0"]
    mm_positions = [DummyPlaceholderRange(offset=2, length=4)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    expected_vals = [mm_token_surrogate_id(mm_hashes[0], i) for i in range(4)]
    assert out[2:6].tolist() == expected_vals
    assert all(x >= 2**62 for x in expected_vals)
    assert len(set(expected_vals)) == 4


def test_apply_mm_hashes_to_token_ids_out_of_bounds_is_safe() -> None:
    token_ids = torch.zeros(5, dtype=torch.long)
    mm_hashes = ["deadbeef"]
    mm_positions = [DummyPlaceholderRange(offset=999, length=10)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    assert out.tolist() == token_ids.tolist()


def test_apply_mm_hashes_to_token_ids_multiple_placeholders() -> None:
    token_ids = torch.zeros(12, dtype=torch.long)
    mm_hashes = ["deadbeef", "chatcmpl-a2a48871c4aad192-image-0"]
    mm_positions = [
        DummyPlaceholderRange(offset=0, length=3),
        DummyPlaceholderRange(offset=5, length=4),
    ]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    v0 = [mm_token_surrogate_id(mm_hashes[0], i) for i in range(3)]
    v1 = [mm_token_surrogate_id(mm_hashes[1], i) for i in range(4)]

    assert out[0:3].tolist() == v0
    assert out[5:9].tolist() == v1
    assert out[3:5].tolist() != [0, 0]
    assert out[9:12].tolist() != [0, 0, 0]
    assert all(x >= 2**62 for x in out[3:5].tolist())
    assert all(x >= 2**62 for x in out[9:12].tolist())


def test_apply_mm_hashes_to_token_ids_fails_closed_on_length_mismatch() -> None:
    token_ids = torch.arange(0, 8, dtype=torch.long)
    mm_hashes = ["image-a", "image-b"]
    mm_positions = [DummyPlaceholderRange(offset=1, length=2)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)

    assert out is None


def test_apply_mm_hashes_to_token_list_returns_rewritten_copy() -> None:
    token_ids = [10, 11, 12, 13, 14]
    mm_hashes = ["0x10000"]
    mm_positions = [DummyPlaceholderRange(offset=1, length=3)]

    out = apply_mm_hashes_to_token_list(token_ids, mm_hashes, mm_positions)

    assert token_ids == [10, 11, 12, 13, 14]
    assert out is not None
    assert out[:4] == [
        10,
        mm_token_surrogate_id("0x10000", 0),
        mm_token_surrogate_id("0x10000", 1),
        mm_token_surrogate_id("0x10000", 2),
    ]
    assert out[4] != 14
    assert out[4] >= 2**62


def test_apply_mm_hashes_salts_text_after_placeholder() -> None:
    token_ids = [10, 11, 12, 13, 14, 15]
    mm_positions = [DummyPlaceholderRange(offset=1, length=2)]

    image_a = apply_mm_hashes_to_token_list(token_ids, ["image-a"], mm_positions)
    image_b = apply_mm_hashes_to_token_list(token_ids, ["image-b"], mm_positions)

    assert image_a is not None
    assert image_b is not None
    assert image_a[0] == image_b[0] == 10
    assert image_a[1:3] != image_b[1:3]
    assert image_a[3:] != token_ids[3:]
    assert image_b[3:] != token_ids[3:]
    assert image_a[3:] != image_b[3:]


def test_mm_token_surrogate_id_is_position_and_namespace_safe() -> None:
    same_offset = mm_token_surrogate_id("image-a", 3)
    assert same_offset == mm_token_surrogate_id("image-a", 3)
    assert same_offset != mm_token_surrogate_id("image-a", 4)
    assert same_offset != mm_token_surrogate_id("image-b", 3)
    assert same_offset >= 2**62
    assert same_offset != 3


def test_get_mm_aware_token_ids_text_only_returns_copy() -> None:
    request = DummyRequest(all_token_ids=[1, 2, 3])

    out = get_mm_aware_token_ids(request)

    assert out == [1, 2, 3]
    assert out is not request.all_token_ids


def test_get_mm_aware_token_ids_uses_mm_features() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4, 5],
        mm_features=[
            DummyMMFeature(
                identifier="0x10000",
                mm_position=DummyPlaceholderRange(offset=2, length=2),
            )
        ],
    )

    out = get_mm_aware_token_ids(request)

    assert out[:4] == [
        1,
        2,
        mm_token_surrogate_id("0x10000", 0),
        mm_token_surrogate_id("0x10000", 1),
    ]
    assert out[4] != 5
    assert out[4] >= 2**62


def test_get_mm_aware_token_ids_uses_legacy_fields() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4, 5, 6],
        mm_hashes=["image-a", "image-b"],
        mm_positions=[
            DummyPlaceholderRange(offset=1, length=2),
            DummyPlaceholderRange(offset=4, length=5),
        ],
    )

    out = get_mm_aware_token_ids(request)

    assert out[:3] == [
        1,
        mm_token_surrogate_id("image-a", 0),
        mm_token_surrogate_id("image-a", 1),
    ]
    assert out[3] != 4
    assert out[3] >= 2**62
    assert out[4:] == [
        mm_token_surrogate_id("image-b", 0),
        mm_token_surrogate_id("image-b", 1),
    ]
    assert request.all_token_ids == [1, 2, 3, 4, 5, 6]


def test_get_mm_aware_token_ids_fails_closed_on_empty_identifier() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4],
        mm_features=[
            DummyMMFeature(
                identifier="",
                mm_position=DummyPlaceholderRange(offset=1, length=2),
            )
        ],
    )

    assert try_get_mm_aware_token_ids(request) is None
    assert has_unsafe_mm_metadata(request)


def test_get_mm_aware_token_ids_fails_closed_on_invalid_span() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4],
        mm_features=[
            DummyMMFeature(
                identifier="image-a",
                mm_position=DummyPlaceholderRange(offset=-1, length=2),
            )
        ],
    )

    assert try_get_mm_aware_token_ids(request) is None
    assert has_unsafe_mm_metadata(request)


def test_try_get_mm_aware_token_ids_rejects_missing_positions() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4],
        mm_hashes=["image-a"],
        mm_positions=None,
    )

    assert try_get_mm_aware_token_ids(request) is None
    assert has_unsafe_mm_metadata(request)


def test_try_get_mm_aware_token_ids_rejects_missing_hashes() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4],
        mm_hashes=None,
        mm_positions=[DummyPlaceholderRange(offset=1, length=2)],
    )

    assert try_get_mm_aware_token_ids(request) is None
    assert has_unsafe_mm_metadata(request)


def test_try_get_mm_aware_token_ids_rejects_empty_legacy_hashes_with_positions() -> (
    None
):
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4],
        mm_hashes=[],
        mm_positions=[DummyPlaceholderRange(offset=1, length=2)],
    )

    assert try_get_mm_aware_token_ids(request) is None
    assert has_unsafe_mm_metadata(request)


def test_get_mm_aware_token_ids_raises_for_unsafe_mm_metadata() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4],
        mm_hashes=["image-a"],
        mm_positions=None,
    )

    with pytest.raises(ValueError):
        get_mm_aware_token_ids(request)
