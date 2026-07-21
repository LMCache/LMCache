# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple

# Third Party
import torch

# First Party
from lmcache.integration.mm_utils import (
    apply_mm_hashes_to_token_ids,
    hex_hash_to_int16,
)


def test_hex_hash_to_int16_hex_values() -> None:
    assert hex_hash_to_int16("0000") == 0
    assert hex_hash_to_int16("ffff") == 0xFFFF
    assert hex_hash_to_int16("0xFFFF") == 0xFFFF
    assert hex_hash_to_int16("0x0001") == 1


def test_hex_hash_to_int16_non_hex_deterministic() -> None:
    s = "chatcmpl-a2a48871c4aad192-image-0"
    v1 = hex_hash_to_int16(s)
    v2 = hex_hash_to_int16(s)
    assert isinstance(v1, int)
    assert 0 <= v1 <= 0xFFFF
    assert v1 == v2


def test_hex_hash_to_int16_whitespace_and_truncation() -> None:
    assert hex_hash_to_int16(" FfFf ") == 0xFFFF
    assert hex_hash_to_int16("\n0x00aB\t") == 0x00AB
    assert hex_hash_to_int16("123456") == 0x3456
    assert hex_hash_to_int16("0x123456") == 0x3456


def test_hex_hash_to_int16_empty_and_invalid() -> None:
    for s in ("", "   ", "0x"):
        v1 = hex_hash_to_int16(s)
        v2 = hex_hash_to_int16(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2

    for s in ("0xGG", "deadbeeg", "0x12xz"):
        v1 = hex_hash_to_int16(s)
        v2 = hex_hash_to_int16(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2


def test_hex_hash_to_int16_non_string_inputs() -> None:
    for val in (None, 0, 12345, 3.14, b"deadbeef"):
        v1 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        v2 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2


def test_apply_mm_hashes_to_token_ids_tensor() -> None:
    token_ids = torch.arange(0, 10, dtype=torch.long)
    mm_hashes = ["deadbeef"]
    mm_positions: List[Tuple[int, int]] = [(2, 4)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    expected_val = hex_hash_to_int16("deadbeef")
    assert out[2:6].tolist() == [expected_val] * 4


def test_apply_mm_hashes_to_token_ids_list() -> None:
    token_ids: List[int] = list(range(10))
    mm_hashes = ["deadbeef"]
    mm_positions: List[Tuple[int, int]] = [(2, 4)]

    out = apply_mm_hashes_to_token_ids(token_ids[:], mm_hashes, mm_positions)
    expected_val = hex_hash_to_int16("deadbeef")
    assert out[2:6] == [expected_val] * 4


def test_apply_mm_hashes_to_token_ids_out_of_bounds() -> None:
    token_ids = torch.zeros(5, dtype=torch.long)
    mm_hashes = ["deadbeef"]
    mm_positions: List[Tuple[int, int]] = [(999, 10)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    assert out.tolist() == token_ids.tolist()


def test_apply_mm_hashes_to_token_ids_multiple_placeholders() -> None:
    token_ids = torch.zeros(12, dtype=torch.long)
    mm_hashes = ["deadbeef", "chatcmpl-image-0"]
    mm_positions: List[Tuple[int, int]] = [(0, 3), (5, 4)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    v0 = hex_hash_to_int16(mm_hashes[0])
    v1 = hex_hash_to_int16(mm_hashes[1])

    assert out[0:3].tolist() == [v0] * 3
    assert out[5:9].tolist() == [v1] * 4
    assert out[3:5].tolist() == [0, 0]
    assert out[9:12].tolist() == [0, 0, 0]


def test_apply_mm_hashes_to_token_ids_list_returns_modified() -> None:
    token_ids: List[int] = [0] * 10
    mm_hashes = ["deadbeef"]
    mm_positions: List[Tuple[int, int]] = [(1, 2)]

    result = apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
    val = hex_hash_to_int16("deadbeef")
    assert result[1] == val
    assert result[2] == val
