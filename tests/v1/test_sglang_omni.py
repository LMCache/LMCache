# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# Third Party
import torch


@dataclass
class _StoreMetadata:
    last_node: Any
    token_ids: List[int]
    kv_indices: torch.Tensor
    offset: int
    request_id: str = ""
    mm_hashes: Optional[List[str]] = None
    mm_positions: Optional[List[Tuple[int, int]]] = None


@dataclass
class _LoadMetadata:
    token_ids: List[int]
    slot_mapping: torch.Tensor
    offset: int
    prefix_pad: int = 0
    request_id: str = ""
    mm_hashes: Optional[List[str]] = None
    mm_positions: Optional[List[Tuple[int, int]]] = None


def test_store_metadata_with_mm_hashes() -> None:
    meta = _StoreMetadata(
        last_node=None,
        token_ids=[1, 2, 3, 4, 5],
        kv_indices=torch.arange(5, dtype=torch.long),
        offset=0,
        mm_hashes=["deadbeef"],
        mm_positions=[(2, 2)],
    )
    assert meta.mm_hashes == ["deadbeef"]
    assert meta.mm_positions == [(2, 2)]


def test_load_metadata_with_mm_hashes() -> None:
    meta = _LoadMetadata(
        token_ids=[1, 2, 3, 4, 5],
        slot_mapping=torch.arange(5, dtype=torch.long),
        offset=0,
        mm_hashes=["deadbeef"],
        mm_positions=[(1, 3)],
    )
    assert meta.mm_hashes == ["deadbeef"]
    assert meta.mm_positions == [(1, 3)]


def test_store_metadata_without_mm_hashes() -> None:
    meta = _StoreMetadata(
        last_node=None,
        token_ids=[1, 2, 3],
        kv_indices=torch.arange(3, dtype=torch.long),
        offset=0,
    )
    assert meta.mm_hashes is None
    assert meta.mm_positions is None


def test_load_metadata_without_mm_hashes() -> None:
    meta = _LoadMetadata(
        token_ids=[1, 2, 3],
        slot_mapping=torch.arange(3, dtype=torch.long),
        offset=0,
    )
    assert meta.mm_hashes is None
    assert meta.mm_positions is None


def test_apply_mm_hashes_empty() -> None:
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    # First Party
    from lmcache.integration.mm_utils import apply_mm_hashes_to_token_ids as _apply

    result = _apply(token_ids.clone(), None, None)  # type: ignore[arg-type]
    assert result.tolist() == [1, 2, 3, 4, 5]


def test_apply_mm_hashes_with_positions() -> None:
    # First Party
    from lmcache.integration.mm_utils import apply_mm_hashes_to_token_ids as _apply
    from lmcache.integration.mm_utils import hex_hash_to_int16

    token_ids = torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.long)
    result = _apply(token_ids, ["deadbeef"], [(2, 3)])

    expected_val = hex_hash_to_int16("deadbeef")
    expected = [100, 101, expected_val, expected_val, expected_val, 105]
    assert result.tolist() == expected


def test_apply_mm_hashes_partial_overlap() -> None:
    # First Party
    from lmcache.integration.mm_utils import apply_mm_hashes_to_token_ids as _apply
    from lmcache.integration.mm_utils import hex_hash_to_int16

    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    result = _apply(token_ids, ["deadbeef"], [(1, 10)])

    expected_val = hex_hash_to_int16("deadbeef")
    assert result.tolist() == [1, expected_val, expected_val]
