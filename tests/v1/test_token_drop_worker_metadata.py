# SPDX-License-Identifier: Apache-2.0
"""Absolute post-compaction resident KV lengths in worker metadata."""

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPWorkerMetadata,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({}, {"req": 8}, {"req": 8}),
        ({"req": 8}, {}, {"req": 8}),
        ({"req": 8}, {"req": 8}, {"req": 8}),
        ({"first": 4}, {"second": 8}, {"first": 4, "second": 8}),
    ],
)
def test_aggregate_resident_lengths(
    left: dict[str, int], right: dict[str, int], expected: dict[str, int]
) -> None:
    lhs = LMCacheMPWorkerMetadata({"req": 1}, resident_kv_updates=left)
    rhs = LMCacheMPWorkerMetadata({"req": 2}, resident_kv_updates=right)
    merged = lhs.aggregate(rhs)

    assert isinstance(merged, LMCacheMPWorkerMetadata)
    assert merged.completed_store_requests == {"req": 3}
    assert merged.resident_kv_updates == expected
    assert merged.resident_kv_updates is not left
    assert merged.resident_kv_updates is not right


def test_conflicting_resident_lengths_fail() -> None:
    lhs = LMCacheMPWorkerMetadata({}, resident_kv_updates={"req": 8})
    rhs = LMCacheMPWorkerMetadata({}, resident_kv_updates={"req": 9})

    with pytest.raises(ValueError, match="Conflicting resident KV length"):
        lhs.aggregate(rhs)
