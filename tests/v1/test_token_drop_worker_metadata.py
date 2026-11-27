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
        ({}, {"req": (1, 8)}, {"req": (1, 8)}),
        ({"req": (1, 8)}, {}, {"req": (1, 8)}),
        ({"req": (1, 8)}, {"req": (1, 8)}, {"req": (1, 8)}),
        ({"first": (1, 4)}, {"second": (2, 8)}, {"first": (1, 4), "second": (2, 8)}),
    ],
)
def test_aggregate_resident_lengths(
    left: dict[str, tuple[int, int]],
    right: dict[str, tuple[int, int]],
    expected: dict[str, tuple[int, int]],
) -> None:
    lhs = LMCacheMPWorkerMetadata({"req": 1}, resident_kv_updates=left)
    rhs = LMCacheMPWorkerMetadata({"req": 2}, resident_kv_updates=right)
    merged = lhs.aggregate(rhs)

    assert isinstance(merged, LMCacheMPWorkerMetadata)
    assert merged.completed_store_requests == {"req": 3}
    assert merged.resident_kv_updates == expected
    assert merged.resident_kv_updates is not left
    assert merged.resident_kv_updates is not right


def test_conflicting_generation_reports_rejected():
    lhs = LMCacheMPWorkerMetadata({}, resident_kv_updates={"td": (1, 8)})
    rhs = LMCacheMPWorkerMetadata({}, resident_kv_updates={"td": (2, 6)})
    with pytest.raises(ValueError, match="Conflicting"):
        lhs.aggregate(rhs)
