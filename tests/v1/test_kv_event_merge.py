# SPDX-License-Identifier: Apache-2.0
"""Tests for merging per-worker KV event batches."""

# First Party
from lmcache.integration.vllm.kv_event_merge import merge_worker_kv_events


def test_merge_is_a_first_seen_ordered_union() -> None:
    assert merge_worker_kv_events([["a", "b"], ["b", "c"], ["a"]]) == ["a", "b", "c"]


def test_merge_keeps_an_event_one_worker_reported_alone() -> None:
    """Per-rank completion skew must not lose a store that only one
    tensor-parallel worker reported in this step."""
    assert merge_worker_kv_events([["store-req-1"], []]) == ["store-req-1"]
    assert merge_worker_kv_events([[], ["store-req-1"]]) == ["store-req-1"]


def test_merge_keeps_unhashable_events_as_they_are() -> None:
    assert merge_worker_kv_events([[[1]], [[1]]]) == [[1], [1]]


def test_merge_of_nothing_is_empty() -> None:
    assert merge_worker_kv_events([]) == []
    assert merge_worker_kv_events([[], []]) == []
