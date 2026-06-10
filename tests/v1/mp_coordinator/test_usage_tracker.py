# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator UsageTracker."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.l2.usage_tracker import UsageTracker


def test_record_stored():
    t = UsageTracker()
    t.record_stored("a", 100)
    assert t.get("a") == 100
    assert t.get_total() == 100


def test_record_stored_accumulates():
    t = UsageTracker()
    t.record_stored("a", 100)
    t.record_stored("a", 200)
    assert t.get("a") == 300
    assert t.get_total() == 300


def test_record_evicted():
    t = UsageTracker()
    t.record_stored("a", 100)
    t.record_evicted("a", 40)
    assert t.get("a") == 60
    assert t.get_total() == 60


def test_evict_clamps_at_zero():
    t = UsageTracker()
    t.record_stored("a", 50)
    t.record_evicted("a", 100)
    assert t.get("a") == 0
    assert t.get_total() == 0


def test_evict_removes_zero_entry():
    t = UsageTracker()
    t.record_stored("a", 100)
    t.record_evicted("a", 100)
    assert t.get_all() == {}


def test_multiple_salts():
    t = UsageTracker()
    t.record_stored("a", 100)
    t.record_stored("b", 200)
    assert t.get("a") == 100
    assert t.get("b") == 200
    assert t.get_total() == 300


def test_get_unknown_returns_zero():
    t = UsageTracker()
    assert t.get("unknown") == 0


def test_get_all():
    t = UsageTracker()
    t.record_stored("a", 100)
    t.record_stored("b", 200)
    assert t.get_all() == {"a": 100, "b": 200}


def test_get_all_empty():
    t = UsageTracker()
    assert t.get_all() == {}


def test_zero_bytes_is_noop():
    t = UsageTracker()
    t.record_stored("a", 0)
    assert t.get("a") == 0
    assert t.get_all() == {}


def test_negative_store_raises():
    t = UsageTracker()
    with pytest.raises(ValueError, match="non-negative"):
        t.record_stored("a", -1)


def test_negative_evict_raises():
    t = UsageTracker()
    with pytest.raises(ValueError, match="non-negative"):
        t.record_evicted("a", -1)
