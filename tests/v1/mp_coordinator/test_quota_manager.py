# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator L2QuotaManager."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.l2.quota_manager import L2QuotaManager


def test_set_and_get():
    store = L2QuotaManager()
    store.set("salt-a", 1000)
    assert store.get("salt-a") == 1000


def test_get_unregistered_returns_none():
    store = L2QuotaManager()
    assert store.get("unknown") is None


def test_set_overwrites():
    store = L2QuotaManager()
    store.set("salt-a", 1000)
    store.set("salt-a", 2000)
    assert store.get("salt-a") == 2000


def test_delete():
    store = L2QuotaManager()
    store.set("salt-a", 1000)
    assert store.delete("salt-a") is True
    assert store.get("salt-a") is None


def test_delete_nonexistent():
    store = L2QuotaManager()
    assert store.delete("unknown") is False


def test_list_all():
    store = L2QuotaManager()
    store.set("a", 100)
    store.set("b", 200)
    entries = store.list_all()
    by_salt = {e.cache_salt: e.limit_bytes for e in entries}
    assert by_salt == {"a": 100, "b": 200}


def test_list_all_empty():
    store = L2QuotaManager()
    assert store.list_all() == []


def test_negative_limit_raises():
    store = L2QuotaManager()
    with pytest.raises(ValueError, match="non-negative"):
        store.set("salt-a", -1)


def test_zero_limit_accepted():
    store = L2QuotaManager()
    store.set("salt-a", 0)
    assert store.get("salt-a") == 0
