# Copyright 2024-2025 LMCache Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TPU integration LMCacheStorageKVStore raw-bytes backend.

Host-only (no torch_xla / JAX); runs anywhere LMCache imports.
"""
import os

import pytest

from lmcache.integration.tpu.lmcache_storage_kv_store import (
    LMCacheStorageKVStore, _stable_hash_int)


def test_put_get_remove(tmp_path):
    store = LMCacheStorageKVStore(root=str(tmp_path), model_name="m",
                                  world_size=1, worker_id=0)
    data = b"\x00\x01\x02hello-kv-block\xff" * 100
    assert store.get("k1") is None
    assert not store.contains("k1")
    store.put("k1", data)
    assert store.contains("k1")
    assert store.get("k1") == data
    store.remove("k1")
    assert store.get("k1") is None
    assert not store.contains("k1")


def test_overwrite_updates_usage(tmp_path):
    store = LMCacheStorageKVStore(root=str(tmp_path), model_name="m")
    store.put("k", b"a" * 100)
    store.put("k", b"b" * 50)  # overwrite
    assert store.get("k") == b"b" * 50


def test_content_addressed_keys(tmp_path):
    """Same content-hash key -> same on-disk file (cross-instance sharing)."""
    s1 = LMCacheStorageKVStore(root=str(tmp_path), model_name="qwen", world_size=4, worker_id=2)
    s1.put("h:prefixhash123", b"kvdata")
    files = [f for f in os.listdir(tmp_path) if f.endswith(".kvb")]
    assert len(files) == 1
    assert "qwen@4@2@" in files[0] and "@uint8" in files[0]


def test_lru_disk_eviction(tmp_path):
    # 1 KB cap, put three 512B blocks -> oldest evicted
    store = LMCacheStorageKVStore(root=str(tmp_path), model_name="m",
                                  max_disk_gb=1024 / (1024 ** 3))  # 1 KB
    store.put("a", b"a" * 512)
    store.put("b", b"b" * 512)
    store.put("c", b"c" * 512)  # should evict "a"
    assert store.get("a") is None
    assert store.get("c") == b"c" * 512


def test_stable_hash_deterministic():
    assert _stable_hash_int("foo") == _stable_hash_int("foo")
    assert _stable_hash_int("foo") != _stable_hash_int("bar")


def test_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TPU_OFFLOAD_LMCACHE_PATH", str(tmp_path))
    monkeypatch.setenv("LMCACHE_TPU_WORLD_SIZE", "8")
    monkeypatch.setenv("LMCACHE_TPU_WORKER_ID", "3")
    store = LMCacheStorageKVStore.from_env(model_name="mymodel")
    assert store._world == 8 and store._worker == 3
    store.put("x", b"data")
    assert store.get("x") == b"data"
