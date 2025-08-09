# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.storage_backend.cache_policy import get_cache_policy


def test_lru():
    policy = get_cache_policy("LRU")
    assert policy is not None


def test_fifo():
    policy = get_cache_policy("FIFO")
    assert policy is not None


def test_lfu():
    policy = get_cache_policy("LFU")
    assert policy is not None
