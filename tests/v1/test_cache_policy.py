# SPDX-License-Identifier: Apache-2.0
# Third Party
from utils import dumb_cache_engine_key

# First Party
from lmcache.v1.storage_backend.cache_policy import get_cache_policy


class DummyMemoryObj:
    def __init__(self, can_evict: bool = True):
        self.can_evict = can_evict


def test_lru():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lru_with_pin():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key3, key1]


def test_fifo():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key1, key2]


def test_fifo_with_pin():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lfu():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key1, key3]


def test_lfu_with_pin():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key3, key2]


def test_s3fifo_fifo_mode():
    """
    Test S3-FIFO cache policy without pinned keys
    when capacity is not set and eviction does not occur
    """
    policy = get_cache_policy("S3-FIFO")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    obj4 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)
    key4 = dumb_cache_engine_key(4)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    cache_dict[key4] = obj4
    policy.update_on_put(key4)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key1, key2]


def test_s3fifo_fifo_mode_with_pin():
    """
    Test S3-FIFO cache policy with pinned keys
    when capacity is not set and eviction does not occur
    """
    policy = get_cache_policy("S3-FIFO")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    obj4 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)
    key4 = dumb_cache_engine_key(4)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)

    cache_dict[key4] = obj4
    policy.update_on_put(key4)

    cache_dict.pop(key3, None)
    policy.update_on_force_evict(key3)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert key1 not in evict_candidates
    assert evict_candidates == [key2, key4]


def test_s3fifo_s3fifo_mode():
    """
    Test S3-FIFO cache policy without pinned keys
    when capacity is not set and eviction does occur
    """
    policy = get_cache_policy("S3-FIFO")
    cache_dict = policy.init_mutable_mapping()

    test_kvs = [(dumb_cache_engine_key(i), DummyMemoryObj()) for i in range(20)]
    for i in range(10):
        key, obj = test_kvs[i]
        cache_dict[key] = obj
        policy.update_on_put(key)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
    cache_dict.pop(evict_candidates[0], None)
    # by now cache_policy should be:
    # s queue (cap: 1): [9]
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []
    key0, obj0 = test_kvs[0]
    assert evict_candidates == [key0]

    key10, obj10 = test_kvs[10]
    cache_dict[key10] = obj10
    policy.update_on_put(key10)
    # by now cache_policy should be:
    # s queue (cap: 1): [10]
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): [9]
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    for key in evict_candidates:
        cache_dict.pop(key, None)
    # by now cache_policy should be:
    # s queue (cap: 1): []
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []
    key9, _ = test_kvs[9]
    assert evict_candidates == [key9, key10]

    key1, _ = test_kvs[1]
    for _ in range(3):
        policy.update_on_hit(key1, cache_dict)

    cache_dict[key0] = obj0
    policy.update_on_put(key0)
    policy.update_on_hit(key0, cache_dict)
    # by now cache_policy should be:
    # s queue (cap: 1): [0]
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []

    cache_dict[key10] = obj10
    policy.update_on_put(key10)
    policy.update_on_hit(key10, cache_dict)
    # by now cache_policy should be:
    # s queue (cap: 1): [10]
    # m queue (cap: 9): [0, 8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []

    for key, obj in test_kvs[11:20]:
        cache_dict[key] = obj
        policy.update_on_put(key)
    # by now cache_policy should be:
    # s queue (cap: 1): [19]
    # m queue (cap: 9): [1, 10, 0, 8, 7, 6, 5, 4, 3] -- [2]
    # g queue (cap: 9): [18, 17, 16, 15, 14, 13, 12, 11]

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    for key in evict_candidates:
        cache_dict.pop(key, None)
    # by now cache_policy should be:
    # s queue (cap: 1): [19]
    # m queue (cap: 9): [1, 10, 0, 8, 7, 6, 5, 4, 3]
    # g queue (cap: 9): [18, 17, 16, 15, 14, 13, 12]
    key2, _ = test_kvs[2]
    key11, _ = test_kvs[11]
    assert evict_candidates == [key2, key11]


def test_s3fifo_s3fifo_mode_with_pin():
    """
    Test S3-FIFO cache policy with pinned keys
    when capacity is not set and eviction does occur
    """
    policy = get_cache_policy("S3-FIFO")
    cache_dict = policy.init_mutable_mapping()

    test_kvs = [(dumb_cache_engine_key(i), DummyMemoryObj()) for i in range(20)]
    _, obj2 = test_kvs[2]
    obj2.can_evict = False
    for i in range(10):
        key, obj = test_kvs[i]
        cache_dict[key] = obj
        policy.update_on_put(key)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
    cache_dict.pop(evict_candidates[0], None)
    # by now cache_policy should be:
    # s queue (cap: 1): [9]
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []
    key0, obj0 = test_kvs[0]
    assert evict_candidates == [key0]

    key10, obj10 = test_kvs[10]
    cache_dict[key10] = obj10
    policy.update_on_put(key10)
    # by now cache_policy should be:
    # s queue (cap: 1): [10]
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): [9]
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    for key in evict_candidates:
        cache_dict.pop(key, None)
    # by now cache_policy should be:
    # s queue (cap: 1): []
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []
    key9, _ = test_kvs[9]
    assert evict_candidates == [key9, key10]

    key1, _ = test_kvs[1]
    for _ in range(3):
        policy.update_on_hit(key1, cache_dict)

    cache_dict[key0] = obj0
    policy.update_on_put(key0)
    policy.update_on_hit(key0, cache_dict)
    # by now cache_policy should be:
    # s queue (cap: 1): [0]
    # m queue (cap: 9): [8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []

    cache_dict[key10] = obj10
    policy.update_on_put(key10)
    policy.update_on_hit(key10, cache_dict)
    # by now cache_policy should be:
    # s queue (cap: 1): [10]
    # m queue (cap: 9): [0, 8, 7, 6, 5, 4, 3, 2, 1]
    # g queue (cap: 9): []

    for key, obj in test_kvs[11:20]:
        cache_dict[key] = obj
        policy.update_on_put(key)
    # by now cache_policy should be:
    # s queue (cap: 1): [19]
    # m queue (cap: 9): [1, 10, 0, 8, 7, 6, 5, 4, 3] -- [2]
    # g queue (cap: 9): [18, 17, 16, 15, 14, 13, 12, 11]

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    for key in evict_candidates:
        cache_dict.pop(key, None)
    # by now cache_policy should be:
    # s queue (cap: 1): [19]
    # m queue (cap: 9): [1, 10, 0, 8, 7, 6, 5, 4, 3] -- [2]
    # g queue (cap: 9): [18, 17, 16, 15, 14, 13]
    key11, _ = test_kvs[11]
    key12, _ = test_kvs[12]
    assert evict_candidates == [key11, key12]
