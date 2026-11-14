# SPDX-License-Identifier: Apache-2.0
"""Mixed L1/FS prefix lookup and capacity contracts."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.engine_context import partition_streaming_budget
from lmcache.v1.multiprocess.mixed_streaming import plan_mixed_streaming


class L1:
    def __init__(self, keys, manager_id=7):
        self.keys = set(keys)
        self.l1_manager_id = manager_id
        self.locks = {}

    def reserve_read(self, keys, read_locks=1):
        results = {}
        for key in keys:
            present = key in self.keys
            if present:
                self.locks[key] = self.locks.get(key, 0) + read_locks
            results[key] = (L1Error.SUCCESS if present else L1Error.KEY_NOT_EXIST, None)
        return results

    def finish_read(self, keys, read_locks=1):
        for key in keys:
            self.locks[key] -= read_locks
            if not self.locks[key]:
                del self.locks[key]


def keys(count):
    return [
        ObjectKey(chunk_hash=bytes([i]), model_name="m", kv_rank=0)
        for i in range(count)
    ]


def spec_for(obj_keys):
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    return PrefetchTaskSpec([GroupedObjectKeys(obj_keys, 0, layout)])


def reader_for(disk_keys):
    reader = MagicMock()

    def lookup(keys, _layouts):
        bitmap = Bitmap(len(keys))
        for index, key in enumerate(keys):
            if key in disk_keys:
                bitmap.set(index)
        return bitmap

    reader.lookup.side_effect = lookup
    return reader


def test_l1_then_fs_prefix_stops_before_missing_and_releases_later_l1():
    obj_keys = keys(4)
    l1 = L1([obj_keys[0], obj_keys[3]])
    result = plan_mixed_streaming(spec_for(obj_keys), [l1], reader_for({obj_keys[1]}))
    assert result is not None
    assert [result.hit_cells[0].test(i) for i in range(4)] == [True, True, False, False]
    assert [result.l1_hit_cells[0].test(i) for i in range(4)] == [
        True,
        False,
        False,
        False,
    ]
    assert [result.l2_hit_cells[0].test(i) for i in range(4)] == [
        False,
        True,
        False,
        False,
    ]
    assert result.l1_owners == {obj_keys[0]: 7}
    assert l1.locks == {obj_keys[0]: 1}


def test_no_streamed_prefix_releases_all_l1_locks():
    obj_keys = keys(3)
    l1 = L1([obj_keys[0], obj_keys[2]])
    assert (
        plan_mixed_streaming(spec_for(obj_keys), [l1], reader_for({obj_keys[2]}))
        is None
    )
    assert l1.locks == {}


def test_lookup_timeout_releases_l1_locks():
    obj_keys = keys(2)
    l1 = L1([obj_keys[0]])
    reader = MagicMock()
    reader.lookup.side_effect = LMCacheTimeoutError("timeout")
    with pytest.raises(LMCacheTimeoutError):
        plan_mixed_streaming(spec_for(obj_keys), [l1], reader)
    assert l1.locks == {}


def test_multiple_ranks_require_every_shard_and_release_locks_after_gap():
    first = keys(3)
    second = [
        ObjectKey(chunk_hash=k.chunk_hash, model_name="m", kv_rank=1) for k in first
    ]
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    spec = PrefetchTaskSpec(
        [
            GroupedObjectKeys(first, 0, layout),
            GroupedObjectKeys(second, 0, layout),
        ]
    )
    l1 = L1([first[0], second[0], first[2], second[2]])
    result = plan_mixed_streaming(spec, [l1], reader_for({first[1]}))
    assert result is None  # second rank is missing at the first FS chunk
    assert l1.locks == {}


def test_fs_then_l1_with_multiple_reader_locks():
    obj_keys = keys(2)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    spec = PrefetchTaskSpec([GroupedObjectKeys(obj_keys, 0, layout, num_kv_readers=2)])
    l1 = L1([obj_keys[1]])
    result = plan_mixed_streaming(spec, [l1], reader_for({obj_keys[0]}))
    assert result is not None
    assert result.l2_hit_count == result.l1_hit_count == 1
    assert l1.locks == {obj_keys[1]: 2}


def test_multi_rank_mixed_prefix_keeps_only_required_l1_owners():
    first = keys(4)
    second = [
        ObjectKey(chunk_hash=k.chunk_hash, model_name="m", kv_rank=1) for k in first
    ]
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    spec = PrefetchTaskSpec(
        [GroupedObjectKeys(first, 0, layout), GroupedObjectKeys(second, 0, layout)]
    )
    l1a = L1([first[0], first[3]], 7)
    l1b = L1([second[0], second[3]], 8)
    result = plan_mixed_streaming(spec, [l1a, l1b], reader_for({first[1], second[1]}))
    assert result is not None
    assert result.l1_owners == {first[0]: 7, second[0]: 8}
    assert result.l2_hit_count == 2
    assert l1a.locks == {first[0]: 1}
    assert l1b.locks == {second[0]: 1}


def test_staging_capacity_is_part_of_l1_budget(tmp_path):
    l1 = L1ManagerConfig(L1MemoryManagerConfig(size_in_bytes=8192, use_lazy=True))
    config = StorageManagerConfig(
        l1_manager_config=l1,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            [FSL2AdapterConfig(str(tmp_path), streaming_buffer_size_bytes=4096)]
        ),
    )
    partitioned = partition_streaming_budget(config)
    assert partitioned.l1_manager_config.memory_config.size_in_bytes == 4096
    assert config.l1_manager_config.memory_config.size_in_bytes == 8192
    with pytest.raises(ValueError, match="smaller"):
        partition_streaming_budget(partitioned)
