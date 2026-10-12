# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPWorkerMetadata
from lmcache.integration.vllm.token_drop_allocator_adapter import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)


@pytest.fixture(autouse=True)
def clear_state():
    clear_resident_kv_tokens("td")
    yield
    clear_resident_kv_tokens("td")


def connector(*, private=True):
    instance = LMCacheMPConnector.__new__(LMCacheMPConnector)
    row = [
        SimpleNamespace(block_id=i, ref_cnt=1 if private else 2, block_hash=None)
        for i in (7, 8, 9, 10)
    ]
    instance._token_drop_allocations = {"td": SimpleNamespace(blocks=(row,))}
    instance._token_drop_finished_before_output = set()
    instance._gpu_block_pool = MagicMock()
    instance._group_tokens_per_block = [4]
    instance._kv_cache_events = None
    instance.lazy_offload = False
    tracker = SimpleNamespace(
        request_configs={"lmcache.token_drop": {"algorithm": "rkv", "config": {}}},
        num_scheduled_tokens=16,
        allocated_block_ids={0: [7, 8, 9, 10]},
    )
    instance.request_trackers = {"td": tracker}
    return instance, row, tracker


def test_completed_worker_update_reclaims_tail_and_commits_p():
    c, row, tracker = connector()
    set_resident_kv_tokens("td", 16)
    c.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={}, resident_kv_updates={"td": 7}
            ),
        )
    )
    assert get_resident_kv_tokens("td") == 7
    assert [block.block_id for block in row] == [7, 8]
    assert tracker.allocated_block_ids[0] == [7, 8]
    freed = list(c._gpu_block_pool.free_blocks.call_args.args[0])
    assert [block.block_id for block in freed] == [10, 9]


def test_exclusive_block_guard_prevents_reclaim():
    c, row, _ = connector(private=False)
    set_resident_kv_tokens("td", 16)
    with pytest.raises(ValueError, match="exclusively owned"):
        c._commit_token_drop_resident_updates({"td": 7})
    assert len(row) == 4
    assert get_resident_kv_tokens("td") == 16
    c._gpu_block_pool.free_blocks.assert_not_called()


def test_invalid_length_does_not_mutate_pool():
    c, row, _ = connector()
    set_resident_kv_tokens("td", 16)
    with pytest.raises(ValueError, match="Invalid post-compaction"):
        c._commit_token_drop_resident_updates({"td": 17})
    assert len(row) == 4
    assert get_resident_kv_tokens("td") == 16
    c._gpu_block_pool.free_blocks.assert_not_called()


def test_late_update_after_finish_is_ignored():
    c, row, _ = connector()
    c._token_drop_finished_before_output.add("td")
    c.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={}, resident_kv_updates={"td": 7}
            ),
        )
    )
    assert len(row) == 4
    c._gpu_block_pool.free_blocks.assert_not_called()


def test_pinned_vllm_allocator_reclaim_and_reallocate():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Resident allocation requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import RequestStatus
    import torch

    # First Party
    from lmcache.integration.vllm.token_drop_allocator_adapter import (
        _allocate_with_resident_frontier,
    )

    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer0"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=128,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=False,
    )
    req = SimpleNamespace(
        request_id="td",
        num_computed_tokens=0,
        num_prompt_tokens=48,
        num_tokens=128,
        status=RequestStatus.RUNNING,
        kv_transfer_params={"lmcache.token_drop": {"algorithm": "rkv", "config": {}}},
    )
    free_before = manager.block_pool.get_num_free_blocks()
    assert manager.allocate_slots(req, 48) is not None
    canonical = manager.coordinator.get_blocks("td")[0]
    assert len(canonical) == 3

    c = LMCacheMPConnector.__new__(LMCacheMPConnector)
    c._gpu_block_pool = manager.block_pool
    c._group_tokens_per_block = [16]
    c._token_drop_allocations = {"td": KVCacheBlocks((canonical,))}
    tracker = SimpleNamespace(
        request_configs=req.kv_transfer_params,
        num_scheduled_tokens=48,
        allocated_block_ids={0: [block.block_id for block in canonical]},
    )
    c.request_trackers = {"td": tracker}
    set_resident_kv_tokens("td", 48)
    c._commit_token_drop_resident_updates({"td": 20})
    assert len(canonical) == 2
    assert manager.coordinator.get_blocks("td")[0] is canonical
    assert tracker.allocated_block_ids[0] == [block.block_id for block in canonical]
    assert manager.block_pool.get_num_free_blocks() == free_before - 2

    req.num_computed_tokens = 48
    assert (
        _allocate_with_resident_frontier(
            KVCacheManager.allocate_slots, manager, req, 13
        )
        is not None
    )
    assert req.num_computed_tokens == 48
    assert get_resident_kv_tokens("td") == 33
    assert len(canonical) == 3

    manager.free(req)
    assert manager.block_pool.get_num_free_blocks() == free_before


def test_scheduler_preemption_clears_resident_state_and_ignores_late_update():
    c, row, _ = connector()
    set_resident_kv_tokens("td", 7)
    c._process_retrieve_requests = MagicMock()
    c._process_new_requests = MagicMock()
    c._process_cached_requests = MagicMock()
    c._report_block_allocation_deltas = MagicMock()

    metadata = c.build_connector_meta(
        SimpleNamespace(preempted_req_ids={"td"}, scheduled_cached_reqs=None)
    )
    assert metadata.need_flush_before_forward
    assert "td" not in c._token_drop_allocations
    assert get_resident_kv_tokens("td") is None
    assert c._token_drop_finished_before_output == {"td"}

    c.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={}, resident_kv_updates={"td": 3}
            ),
        )
    )
    assert get_resident_kv_tokens("td") is None
    assert len(row) == 4
    c._gpu_block_pool.free_blocks.assert_not_called()
    assert not c._token_drop_finished_before_output
