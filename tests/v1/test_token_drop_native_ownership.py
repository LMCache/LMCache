# SPDX-License-Identifier: Apache-2.0
"""Native scheduler ownership across TD reclaim, reuse and teardown."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")


def test_native_td_reclaim_preserves_reallocated_block_ownership():
    # Third Party
    import vllm

    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Native ownership contract requires vLLM 0.25.1")

    # Third Party
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_scheduler import (
        TokenDropScheduler,
        get_resident_kv_state,
    )
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestState,
        LMCacheMPRequestTracker,
        LMCacheMPWorkerMetadata,
    )

    manager = KVCacheManager(
        kv_cache_config=KVCacheConfig(
            num_blocks=4,
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
        ),
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=False,
    )
    pool = manager.block_pool
    baseline = pool.get_num_free_blocks()
    assert baseline == 3  # One native null block is reserved.
    a = Request(
        request_id="native-ownership-td",
        prompt_token_ids=list(range(48)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "lmcache.token_drop": {"algorithm": "rkv", "config": {}}
                }
            },
        ),
        pooling_params=None,
    )
    b = Request(
        request_id="native-ownership-normal",
        prompt_token_ids=list(range(32)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    tracker = LMCacheMPRequestTracker(a, enable_token_drop=True)
    tracker.state = LMCacheMPRequestState.BYPASS_LMCACHE
    tracker.num_scheduled_tokens = 48
    tracker.token_drop_generation = 1
    td_scheduler = TokenDropScheduler()
    td_scheduler.bind_block_pool(pool)
    resident_state = get_resident_kv_state(pool)
    connector = SimpleNamespace(
        request_trackers={a.request_id: tracker},
        _get_request_tracker={a.request_id: tracker}.__getitem__,
        _token_drop_scheduler=td_scheduler,
        _gpu_block_pool=pool,
        _group_tokens_per_block=[16],
        _mamba_relocation_window=0,
    )
    resident_state.clear(a.request_id)
    try:
        assert manager.allocate_slots(a, 48) is not None
        assert pool.get_num_free_blocks() == 0
        native_manager = manager.coordinator.single_type_managers[0]
        native_list = native_manager.req_to_blocks[a.request_id]
        LMCacheMPConnector.update_state_after_alloc(
            connector, a, manager.get_blocks(a.request_id), 0
        )
        retained, *tail = native_list
        assert len(tail) == 2

        td_scheduler.commit_completed_updates(
            LMCacheMPWorkerMetadata(
                completed_store_requests={},
                resident_kv_updates={a.request_id: (1, 16)},
            ),
            connector.request_trackers,
            pool,
            16,
        )
        assert native_manager.req_to_blocks[a.request_id] is native_list
        assert native_list == [retained]
        assert tracker.allocated_block_ids[0] == [retained.block_id]
        assert resident_state.get(a.request_id) == 16
        assert all(block.ref_cnt == 0 for block in tail)
        assert pool.get_num_free_blocks() == 2

        assert manager.allocate_slots(b, 32) is not None
        b_blocks = native_manager.req_to_blocks[b.request_id]
        assert {block.block_id for block in b_blocks} == {
            block.block_id for block in tail
        }
        assert all(block.ref_cnt == 1 for block in b_blocks)
        assert pool.get_num_free_blocks() == 0

        manager.free(a)
        assert a.request_id not in native_manager.req_to_blocks
        assert all(block.ref_cnt == 1 for block in b_blocks)
        assert pool.get_num_free_blocks() == 1
        manager.free(b)
        assert b.request_id not in native_manager.req_to_blocks
        assert all(block.ref_cnt == 0 for block in b_blocks)
        assert pool.get_num_free_blocks() == baseline
    finally:
        resident_state.clear(a.request_id)
