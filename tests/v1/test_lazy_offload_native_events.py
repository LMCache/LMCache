# SPDX-License-Identifier: Apache-2.0
"""Native STORE descriptors survive deferred, coalesced offload."""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

# Third Party
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_manager import LazyOffloadManager
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)
from lmcache.v1.multiprocess.custom_types import EventBinding, KVEventProfile


@pytest.mark.parametrize("parts", [1, 2, 3])
@pytest.mark.parametrize("native_events", [False, True])
def test_lazy_offload_preserves_native_bindings(
    parts: int, native_events: bool
) -> None:
    """The emitted STORE keeps every chunk's hashes, parent and original tokens."""
    init_none_hash(sha256)
    block_size, chunk_size = 16, 64
    request = Request(
        request_id="lazy-native",
        prompt_token_ids=list(range(parts * chunk_size)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )
    pool = BlockPool(
        32, enable_caching=True, hash_block_size=block_size, enable_kv_cache_events=True
    )
    blocks = pool.get_new_blocks(parts * chunk_size // block_size)
    pool.cache_full_blocks(request, blocks, 0, len(blocks), block_size, 0)
    tracker = LMCacheMPRequestTracker(request)
    tracker.native_block_hashes = request.block_hashes
    tracker.append_block_ids(([block.block_id for block in blocks],))
    profile = KVEventProfile("native-test", block_size) if native_events else None
    completion_tracker = Mock()
    completion_tracker.update_pending_store_count.return_value = True
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "FIFO",
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 1,
        },
        [block_size],
        completion_tracker,
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived(request.request_id)
    bindings: list[EventBinding] = []
    for _ in range(parts):
        tracker.increase_num_scheduled_tokens(chunk_size)
        metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            chunk_size,
            [block_size],
            event_profile=profile,
            hash_block_size=block_size,
        )
        assert metadata is not None
        bindings.extend(metadata.op.event_bindings)
        manager.add_store_candidate(metadata)
    manager.on_request_finished(request.request_id)
    step = SimpleNamespace(
        total_num_scheduled_tokens=1,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
    )
    try:
        (merged,) = manager.on_scheduler_step(
            cast(SchedulerOutput, step)
        ).stores_to_submit
        assert (merged.op.start, merged.op.end) == (0, parts * chunk_size)
        assert merged.op.block_ids == [[block.block_id for block in blocks]]
        assert merged.op.event_bindings == tuple(bindings)
        assert len(bindings) == (parts if native_events else 0)
    finally:
        manager.on_store_results(set(), {request.request_id: 1})
