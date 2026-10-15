# SPDX-License-Identifier: Apache-2.0
"""Existing request cap through real metadata, lazy managers and CUDA copies."""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
import argparse
import hashlib
import time

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="Requires actual vLLM request and block-pool APIs")

# Third Party
from vllm import SamplingParams  # noqa: E402
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

# First Party
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadManager,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)
from lmcache.v1.distributed.api import (  # noqa: E402
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.config import (  # noqa: E402
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import StorageManager  # noqa: E402
from tests.v1.distributed.utils import single_row_spec  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator
MODES = ["eager", "FIFO", "EVICTION_AWARE"]
CAPS = [0, 3, 8, 19, None]
CHUNK = 8
BLOCK = 4
TOKENS = 40


class SingleWorkerReceipts:
    """A test worker reports one terminal receipt per submitted batch."""

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Accept exactly one worker receipt; reject an invalid test protocol."""
        assert request_id == "prefix-cap"
        assert count == 1
        return True


@dataclass
class PlannedStores:
    """The real scheduler-side objects kept alive until copies complete."""

    pool: BlockPool
    blocks: list[KVCacheBlock]
    manager: LazyOffloadManager | None
    stores: list[LMCacheMPRequestMetadata]

    def complete(self) -> None:
        """Apply the worker receipt or release the eager request's ownership."""
        if self.manager is None:
            self.pool.free_blocks(self.blocks)
        elif self.stores:
            actions = self.manager.on_store_results(set(), {"prefix-cap": 1})
            assert actions.sessions_to_end == ["prefix-cap"]
            assert not self.manager.has_inflight_store_work()
        assert self.pool.get_num_free_blocks() == self.pool.num_gpu_blocks - 1


def scheduler_step(tokens: int = BLOCK) -> SchedulerOutput:
    """Supply only the scheduler fields consumed by the real lazy manager."""
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            total_num_scheduled_tokens=tokens,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
        ),
    )


def plan_stores(mode: str, cap: int | None, reset: bool = False) -> PlannedStores:
    """Produce chunked-prefill candidates and dispatch via the selected path.

    Uses real vLLM Request, BlockPool and cache-block hashes. Only the scheduler
    output envelope and one-worker receipt transport are test doubles.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=BLOCK)
    blocks = pool.get_new_blocks(TOKENS // BLOCK)
    for index, block in enumerate(blocks):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(index.to_bytes(8, "big")), 0)
        )
    request_configs = {} if cap is None else {"lmcache.max_offload_tokens": cap}
    request = Request(
        request_id="prefix-cap",
        prompt_token_ids=list(range(TOKENS)),
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"kv_transfer_params": request_configs}
        ),
        pooling_params=None,
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [block.block_id for block in blocks]}
    candidates = []
    for computed in (4, 16, TOKENS):
        tracker.num_scheduled_tokens = computed
        metadata = LMCacheMPRequestMetadata.GetStoreMetadata(tracker, CHUNK, [BLOCK])
        if metadata is not None:
            candidates.append(metadata)
    admitted_tokens = TOKENS if cap is None else min(cap, TOKENS) // CHUNK * CHUNK
    assert tracker.num_stored_tokens == admitted_tokens
    assert sum(meta.op.end - meta.op.start for meta in candidates) == admitted_tokens
    assert [meta.op.start for meta in candidates] == [
        0,
        *[meta.op.end for meta in candidates[:-1]],
    ][: len(candidates)]
    assert LMCacheMPRequestMetadata.GetStoreMetadata(tracker, CHUNK, [BLOCK]) is None
    if mode == "eager":
        return PlannedStores(pool, blocks, None, candidates)
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": mode,
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 1,
            "lmcache.mp.lazy_offload_horizon_steps": 1000.0,
        },
        [BLOCK],
        SingleWorkerReceipts(),
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived(request.request_id)
    for metadata in candidates:
        manager.add_store_candidate(metadata)
    if reset:
        manager.on_request_reset(request.request_id)
    manager.on_request_finished(request.request_id)
    pool.free_blocks(blocks)
    assert manager.on_scheduler_step(scheduler_step(0)).stores_to_submit == []
    stores = manager.on_scheduler_step(scheduler_step()).stores_to_submit
    assert sum(meta.op.end - meta.op.start for meta in stores) == (
        0 if reset else admitted_tokens
    )
    for metadata in stores:
        assert metadata.op.start == 0
        assert metadata.op.end == admitted_tokens
        assert metadata.op.block_ids == [
            [b.block_id for b in blocks[: admitted_tokens // BLOCK]]
        ]
    return PlannedStores(pool, blocks, manager, stores)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cap", CAPS)
def test_cap_survives_metadata_and_eager_or_lazy_dispatch(
    mode: str, cap: int | None
) -> None:
    planned = plan_stores(mode, cap)
    planned.complete()


@pytest.mark.parametrize("mode", ["FIFO", "EVICTION_AWARE"])
def test_preemption_drops_capped_pending_operations(mode: str) -> None:
    planned = plan_stores(mode, 19, reset=True)
    assert planned.stores == []
    planned.complete()


@pytest.mark.cuda
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cap", [0, 8, 19, None])
def test_cap_controls_actual_cuda_and_storage_manager_payload(
    mode: str, cap: int | None
) -> None:
    """Only the selected prefix crosses CUDA/L1; the excluded suffix is untouched.

    The scheduler objects and StorageManager are real. Torch copies stand in
    for the MP worker/IPC executor; this is not a model-serving benchmark.
    """
    if not torch.cuda.is_available():
        pytest.skip("Requires actual CUDA transfers")
    planned = plan_stores(mode, cap)
    parser = argparse.ArgumentParser()
    add_storage_manager_args(parser)
    config = parse_args_to_config(
        parser.parse_args(
            ["--l1-size-gb", "0.0625", "--no-l1-use-lazy", "--eviction-policy", "noop"]
        )
    )
    storage = StorageManager(config)
    try:
        expected = torch.arange(TOKENS * 2, dtype=torch.float32, device="cuda").view(
            TOKENS, 2
        )
        pages = torch.full((16, BLOCK, 2), -2.0, device="cuda")
        for index, block in enumerate(planned.blocks):
            pages[block.block_id].copy_(expected[index * BLOCK : (index + 1) * BLOCK])
        layout = MemoryLayoutDesc([torch.Size([CHUNK, 2])], [torch.float32])
        keys = [
            ObjectKey(hashlib.sha256(bytes(range(end))).digest(), "prefix-cap", 0)
            for end in range(CHUNK, TOKENS + 1, CHUNK)
        ]
        copied_bytes = 0
        for metadata in planned.stores:
            op = metadata.op
            payload = torch.cat([pages[block_id] for block_id in op.block_ids[0]])
            selected_keys = keys[op.start // CHUNK : op.end // CHUNK]
            writable = storage.reserve_write(selected_keys, layout)
            assert set(writable) == set(selected_keys)
            for index, key in enumerate(selected_keys):
                target = writable[key].tensor
                assert target is not None
                target.copy_(
                    payload[index * CHUNK : (index + 1) * CHUNK], non_blocking=True
                )
                copied_bytes += target.numel() * target.element_size()
            torch.cuda.synchronize()
            storage.finish_write(selected_keys)
        planned.complete()
        admitted = TOKENS if cap is None else min(cap, TOKENS) // CHUNK * CHUNK
        assert copied_bytes == admitted * 2 * 4
        handle = storage.submit_prefetch_task(single_row_spec(keys, layout))
        deadline = time.monotonic() + 15
        result = None
        while result is None and time.monotonic() < deadline:
            result = storage.query_prefetch_status(handle)
            if result is None:
                time.sleep(0.01)
        assert result is not None
        assert result.hit_cells[0].popcount() == admitted // CHUNK
        restored = torch.full_like(expected, -1)
        found_keys = keys[: admitted // CHUNK]
        if found_keys:
            with storage.read_prefetched_results(
                found_keys, result.l1_owners
            ) as objects:
                assert objects is not None
                for index, obj in enumerate(objects):
                    assert obj.tensor is not None
                    restored[index * CHUNK : (index + 1) * CHUNK].copy_(
                        obj.tensor, non_blocking=True
                    )
                torch.cuda.synchronize()
            storage.finish_read_prefetched(found_keys, l1_owners=result.l1_owners)
            assert storage.delete_l1_keys(found_keys) == (len(found_keys), 0)
        torch.testing.assert_close(
            restored[:admitted], expected[:admitted], rtol=0, atol=0
        )
        assert torch.all(restored[admitted:] == -1)
        assert storage.report_status()["is_healthy"]
    finally:
        storage.close()
