# SPDX-License-Identifier: Apache-2.0
"""Existing request cap through real metadata, lazy managers and CUDA copies."""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import MagicMock
import argparse
import time

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="Requires actual vLLM request and block-pool APIs")

# Third Party
from vllm import SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheBlocks  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPWorkerMetadata,
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
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402
from tests.v1.distributed.utils import single_row_spec  # noqa: E402
import lmcache.integration.vllm.lmcache_mp_connector as connector_module  # noqa: E402

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
    connector: LMCacheMPConnector
    stores: list[LMCacheMPRequestMetadata]

    def complete(self) -> None:
        """Apply the worker receipt or release the eager request's ownership."""
        adapter = cast(MagicMock, self.connector.scheduler_adapter)
        if not self.connector.lazy_offload:
            self.pool.free_blocks(self.blocks)
        elif self.stores:
            adapter.end_session.assert_not_called()
            assert self.connector.has_pending_push_work()
            self.connector.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                        completed_store_requests={"prefix-cap": 1}
                    )
                )
            )
        adapter.end_session.assert_called_once_with("prefix-cap")
        assert "prefix-cap" not in self.connector.request_trackers
        assert not self.connector.has_pending_push_work()
        assert all(block.ref_cnt == 0 for block in self.blocks)
        assert self.pool.get_num_free_blocks() == self.pool.num_gpu_blocks - 1
        self.connector.shutdown()
        adapter.shutdown.assert_called_once_with()


def scheduler_step(
    tokens: int = BLOCK,
    path: Literal["idle", "new", "cached"] = "idle",
    block_ids: tuple[list[int], ...] = (),
) -> SchedulerOutput:
    """Supply the scheduler fields consumed by the real connector entry point."""
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            total_num_scheduled_tokens=tokens,
            scheduled_new_reqs=(
                [SimpleNamespace(req_id="prefix-cap", block_ids=block_ids)]
                if path == "new"
                else []
            ),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["prefix-cap"] if path == "cached" else [],
                new_block_ids=[None] if path == "cached" else [],
                resumed_req_ids=set(),
            ),
            num_scheduled_tokens={"prefix-cap": tokens} if path != "idle" else {},
            preempted_req_ids=set(),
        ),
    )


def plan_stores(
    mode: str,
    cap: int | None,
    monkeypatch: pytest.MonkeyPatch,
    reset: bool = False,
    first_step: int = 4,
) -> PlannedStores:
    """Produce chunked-prefill candidates and dispatch via the selected path.

    Uses the public connector constructor and scheduler entry points. Config,
    scheduler output and the adapter transport are test doubles, not dispatch.
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
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK
    adapter.check_lookup_result.return_value = 0
    adapter.update_pending_store_count.side_effect = (
        SingleWorkerReceipts().update_pending_store_count
    )
    monkeypatch.setattr(
        connector_module, "LMCacheMPSchedulerAdapter", lambda **kwargs: adapter
    )
    config = cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(model="prefix-cap", use_mla=False),
            cache_config=SimpleNamespace(
                block_size=BLOCK, enable_prefix_caching=True, mamba_cache_mode="none"
            ),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            kv_events_config=None,
            speculative_config=None,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheMPConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "lmcache.mp.lazy_offload": mode != "eager",
                    "lmcache.mp.lazy_offload_policy": mode,
                    "lmcache.mp.lazy_offload_threshold": 1,
                    "lmcache.mp.lazy_offload_select_count": 1,
                    "lmcache.mp.lazy_offload_horizon_steps": 1000.0,
                },
            ),
        ),
    )
    connector = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    connector.bind_gpu_block_pool(pool)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(request, KVCacheBlocks((blocks,)), 0)
    stores = []
    previous_computed = previous_stored = 0
    for index, computed in enumerate(
        [first_step, *[value for value in (16, TOKENS) if value > first_step]]
    ):
        step = scheduler_step(
            computed - previous_computed,
            path="new" if index == 0 else "cached",
            block_ids=([block.block_id for block in blocks],),
        )
        output = connector.build_connector_meta(step)
        assert isinstance(output, LMCacheMPConnectorMetadata)
        expected_end = min(computed, TOKENS if cap is None else cap) // CHUNK * CHUNK
        tracker = connector.request_trackers[request.request_id]
        assert tracker.num_stored_tokens == expected_end
        if mode == "eager":
            assert [(meta.op.start, meta.op.end) for meta in output.requests] == (
                [(previous_stored, expected_end)]
                if expected_end > previous_stored
                else []
            )
            stores.extend(output.requests)
        else:
            assert output.requests == []
        previous_computed, previous_stored = computed, expected_end
    admitted_tokens = TOKENS if cap is None else min(cap, TOKENS) // CHUNK * CHUNK
    assert tracker.num_stored_tokens == admitted_tokens
    if reset:
        request.status = RequestStatus.PREEMPTED
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    retain_for_worker, return_params = connector.request_finished(
        request, [block.block_id for block in blocks]
    )
    assert retain_for_worker is (mode == "eager")
    assert return_params is None
    assert request.request_id not in connector.request_trackers
    if connector.lazy_offload:
        pool.free_blocks(blocks)
        idle_output = connector.build_connector_meta(scheduler_step(0))
        assert isinstance(idle_output, LMCacheMPConnectorMetadata)
        assert idle_output.requests == []
        drain_output = connector.build_connector_meta(scheduler_step())
        assert isinstance(drain_output, LMCacheMPConnectorMetadata)
        stores = drain_output.requests
    if connector.lazy_offload and stores:
        adapter.end_session.assert_not_called()
    else:
        adapter.end_session.assert_called_once_with(request.request_id)
    assert sum(meta.op.end - meta.op.start for meta in stores) == (
        0 if reset else admitted_tokens
    )
    for metadata in stores:
        assert metadata.op.token_ids == list(range(TOKENS))
        assert metadata.op.block_ids == [
            [
                b.block_id
                for b in blocks[metadata.op.start // BLOCK : metadata.op.end // BLOCK]
            ]
        ]
    pinned = {block_id for meta in stores for block_id in meta.op.flat_block_ids}
    for block in blocks:
        assert block.ref_cnt == (
            1 if mode == "eager" or block.block_id in pinned else 0
        )
    competing = pool.get_new_blocks(pool.get_num_free_blocks())
    assert pinned.isdisjoint(block.block_id for block in competing)
    pool.free_blocks(competing)
    return PlannedStores(pool, blocks, connector, stores)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cap", CAPS)
@pytest.mark.parametrize("first_step", [4, 16])
def test_cap_survives_metadata_and_eager_or_lazy_dispatch(
    mode: str, cap: int | None, first_step: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = plan_stores(mode, cap, monkeypatch, first_step=first_step)
    planned.complete()


@pytest.mark.parametrize("mode", ["FIFO", "EVICTION_AWARE"])
def test_preemption_drops_capped_pending_operations(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = plan_stores(mode, 19, monkeypatch, reset=True)
    assert planned.stores == []
    planned.complete()


@pytest.mark.cuda
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cap", [0, 8, 19, None])
def test_cap_controls_actual_cuda_and_storage_manager_payload(
    mode: str, cap: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the selected prefix crosses CUDA/L1; the excluded suffix is untouched.

    The connector, Request, BlockPool and StorageManager are real, with mocked
    configuration/scheduler envelopes and adapter transport. Torch copies
    replace the MP worker/IPC executor; this is not a serving benchmark.
    """
    if not torch.cuda.is_available():
        pytest.skip("Requires actual CUDA transfers")
    planned = plan_stores(mode, cap, monkeypatch)
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
        hasher = TokenHasher(chunk_size=CHUNK)
        keys = [
            ObjectKey(value, "prefix-cap", 0)
            for value in hasher.compute_chunk_hashes(list(range(TOKENS)))
        ]
        copied_bytes = 0
        for metadata in planned.stores:
            op = metadata.op
            payload = torch.cat([pages[block_id] for block_id in op.block_ids[0]])
            emitted_hashes = hasher.compute_chunk_hashes(op.token_ids)
            selected_keys = [
                ObjectKey(value, "prefix-cap", 0)
                for value in emitted_hashes[op.start // CHUNK : op.end // CHUNK]
            ]
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
