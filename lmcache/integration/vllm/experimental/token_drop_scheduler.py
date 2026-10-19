# SPDX-License-Identifier: Apache-2.0
"""Experimental scheduler lifecycle, physical KV state and reclaim."""

from __future__ import annotations

from lmcache.integration.vllm.experimental.token_drop import (
    LMCacheMPTokenDropRequestState,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPWorkerMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput


_resident_kv_tokens: dict[str, int] = {}


def get_resident_kv_tokens(request_id: str) -> int | None:
    return _resident_kv_tokens.get(request_id)


def set_resident_kv_tokens(request_id: str, num_tokens: int) -> None:
    _resident_kv_tokens[request_id] = num_tokens


def clear_resident_kv_tokens(request_id: str) -> None:
    _resident_kv_tokens.pop(request_id, None)


def require_private_token_drop_blocks(self, request_id: str):
    allocation = self._token_drop_allocations.get(request_id)
    if allocation is None or len(allocation.blocks) != 1:
        raise RuntimeError(
            f"Token dropping requires one registered KV group: {request_id}"
        )
    blocks = allocation.blocks[0]
    if any(block.ref_cnt != 1 or block.block_hash is not None for block in blocks):
        raise RuntimeError(
            f"Token dropping requires exclusively owned, unhashed blocks: {request_id}"
        )
    return blocks


def commit_token_drop_resident_updates(self, updates: dict[str, int]) -> None:
    """Commit only completed worker updates; return private tail blocks."""
    if not updates:
        return
    if self._gpu_block_pool is None:
        raise RuntimeError("Missing GPU KV block pool")
    block_size = self._group_tokens_per_block[0]
    # Validate all reports BEFORE freeing any blocks.
    planned = []
    for request_id, physical_length in updates.items():
        tracker = self._get_request_tracker(request_id)
        if tracker.token_drop_spec is None:
            raise RuntimeError("Resident update for non-token-drop request")
        blocks = require_private_token_drop_blocks(self, request_id)
        current_length = get_resident_kv_tokens(request_id)
        if current_length is None:
            current_length = tracker.num_scheduled_tokens
        if not 0 < physical_length <= current_length:
            raise ValueError(
                f"Invalid post-compaction resident length {physical_length} "
                f"for {request_id} (current={current_length})"
            )
        count = (physical_length + block_size - 1) // block_size
        if count > len(blocks):
            raise ValueError("Resident length exceeds allocated capacity")
        planned.append((request_id, tracker, blocks, physical_length, count))

    for request_id, tracker, blocks, physical_length, count in planned:
        tail = blocks[count:]
        del blocks[count:]
        if tail:
            self._gpu_block_pool.free_blocks(reversed(tail))
        tracker.allocated_block_ids[0] = [block.block_id for block in blocks]
        set_resident_kv_tokens(request_id, physical_length)


def reset_token_drop_request(self, request_id: str) -> None:
    clear_resident_kv_tokens(request_id)
    self._token_drop_allocations.pop(request_id, None)
    self._token_drop_finished_before_output.add(request_id)


def reset_preempted_token_drop_requests(
    self, scheduler_output: SchedulerOutput, metadata: LMCacheMPConnectorMetadata
) -> None:
    for request_id in scheduler_output.preempted_req_ids or ():
        if request_id in self._token_drop_allocations:
            if request_id in scheduler_output.num_scheduled_tokens:
                # vLLM can reallocate a preempted request in the same step.
                # Preserve its NEW blocks; only discard its old physical frontier.
                clear_resident_kv_tokens(request_id)
            else:
                reset_token_drop_request(self, request_id)
            metadata.token_drop_reset_ids.add(request_id)


def commit_completed_token_drop_updates(self, worker_meta) -> None:
    # Drop reports for requests already finished or preempted before output.
    stale_ids = self._token_drop_finished_before_output
    self._token_drop_finished_before_output = set()
    if not isinstance(worker_meta, LMCacheMPWorkerMetadata):
        return
    updates = {
        request_id: length
        for request_id, length in worker_meta.resident_kv_updates.items()
        if request_id not in stale_ids
    }
    commit_token_drop_resident_updates(self, updates)


def admit_token_drop_request(self, request) -> None:
    """Validate supported TD mode and isolate vLLM prefix-cache access."""
    cfg = self._vllm_config
    if getattr(cfg.scheduler_config, "enable_chunked_prefill", False):
        raise ValueError("Token dropping does not support chunked prefill")
    if getattr(cfg, "speculative_config", None) is not None:
        raise ValueError("Token dropping does not support speculative decoding")
    if getattr(cfg.scheduler_config, "async_scheduling", False):
        raise ValueError("Token dropping requires synchronous scheduling")
    if cfg.parallel_config.world_size != 1:
        raise ValueError("Token dropping requires a single GPU")
    if len(self._group_tokens_per_block) != 1:
        raise ValueError("Token dropping requires one KV cache group")
    # Install vLLM-private hooks only after request admission succeeds.
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_allocator_adapter,
        install_token_drop_apc_guard,
    )

    install_token_drop_apc_guard()
    install_token_drop_allocator_adapter()
    request.skip_reading_prefix_cache = True


def add_token_drop_states(
    self, scheduler_output: SchedulerOutput, metadata: LMCacheMPConnectorMetadata
) -> None:
    # Publish scheduler-owned resident P and private block IDs to the worker.
    request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs] + list(
        scheduler_output.scheduled_cached_reqs.req_ids
    )
    for request_id in request_ids:
        tracker = self.request_trackers.get(request_id)
        if tracker is None or tracker.token_drop_spec is None:
            continue
        scheduled = int(scheduler_output.num_scheduled_tokens[request_id])
        if scheduled <= 0:
            raise RuntimeError("Token-drop step must schedule at least one token")
        blocks = require_private_token_drop_blocks(self, request_id)
        resident = get_resident_kv_tokens(request_id)
        has_override = resident is not None
        if resident is None:
            resident = tracker.num_scheduled_tokens
        if resident < scheduled:
            raise RuntimeError("Token-drop resident frontier precedes scheduled KV")
        logical_before = tracker.num_scheduled_tokens - scheduled
        metadata.token_drop_requests.append(
            LMCacheMPTokenDropRequestState(
                request_id=request_id,
                algorithm=tracker.token_drop_spec.algorithm,
                config=dict(tracker.token_drop_spec.config),
                resident_kv_tokens=resident,
                physical_block_ids=(
                    [block.block_id for block in blocks] if has_override else None
                ),
                is_genuine_decode=(
                    logical_before >= tracker.num_prompt_tokens and scheduled == 1
                ),
                num_decoded_tokens=max(0, logical_before - tracker.num_prompt_tokens),
                num_new_tokens=scheduled,
            )
        )
