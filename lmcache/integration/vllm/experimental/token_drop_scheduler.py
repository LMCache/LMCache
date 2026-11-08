# SPDX-License-Identifier: Apache-2.0
"""Track each request's resident KV length separately from its logical length.

The allocator updates it after successful allocation.
Scheduler owns physical P, and completed compaction reports update it before reclaim.
"""

# Future
from __future__ import annotations

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPWorkerMetadata,
)

_resident_kv_tokens_by_request_id: dict[str, int] = {}


def get_resident_kv_tokens(request_id: str) -> int | None:
    return _resident_kv_tokens_by_request_id.get(request_id)


def set_resident_kv_tokens(request_id: str, num_tokens: int) -> None:
    _resident_kv_tokens_by_request_id[request_id] = num_tokens


def clear_resident_kv_tokens(request_id: str) -> None:
    _resident_kv_tokens_by_request_id.pop(request_id, None)


def require_private_token_drop_blocks(self, request_id: str):
    allocation = self._token_drop_allocations.get(request_id)
    if allocation is None or len(allocation.blocks) != 1:
        raise RuntimeError(
            f"Token dropping requires one registered KV group: {request_id}"
        )
    blocks = allocation.blocks[0]
    if not blocks:
        raise RuntimeError(
            f"Token dropping requires nonempty native private blocks: {request_id}"
        )
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
    updates = {}
    for request_id, (generation, length) in worker_meta.resident_kv_updates.items():
        if request_id in stale_ids:
            continue
        tracker = self.request_trackers.get(request_id)
        if tracker is None or (
            tracker.token_drop_spec is None
            or tracker.token_drop_generation != generation
        ):
            # A finished, preempted, or reused request id has no claim on
            # blocks owned by the current generation.
            continue
        updates[request_id] = length
    commit_token_drop_resident_updates(self, updates)
