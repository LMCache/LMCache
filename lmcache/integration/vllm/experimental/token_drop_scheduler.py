# SPDX-License-Identifier: Apache-2.0
"""Track each request's resident KV length separately from its logical length.

The allocator updates it after successful allocation.
Scheduler-side updates after compaction and cleanup are handled in follow-up PRs.
"""

_resident_kv_tokens_by_request_id: dict[str, int] = {}


def get_resident_kv_tokens(request_id: str) -> int | None:
    return _resident_kv_tokens_by_request_id.get(request_id)


def set_resident_kv_tokens(request_id: str, num_tokens: int) -> None:
    _resident_kv_tokens_by_request_id[request_id] = num_tokens


def clear_resident_kv_tokens(request_id: str) -> None:
    _resident_kv_tokens_by_request_id.pop(request_id, None)


def require_private_token_drop_blocks(connector, request_id: str):
    """Return exclusively owned, unhashed blocks for one token-drop request."""
    allocation = connector._token_drop_allocations.get(request_id)
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


def commit_token_drop_resident_updates(
    connector, completed_resident_kv_tokens: dict[str, int]
) -> None:
    """Commit completed GPU compactions and reclaim their unused private tails.

    Caller must wait for worker completion before invoking this function.
    """
    if not completed_resident_kv_tokens:
        return
    if connector._gpu_block_pool is None:
        raise RuntimeError("Missing GPU KV block pool")

    block_size = connector._group_tokens_per_block[0]
    planned = []
    # Validate the entire batch before changing any allocation or physical P.
    for request_id, resident_length in completed_resident_kv_tokens.items():
        tracker = connector._get_request_tracker(request_id)
        if tracker.token_drop_spec is None:
            raise RuntimeError("Resident update for non-token-drop request")

        blocks = require_private_token_drop_blocks(connector, request_id)
        previous_length = get_resident_kv_tokens(request_id)
        if previous_length is None:
            previous_length = tracker.num_scheduled_tokens
        if not 0 < resident_length <= previous_length:
            raise ValueError(
                f"Invalid completed resident length {resident_length} for "
                f"{request_id} (previous={previous_length})"
            )

        retained_blocks = (resident_length + block_size - 1) // block_size
        if retained_blocks > len(blocks):
            raise ValueError("Resident length exceeds allocated capacity")
        planned.append((request_id, tracker, blocks, resident_length, retained_blocks))

    for request_id, tracker, blocks, resident_length, retained_blocks in planned:
        tail_blocks = blocks[retained_blocks:]
        del blocks[retained_blocks:]
        if tail_blocks:
            connector._gpu_block_pool.free_blocks(reversed(tail_blocks))
        tracker.allocated_block_ids[0] = [block.block_id for block in blocks]
        # Native vLLM tracks P for allocation; older versions use LMCache state.
        manager = getattr(connector, "_kv_cache_manager", None)
        if manager is not None and hasattr(manager, "set_resident_kv_tokens"):
            manager.set_resident_kv_tokens(request_id, resident_length)
        set_resident_kv_tokens(request_id, resident_length)


def reset_token_drop_request(connector, request_id: str) -> None:
    """Forget physical P and ignore any in-flight completion for this request."""
    clear_resident_kv_tokens(request_id)
    connector._token_drop_allocations.pop(request_id, None)
    connector._token_drop_finished_before_output.add(request_id)


def reset_preempted_token_drop_requests(connector, scheduler_output) -> set[str]:
    """Reset TD state without discarding new same-step reschedule allocations."""
    reset_request_ids: set[str] = set()
    for request_id in scheduler_output.preempted_req_ids or ():
        if request_id not in connector._token_drop_allocations:
            continue

        if request_id in scheduler_output.num_scheduled_tokens:
            # A preempted request can be rescheduled with NEW blocks immediately.
            # Its old P must be cleared, while the new allocation stays intact.
            clear_resident_kv_tokens(request_id)
        else:
            reset_token_drop_request(connector, request_id)
        reset_request_ids.add(request_id)
    return reset_request_ids


def commit_completed_token_drop_updates(
    connector, worker_resident_kv_updates: dict[str, int]
) -> None:
    """Ignore completions invalidated by finish/preemption, then reclaim."""
    finished_before_output = connector._token_drop_finished_before_output
    connector._token_drop_finished_before_output = set()
    completed = {
        request_id: resident_length
        for request_id, resident_length in worker_resident_kv_updates.items()
        if request_id not in finished_before_output
    }
    commit_token_drop_resident_updates(connector, completed)
