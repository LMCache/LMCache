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
from lmcache.integration.vllm.experimental.token_drop import (
    LMCacheMPTokenDropRequestState,
)
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
    if cfg.use_v2_model_runner:
        raise ValueError("Token dropping requires classic GPU model runner (V1)")
    if cfg.parallel_config.use_ubatching:
        raise ValueError("Token dropping does not support microbatching")
    if len(self._group_tokens_per_block) != 1:
        raise ValueError("Token dropping requires one KV cache group")

    # Do not infer full attention from a single group: sliding/local KV
    # managers evict blocks that TD requires to remain privately resident.
    # Third Party
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVQuantMode,
        UniformTypeKVCacheSpecs,
    )

    kv_cache_config = getattr(self, "_kv_cache_config", None)
    if kv_cache_config is None or len(kv_cache_config.kv_cache_groups) != 1:
        raise ValueError("Token dropping requires one resolved KV cache group")
    group_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    specs = (
        tuple(group_spec.kv_cache_specs.values())
        if isinstance(group_spec, UniformTypeKVCacheSpecs)
        else (group_spec,)
    )
    if not specs or any(
        type(spec) is not FullAttentionSpec
        or spec.sliding_window is not None
        or spec.attention_chunk_size is not None
        or spec.non_causal
        or spec.kv_quant_mode != KVQuantMode.NONE
        or spec.head_size_v != spec.head_size
        for spec in specs
    ):
        raise ValueError(
            "Token dropping requires an ordinary full-attention KV cache group"
        )

    # The post-RoPE Q hook and per-request KV addressing require FlashAttention.
    # None means vLLM selects the backend; validate its actual metadata on worker.
    # Third Party
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    backend = cfg.attention_config.backend
    if backend is not None and backend != AttentionBackendEnum.FLASH_ATTN:
        raise ValueError("Token dropping requires FlashAttention backend")
    # Full decode CUDA Graph replay skips the Python post-RoPE Q hook.
    # PIECEWISE leaves attention eager while retaining other graph optimizations.
    # Third Party
    from vllm.config.compilation import CUDAGraphMode

    if cfg.compilation_config.cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
    ):
        raise ValueError(
            "Token dropping requires CUDA Graph mode NONE or PIECEWISE; "
            "FULL decode graphs cannot capture per-step Q"
        )
    # Install vLLM-private hooks only after request admission succeeds.
    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_allocator_adapter,
        install_token_drop_prefix_caching_publication_guard,
    )

    install_token_drop_prefix_caching_publication_guard()
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
                phase=(
                    "decode"
                    if logical_before >= tracker.num_prompt_tokens and scheduled == 1
                    else "prefill"
                ),
                generation=tracker.token_drop_generation,
                num_decoded_tokens=max(0, logical_before - tracker.num_prompt_tokens),
                num_new_tokens=scheduled,
            )
        )
