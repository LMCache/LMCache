# SPDX-License-Identifier: Apache-2.0
"""Track each request's resident KV length separately from its logical length.

The allocator updates it after successful allocation.
Scheduler owns physical P; completed compaction reports commit it after reclaim.
Each native block pool binds one resident-state instance.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import TYPE_CHECKING
from weakref import ReferenceType, WeakKeyDictionary, ref

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput

# First Party
from lmcache.integration.vllm.experimental.token_drop import (
    LMCacheMPTokenDropRequestState,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestTracker,
    LMCacheMPWorkerMetadata,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class ResidentKVState:
    """Absolute resident frontiers for one synchronous scheduler."""

    def __init__(self) -> None:
        """Initialize the resident override map for one scheduler."""
        self._resident_kv_tokens_by_request_id: dict[str, int] = {}

    def get(self, request_id: str) -> int | None:
        """Return the current resident frontier.

        Args:
            request_id: Request ID in this scheduler's namespace.

        Returns:
            Absolute physical KV token count, or None without an override.
        """
        return self._resident_kv_tokens_by_request_id.get(request_id)

    def set(self, request_id: str, num_tokens: int) -> None:
        """Record an absolute resident length after allocation or compaction.

        Args:
            request_id: Request ID in this scheduler's namespace.
            num_tokens: Validated physical KV length. The caller has
                completed the corresponding native allocation or reclaim.
        """
        self._resident_kv_tokens_by_request_id[request_id] = num_tokens

    def clear(self, request_id: str) -> None:
        """Forget a request's resident override after reset or completion.

        Args:
            request_id: Request ID in this scheduler's namespace.
        """
        self._resident_kv_tokens_by_request_id.pop(request_id, None)


# The process-wide binding table owns neither the pool nor its resident state.
_resident_states_by_pool: WeakKeyDictionary[
    BlockPool, ReferenceType[ResidentKVState]
] = WeakKeyDictionary()
_resident_state_bindings_lock = Lock()


def bind_resident_kv_state(pool: BlockPool, state: ResidentKVState) -> None:
    """Bind one scheduler's resident state to its native block pool.

    Repeated binding of the same pair is harmless. A different owner for the
    same pool, or the same state for another live pool, is rejected.

    Args:
        pool: Native allocator identity.
        state: Scheduler-owned resident state.

    Raises:
        RuntimeError: Conflicting live binding.
    """
    with _resident_state_bindings_lock:
        current_ref = _resident_states_by_pool.get(pool)
        current = current_ref() if current_ref is not None else None
        if current is state:
            return
        if current is not None or any(
            existing() is state for existing in _resident_states_by_pool.values()
        ):
            raise RuntimeError("Resident KV state or block pool already bound")
        _resident_states_by_pool[pool] = ref(state)


def get_resident_kv_state(pool: BlockPool) -> ResidentKVState | None:
    """Resolve the live resident state without transferring ownership.

    Args:
        pool: Native allocator's block pool.

    Returns:
        Bound scheduler state, or None if this pool has no live TD owner.
    """
    with _resident_state_bindings_lock:
        state_ref = _resident_states_by_pool.get(pool)
        return state_ref() if state_ref is not None else None


def _unbind_resident_kv_state(pool: BlockPool, state: ResidentKVState) -> None:
    """Remove only the caller's binding; never unbind a replacement owner."""
    with _resident_state_bindings_lock:
        state_ref = _resident_states_by_pool.get(pool)
        if state_ref is not None and state_ref() is state:
            del _resident_states_by_pool[pool]


def admit_token_drop_request(
    request: Request,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig | None,
    group_tokens_per_block: Sequence[int],
) -> None:
    """Validate supported TD mode and isolate vLLM prefix-cache access.

    Args:
        request: TD-tagged native request, before prefix lookup or allocation.
        vllm_config: Engine execution and backend configuration.
        kv_cache_config: Resolved native KV configuration.
        group_tokens_per_block: Token capacity per native KV group.

    Raises:
        ValueError: Unsupported request execution or KV layout.
        RuntimeError: Unsupported pinned vLLM version.
    """
    cfg = vllm_config
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
    if len(group_tokens_per_block) != 1:
        raise ValueError("Token dropping requires one KV cache group")

    # Do not infer full attention from a single group: sliding/local KV
    # managers evict blocks that TD requires to remain privately resident.
    # Third Party
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVQuantMode,
        UniformTypeKVCacheSpecs,
    )

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
    # Install vLLM-private hooks only after request admission succeeds.
    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_allocator_adapter,
        install_token_drop_prefix_caching_publication_guard,
    )

    install_token_drop_prefix_caching_publication_guard()
    install_token_drop_allocator_adapter()
    request.skip_reading_prefix_cache = True


class TokenDropScheduler:
    """Own request-local TD scheduling state, not the native allocator.

    One instance belongs to one scheduler connector. The native allocation
    references are borrowed and their inner block lists must never be copied.
    """

    def __init__(self) -> None:
        """Initialize independent allocation, generation, finish and resident state."""
        self._resident_state = ResidentKVState()
        self._token_drop_allocations: dict[str, KVCacheBlocks] = {}
        self._token_drop_next_generation = 0
        self._token_drop_finished_before_output: set[str] = set()
        self._token_drop_pending_finished_ids: set[str] = set()

    def bind_block_pool(self, pool: BlockPool) -> None:
        """Associate the allocator's pool with this scheduler's resident state.

        Args:
            pool: The native BlockPool supplied by the vLLM scheduler.

        Raises:
            RuntimeError: The pool or resident state has another live binding.
        """
        bind_resident_kv_state(pool, self._resident_state)

    def unbind_block_pool(self, pool: BlockPool) -> None:
        """Release this scheduler's binding on shutdown; idempotent.

        Args:
            pool: Previously bound native BlockPool. A replacement owner's
                binding is left untouched.
        """
        _unbind_resident_kv_state(pool, self._resident_state)

    def next_generation(self) -> int:
        """Allocate the next generation for a newly created TD request tracker.

        Returns:
            Positive, monotonically increasing generation within this scheduler.
        """
        self._token_drop_next_generation += 1
        return self._token_drop_next_generation

    def register_allocation(self, request_id: str, blocks: KVCacheBlocks) -> None:
        """Validate and borrow the native allocation, preserving list identity.

        Args:
            request_id: TD request receiving this allocation.
            blocks: All native blocks for that request, not just newly
                allocated blocks. Inner lists must not be copied.

        Raises:
            RuntimeError: Missing, empty, shared or hashed native allocation.
        """
        self._validate_private_blocks(blocks, request_id)
        self._token_drop_allocations[request_id] = blocks

    def finish_request(self, request_id: str) -> None:
        """Clear physical state and queue a worker finish notification.

        Args:
            request_id: Finished or aborted TD request.
        """
        self._reset_request(request_id)
        self._token_drop_pending_finished_ids.add(request_id)

    def prepare_step(
        self, scheduler_output: SchedulerOutput, metadata: LMCacheMPConnectorMetadata
    ) -> None:
        """Publish finish notifications and reset preempted TD requests.

        Args:
            scheduler_output: Current native schedule, including preemptions.
            metadata: This step's connector metadata, populated in place.
        """
        metadata.token_drop_finished_ids = self._token_drop_pending_finished_ids
        self._token_drop_pending_finished_ids = set()
        for request_id in scheduler_output.preempted_req_ids or ():
            if request_id in self._token_drop_allocations:
                if request_id in scheduler_output.num_scheduled_tokens:
                    # vLLM can reallocate a preempted request in the same step.
                    # Preserve its NEW blocks; only discard its old physical frontier.
                    self._resident_state.clear(request_id)
                else:
                    self._reset_request(request_id)
                metadata.token_drop_reset_ids.add(request_id)

    def commit_completed_updates(
        self,
        worker_meta: object,
        request_trackers: Mapping[str, LMCacheMPRequestTracker],
        block_pool: BlockPool | None,
        block_size: int,
    ) -> None:
        """Consume generation-qualified reports and reclaim native private tails.

        Args:
            worker_meta: Completed worker metadata, if present.
            request_trackers: Connector-owned trackers for current requests.
            block_pool: Native pool bound to this scheduler.
            block_size: Tokens per block in the supported single KV group.

        Raises:
            RuntimeError: Invalid native ownership or missing pool binding.
            ValueError: Invalid resident length or allocated capacity.
        """
        # Drop reports for requests already finished or preempted before output.
        stale_ids = self._token_drop_finished_before_output
        self._token_drop_finished_before_output = set()
        if not isinstance(worker_meta, LMCacheMPWorkerMetadata):
            return
        updates = {}
        for request_id, (generation, length) in worker_meta.resident_kv_updates.items():
            if request_id in stale_ids:
                continue
            tracker = request_trackers.get(request_id)
            if tracker is None or (
                tracker.token_drop_spec is None
                or tracker.token_drop_generation != generation
            ):
                # A finished, preempted, or reused request id has no claim on
                # blocks owned by the current generation.
                continue
            updates[request_id] = length
        self._commit_resident_updates(updates, request_trackers, block_pool, block_size)

    def populate_request_states(
        self,
        scheduler_output: SchedulerOutput,
        request_trackers: Mapping[str, LMCacheMPRequestTracker],
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        """Publish physical frontiers and scheduled TD request facts.

        Args:
            scheduler_output: Current native schedule.
            request_trackers: Connector-owned trackers with current progress.
            metadata: Step metadata, receiving scheduled TD request states.

        Raises:
            RuntimeError: Invalid frontier, schedule or native private allocation.
        """
        # Publish scheduler-owned resident P and private block IDs to the worker.
        request_ids = [
            req.req_id for req in scheduler_output.scheduled_new_reqs
        ] + list(scheduler_output.scheduled_cached_reqs.req_ids)
        for request_id in request_ids:
            tracker = request_trackers.get(request_id)
            if tracker is None or tracker.token_drop_spec is None:
                continue
            scheduled = int(scheduler_output.num_scheduled_tokens[request_id])
            if scheduled <= 0:
                raise RuntimeError("Token-drop step must schedule at least one token")
            blocks = self._require_private_blocks(request_id)
            resident = self._resident_state.get(request_id)
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
                        if logical_before >= tracker.num_prompt_tokens
                        and scheduled == 1
                        else "prefill"
                    ),
                    generation=tracker.token_drop_generation,
                    num_decoded_tokens=max(
                        0, logical_before - tracker.num_prompt_tokens
                    ),
                    num_new_tokens=scheduled,
                )
            )

    def _require_private_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self._validate_private_blocks(
            self._token_drop_allocations.get(request_id), request_id
        )

    @staticmethod
    def _validate_private_blocks(
        allocation: KVCacheBlocks | None, request_id: str
    ) -> list[KVCacheBlock]:
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
                "Token dropping requires exclusively owned, unhashed blocks: "
                f"{request_id}"
            )
        return blocks

    def _commit_resident_updates(
        self,
        updates: dict[str, int],
        request_trackers: Mapping[str, LMCacheMPRequestTracker],
        block_pool: BlockPool | None,
        block_size: int,
    ) -> None:
        """Commit completed worker updates and return private tail blocks.

        The registered allocation's inner block list must be the same mutable list
        owned by vLLM's request-to-block mapping, not a copy. Trim it in place before
        freeing the tail so native request teardown cannot free reclaimed blocks again.
        """
        if not updates:
            return
        if block_pool is None:
            raise RuntimeError("Missing GPU KV block pool")
        if get_resident_kv_state(block_pool) is not self._resident_state:
            raise RuntimeError("GPU KV block pool is not bound to this TD scheduler")
        # Validate all reports BEFORE freeing any blocks.
        planned = []
        for request_id, physical_length in updates.items():
            tracker = request_trackers[request_id]
            if tracker.token_drop_spec is None:
                raise RuntimeError("Resident update for non-token-drop request")
            blocks = self._require_private_blocks(request_id)
            current_length = self._resident_state.get(request_id)
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
                block_pool.free_blocks(reversed(tail))
            tracker.allocated_block_ids[0] = [block.block_id for block in blocks]
            self._resident_state.set(request_id, physical_length)

    def _reset_request(self, request_id: str) -> None:
        self._resident_state.clear(request_id)
        self._token_drop_allocations.pop(request_id, None)
        self._token_drop_finished_before_output.add(request_id)
