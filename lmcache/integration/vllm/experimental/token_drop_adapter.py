# SPDX-License-Identifier: Apache-2.0
"""vLLM integration adaptors for token dropping."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

# Third Party
import numpy as np
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)
from lmcache.integration.vllm.utils import extract_request_configs_from_request

if TYPE_CHECKING:
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # First Party
    from lmcache.integration.vllm.experimental.token_drop import (
        LMCacheMPTokenDropRequestState,
    )


_SUPPORTED_VLLM_VERSION = "0.25.1"


def _check_vllm_version() -> None:
    # Third Party
    from vllm.version import __version__ as vllm_version

    # Strip local version labels (+cpu, +cu130, ...) and keep the release only.
    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Token dropping requires vLLM {_SUPPORTED_VLLM_VERSION}, "
            f"got {vllm_version}"
        )


class _ResidentKVRequestView:
    """Let the allocator read resident KV length without changing the request."""

    def __init__(self, request, resident_kv_length):
        self._request = request
        self._resident_kv_length = resident_kv_length

    @property
    def num_computed_tokens(self):
        return self._resident_kv_length

    def __getattr__(self, name):
        """Read other attributes from the original request."""
        return getattr(self._request, name)


def _allocate_with_resident_frontier(
    original, manager, request, num_new_tokens, *args, **kwargs
):
    request_id = request.request_id
    resident_kv_length = get_resident_kv_tokens(request_id)
    if resident_kv_length is None:
        return original(manager, request, num_new_tokens, *args, **kwargs)
    if request.num_computed_tokens == 0 or "lmcache.token_drop" not in (
        extract_request_configs_from_request(request) or {}
    ):
        # A preempted request must recompute from the logical beginning.
        clear_resident_kv_tokens(request_id)
        return original(manager, request, num_new_tokens, *args, **kwargs)

    # These are vLLM 0.25.1 allocate_slots() options. Prefix/external KV,
    # speculative or encoder tokens, and full-sequence admission need
    # extra allocation handling that token dropping doesn't support in the MVP.
    if (
        args
        or kwargs.get("new_computed_blocks") is not None
        or any(
            kwargs.get(name)
            for name in (
                "num_new_computed_tokens",
                "num_external_computed_tokens",
                "num_lookahead_tokens",
                "num_encoder_tokens",
                "full_sequence_must_fit",
            )
        )
    ):
        raise ValueError("Unsupported token-drop physical allocation mode")

    new_blocks = original(
        manager,
        _ResidentKVRequestView(request, resident_kv_length),
        num_new_tokens,
        *args,
        **kwargs,
    )

    if new_blocks is not None:
        new_resident_kv_length = min(
            resident_kv_length + num_new_tokens, manager.max_model_len
        )
        set_resident_kv_tokens(request_id, new_resident_kv_length)
    return new_blocks


def install_token_drop_allocator_adapter() -> None:
    """Install the resident-frontier allocation hook for vLLM 0.25.1."""
    _check_vllm_version()

    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_allocate_slots = KVCacheManager.allocate_slots
    adapter_marker = "_lmcache_token_drop_allocator_adapter"
    if getattr(original_allocate_slots, adapter_marker, False):
        return

    @wraps(original_allocate_slots)
    def wrapped_allocate_slots(self, *args, **kwargs):
        return _allocate_with_resident_frontier(
            original_allocate_slots, self, *args, **kwargs
        )

    # Prevent wrapping the allocator more than once.
    setattr(wrapped_allocate_slots, adapter_marker, True)
    KVCacheManager.allocate_slots = wrapped_allocate_slots


def install_token_drop_prefix_caching_publication_guard() -> None:
    """Prevent token-dropping requests from publishing KV blocks to the prefix cache.

    In-place KV compaction requires private blocks from initial allocation.
    Prefix-cache lookup must be disabled separately before enabling token dropping.
    """
    _check_vllm_version()

    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # Wrap BlockPool.cache_full_blocks() to skip publication for
    # token-dropping requests.
    original_cache_full_blocks = BlockPool.cache_full_blocks
    guard_marker = "_lmcache_token_drop_prefix_caching_publication_guard"
    if getattr(original_cache_full_blocks, guard_marker, False):
        return

    @wraps(original_cache_full_blocks)
    def cache_full_blocks_with_publication_guard(self, request, *args, **kwargs):
        if "lmcache.token_drop" in (
            extract_request_configs_from_request(request) or {}
        ):
            return
        return original_cache_full_blocks(self, request, *args, **kwargs)

    # Prevent wrapping the prefix-cache publication method more than once.
    setattr(cache_full_blocks_with_publication_guard, guard_marker, True)
    BlockPool.cache_full_blocks = cache_full_blocks_with_publication_guard


def install_token_drop_query_hooks(worker, no_compile_layers: dict[str, Any]) -> None:
    """Capture post-RoPE queries without changing attention forward outputs."""
    _check_vllm_version()
    if worker._original_attention_forwards:
        return

    original_forwards = {}
    for layer_name in worker._kv_caches:
        impl = no_compile_layers[layer_name].impl
        original_forwards[layer_name] = (
            impl,
            impl.forward,
            "forward" in vars(impl),
        )

    worker._original_attention_forwards = original_forwards
    for impl, original_forward, _ in original_forwards.values():

        def forward_with_query_capture(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            *args,
            _original_forward=original_forward,
            **kwargs,
        ):
            if attn_metadata is not None:
                worker.capture_query(
                    attn_layer.layer_name, query[: attn_metadata.num_actual_tokens]
                )
            return _original_forward(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                *args,
                **kwargs,
            )

        try:
            impl.forward = forward_with_query_capture
        except Exception:
            # Roll back earlier hooks if a later layer rejects installation.
            remove_token_drop_query_hooks(worker)
            raise


def remove_token_drop_query_hooks(worker) -> None:
    """Restore original attention forwards after query capture."""
    for (
        impl,
        original_forward,
        had_instance_forward,
    ) in worker._original_attention_forwards.values():
        if had_instance_forward:
            impl.forward = original_forward
        elif "forward" in vars(impl):
            # Remove the instance override to restore the original class method.
            delattr(impl, "forward")
    worker._original_attention_forwards.clear()


def _find_token_drop_requests(
    metadata: KVConnectorMetadata | None,
) -> list[LMCacheMPTokenDropRequestState] | None:
    """Retrieve request-local metadata, including inside MultiConnector."""
    if metadata is None:
        return None
    states = getattr(metadata, "token_drop_requests", None)
    if states:
        return states
    found = None
    for child in getattr(metadata, "metadata", ()) or ():
        child_states = _find_token_drop_requests(child)
        if child_states:
            if found is not None:
                raise RuntimeError(
                    "Multiple active token-drop connectors are unsupported"
                )
            found = child_states
    return found


def _prepare_inputs_with_physical_frontier(
    original: Callable[..., tuple[torch.Tensor, SpecDecodeMetadata | None]],
    runner: GPUModelRunner,
    scheduler_output: SchedulerOutput,
    num_scheduled_tokens: np.ndarray,
) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
    """Use resident P for KV slot addresses, preserving logical L for RoPE."""
    states = _find_token_drop_requests(
        getattr(scheduler_output, "kv_connector_metadata", None)
    )
    if not states:
        return original(runner, scheduler_output, num_scheduled_tokens)

    batch = runner.input_batch
    num_reqs = batch.num_reqs
    req_ids = list(batch.req_ids[:num_reqs])
    by_row = {req_id: row for row, req_id in enumerate(req_ids)}
    if len(by_row) != num_reqs or len({state.request_id for state in states}) != len(
        states
    ):
        raise RuntimeError("Duplicate request ids in token-drop batch metadata")
    for state in states:
        if state.request_id not in by_row:
            raise RuntimeError("Token-drop request missing from worker batch")
        state.worker_row = by_row[state.request_id]
    # Keep the scheduler-provided metadata order unchanged.
    # Worker observation uses explicit row indices, not metadata order.

    # First, plan every TD row. Do not rewrite any native block table until
    # every physical length and source ownership mapping is valid.
    table = batch.block_table
    changes = []
    sizes = [int(n) for n in num_scheduled_tokens[:num_reqs]]
    for state in states:
        row = state.worker_row
        n = sizes[row]
        p = int(state.resident_kv_tokens)
        ids = state.physical_block_ids
        if ids is None:
            expected = int(batch.num_computed_tokens_cpu[row]) + n
            if p != expected:
                raise RuntimeError(
                    "Token-drop resident KV diverges from logical length "
                    "without authoritative private block ids"
                )
            continue
        if n <= 0 or p < n or not ids:
            raise RuntimeError("Invalid resident KV frontier for token-drop row")
        if len(table.block_tables) != 1:
            raise RuntimeError("Token dropping supports one KV group")
        block_size = table.block_tables[0].block_size
        if len(ids) * block_size < p or any(block <= 0 for block in ids):
            raise RuntimeError("Token-drop physical block mapping is incomplete")
        changes.append((row, p, ids))
    if not changes:
        return original(runner, scheduler_output, num_scheduled_tokens)

    overrides = {row: p for row, p, _ in changes}

    # CPU-packed P for all rows; only the changed TD rows differ from L.
    # Normal requests keep their exact logical slot positions and KV row.
    total = sum(sizes)
    physical_positions = np.empty(total, dtype=np.int64)
    logical_before = batch.num_computed_tokens_cpu[:num_reqs]
    offset = 0
    physical_seq_by_row = {}
    for row, n in enumerate(sizes):
        start = int(logical_before[row])
        if row in overrides:
            start = overrides[row] - n
            physical_seq_by_row[row] = overrides[row]
        physical_positions[offset : offset + n] = np.arange(n) + start
        offset += n

    # Commit TD block ownership before native _prepare_inputs() copies the
    # table. Scope interception to THIS instance for the native first mapping.
    for row, p, ids in changes:
        table.add_row((ids,), row)
    positions_gpu = torch.from_numpy(physical_positions).to(
        device=runner.positions.device
    )
    original_mapping = table.compute_slot_mapping
    has_instance_mapping = "compute_slot_mapping" in vars(table)

    def physical_mapping(
        num_reqs_arg: int,
        query_start_loc: torch.Tensor,
        _logical_positions: torch.Tensor,
    ) -> None:
        return original_mapping(num_reqs_arg, query_start_loc, positions_gpu)

    try:
        table.compute_slot_mapping = physical_mapping
        return_value = original(runner, scheduler_output, num_scheduled_tokens)
    finally:
        # Restore precisely the prior instance/class attribute resolution.
        if has_instance_mapping:
            table.compute_slot_mapping = original_mapping
        else:
            delattr(table, "compute_slot_mapping")

    # The native code must still calculate logical RoPE positions from L.
    # Replace KV-facing lengths for TD rows only, before attention metadata.
    rows = list(physical_seq_by_row)
    lens = list(physical_seq_by_row.values())
    runner.seq_lens[rows] = torch.tensor(
        lens, dtype=runner.seq_lens.dtype, device=runner.seq_lens.device
    )
    runner.optimistic_seq_lens_cpu[rows] = torch.tensor(
        lens, dtype=runner.optimistic_seq_lens_cpu.dtype
    )
    return return_value


def install_token_drop_worker_adapter() -> None:
    """Install pinned, opt-in resident KV addressing without upstream edits."""
    _check_vllm_version()
    # Third Party
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._prepare_inputs
    marker = "_lmcache_token_drop_worker_adapter"
    if getattr(original, marker, False):
        return

    @wraps(original)
    def wrapped(
        self: GPUModelRunner,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
        return _prepare_inputs_with_physical_frontier(
            original, self, scheduler_output, num_scheduled_tokens
        )

    setattr(wrapped, marker, True)
    GPUModelRunner._prepare_inputs = wrapped
