# SPDX-License-Identifier: Apache-2.0
"""vLLM integration adaptors for token dropping."""

# Future
from __future__ import annotations

# Standard
from functools import wraps
from typing import TYPE_CHECKING

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


def _find_token_drop_states(
    metadata: KVConnectorMetadata | None,
) -> list[LMCacheMPTokenDropRequestState] | None:
    if metadata is None:
        return None

    states = getattr(metadata, "token_drop_requests", None)
    if states:
        return states

    for child in getattr(metadata, "metadata", ()):
        child_states = _find_token_drop_states(child)
        if child_states:
            return child_states
    return None


def _prepare_inputs_with_resident_kv_addressing(
    original,
    runner: GPUModelRunner,
    scheduler_output: SchedulerOutput,
    num_scheduled_tokens: np.ndarray,
) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
    states = _find_token_drop_states(
        getattr(scheduler_output, "kv_connector_metadata", None)
    )
    if not states:
        return original(runner, scheduler_output, num_scheduled_tokens)

    num_reqs = runner.input_batch.num_reqs
    req_ids = list(runner.input_batch.req_ids[:num_reqs])
    row_by_request_id = {req_id: row for row, req_id in enumerate(req_ids)}
    missing = sorted({state.request_id for state in states} - row_by_request_id.keys())
    if missing:
        raise RuntimeError(f"Token-drop states absent from worker batch: {missing}")

    # Validate all TD rows before touching the block table.
    for state in states:
        row = row_by_request_id[state.request_id]
        scheduled = int(num_scheduled_tokens[row])
        if state.resident_kv_tokens < scheduled:
            raise RuntimeError(
                f"Invalid token-drop resident KV length for {state.request_id!r}: "
                f"resident={state.resident_kv_tokens}, scheduled={scheduled}"
            )
        if state.physical_block_ids is not None and not state.physical_block_ids:
            raise RuntimeError(
                "Token-drop request is missing scheduler-provided KV block IDs"
            )

    state_by_row = {}
    physical_addressing_needed = False
    logical_frontiers = runner.input_batch.num_computed_tokens_cpu
    for state in states:
        row = row_by_request_id[state.request_id]
        state.worker_row = row
        state_by_row[row] = state
        if state.physical_block_ids is not None:
            # The native _prepare_inputs() commits block tables before mapping slots.
            runner.input_batch.block_table.add_row((state.physical_block_ids,), row)
        if state.resident_kv_tokens != (
            int(logical_frontiers[row]) + int(num_scheduled_tokens[row])
        ):
            physical_addressing_needed = True

    # vLLM prepares model/RoPE positions and its initial slot mapping normally.
    result = original(runner, scheduler_output, num_scheduled_tokens)
    if not physical_addressing_needed:
        return result

    num_tokens = int(num_scheduled_tokens.sum())
    # Preserve native positions for every normal request, including mixed batches.
    kv_positions = getattr(runner, "_lmcache_token_drop_kv_write_positions", None)
    if kv_positions is None:
        kv_positions = torch.empty_like(runner.positions)
        runner._lmcache_token_drop_kv_write_positions = kv_positions
    kv_positions[:num_tokens].copy_(runner.positions[:num_tokens])

    offset = 0
    for row, scheduled_raw in enumerate(num_scheduled_tokens):
        scheduled = int(scheduled_raw)
        state = state_by_row.get(row)
        if state is not None and scheduled:
            # P includes the current step's scheduled tokens; writes start at P-N.
            torch.add(
                runner.query_pos.gpu[offset : offset + scheduled],
                state.resident_kv_tokens - scheduled,
                out=kv_positions[offset : offset + scheduled],
            )
        offset += scheduled

    # Single hook: remap slots once more, after native preparation.
    runner.input_batch.block_table.compute_slot_mapping(
        num_reqs,
        runner.query_start_loc.gpu[: num_reqs + 1],
        kv_positions[:num_tokens],
    )
    for state in states:
        runner.seq_lens[state.worker_row].fill_(state.resident_kv_tokens)
    return result


def install_token_drop_worker_kv_address_adapter() -> None:
    """Install the single-hook worker KV adaptor for pinned vLLM 0.25.1."""
    # Third Party
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _check_vllm_version()

    original_prepare_inputs = GPUModelRunner._prepare_inputs
    guard_marker = "_lmcache_token_drop_worker_adapter"
    if getattr(original_prepare_inputs, guard_marker, False):
        return

    @wraps(original_prepare_inputs)
    def wrapped_prepare_inputs(
        self: GPUModelRunner,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
        return _prepare_inputs_with_resident_kv_addressing(
            original_prepare_inputs, self, scheduler_output, num_scheduled_tokens
        )

    setattr(wrapped_prepare_inputs, guard_marker, True)
    GPUModelRunner._prepare_inputs = wrapped_prepare_inputs
