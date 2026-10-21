# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.25.1 token-dropping allocator and worker KV-address adaptors."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from contextvars import ContextVar
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
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # First Party
    from lmcache.integration.vllm.experimental.token_drop import (
        LMCacheMPTokenDropRequestState,
    )


_SUPPORTED_VLLM_VERSION = "0.25.1"


def _check_vllm_version() -> None:
    # Third Party
    from vllm.version import __version__ as vllm_version

    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Token dropping requires vLLM {_SUPPORTED_VLLM_VERSION}, "
            f"got {vllm_version}"
        )


class _PhysicalKVRequestView:
    """Let the allocator read resident KV length without changing the request."""

    def __init__(self, request, physical_kv_length):
        self._request = request
        self.physical_kv_length = physical_kv_length

    @property
    def num_computed_tokens(self):
        return self.physical_kv_length

    def __getattr__(self, name):
        return getattr(self._request, name)


def _allocate_with_resident_frontier(
    original, manager, request, num_new_tokens, *args, **kwargs
):
    request_id = request.request_id
    resident = get_resident_kv_tokens(request_id)
    if resident is None:
        return original(manager, request, num_new_tokens, *args, **kwargs)

    resident_kv_length: int = resident
    if request.num_computed_tokens == 0 or "lmcache.token_drop" not in (
        extract_request_configs_from_request(request) or {}
    ):
        # A preempted request must recompute from the logical beginning.
        clear_resident_kv_tokens(request_id)
        return original(manager, request, num_new_tokens, *args, **kwargs)

    # These are vLLM 0.25.1 allocate_slots() options. Prefix/external KV,
    # speculative or encoder tokens, and full-sequence admission need
    # allocation accounting not supported by compacted KV states.
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

    result = original(
        manager,
        _PhysicalKVRequestView(request, resident_kv_length),
        num_new_tokens,
        *args,
        **kwargs,
    )

    if result is not None:
        updated_length = min(resident_kv_length + num_new_tokens, manager.max_model_len)
        set_resident_kv_tokens(request_id, updated_length)
    return result


def install_token_drop_allocator_adapter() -> None:
    """Install the resident-frontier allocation hook for vLLM 0.25.1."""
    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _check_vllm_version()

    original_allocate = KVCacheManager.allocate_slots
    if getattr(original_allocate, "_lmcache_token_drop_allocator_adapter", False):
        return

    @wraps(original_allocate)
    def wrapped_allocate(self, *args, **kwargs):
        return _allocate_with_resident_frontier(
            original_allocate, self, *args, **kwargs
        )

    # Mark the wrapper so repeated installation does not stack allocator hooks.
    wrapped_allocate._lmcache_token_drop_allocator_adapter = True  # type: ignore[attr-defined]
    KVCacheManager.allocate_slots = wrapped_allocate


# (block_table_id, KV write positions), scoped to this input-preparation call.
_kv_write_positions_override: ContextVar[tuple[int, torch.Tensor] | None] = ContextVar(
    "lmcache_token_drop_kv_write_positions_override",
    default=None,
)


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


def _ensure_kv_write_positions_buffer(runner: GPUModelRunner) -> CpuGpuBuffer:
    kv_write_positions = getattr(runner, "_lmcache_token_drop_kv_write_positions", None)
    if kv_write_positions is None:
        kv_write_positions = runner._make_buffer(
            runner.max_num_tokens, dtype=torch.int64
        )
        runner._lmcache_token_drop_kv_write_positions = kv_write_positions
    return kv_write_positions


def _prepare_inputs_with_resident_kv_addressing(
    original: Callable[..., tuple[torch.Tensor, SpecDecodeMetadata | None]],
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
    states_by_request_id = {state.request_id: state for state in states}
    row_by_request_id = {request_id: row for row, request_id in enumerate(req_ids)}
    missing_request_ids = sorted(set(states_by_request_id) - set(row_by_request_id))
    if missing_request_ids:
        raise RuntimeError(
            f"Token-drop states absent from worker batch: {missing_request_ids}"
        )

    # Worker rows are explicit: no reordering or mutation of the metadata list.
    for state in states:
        state.worker_row = row_by_request_id[state.request_id]

    logical_frontiers = runner.input_batch.num_computed_tokens_cpu
    # Block IDs may be omitted; resident KV length still controls addressing.
    needs_kv_addressing = any(
        state.physical_block_ids is not None
        or state.resident_kv_tokens
        != int(logical_frontiers[state.worker_row])
        + int(num_scheduled_tokens[state.worker_row])
        for state in states
    )
    if not needs_kv_addressing:
        return original(runner, scheduler_output, num_scheduled_tokens)

    kv_write_positions = _ensure_kv_write_positions_buffer(runner)
    total_scheduled_tokens = int(num_scheduled_tokens.sum())
    batch_offset = 0
    for row, num_new_tokens_raw in enumerate(num_scheduled_tokens):
        num_new_tokens = int(num_new_tokens_raw)
        request_id = req_ids[row]
        request_state = states_by_request_id.get(request_id)
        kv_write_start = int(logical_frontiers[row])
        if request_state is not None:
            resident_kv_tokens = request_state.resident_kv_tokens
            if resident_kv_tokens < num_new_tokens:
                raise RuntimeError(
                    f"Invalid token-drop resident KV length for "
                    f"{request_state.request_id!r}: resident={resident_kv_tokens}, "
                    f"scheduled={num_new_tokens}"
                )
            if request_state.physical_block_ids is not None:
                block_ids = request_state.physical_block_ids
                if not block_ids:
                    raise RuntimeError(
                        "Token-drop request is missing scheduler-provided KV block IDs"
                    )
                runner.input_batch.block_table.add_row((block_ids,), row)
            kv_write_start = resident_kv_tokens - num_new_tokens
        if num_new_tokens:
            kv_write_positions.np[batch_offset : batch_offset + num_new_tokens] = (
                runner.arange_np[:num_new_tokens] + kv_write_start
            )
        batch_offset += num_new_tokens

    kv_write_positions.copy_to_gpu(total_scheduled_tokens)

    block_table = runner.input_batch.block_table
    override_token = _kv_write_positions_override.set(
        (id(block_table), kv_write_positions.gpu[:total_scheduled_tokens])
    )
    try:
        result = original(runner, scheduler_output, num_scheduled_tokens)
    finally:
        _kv_write_positions_override.reset(override_token)

    # Only attention KV lengths change; logical model/RoPE positions stay intact.
    for state in states:
        runner.seq_lens[state.worker_row].fill_(state.resident_kv_tokens)
    return result


def _compute_slot_mapping_with_kv_write_positions(
    original: Callable[..., None],
    block_table: MultiGroupBlockTable,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    kv_write_positions_override = _kv_write_positions_override.get()
    if kv_write_positions_override is not None:
        block_table_id, kv_write_positions = kv_write_positions_override
        if block_table_id == id(block_table):
            positions = kv_write_positions
    return original(block_table, num_reqs, query_start_loc, positions)


def install_token_drop_worker_kv_address_adapter() -> None:
    """Install the worker-side token-drop KV adaptor for pinned vLLM 0.25.1."""
    # Third Party
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _check_vllm_version()

    original_prepare_inputs = GPUModelRunner._prepare_inputs
    if not getattr(
        original_prepare_inputs,
        "_lmcache_token_drop_worker_adapter",
        False,
    ):

        @wraps(original_prepare_inputs)
        def wrapped_prepare_inputs(
            self: GPUModelRunner,
            scheduler_output: SchedulerOutput,
            num_scheduled_tokens: np.ndarray,
        ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
            return _prepare_inputs_with_resident_kv_addressing(
                original_prepare_inputs,
                self,
                scheduler_output,
                num_scheduled_tokens,
            )

        wrapped_prepare_inputs._lmcache_token_drop_worker_adapter = True  # type: ignore[attr-defined]
        GPUModelRunner._prepare_inputs = wrapped_prepare_inputs

    original_compute_slot_mapping = MultiGroupBlockTable.compute_slot_mapping
    if not getattr(
        original_compute_slot_mapping,
        "_lmcache_token_drop_worker_adapter",
        False,
    ):

        @wraps(original_compute_slot_mapping)
        def wrapped_compute_slot_mapping(
            self: MultiGroupBlockTable,
            num_reqs: int,
            query_start_loc: torch.Tensor,
            positions: torch.Tensor,
        ) -> None:
            return _compute_slot_mapping_with_kv_write_positions(
                original_compute_slot_mapping,
                self,
                num_reqs,
                query_start_loc,
                positions,
            )

        wrapped_compute_slot_mapping._lmcache_token_drop_worker_adapter = True  # type: ignore[attr-defined]
        MultiGroupBlockTable.compute_slot_mapping = wrapped_compute_slot_mapping
