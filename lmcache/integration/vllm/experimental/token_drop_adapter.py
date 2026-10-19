# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.25.1 token-dropping adaptor: scheduler, worker and attention hooks."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)
from lmcache.integration.vllm.utils import extract_request_configs_from_request

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from lmcache.integration.vllm.experimental.token_drop import (
        LMCacheMPTokenDropRequestState,
    )


_SUPPORTED_VLLM_VERSION = "0.25.1"


def _check_vllm_version() -> None:
    from vllm.version import __version__ as vllm_version

    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Token dropping requires vLLM {_SUPPORTED_VLLM_VERSION}, "
            f"got {vllm_version}"
        )


def _allocate_with_resident_frontier(
    original: Callable[..., KVCacheBlocks | None],
    manager: KVCacheManager,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
    has_scheduled_reqs: bool = True,
) -> KVCacheBlocks | None:
    request_id = request.request_id
    is_token_drop = "lmcache.token_drop" in (
        extract_request_configs_from_request(request) or {}
    )
    resident = get_resident_kv_tokens(request_id)

    # Admission already validated the spec. Only the opt-in bit matters here.
    use_physical = is_token_drop and resident is not None
    if not is_token_drop or request.num_computed_tokens == 0:
        clear_resident_kv_tokens(request_id)
        use_physical = False

    if use_physical and (
        num_new_computed_tokens
        or new_computed_blocks is not None
        or num_lookahead_tokens
        or num_external_computed_tokens
        or num_encoder_tokens
        or full_sequence_must_fit
    ):
        raise ValueError(
            "Token-drop resident allocation only supports normal synchronous "
            "full-attention decoding"
        )

    logical = request.num_computed_tokens
    if use_physical:
        request.num_computed_tokens = resident
    try:
        result = original(
            manager,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )
    finally:
        if use_physical:
            request.num_computed_tokens = logical

    if use_physical and result is not None:
        set_resident_kv_tokens(
            request_id, min(resident + num_new_tokens, manager.max_model_len)
        )
    return result


def install_token_drop_allocator_adapter() -> None:
    """Install resident-frontier and private-block adaptors for vLLM 0.25.1."""
    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _check_vllm_version()

    original_allocate = KVCacheManager.allocate_slots
    if not getattr(
        original_allocate,
        "_lmcache_token_drop_allocator_adapter",
        False,
    ):

        @wraps(original_allocate)
        def wrapped_allocate(self, *args, **kwargs):
            return _allocate_with_resident_frontier(
                original_allocate, self, *args, **kwargs
            )

        wrapped_allocate._lmcache_token_drop_allocator_adapter = True  # type: ignore[attr-defined]
        KVCacheManager.allocate_slots = wrapped_allocate


def install_token_drop_apc_guard() -> None:
    """Install the opt-in APC publication guard for pinned vLLM 0.25.1."""
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    _check_vllm_version()

    original = BlockPool.cache_full_blocks
    if getattr(original, "_lmcache_token_drop_apc_guard", False):
        return

    @wraps(original)
    def wrapped(self, request, *args, **kwargs):
        if "lmcache.token_drop" in (
            extract_request_configs_from_request(request) or {}
        ):
            return
        return original(self, request, *args, **kwargs)

    wrapped._lmcache_token_drop_apc_guard = True  # type: ignore[attr-defined]
    BlockPool.cache_full_blocks = wrapped


_slot_mapping_override: ContextVar[tuple[int, torch.Tensor] | None] = ContextVar(
    "lmcache_token_drop_slot_mapping_override",
    default=None,
)


def _find_token_drop_requests(
    metadata: KVConnectorMetadata | None,
) -> list[LMCacheMPTokenDropRequestState] | None:
    if metadata is None:
        return None

    requests = getattr(metadata, "token_drop_requests", None)
    if requests is not None:
        return requests

    for child in getattr(metadata, "metadata", ()):
        found = _find_token_drop_requests(child)
        if found is not None:
            return found
    return None


def _ensure_physical_buffers(
    runner: GPUModelRunner,
) -> tuple[CpuGpuBuffer, CpuGpuBuffer]:
    positions = getattr(runner, "_lmcache_token_drop_physical_positions", None)
    seq_lens = getattr(runner, "_lmcache_token_drop_physical_seq_lens", None)
    if positions is None:
        positions = runner._make_buffer(runner.max_num_tokens, dtype=torch.int64)
        runner._lmcache_token_drop_physical_positions = positions
    if seq_lens is None:
        seq_lens = runner._make_buffer(runner.max_num_reqs, dtype=torch.int32)
        runner._lmcache_token_drop_physical_seq_lens = seq_lens
    return positions, seq_lens


def _prepare_inputs_with_physical_frontier(
    original: Callable[..., tuple[torch.Tensor, SpecDecodeMetadata | None]],
    runner: GPUModelRunner,
    scheduler_output: SchedulerOutput,
    num_scheduled_tokens: np.ndarray,
) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
    requests = _find_token_drop_requests(
        getattr(scheduler_output, "kv_connector_metadata", None)
    )
    if not requests:
        return original(runner, scheduler_output, num_scheduled_tokens)

    num_reqs = runner.input_batch.num_reqs
    req_ids = list(runner.input_batch.req_ids[:num_reqs])
    by_request_id = {state.request_id: state for state in requests}
    row_by_request_id = {request_id: row for row, request_id in enumerate(req_ids)}
    extra = sorted(set(by_request_id) - set(row_by_request_id))
    if extra:
        raise RuntimeError(
            f"Token-drop metadata has requests absent from worker batch: {extra}"
        )

    # Keep only token-dropping requests in this list, but order those sparse
    # rows exactly like the model-runner batch. Normal rows remain vanilla.
    ordered = sorted(requests, key=lambda state: row_by_request_id[state.request_id])
    for state in ordered:
        state.worker_row = row_by_request_id[state.request_id]
    requests[:] = ordered

    if not any(state.physical_block_ids is not None for state in ordered):
        return original(runner, scheduler_output, num_scheduled_tokens)

    physical_positions, physical_seq_lens = _ensure_physical_buffers(runner)
    total_tokens = int(num_scheduled_tokens.sum())
    logical_frontiers = runner.input_batch.num_computed_tokens_cpu
    offset = 0
    for row, num_new_tokens_raw in enumerate(num_scheduled_tokens):
        num_new_tokens = int(num_new_tokens_raw)
        request_id = req_ids[row]
        request_state = by_request_id.get(request_id)
        logical_start = int(logical_frontiers[row])
        start = logical_start
        seq_len = logical_start + num_new_tokens

        if request_state is not None and request_state.physical_block_ids is not None:
            resident = request_state.resident_kv_tokens
            if resident < num_new_tokens:
                raise RuntimeError(
                    f"Invalid token-drop physical frontier for "
                    f"{request_state.request_id!r}: resident={resident}, "
                    f"scheduled={num_new_tokens}"
                )
            block_ids = request_state.physical_block_ids
            if not block_ids:
                raise RuntimeError(
                    "Token-drop metadata is missing authoritative KV block ids"
                )
            runner.input_batch.block_table.add_row((block_ids,), row)
            start = resident - num_new_tokens
            seq_len = resident

        physical_seq_lens.np[row] = seq_len
        if num_new_tokens:
            physical_positions.np[offset : offset + num_new_tokens] = (
                runner.arange_np[:num_new_tokens] + start
            )
        offset += num_new_tokens

    physical_positions.copy_to_gpu(total_tokens)
    physical_seq_lens.copy_to_gpu(num_reqs)

    block_table = runner.input_batch.block_table
    token = _slot_mapping_override.set(
        (id(block_table), physical_positions.gpu[:total_tokens])
    )
    try:
        result = original(runner, scheduler_output, num_scheduled_tokens)
    finally:
        _slot_mapping_override.reset(token)

    # RoPE/model positions remain logical. KV-facing lengths use P only for
    # token-dropping rows; normal rows are copied back with their vanilla L.
    runner.seq_lens[:num_reqs].copy_(
        physical_seq_lens.gpu[:num_reqs],
        non_blocking=True,
    )
    return result


def _compute_slot_mapping_with_physical_positions(
    original: Callable[..., None],
    block_table: MultiGroupBlockTable,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    override = _slot_mapping_override.get()
    if override is not None and override[0] == id(block_table):
        positions = override[1]
    return original(block_table, num_reqs, query_start_loc, positions)


def install_token_drop_worker_adapter() -> None:
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
            return _prepare_inputs_with_physical_frontier(
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
            return _compute_slot_mapping_with_physical_positions(
                original_compute_slot_mapping,
                self,
                num_reqs,
                query_start_loc,
                positions,
            )

        wrapped_compute_slot_mapping._lmcache_token_drop_worker_adapter = True  # type: ignore[attr-defined]
        MultiGroupBlockTable.compute_slot_mapping = wrapped_compute_slot_mapping


def install_query_hooks(worker, no_compile_layers: dict[str, "Attention"]) -> None:
    if worker._query_hook_originals:
        return

    originals: dict[str, tuple[FlashAttentionImpl, Callable[..., torch.Tensor]]] = {}
    for layer_name in worker._layer_names:
        layer = no_compile_layers[layer_name]
        impl = getattr(layer, "impl", None)
        original_forward = impl.forward
        originals[layer_name] = (impl, original_forward)

    for impl, forward_impl in originals.values():

        def forward_with_query_capture(
            attn_layer: "Attention",
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: "FlashAttentionMetadata",
            *args: Any,
            _forward: Callable[..., torch.Tensor] = forward_impl,
            **kwargs: Any,
        ) -> torch.Tensor:
            # Under PIECEWISE cudagraph this backend call executes on every
            # model step. Ignore any padded rows beyond the real batch.
            num_actual_tokens = getattr(
                attn_metadata, "num_actual_tokens", query.shape[0]
            )
            worker.capture_query(
                attn_layer.layer_name,
                query[:num_actual_tokens],
            )
            return _forward(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                *args,
                **kwargs,
            )

        setattr(impl, "forward", forward_with_query_capture)  # noqa: B010

    worker._query_hook_originals = originals


def remove_query_hooks(worker) -> None:
    if not worker._query_hook_originals:
        return

    for impl, original_forward in worker._query_hook_originals.values():
        setattr(impl, "forward", original_forward)  # noqa: B010
    worker._query_hook_originals.clear()
