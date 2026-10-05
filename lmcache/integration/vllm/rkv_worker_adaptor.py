# SPDX-License-Identifier: Apache-2.0
"""Version-pinned vLLM worker adaptor for LMCache R-KV physical KV state."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any, Callable

import numpy as np
import torch

_SUPPORTED_VLLM_VERSION = "0.25.1"
_EXPECTED_PREPARE_INPUTS_PARAMS = (
    "self",
    "scheduler_output",
    "num_scheduled_tokens",
)
_EXPECTED_COMPUTE_SLOT_MAPPING_PARAMS = (
    "self",
    "num_reqs",
    "query_start_loc",
    "positions",
)


@dataclass(frozen=True)
class _SlotMappingOverride:
    block_table_id: int
    positions: torch.Tensor


_slot_mapping_override: ContextVar[_SlotMappingOverride | None] = ContextVar(
    "lmcache_rkv_slot_mapping_override",
    default=None,
)


def _find_rkv_requests(metadata: Any) -> list[Any] | None:
    if metadata is None:
        return None

    requests = getattr(metadata, "rkv_requests", None)
    if requests is not None:
        return requests

    nested = getattr(metadata, "metadata", None)
    if nested is None:
        return None

    matches = [
        requests
        for child in nested
        if (requests := _find_rkv_requests(child)) is not None
    ]
    if len(matches) > 1:
        raise RuntimeError("Multiple R-KV connector metadata entries are unsupported")
    return matches[0] if matches else None


def _ensure_physical_buffers(runner: Any) -> tuple[Any, Any]:
    positions = getattr(runner, "_lmcache_rkv_physical_positions", None)
    seq_lens = getattr(runner, "_lmcache_rkv_physical_seq_lens", None)
    if positions is None:
        positions = runner._make_buffer(runner.max_num_tokens, dtype=torch.int64)
        runner._lmcache_rkv_physical_positions = positions
    if seq_lens is None:
        seq_lens = runner._make_buffer(runner.max_num_reqs, dtype=torch.int32)
        runner._lmcache_rkv_physical_seq_lens = seq_lens
    return positions, seq_lens


def _prepare_inputs_with_physical_frontier(
    original: Callable[..., Any],
    runner: Any,
    scheduler_output: Any,
    num_scheduled_tokens: np.ndarray,
) -> Any:
    requests = _find_rkv_requests(
        getattr(scheduler_output, "kv_connector_metadata", None)
    )
    if not requests:
        return original(runner, scheduler_output, num_scheduled_tokens)

    num_reqs = runner.input_batch.num_reqs
    req_ids = list(runner.input_batch.req_ids[:num_reqs])
    if len(num_scheduled_tokens) != num_reqs:
        raise RuntimeError("R-KV scheduled-token rows do not match the worker batch")

    by_request_id = {state.request_id: state for state in requests}
    if len(by_request_id) != len(requests):
        raise RuntimeError("R-KV request metadata contains duplicate request ids")

    try:
        ordered = [by_request_id[request_id] for request_id in req_ids]
    except KeyError as exc:
        raise RuntimeError(
            f"R-KV metadata is missing worker request {exc.args[0]!r}"
        ) from exc
    if len(ordered) != len(requests):
        extra = sorted(set(by_request_id) - set(req_ids))
        raise RuntimeError(
            f"R-KV metadata has requests absent from worker batch: {extra}"
        )

    # The same metadata object is consumed later by start_load_kv(). Keep it in
    # authoritative model-runner row order so query observation needs no GPU
    # block-table -> CPU synchronization to recover row identity.
    requests[:] = ordered

    if not any(bool(state.has_physical_override) for state in ordered):
        return original(runner, scheduler_output, num_scheduled_tokens)

    physical_positions, physical_seq_lens = _ensure_physical_buffers(runner)
    total_tokens = int(num_scheduled_tokens.sum())
    offset = 0
    for row, (state, num_new_tokens) in enumerate(
        zip(ordered, num_scheduled_tokens, strict=True)
    ):
        num_new_tokens = int(num_new_tokens)
        resident = state.resident_kv_tokens
        if resident is None:
            raise RuntimeError("R-KV metadata is missing resident KV length")
        resident = int(resident)
        if num_new_tokens < 0 or resident < num_new_tokens:
            raise RuntimeError(
                f"Invalid R-KV physical frontier for {state.request_id!r}: "
                f"resident={resident}, scheduled={num_new_tokens}"
            )

        physical_seq_lens.np[row] = resident
        if num_new_tokens:
            start = resident - num_new_tokens
            physical_positions.np[offset : offset + num_new_tokens] = (
                runner.arange_np[:num_new_tokens] + start
            )
            offset += num_new_tokens

    if offset != total_tokens:
        raise RuntimeError("R-KV physical-position construction did not cover the step")

    physical_positions.copy_to_gpu(total_tokens)
    physical_seq_lens.copy_to_gpu(num_reqs)

    block_table = runner.input_batch.block_table
    token = _slot_mapping_override.set(
        _SlotMappingOverride(
            block_table_id=id(block_table),
            positions=physical_positions.gpu[:total_tokens],
        )
    )
    try:
        result = original(runner, scheduler_output, num_scheduled_tokens)
    finally:
        _slot_mapping_override.reset(token)

    # RoPE/model positions remain logical. Only KV-facing attention lengths
    # use the scheduler-committed resident frontier.
    runner.seq_lens[:num_reqs].copy_(
        physical_seq_lens.gpu[:num_reqs],
        non_blocking=True,
    )
    return result


def _compute_slot_mapping_with_physical_positions(
    original: Callable[..., Any],
    block_table: Any,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> Any:
    override = _slot_mapping_override.get()
    if override is not None and override.block_table_id == id(block_table):
        positions = override.positions
    return original(block_table, num_reqs, query_start_loc, positions)


def install_rkv_worker_adaptor() -> None:
    """Install the worker-side physical-KV adaptor for pinned vLLM 0.25.1."""
    # Third Party
    from vllm.version import __version__ as vllm_version
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"R-KV MVP requires vLLM {_SUPPORTED_VLLM_VERSION}, got {vllm_version}"
        )

    original_prepare_inputs = GPUModelRunner._prepare_inputs
    if not getattr(original_prepare_inputs, "_lmcache_rkv_worker_adaptor", False):
        params = tuple(signature(original_prepare_inputs).parameters)
        if params != _EXPECTED_PREPARE_INPUTS_PARAMS:
            raise RuntimeError(
                f"Unsupported vLLM GPUModelRunner._prepare_inputs signature: {params}"
            )

        @wraps(original_prepare_inputs)
        def wrapped_prepare_inputs(
            self: Any,
            scheduler_output: Any,
            num_scheduled_tokens: np.ndarray,
        ) -> Any:
            return _prepare_inputs_with_physical_frontier(
                original_prepare_inputs,
                self,
                scheduler_output,
                num_scheduled_tokens,
            )

        wrapped_prepare_inputs._lmcache_rkv_worker_adaptor = True  # type: ignore[attr-defined]
        GPUModelRunner._prepare_inputs = wrapped_prepare_inputs

    original_compute_slot_mapping = MultiGroupBlockTable.compute_slot_mapping
    if not getattr(
        original_compute_slot_mapping,
        "_lmcache_rkv_worker_adaptor",
        False,
    ):
        params = tuple(signature(original_compute_slot_mapping).parameters)
        if params != _EXPECTED_COMPUTE_SLOT_MAPPING_PARAMS:
            raise RuntimeError(
                "Unsupported vLLM MultiGroupBlockTable.compute_slot_mapping "
                f"signature: {params}"
            )

        @wraps(original_compute_slot_mapping)
        def wrapped_compute_slot_mapping(
            self: Any,
            num_reqs: int,
            query_start_loc: torch.Tensor,
            positions: torch.Tensor,
        ) -> Any:
            return _compute_slot_mapping_with_physical_positions(
                original_compute_slot_mapping,
                self,
                num_reqs,
                query_start_loc,
                positions,
            )

        wrapped_compute_slot_mapping._lmcache_rkv_worker_adaptor = True  # type: ignore[attr-defined]
        MultiGroupBlockTable.compute_slot_mapping = wrapped_compute_slot_mapping
