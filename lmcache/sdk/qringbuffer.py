# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence
import math
import threading
import time

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import vllm_layout_hints
from lmcache.utils import init_logger as lmcache_init_logger
from lmcache.v1.gpu_connector.utils import get_device
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPConnectorMetadata,
    )
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        StoreResult,
        _IpcEvent,
    )
    from lmcache.v1.multiprocess.mq import MessagingFuture

logger = lmcache_init_logger(__name__)


@dataclass(frozen=True)
class QRingBufferUsageSnapshot:
    """Immutable block-usage snapshot for a query ring.

    Attributes:
        capacity_blocks: Total number of blocks in the ring.
        used_blocks: Blocks currently reserved by capture or an in-flight store.
        free_blocks: Blocks currently available for capture.
        high_watermark_blocks: Maximum simultaneous block usage since creation.
    """

    capacity_blocks: int
    used_blocks: int
    free_blocks: int
    high_watermark_blocks: int


@dataclass(frozen=True)
class QCaptureMetricsSnapshot:
    """Cumulative query-capture metrics plus current ring pressure.

    Skip reasons use fixed fields rather than request IDs or free-form labels,
    so exporting a snapshot cannot create unbounded metric cardinality.

    Attributes:
        steps_attempted: Query-capture plans attempted.
        steps_captured: Steps where at least one request was captured.
        steps_skipped: Steps where no request was captured.
        store_requests_seen: STORE requests inspected while building plans.
        requests_captured: Requests successfully assigned ring blocks.
        tokens_captured: Query-token rows assigned to ring slots.
        blocks_reserved: Ring blocks reserved by successful captures.
        plan_duration_ns_total: Total host time spent building capture plans.
        plan_duration_ns_max: Maximum host time spent building one plan.
        skipped_ring_unavailable_steps: Steps skipped before ring setup.
        skipped_missing_slot_mapping_steps: Steps skipped because rows could
            not be mapped back to KV slots.
        skipped_no_store_steps: Steps containing no STORE request.
        skipped_nonpositive_requests: STORE requests with empty token ranges.
        skipped_unaligned_requests: STORE requests not aligned to ring blocks.
        skipped_unsupported_layout_requests: STORE requests whose GPU block
            layout query capture does not support.
        skipped_partial_step_requests: STORE requests whose complete token
            range was not present in the current forward step.
        skipped_ring_exhausted_requests: STORE requests rejected by ring
            backpressure.
        ring_exhausted_requested_blocks: Blocks requested by rejected stores.
        ring_exhausted_shortfall_blocks: Additional blocks required for those
            rejected stores at allocation time.
        ring: Current and high-water ring block usage.
    """

    steps_attempted: int
    steps_captured: int
    steps_skipped: int
    store_requests_seen: int
    requests_captured: int
    tokens_captured: int
    blocks_reserved: int
    plan_duration_ns_total: int
    plan_duration_ns_max: int
    skipped_ring_unavailable_steps: int
    skipped_missing_slot_mapping_steps: int
    skipped_no_store_steps: int
    skipped_nonpositive_requests: int
    skipped_unaligned_requests: int
    skipped_unsupported_layout_requests: int
    skipped_partial_step_requests: int
    skipped_ring_exhausted_requests: int
    ring_exhausted_requested_blocks: int
    ring_exhausted_shortfall_blocks: int
    ring: QRingBufferUsageSnapshot


@dataclass(frozen=True)
class QStoreMetricsSnapshot:
    """Cumulative query-store metrics and current in-flight pressure.

    Attributes:
        requests_submitted: Stores accepted by the transfer context.
        requests_completed: Submitted stores completing successfully.
        requests_failed: Submitted stores returning failure or raising while
            their completion is reclaimed.
        requests_submission_failed: Stores raising synchronously at submit.
        requests_dropped_unavailable: Stores dropped because the ring or
            transfer context was unavailable.
        requests_dropped_unhealthy: Stores dropped before submit because the
            worker adapter was unhealthy.
        requests_dropped_missing_token_ids: Stores dropped because a cache key
            could not be constructed.
        requests_dropped_missing_event: Captures dropped at forward exit when
            vLLM supplied no completion event.
        requests_abandoned_unhealthy: Previously submitted stores reclaimed
            without polling after the adapter became unhealthy.
        blocks_submitted: Ring blocks handed to the transfer context.
        blocks_released: Ring blocks returned on every terminal path.
        completion_duration_ns_total: Total host-observed submit-to-reclaim
            time for completed and failed asynchronous stores.
        completion_duration_ns_max: Maximum host-observed submit-to-reclaim
            time for one asynchronous store.
        in_flight_requests: Submitted stores awaiting completion.
        in_flight_blocks: Ring blocks held by in-flight stores.
        high_watermark_in_flight_requests: Maximum simultaneous stores.
        high_watermark_in_flight_blocks: Maximum blocks held by stores.
    """

    requests_submitted: int
    requests_completed: int
    requests_failed: int
    requests_submission_failed: int
    requests_dropped_unavailable: int
    requests_dropped_unhealthy: int
    requests_dropped_missing_token_ids: int
    requests_dropped_missing_event: int
    requests_abandoned_unhealthy: int
    blocks_submitted: int
    blocks_released: int
    completion_duration_ns_total: int
    completion_duration_ns_max: int
    in_flight_requests: int
    in_flight_blocks: int
    high_watermark_in_flight_requests: int
    high_watermark_in_flight_blocks: int


@dataclass(frozen=True)
class QPipelineMetricsSnapshot:
    """Point-in-time query capture and asynchronous-store metrics.

    Attributes:
        capture: Capture-plan outcomes and ring pressure.
        store: Store outcomes, latency, and in-flight pressure.
    """

    capture: QCaptureMetricsSnapshot
    store: QStoreMetricsSnapshot


@dataclass
class _QCaptureAttempt:
    """Metrics accumulated locally while building one capture plan."""

    store_requests_seen: int = 0
    requests_captured: int = 0
    tokens_captured: int = 0
    blocks_reserved: int = 0
    skipped_ring_unavailable_steps: int = 0
    skipped_missing_slot_mapping_steps: int = 0
    skipped_no_store_steps: int = 0
    skipped_nonpositive_requests: int = 0
    skipped_unaligned_requests: int = 0
    skipped_unsupported_layout_requests: int = 0
    skipped_partial_step_requests: int = 0
    skipped_ring_exhausted_requests: int = 0
    ring_exhausted_requested_blocks: int = 0
    ring_exhausted_shortfall_blocks: int = 0


class _QCaptureMetricsRecorder:
    """Thread-safe cumulative query-capture metric recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps_attempted = 0
        self._steps_captured = 0
        self._store_requests_seen = 0
        self._requests_captured = 0
        self._tokens_captured = 0
        self._blocks_reserved = 0
        self._plan_duration_ns_total = 0
        self._plan_duration_ns_max = 0
        self._skipped_ring_unavailable_steps = 0
        self._skipped_missing_slot_mapping_steps = 0
        self._skipped_no_store_steps = 0
        self._skipped_nonpositive_requests = 0
        self._skipped_unaligned_requests = 0
        self._skipped_unsupported_layout_requests = 0
        self._skipped_partial_step_requests = 0
        self._skipped_ring_exhausted_requests = 0
        self._ring_exhausted_requested_blocks = 0
        self._ring_exhausted_shortfall_blocks = 0

    def record(
        self,
        attempt: _QCaptureAttempt,
        captured: bool,
        duration_ns: int,
    ) -> None:
        """Merge one capture attempt into the cumulative counters."""
        with self._lock:
            self._steps_attempted += 1
            self._steps_captured += int(captured)
            self._store_requests_seen += attempt.store_requests_seen
            self._requests_captured += attempt.requests_captured
            self._tokens_captured += attempt.tokens_captured
            self._blocks_reserved += attempt.blocks_reserved
            self._plan_duration_ns_total += duration_ns
            self._plan_duration_ns_max = max(self._plan_duration_ns_max, duration_ns)
            self._skipped_ring_unavailable_steps += (
                attempt.skipped_ring_unavailable_steps
            )
            self._skipped_missing_slot_mapping_steps += (
                attempt.skipped_missing_slot_mapping_steps
            )
            self._skipped_no_store_steps += attempt.skipped_no_store_steps
            self._skipped_nonpositive_requests += attempt.skipped_nonpositive_requests
            self._skipped_unaligned_requests += attempt.skipped_unaligned_requests
            self._skipped_unsupported_layout_requests += (
                attempt.skipped_unsupported_layout_requests
            )
            self._skipped_partial_step_requests += attempt.skipped_partial_step_requests
            self._skipped_ring_exhausted_requests += (
                attempt.skipped_ring_exhausted_requests
            )
            self._ring_exhausted_requested_blocks += (
                attempt.ring_exhausted_requested_blocks
            )
            self._ring_exhausted_shortfall_blocks += (
                attempt.ring_exhausted_shortfall_blocks
            )

    def snapshot(self, ring: QRingBufferUsageSnapshot) -> QCaptureMetricsSnapshot:
        """Return an immutable, internally consistent counter snapshot."""
        with self._lock:
            steps_attempted = self._steps_attempted
            steps_captured = self._steps_captured
            return QCaptureMetricsSnapshot(
                steps_attempted=steps_attempted,
                steps_captured=steps_captured,
                steps_skipped=steps_attempted - steps_captured,
                store_requests_seen=self._store_requests_seen,
                requests_captured=self._requests_captured,
                tokens_captured=self._tokens_captured,
                blocks_reserved=self._blocks_reserved,
                plan_duration_ns_total=self._plan_duration_ns_total,
                plan_duration_ns_max=self._plan_duration_ns_max,
                skipped_ring_unavailable_steps=(self._skipped_ring_unavailable_steps),
                skipped_missing_slot_mapping_steps=(
                    self._skipped_missing_slot_mapping_steps
                ),
                skipped_no_store_steps=self._skipped_no_store_steps,
                skipped_nonpositive_requests=self._skipped_nonpositive_requests,
                skipped_unaligned_requests=self._skipped_unaligned_requests,
                skipped_unsupported_layout_requests=(
                    self._skipped_unsupported_layout_requests
                ),
                skipped_partial_step_requests=(self._skipped_partial_step_requests),
                skipped_ring_exhausted_requests=(self._skipped_ring_exhausted_requests),
                ring_exhausted_requested_blocks=(self._ring_exhausted_requested_blocks),
                ring_exhausted_shortfall_blocks=(self._ring_exhausted_shortfall_blocks),
                ring=ring,
            )


class _QStoreMetricsRecorder:
    """Thread-safe cumulative query-store metric recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_submitted = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._requests_submission_failed = 0
        self._requests_dropped_unavailable = 0
        self._requests_dropped_unhealthy = 0
        self._requests_dropped_missing_token_ids = 0
        self._requests_dropped_missing_event = 0
        self._requests_abandoned_unhealthy = 0
        self._blocks_submitted = 0
        self._blocks_released = 0
        self._completion_duration_ns_total = 0
        self._completion_duration_ns_max = 0
        self._in_flight_requests = 0
        self._in_flight_blocks = 0
        self._high_watermark_in_flight_requests = 0
        self._high_watermark_in_flight_blocks = 0

    def record_submitted(self, blocks: int) -> None:
        """Record one successfully submitted asynchronous store."""
        with self._lock:
            self._requests_submitted += 1
            self._blocks_submitted += blocks
            self._in_flight_requests += 1
            self._in_flight_blocks += blocks
            self._high_watermark_in_flight_requests = max(
                self._high_watermark_in_flight_requests,
                self._in_flight_requests,
            )
            self._high_watermark_in_flight_blocks = max(
                self._high_watermark_in_flight_blocks,
                self._in_flight_blocks,
            )

    def record_finished(
        self,
        blocks: int,
        succeeded: bool,
        duration_ns: int,
    ) -> None:
        """Record one reclaimed asynchronous store."""
        with self._lock:
            if succeeded:
                self._requests_completed += 1
            else:
                self._requests_failed += 1
            self._blocks_released += blocks
            self._completion_duration_ns_total += duration_ns
            self._completion_duration_ns_max = max(
                self._completion_duration_ns_max, duration_ns
            )
            self._in_flight_requests -= 1
            self._in_flight_blocks -= blocks

    def record_submission_failed(self, blocks: int) -> None:
        """Record a synchronous submission failure and released blocks."""
        with self._lock:
            self._requests_submission_failed += 1
            self._blocks_released += blocks

    def record_dropped_unavailable(self, blocks: int) -> None:
        """Record a store dropped before submit because transport is absent."""
        with self._lock:
            self._requests_dropped_unavailable += 1
            self._blocks_released += blocks

    def record_dropped_unhealthy(self, blocks: int) -> None:
        """Record a store dropped before submit because transport is unhealthy."""
        with self._lock:
            self._requests_dropped_unhealthy += 1
            self._blocks_released += blocks

    def record_dropped_missing_token_ids(self, blocks: int) -> None:
        """Record a store dropped because its key cannot be built."""
        with self._lock:
            self._requests_dropped_missing_token_ids += 1
            self._blocks_released += blocks

    def record_dropped_missing_event(self, blocks: int) -> None:
        """Record a capture discarded because no forward event exists."""
        with self._lock:
            self._requests_dropped_missing_event += 1
            self._blocks_released += blocks

    def record_abandoned_unhealthy(self, requests: int, blocks: int) -> None:
        """Record submitted stores reclaimed after transport became unhealthy."""
        with self._lock:
            self._requests_abandoned_unhealthy += requests
            self._blocks_released += blocks
            self._in_flight_requests -= requests
            self._in_flight_blocks -= blocks

    def snapshot(self) -> QStoreMetricsSnapshot:
        """Return an immutable, internally consistent store snapshot."""
        with self._lock:
            return QStoreMetricsSnapshot(
                requests_submitted=self._requests_submitted,
                requests_completed=self._requests_completed,
                requests_failed=self._requests_failed,
                requests_submission_failed=self._requests_submission_failed,
                requests_dropped_unavailable=(self._requests_dropped_unavailable),
                requests_dropped_unhealthy=self._requests_dropped_unhealthy,
                requests_dropped_missing_token_ids=(
                    self._requests_dropped_missing_token_ids
                ),
                requests_dropped_missing_event=self._requests_dropped_missing_event,
                requests_abandoned_unhealthy=(self._requests_abandoned_unhealthy),
                blocks_submitted=self._blocks_submitted,
                blocks_released=self._blocks_released,
                completion_duration_ns_total=(self._completion_duration_ns_total),
                completion_duration_ns_max=self._completion_duration_ns_max,
                in_flight_requests=self._in_flight_requests,
                in_flight_blocks=self._in_flight_blocks,
                high_watermark_in_flight_requests=(
                    self._high_watermark_in_flight_requests
                ),
                high_watermark_in_flight_blocks=(self._high_watermark_in_flight_blocks),
            )


@dataclass
class _QLayerStore:
    """One request's query tensor store for the current forward step.

    Attributes:
        request_id: The request ID of the stored query tensor.
        op: The store op whose ``[start, end)`` token range is stored.
        ring_block_ids: Ring blocks (in token order) holding this op's query.
        cache_salt: Per-user isolation salt for the query key.
    """

    request_id: str
    op: "LoadStoreOp"
    ring_block_ids: list[int]
    cache_salt: str


@dataclass
class _QStepState:
    """Plan to store query tensors for the current forward step, built on the
    first layer's save_kv_layer call and reused by every layer in the step.

    Attributes:
        ring_slots: ``int64`` tensor of shape ``[num_tokens]`` mapping each
            token to ring slot ID, or ``-1`` to drop the row.
        stores: one ``_QLayerStore`` per store request in the step.
    """

    ring_slots: torch.Tensor
    stores: list[_QLayerStore]


@dataclass
class _QStoreFutureState:
    """One submitted query store awaiting asynchronous completion.

    Attributes:
        future: Transfer-context future for the store result.
        ring_block_ids: Ring blocks held until the future is terminal.
        submitted_ns: Monotonic host timestamp immediately before submission.
    """

    future: "MessagingFuture[StoreResult]"
    ring_block_ids: list[int]
    submitted_ns: int


class QRingBuffer:
    """A ring buffer for storing query tensors in LMCache.
    The buffer has shape [num_layers, num_blocks, block_size, hidden_dim].
    The buffer is paged in blocks to use existing LMCache machinery for
    transferring paged KV tensors.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Initialize the QRingBuffer."""
        if num_layers <= 0 or num_blocks <= 0 or block_size <= 0 or hidden_dim <= 0:
            raise ValueError(
                "QRingBuffer requires positive num_layers, num_blocks, "
                f"block_size and hidden_dim; got num_layers={num_layers}, "
                f"num_blocks={num_blocks}, block_size={block_size}, "
                f"hidden_dim={hidden_dim}"
            )
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device

        self._layer_tensors: list[torch.Tensor] = [
            torch.empty(
                (num_blocks, block_size, hidden_dim), dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        self.tensors: dict[str, torch.Tensor] = {
            f"lmcache_q_layer_{i}": self._layer_tensors[i] for i in range(num_layers)
        }
        self._free_blocks: list[int] = list(range(num_blocks))
        self._allocation_lock = threading.Lock()
        self._high_watermark_blocks = 0

    def num_free_blocks(self) -> int:
        """The number of free blocks in the ring buffer."""
        with self._allocation_lock:
            return len(self._free_blocks)

    def usage_snapshot(self) -> QRingBufferUsageSnapshot:
        """Return a consistent snapshot of current and peak block pressure."""
        with self._allocation_lock:
            free_blocks = len(self._free_blocks)
            return QRingBufferUsageSnapshot(
                capacity_blocks=self.num_blocks,
                used_blocks=self.num_blocks - free_blocks,
                free_blocks=free_blocks,
                high_watermark_blocks=self._high_watermark_blocks,
            )

    def allocate(self, n: int) -> list[int] | None:
        """Allocating n blocks from the ring buffer.

        Returns:
            A list of allocated block IDs if successful.
            None if not enough free blocks are available.

        Raises:
            ValueError: If number of blocks requested is invalid.
        """
        if n < 0:
            raise ValueError(f"cannot allocate a negative block count: {n}")
        with self._allocation_lock:
            if n > len(self._free_blocks):
                return None
            reserved = self._free_blocks[-n:] if n else []
            del self._free_blocks[len(self._free_blocks) - n :]
            used_blocks = self.num_blocks - len(self._free_blocks)
            self._high_watermark_blocks = max(self._high_watermark_blocks, used_blocks)
            return reserved

    def free(self, block_ids: list[int]) -> None:
        """Free the given block IDs back to the ring buffer.
        Drop invalid or already freed block IDs with a warning."""
        with self._allocation_lock:
            free_blocks = set(self._free_blocks)
            for block_id in block_ids:
                if block_id < 0 or block_id >= self.num_blocks:
                    logger.warning(
                        "free() called with invalid block_id %d (valid [0, %d))",
                        block_id,
                        self.num_blocks,
                    )
                    continue
                if block_id in free_blocks:
                    logger.warning(
                        "free() called with already freed block_id %d",
                        block_id,
                    )
                    continue
                free_blocks.add(block_id)
                self._free_blocks.append(block_id)

    def scatter(
        self,
        layer_index: int,
        query: torch.Tensor,
        ring_slots: torch.Tensor,
    ) -> None:
        """Scatter the query tensor into the ring buffer at the given layer."""
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError(
                f"layer_index {layer_index} out of range [0, {self.num_layers})"
            )
        flat_q = query.reshape(query.shape[0], -1)
        if flat_q.shape[1] != self.hidden_dim:
            raise ValueError(
                f"query flattens to width {flat_q.shape[1]} but the ring was "
                f"allocated for hidden_dim {self.hidden_dim}; check the "
                "configured query head count / head size"
            )
        ring = self._layer_tensors[layer_index]
        valid = ring_slots >= 0
        slots = ring_slots[valid]
        if slots.numel() == 0:
            return
        ring[slots // self.block_size, slots % self.block_size] = flat_q[valid].to(
            ring.dtype
        )


def get_tensor(
    args: dict[str, Any],
    arg_names: list[str],
) -> torch.Tensor | None:
    """Return the first matching tensor argument, or None.

    Args:
        args: A dictionary of argument names to values.
        arg_names: A list of argument names to search for.

    Returns:
        The first matching tensor argument, None if none found.
    """
    for name in arg_names:
        if name in args:
            return args[name]
    return None


def attention_layer_names_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
) -> list[str]:
    """Return the attention layers' names.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig``, or ``None``.
        kv_caches: Registered tensors keyed by layer name.

    Returns:
        Attention layer names, ordered as in ``kv_caches``.
    """
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    if not vllm_groups:
        return list(kv_caches.keys())

    # Third party
    # Third Party
    from vllm.v1.kv_cache_interface import AttentionSpec

    attention_names: set[str] = set()
    for group in vllm_groups:
        if isinstance(group.kv_cache_spec, AttentionSpec):
            attention_names.update(group.layer_names)
    return [name for name in kv_caches.keys() if name in attention_names]


class QRingBufferCapture:
    def __init__(
        self,
        worker_adapter: "LMCacheMPWorkerAdapter",
        q_ring_adapter: "QRingBufferAdapter",
    ) -> None:
        self.worker_adapter = worker_adapter
        self.q_ring_adapter = q_ring_adapter
        self.q_layer_index: dict[str, int] = {}
        self.q_step_state: _QStepState | None = None
        self.q_step_disabled: bool = False
        self._metrics = _QCaptureMetricsRecorder()

    def metrics_snapshot(self) -> QCaptureMetricsSnapshot:
        """Return cumulative capture outcomes and current ring pressure."""
        q_ring = self.q_ring_adapter.q_ring
        ring_usage = (
            q_ring.usage_snapshot()
            if q_ring is not None
            else QRingBufferUsageSnapshot(0, 0, 0, 0)
        )
        return self._metrics.snapshot(ring_usage)

    def setup_q_ring(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: "KVCacheConfig | None",
        vllm_config: "VllmConfig",
    ) -> None:
        """Allocate and register the paged-Q ring for the attention layers.
        Builds the mapping between layer names and ring indices (for attention layers).
        Calls the register_q_ring method of the worker adapter to allocate the ring.

        Args:
            kv_caches: Registered KV caches.
            kv_cache_config: vLLM KVCacheConfig, or None.
            vllm_config: vLLM configuration.

        Raises:
            ValueError: If the kv_transfer_config is None.
        """
        if not kv_caches:
            logger.warning("No KV caches registered; skipping Q ring setup.")
            return

        attn_layer_names = attention_layer_names_from_vllm(kv_cache_config, kv_caches)
        if not attn_layer_names:
            logger.warning("No attention layers registered. Skipping Q ring setup.")
            return
        num_layers = len(attn_layer_names)

        self.q_layer_index = {}
        for i, name in enumerate(attn_layer_names):
            self.q_layer_index[name] = i

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        num_q_heads = model_config.get_num_attention_heads(parallel_config)
        head_size = model_config.get_head_size()

        raw_dtype = model_config.dtype
        dtype = (
            raw_dtype
            if isinstance(raw_dtype, torch.dtype)
            else getattr(torch, str(raw_dtype))
        )
        device = get_device(next(iter(kv_caches.values())))
        block_size = vllm_config.cache_config.block_size

        cfg = vllm_config.kv_transfer_config
        if cfg is None:
            raise ValueError("kv_transfer_config is None; cannot register Q ring")

        explicit = cfg.get_from_extra_config("lmcache.q.ring_blocks", None)
        if explicit is not None:
            num_ring_blocks = max(1, int(explicit))
        else:
            depth = max(1, int(cfg.get_from_extra_config("lmcache.q.ring_depth", 2)))
            max_batched = (
                getattr(vllm_config.scheduler_config, "max_num_batched_tokens", None)
                or 8192
            )
            num_ring_blocks = max(1, math.ceil(max_batched / block_size) * depth)

        self.q_ring_adapter.register_q_ring(
            num_layers=num_layers,
            num_q_heads=num_q_heads,
            head_size=head_size,
            dtype=dtype,
            num_ring_blocks=num_ring_blocks,
            device=device,
        )

    def save_q_layer(
        self,
        layer_name: str,
        metadata: "LMCacheMPConnectorMetadata",
        **kwargs,
    ) -> None:
        """Scatter this layer's query into the ring, building the per-step plan on
        the first attention layer of the step.
        """
        if not self.worker_adapter.is_kv_writer or self.q_step_disabled:
            return

        intermediate_tensors = kwargs.get("intermediate_tensors")
        if intermediate_tensors is None:
            return
        query = get_tensor(intermediate_tensors, ["q", "query"])
        if query is None:
            return

        layer_index = self.q_layer_index.get(layer_name)
        if layer_index is None:
            return

        if self.q_step_state is None:
            self.q_step_state = self._build_q_step_state(
                query, metadata, kwargs.get("attn_metadata")
            )
            if self.q_step_state is None:
                self.q_step_disabled = True
                return

        self.q_ring_adapter.scatter_q_layer(
            layer_index, query, self.q_step_state.ring_slots
        )

    def _build_q_step_state(
        self,
        query: torch.Tensor,
        metadata: "LMCacheMPConnectorMetadata",
        attn_metadata: Any = None,
    ) -> "_QStepState | None":
        """Build the per-step query store plan for the current forward step.
        Done only on the first layer's save_kv_layer call of the step.

        Query rows are matched to each store op's tokens through
        ``attn_metadata.slot_mapping``: row ``r`` writes its KV to GPU slot
        ``slot_mapping[r]``, and the op's token ``i`` lives in GPU slot
        ``block_ids[i // block_size] * block_size + i % block_size`` (the op's
        block list is pre-sliced to its ``[start, end)`` range). Intersecting
        the two identifies each op's rows regardless of batch composition or
        ordering, so mixed STORE/RETRIEVE steps and requests whose scheduled
        token count differs from their store-op token count (chunk-unaligned
        prompt tails, APC offsets) no longer shift later requests' rows.

        Args:
            query: The query tensor of shape [num_tokens, num_q_heads, head_size].
            metadata: The LMCache connector metadata.
            attn_metadata: The vLLM per-layer attention metadata; must expose
                ``slot_mapping`` for query capture to run.

        Returns:
            _QStepState: the query capture plan (containing ring slot mapping
                            and per-request store ops).
            None if not storing any query tensors.
        """
        started_ns = time.perf_counter_ns()
        attempt = _QCaptureAttempt()
        q_ring = self.q_ring_adapter.q_ring
        if q_ring is None:
            attempt.skipped_ring_unavailable_steps = 1
            return self._finish_capture_attempt(attempt, None, started_ns)
        block_size = q_ring.block_size

        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            logger.warning(
                "Skip query capture for this step: attention metadata does "
                "not expose slot_mapping, so query rows cannot be attributed "
                "to tokens."
            )
            attempt.skipped_missing_slot_mapping_steps = 1
            return self._finish_capture_attempt(attempt, None, started_ns)

        num_query_rows = query.shape[0]
        # Rows beyond slot_mapping (CUDA-graph padding) and rows with slot -1
        # never write KV and are dropped from capture.
        n_rows = min(num_query_rows, slot_mapping.shape[0])
        row_slots = slot_mapping[:n_rows].to(device=query.device, dtype=torch.int64)
        ring_slots = torch.full(
            (num_query_rows,), -1, dtype=torch.int64, device=query.device
        )
        offsets = torch.arange(block_size, device=query.device)

        stores: list[_QLayerStore] = []
        for meta in metadata.requests:
            if meta.direction != "STORE":
                continue
            attempt.store_requests_seen += 1
            op = meta.op
            num_tokens = op.end - op.start
            if num_tokens <= 0:
                attempt.skipped_nonpositive_requests += 1
                continue

            if num_tokens % block_size != 0:
                attempt.skipped_unaligned_requests += 1
                logger.warning(
                    "Skip query for %s: tokens (%d) undivisible by block size",
                    meta.request_id,
                    num_tokens,
                )
                continue

            gpu_blocks = self._op_gpu_blocks(op)
            if gpu_blocks is None or len(gpu_blocks) * block_size != num_tokens:
                attempt.skipped_unsupported_layout_requests += 1
                logger.warning(
                    "Skip query for request %s: op block layout does not "
                    "cover its token range (%d blocks of %d tokens for %d "
                    "tokens)",
                    meta.request_id,
                    0 if gpu_blocks is None else len(gpu_blocks),
                    block_size,
                    num_tokens,
                )
                continue

            # GPU KV slot of the op's token i, in token order.
            block_tensor = torch.tensor(
                gpu_blocks, dtype=torch.int64, device=query.device
            )
            op_slots = (block_tensor[:, None] * block_size + offsets[None, :]).reshape(
                -1
            )

            # Match rows to op tokens through the KV slot each row writes.
            # Each GPU slot is written by at most one row per step, so a full
            # match is a bijection between the op's tokens and its rows.
            sorted_slots, token_order = torch.sort(op_slots)
            pos = torch.searchsorted(sorted_slots, row_slots)
            pos_clamped = pos.clamp(max=num_tokens - 1)
            hit = (row_slots >= 0) & (sorted_slots[pos_clamped] == row_slots)
            num_matched = int(hit.sum().item())
            if num_matched != num_tokens:
                attempt.skipped_partial_step_requests += 1
                logger.warning(
                    "Skip query for request %s: only %d of %d op tokens are "
                    "in this step's batch (tokens computed in an earlier "
                    "chunked-prefill step have no query rows here)",
                    meta.request_id,
                    num_matched,
                    num_tokens,
                )
                continue

            n_blocks = num_tokens // block_size
            ring_blocks = q_ring.allocate(n_blocks)
            if ring_blocks is None:
                free_blocks = q_ring.num_free_blocks()
                attempt.skipped_ring_exhausted_requests += 1
                attempt.ring_exhausted_requested_blocks += n_blocks
                attempt.ring_exhausted_shortfall_blocks += max(
                    0, n_blocks - free_blocks
                )
                logger.warning(
                    "Skip query for request %s: need %d Q blocks, %d free",
                    meta.request_id,
                    n_blocks,
                    free_blocks,
                )
                continue

            ring_block_tensor = torch.tensor(
                ring_blocks, dtype=torch.int64, device=query.device
            )
            ring_slot_by_token = (
                ring_block_tensor[:, None] * block_size + offsets[None, :]
            ).reshape(-1)
            matched_token_idx = token_order[pos_clamped[hit]]
            ring_slots[:n_rows][hit] = ring_slot_by_token[matched_token_idx]
            stores.append(
                _QLayerStore(
                    request_id=meta.request_id,
                    op=op,
                    ring_block_ids=ring_blocks,
                    cache_salt=meta.cache_salt,
                )
            )
            attempt.requests_captured += 1
            attempt.tokens_captured += num_tokens
            attempt.blocks_reserved += n_blocks

        if not stores:
            if attempt.store_requests_seen == 0:
                attempt.skipped_no_store_steps = 1
            return self._finish_capture_attempt(attempt, None, started_ns)

        state = _QStepState(ring_slots=ring_slots, stores=stores)
        return self._finish_capture_attempt(attempt, state, started_ns)

    def _finish_capture_attempt(
        self,
        attempt: _QCaptureAttempt,
        state: _QStepState | None,
        started_ns: int,
    ) -> _QStepState | None:
        """Record one plan outcome and return it unchanged."""
        duration_ns = max(0, time.perf_counter_ns() - started_ns)
        self._metrics.record(attempt, state is not None, duration_ns)
        return state

    @staticmethod
    def _op_gpu_blocks(op: "LoadStoreOp") -> list[int] | None:
        """The op's GPU block IDs as a flat list, or None when the layout is
        not the single-group one query capture supports.

        Handles both the normal ``list[list[int]]`` format and the
        IPC-flattened ``list[int]`` format (see ``LoadStoreOp.flat_block_ids``).
        """
        block_ids = op.block_ids
        if not block_ids:
            return None
        if isinstance(block_ids[0], int):
            return list(block_ids)  # type: ignore[arg-type]
        if len(block_ids) == 1:
            return list(block_ids[0])
        return None

    def batched_submit_qstore_requests(self, event: "_IpcEvent | None") -> None:
        """
        Submit a batched Q store request to LMCache.
        A copy of batched_submit_store_requests for Q stores.
        The recorded step state has all request IDs, ops, ring block IDs, and
        cache salts for the current forward step.

        Args:
            event: The CUDA event that is recorded after the current
                model inference step
        """
        state = self.q_step_state
        self.q_step_state = None
        self.q_step_disabled = False

        if state is None:
            return
        if event is None:
            for q_store in state.stores:
                self.q_ring_adapter.discard_q_store_without_event(
                    q_store.ring_block_ids
                )
            return

        first_error: Exception | None = None
        for q_store in state.stores:
            try:
                self.q_ring_adapter.submit_q_store_request(
                    q_store.request_id,
                    q_store.op,
                    q_store.ring_block_ids,
                    event,
                    cache_salt=q_store.cache_salt,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class QRingBufferAdapter:
    """Adapter for managing the QRingBuffer and its interaction with LMCache"""

    def __init__(
        self,
        adapter: "LMCacheMPWorkerAdapter",
        q_model_name: str,
    ) -> None:
        self._adapter = adapter
        self.q_model_name = q_model_name

        self.q_ring: QRingBuffer | None = None
        self.q_engine_group_infos: Sequence[EngineGroupInfo] | None = None
        self.q_store_futures: dict[int, _QStoreFutureState] = {}
        self.q_store_events: dict[int, _IpcEvent] = {}
        self._q_store_seq: int = 0
        self._metrics = _QStoreMetricsRecorder()

    def metrics_snapshot(self) -> QStoreMetricsSnapshot:
        """Return cumulative store outcomes and current in-flight pressure."""
        return self._metrics.snapshot()

    def register_q_ring(
        self,
        num_layers: int,
        num_q_heads: int,
        head_size: int,
        dtype: torch.dtype,
        num_ring_blocks: int,
        device: torch.device,
    ) -> None:
        """Register the paged Q ring with LMCache server.

        Args:
            num_layers: The number of layers in the model.
            num_q_heads: The number of query heads in the model.
            head_size: The size of each query head.
            dtype: The data type of the query tensor.
            num_ring_blocks: The number of ring blocks to allocate.
            device: The device to allocate the query tensor on.

        Raises:
            RuntimeError: if the transfer context is not established.
        """
        logger.info("Registering paged-Q ring")
        if not self._adapter.transfer_ctx:
            raise RuntimeError(
                "register_q_ring() requires an established transfer context; "
                "call register_kv_caches() first."
            )
        block_size = (
            self._adapter.lmcache_tokens_per_chunk // self._adapter.blocks_in_chunk
        )
        self.q_ring = QRingBuffer(
            num_layers=num_layers,
            num_blocks=num_ring_blocks,
            block_size=block_size,
            hidden_dim=num_q_heads * head_size,
            dtype=dtype,
            device=device,
        )
        self.q_engine_group_infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(num_layers)),
                tokens_per_block=block_size,
                sw_size_tokens=-1,
            )
        ]

        self._send_register_q_ring_request()

    def _send_register_q_ring_request(self) -> None:
        """Registers the paged Q ring alongside the KV caches via the same
        transfer context so both Q and KV uses the same LMCache storage buffer.

        Raises:
            RuntimeError: if the Q ring or transfer context is not initialized.
            ConnectionError: if the server does not respond within mq_timeout.
        """
        if (
            self.q_ring is None
            or self._adapter.transfer_ctx is None
            or self.q_engine_group_infos is None
        ):
            raise RuntimeError("Q ring is not initialized yet.")
        try:
            self._adapter.transfer_ctx.register_q(
                self._adapter.instance_id,
                self.q_ring.tensors,
                self.q_model_name,
                self._adapter.world_size,
                self._adapter.blocks_in_chunk,
                self._adapter.req_client,
                self._adapter._mq_timeout,
                layout_hints=vllm_layout_hints(),
                engine_group_infos=self.q_engine_group_infos,
            )
        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to Q ring registration within "
                f"{self._adapter._mq_timeout}s."
            ) from None

    def reregister_q_ring(self) -> None:
        """Re-register an already-built Q ring after a server recovery.

        A no-op when the ring was never built (nothing to recover).

        Raises:
            ConnectionError: if the server does not respond within mq_timeout.
        """
        if self.q_ring is None:
            return
        self._send_register_q_ring_request()

    def scatter_q_layer(
        self,
        layer_index: int,
        query: torch.Tensor,
        ring_slots: torch.Tensor,
    ) -> None:
        """Scatter query tensor to QRingBuffer for layer_index at ring_slots.

        Args:
            layer_index: The index of the layer to scatter the query tensor.
            query: The query tensor to scatter [T, num_q_heads, head_size].
            ring_slots: The ring slot IDs for each token in the query tensor.
        """
        if not self.q_ring:
            return
        self.q_ring.scatter(layer_index, query, ring_slots)

    def discard_q_store_without_event(self, ring_block_ids: list[int]) -> None:
        """Release a captured request when no forward completion event exists.

        Args:
            ring_block_ids: Ring blocks reserved for the discarded request.
        """
        if self.q_ring is not None:
            self.q_ring.free(ring_block_ids)
        self._metrics.record_dropped_missing_event(len(ring_block_ids))

    def submit_q_store_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        ring_block_ids: list[int],
        event: "_IpcEvent",
        cache_salt: str = "",
    ) -> None:
        """Submit a store request for QRingBuffer content at ring_block_ids
        to LMCache. This is called after the query tensor is scattered, every
        time wait_for_save is called.

        Args:
            request_id: The ID of the request which query belongs to.
            op: The LoadStoreOp with the token_ids, start and end indices.
            ring_block_ids: The ring block IDs to store.
            event: The IPC event to signal when the store is complete.
            cache_salt: Per-user isolation salt.
        """
        q_ring = self.q_ring
        if q_ring is None:
            self._metrics.record_dropped_unavailable(0)
            return
        if not self._adapter.transfer_ctx:
            q_ring.free(ring_block_ids)
            self._metrics.record_dropped_unavailable(len(ring_block_ids))
            return
        try:
            self._adapter._ensure_heartbeat_started()
            if not self._adapter.is_healthy:
                q_ring.free(ring_block_ids)
                self._metrics.record_dropped_unhealthy(len(ring_block_ids))
                return
            if op.token_ids is None:
                logger.warning("Skipping Q store for %s: token_ids is None", request_id)
                q_ring.free(ring_block_ids)
                self._metrics.record_dropped_missing_token_ids(len(ring_block_ids))
                return
            key = self._adapter._create_key(
                op.token_ids,
                op.start,
                op.end,
                request_id=request_id,
                cache_salt=cache_salt,
            )
            key = replace(key, model_name=self.q_model_name)
            submitted_ns = time.perf_counter_ns()
            future = self._adapter.transfer_ctx.submit_q_store(
                request_id,
                key,
                self._adapter.instance_id,
                q_ring.tensors,
                [ring_block_ids],
                event,
                self._adapter.blocks_in_chunk,
            )
        except Exception:
            q_ring.free(ring_block_ids)
            self._metrics.record_submission_failed(len(ring_block_ids))
            raise
        seq = self._q_store_seq
        self._q_store_seq += 1
        self.q_store_futures[seq] = _QStoreFutureState(
            future=future,
            ring_block_ids=ring_block_ids,
            submitted_ns=submitted_ns,
        )
        self.q_store_events[seq] = event
        self._metrics.record_submitted(len(ring_block_ids))

    def reclaim_finished_q_stores(self) -> None:
        """Reclaim ring blocks when query has been saved to LMCache.
        This is called in get_finished()."""
        if not self.q_ring or not self.q_store_futures:
            return
        if not self._adapter.is_healthy:
            abandoned_requests = len(self.q_store_futures)
            abandoned_blocks = 0
            for state in self.q_store_futures.values():
                abandoned_blocks += len(state.ring_block_ids)
                self.q_ring.free(state.ring_block_ids)
            self.q_store_futures.clear()
            self.q_store_events.clear()
            self._metrics.record_abandoned_unhealthy(
                abandoned_requests, abandoned_blocks
            )
            return
        done: list[int] = []
        for seq, state in self.q_store_futures.items():
            succeeded = False
            try:
                if not state.future.query():
                    continue
                succeeded = bool(state.future.result())
                if not succeeded:
                    logger.error("Q store failed for seq=%d", seq)
            except Exception:
                logger.exception("Q store completion raised for seq=%d", seq)
            self.q_ring.free(state.ring_block_ids)
            duration_ns = max(0, time.perf_counter_ns() - state.submitted_ns)
            self._metrics.record_finished(
                len(state.ring_block_ids), succeeded, duration_ns
            )
            done.append(seq)
        for seq in done:
            self.q_store_futures.pop(seq, None)
            self.q_store_events.pop(seq, None)

    def shutdown_q_ring(self) -> None:
        """Unregister the paged Q ring from LMCache server."""
        if not self.q_ring:
            return
        try:
            self._adapter.req_client.unregister_q_cache(
                self._adapter.instance_id
            ).result(timeout=self._adapter._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LMCache server did not respond to Q ring unregister within "
                "%ss. Proceeding with shutdown.",
                self._adapter._mq_timeout,
            )
