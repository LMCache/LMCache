# SPDX-License-Identifier: Apache-2.0
"""DeepInfer client adapter for LMCache multiprocess mode.

The MP server and transfer core stay engine-neutral.  This module only maps
DeepInfer's page-native CA/ICA views and full runtime state snapshots onto the
DeepInfer-owned multiprocess adapter and recurrent-state object-group semantics.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Any

# Third Party
from deepinfer.kv_transfer.base_connector import (
    KVConnectorTransferResults,
    KVTransferConnector,
    KVTransferRole,
)
import torch
import zmq

# First Party
from lmcache import torch_dev
from lmcache.integration.deepinfer.multi_process_adapter import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MQ_TIMEOUT,
    DeepInferMPParallelConfig,
    DeepInferMPSchedulerAdapter,
    DeepInferMPTransferSpec,
    DeepInferMPWorkerAdapter,
)
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, is_kv_writer
from lmcache.v1.platform.ipc_policy import set_ipc_policy

logger = init_logger(__name__)


@dataclass
class _PendingStore:
    """One asynchronous store owning a stable state-snapshot slot."""

    start: int
    end: int
    state_slot: int | None
    future: MessagingFuture[bool] | None


@dataclass
class _PendingLoad:
    """One asynchronous retrieve targeting request-owned cache locations."""

    state_id: int
    state_slot: int | None
    hit_tokens: int
    future: MessagingFuture[bool] | None


def _page_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return DeepInfer's ``[NB, slots, 1, width]`` page as a 3-D view."""
    if tensor.dim() != 4 or int(tensor.shape[2]) != 1:
        raise ValueError(
            "DeepInfer CA/ICA tensors must have shape [NB, slots, 1, width], "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.squeeze(2)


def _collect_page_caches(
    cache_list: list[Any],
) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
    """Collect locally owned CA/ICA pages without hard-coded layer maps.

    Args:
        cache_list: DeepInfer layer-cache objects exposing CA/ICA tensors.
    Returns:
        The registered tensor views and their registered tensor indices.

    Raises:
        ValueError: If DeepInfer exposes no non-empty CA/ICA cache pages.
    """
    caches: dict[str, torch.Tensor] = {}
    seen: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    for layer_id, cache in enumerate(cache_list):
        for component in ("ica_cache", "ca_cache"):
            tensor = getattr(cache, component, None)
            if tensor is None or tensor.numel() == 0:
                continue
            view = _page_view(tensor)
            identity = (
                view.data_ptr(),
                tuple(int(value) for value in view.shape),
                tuple(int(value) for value in view.stride()),
            )
            if identity in seen:
                continue
            seen.add(identity)
            caches[f"layer.{layer_id}.{component}"] = view
    if not caches:
        raise ValueError("DeepInfer exposed no locally owned CA/ICA cache pages")
    return caches, tuple(range(len(caches)))


def _build_engine_group_infos(
    caches: dict[str, torch.Tensor],
    attention_layer_indices: tuple[int, ...],
    *,
    state_layer_index: int | None,
    block_size: int,
    state_tokens_per_block: int,
    tp_replicas: int,
) -> list[EngineGroupInfo]:
    """Group registered tensors by the same physical identity as the server."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import group_layers_by_identity

    layer_index_groups: list[tuple[int, ...]] = [attention_layer_indices]
    per_layer_engine_group_idx = [0] * len(caches)
    if state_layer_index is not None:
        layer_index_groups.append((state_layer_index,))
        per_layer_engine_group_idx[state_layer_index] = 1

    normalized, formats = normalize_and_discover_per_layer_formats(
        list(caches.values()),
        layer_index_groups,
        EngineType.DEEPINFER,
        {},
    )
    infos: list[EngineGroupInfo] = []
    for identity, indices in group_layers_by_identity(
        normalized,
        formats,
        per_layer_engine_group_idx,
    ):
        recurrent_state = identity.engine_group_idx == 1
        infos.append(
            EngineGroupInfo(
                engine_group_id=identity.engine_group_idx,
                layer_indices=tuple(indices),
                tokens_per_block=(
                    state_tokens_per_block if recurrent_state else block_size
                ),
                sw_size_tokens=(state_tokens_per_block if recurrent_state else -1),
                recurrent_state=recurrent_state,
                tp_replicas=tp_replicas,
            )
        )
    return infos


def _allocate_state_snapshot_pool(
    cache_engine: Any,
    num_state_slots: int,
    state_size: int,
) -> torch.Tensor:
    """Allocate state staging with an IPC-exportable CUDA allocator.

    DeepInfer's page cache uses ``cuda_utils.alloc_tensor`` instead of the
    PyTorch caching allocator.  The snapshot pool is registered in the same
    LMCache IPC batch, so it must use the same allocation family; a
    ``torch.empty`` pool may be VMM-backed when expandable segments are
    enabled and cannot be exported through legacy CUDA IPC.
    """
    device = cache_engine.gpu_states.device
    shape = (num_state_slots, 1, state_size)
    if device.type == "cuda":
        # Third Party
        from deepinfer._C import cuda_utils

        with torch_dev.device(device):
            raw = cuda_utils.alloc_tensor(num_state_slots * state_size)
        if not isinstance(raw, torch.Tensor):
            raise RuntimeError("DeepInfer failed to allocate state snapshot pool")
        pool = raw.view(torch.uint8).view(shape)
    else:
        pool = torch.empty(shape, dtype=torch.uint8, device=device)

    # Block zero is the sparse recurrent group's null mapping for all chunks
    # before the endpoint snapshot.
    pool[0].zero_()
    return pool


def _require_config(config: dict[str, Any], name: str) -> Any:
    """Return one required connector option with a useful error."""
    if name not in config:
        raise ValueError(f"DeepInfer LMCache connector requires config[{name!r}]")
    return config[name]


class LMCacheMPConnector(KVTransferConnector):
    """LMCache MP connector for DeepInfer scheduler and worker roles."""

    def __init__(
        self,
        role: KVTransferRole,
        config: dict[str, Any],
        *,
        model_name: str,
        block_size: int,
        world_size: int,
        is_mla: bool,
        worker_id: int | None = None,
        cache_engine: Any | None = None,
    ) -> None:
        super().__init__(role)
        isolated_ipc = config.get("isolated_ipc", False)
        if not isinstance(isolated_ipc, bool):
            raise TypeError(
                "DeepInfer LMCache connector config['isolated_ipc'] must be a bool"
            )
        # DeepInfer's KV pool is backed by cuMemAlloc rather than PyTorch's
        # caching allocator, so LMCache must use its raw CUDA IPC wrapper.
        set_ipc_policy(isolated_ipc=isolated_ipc)
        self._timeout = float(config.get("mq_timeout", DEFAULT_MQ_TIMEOUT))
        self._block_size = block_size
        server_url = str(_require_config(config, "server_url"))
        context = zmq.Context.instance()

        if self.role == "scheduler":
            parallel = DeepInferMPParallelConfig.for_model(world_size, 0, is_mla=is_mla)
            self.scheduler_adapter = DeepInferMPSchedulerAdapter(
                server_url=server_url,
                context=context,
                model_name=model_name,
                block_size=block_size,
                parallel_config=parallel,
                mq_timeout=self._timeout,
            )
            self._chunk_size = self.scheduler_adapter.lmcache_tokens_per_chunk
            self._scheduled_loads: dict[str, tuple[list[int], int]] = {}
            return

        if worker_id is None or cache_engine is None:
            raise ValueError(
                "DeepInfer worker connector needs worker_id and cache_engine"
            )
        self._cache_engine = cache_engine
        parallel = DeepInferMPParallelConfig.for_model(
            world_size, worker_id, is_mla=is_mla
        )
        self.worker_adapter = DeepInferMPWorkerAdapter(
            server_url=server_url,
            context=context,
            model_name=model_name,
            block_size=block_size,
            parallel_config=parallel,
            mq_timeout=self._timeout,
            heartbeat_interval=float(
                config.get("heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)
            ),
            transfer_mode=config.get("transfer_mode"),
        )
        self._chunk_size = self.worker_adapter.lmcache_tokens_per_chunk
        caches, attention_layer_indices = _collect_page_caches(
            cache_engine.gpu_cache.cache_list,
        )

        state_size = int(cache_engine.gpu_cache.total_bytes_inference)
        self._state_snapshot_pool: torch.Tensor | None = None
        self._free_state_load_slots: deque[int] = deque()
        self._free_state_store_slots: deque[int] = deque()
        self._state_group = state_size > 0
        state_layer_index: int | None = None
        if self._state_group:
            num_load_slots = int(config.get("max_inflight_loads", 1))
            num_store_slots = int(config.get("state_snapshot_pool_size", 4))
            if num_load_slots <= 0:
                raise ValueError("DeepInfer state load staging slots must be positive")
            if num_store_slots <= 1:
                raise ValueError(
                    "DeepInfer state_snapshot_pool_size must be greater than one"
                )
            # Row zero remains the sparse recurrent group's null mapping. Loads
            # and stores use disjoint rows so an incoming state cannot overwrite
            # an asynchronous outgoing snapshot.
            num_state_slots = 1 + num_load_slots + num_store_slots
            self._state_snapshot_pool = _allocate_state_snapshot_pool(
                cache_engine,
                num_state_slots,
                state_size,
            )
            self._free_state_load_slots.extend(range(1, 1 + num_load_slots))
            self._free_state_store_slots.extend(
                range(1 + num_load_slots, num_state_slots)
            )
            state_layer_index = len(caches)
            caches["runtime.state"] = self._state_snapshot_pool
            logger.warning(
                "DeepInfer runtime-state snapshots require the LMCache MP "
                "server option --separate-object-groups"
            )

        groups = _build_engine_group_infos(
            caches,
            attention_layer_indices,
            state_layer_index=state_layer_index,
            block_size=block_size,
            state_tokens_per_block=self.chunk_size,
            tp_replicas=parallel.tp_replicas,
        )

        self._is_kv_writer = is_kv_writer(groups, parallel.kv_worker_id)
        self.worker_adapter.register_kv_caches(caches, engine_group_infos=groups)
        self._pending: dict[str, list[_PendingStore]] = {}
        self._pending_loads: dict[str, _PendingLoad] = {}
        self._stored_tokens: dict[str, int] = {}
        self._submitted_tokens: dict[str, int] = {}
        self._active_sends: set[str] = set()
        self._finished_sends: set[str] = set()
        self._finished_requests: set[str] = set()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _require_scheduler_adapter(self) -> DeepInferMPSchedulerAdapter:
        if self.role != "scheduler":
            raise RuntimeError("Scheduler connector method called on worker role")
        return self.scheduler_adapter

    def _require_worker_adapter(self) -> DeepInferMPWorkerAdapter:
        if self.role != "worker":
            raise RuntimeError("Worker connector method called on scheduler role")
        return self.worker_adapter

    def get_num_new_matched_tokens(
        self, *, request_id: str, token_ids: list[int], end: int
    ) -> tuple[int | None, bool]:
        """Poll the longest jointly recoverable, chunk-aligned prefix."""
        adapter = self._require_scheduler_adapter()
        query_end = min(max(end, 0), len(token_ids))
        query_end = query_end // self.chunk_size * self.chunk_size
        if query_end == 0:
            return 0, False
        adapter.maybe_submit_lookup_request(request_id, token_ids[:query_end])
        result = adapter.check_lookup_result(request_id)
        if result is None:
            return None, True
        matched_tokens = min(int(result), query_end)
        return matched_tokens, matched_tokens > 0

    def update_state_after_alloc(
        self,
        *,
        request_id: str,
        token_ids: list[int],
        num_external_tokens: int,
    ) -> None:
        """Publish the scheduler-side load plan after cache allocation."""
        adapter = self._require_scheduler_adapter()
        if num_external_tokens > 0:
            self._scheduled_loads[request_id] = (
                list(token_ids),
                num_external_tokens,
            )
        adapter.cleanup_lookup_result(request_id)

    def update_connector_output(
        self,
        output: KVConnectorTransferResults,
    ) -> None:
        """Forget completed loads after every worker settled its reader lock."""
        self._require_scheduler_adapter()
        for request_id in output.finished_recving:
            self._scheduled_loads.pop(request_id, None)

    def _require_state_snapshot_pool(self) -> torch.Tensor:
        """Return the registered snapshot pool or fail on invalid lifecycle use."""
        if self._state_snapshot_pool is None:
            raise RuntimeError(
                "DeepInfer runtime-state snapshot pool is not registered"
            )
        return self._state_snapshot_pool

    def _state_blocks(self, state_slot: int, chunks: int) -> list[int]:
        """Map a sparse recurrent snapshot to only the range endpoint."""
        if not self._state_group:
            return []
        if state_slot < 0:
            raise ValueError(
                f"DeepInfer state staging slot cannot be negative, got {state_slot}"
            )
        return [0] * max(0, chunks - 1) + [state_slot]

    def _block_ids(
        self,
        block_ids: list[int],
        state_slot: int,
        start: int,
        end: int,
    ) -> list[list[int]]:
        """Slice full DeepInfer block tables into LMCache engine groups."""
        main = block_ids[start // self._block_size : end // self._block_size]
        result = [main]
        if self._state_group:
            result.append(
                self._state_blocks(state_slot, (end - start) // self.chunk_size)
            )
        return result

    def _release_load_slot(self, state_slot: int | None) -> None:
        if state_slot is not None:
            self._free_state_load_slots.append(state_slot)

    def _release_store_slot(self, state_slot: int | None) -> None:
        if state_slot is not None:
            self._free_state_store_slots.append(state_slot)

    def start_load_kv(
        self,
        *,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        state_id: int,
        hit_tokens: int,
    ) -> None:
        """Start loading CA/ICA pages and the endpoint runtime snapshot."""
        adapter = self._require_worker_adapter()
        if request_id in self._pending_loads:
            raise ValueError(f"Duplicate DeepInfer load request: {request_id}")
        if hit_tokens <= 0 or hit_tokens % self.chunk_size:
            raise ValueError(
                f"DeepInfer load end must align to {self.chunk_size}, got {hit_tokens}"
            )
        state_slot: int | None = None
        if self._state_group and self._free_state_load_slots:
            state_slot = self._free_state_load_slots.popleft()
        spec = DeepInferMPTransferSpec(
            token_ids=token_ids,
            block_ids=self._block_ids(
                block_ids,
                state_slot or 0,
                start=0,
                end=hit_tokens,
            ),
            start=0,
            end=hit_tokens,
        )
        future = (
            adapter.submit_retrieve_request(request_id, spec)
            if not self._state_group or state_slot is not None
            else None
        )
        if future is None:
            self._release_load_slot(state_slot)
            state_slot = None
        self._pending_loads[request_id] = _PendingLoad(
            state_id=state_id,
            state_slot=state_slot,
            hit_tokens=hit_tokens,
            future=future,
        )

    def get_transfer_results(self) -> KVConnectorTransferResults:
        """Poll retrieve futures without blocking the DeepInfer engine."""
        self._require_worker_adapter()
        finished: set[str] = set()
        failed: set[str] = set()
        for request_id, pending in list(self._pending_loads.items()):
            future = pending.future
            if future is None:
                finished.add(request_id)
                failed.add(request_id)
                del self._pending_loads[request_id]
                continue
            try:
                if not future.query():
                    continue
                success = bool(future.result(timeout=0))
                if success and self._state_group:
                    state_staging = self._require_state_snapshot_pool()
                    assert pending.state_slot is not None
                    self._cache_engine.gpu_cache.load_full_state_to_cache(
                        pending.state_id,
                        pending.hit_tokens,
                        state_staging[pending.state_slot, 0],
                        self._cache_engine.gpu_states,
                    )
            except Exception:
                logger.exception(
                    "DeepInfer LMCache load failed for %s",
                    request_id,
                )
                success = False
            finished.add(request_id)
            if not success:
                failed.add(request_id)
            self._release_load_slot(pending.state_slot)
            del self._pending_loads[request_id]
        self._poll_pending_stores()
        finished_sending = self._finished_sends
        self._finished_sends = set()
        return KVConnectorTransferResults(
            finished_sending=finished_sending,
            finished_recving=finished,
            failed_recving=failed,
        )

    def _finish_send_if_drained(self, request_id: str) -> None:
        if request_id in self._pending or request_id not in self._active_sends:
            return
        self._active_sends.remove(request_id)
        self._finished_sends.add(request_id)
        self._submitted_tokens[request_id] = self._stored_tokens.get(request_id, 0)
        if request_id in self._finished_requests:
            self._finished_requests.remove(request_id)
            self._stored_tokens.pop(request_id, None)
            self._submitted_tokens.pop(request_id, None)

    def _poll_pending_stores(self) -> None:
        """Reclaim snapshot slots without waiting for unfinished D2H."""
        for request_id, pending_stores in list(self._pending.items()):
            while pending_stores:
                pending = pending_stores[0]
                future = pending.future
                try:
                    if future is not None and not future.query():
                        break
                    success = future is not None and bool(future.result(timeout=0))
                except Exception:
                    logger.exception(
                        "DeepInfer LMCache store failed for %s",
                        request_id,
                    )
                    success = False
                pending_stores.pop(0)
                self._release_store_slot(pending.state_slot)
                contiguous = self._stored_tokens.get(request_id, 0)
                if success and contiguous == pending.start:
                    self._stored_tokens[request_id] = pending.end
            if not pending_stores:
                del self._pending[request_id]
                self._finish_send_if_drained(request_id)

    def wait_for_save(
        self,
        *,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        state_id: int,
        end: int,
    ) -> None:
        """Asynchronously store newly completed aligned prompt chunks."""
        self._poll_pending_stores()
        # A just-completed older store may have queued a completion that the
        # engine has not polled yet. New work extends the same block-pin epoch,
        # so suppress that stale completion until the new queue also drains.
        self._finished_sends.discard(request_id)
        self._active_sends.add(request_id)
        if not self._is_kv_writer:
            self._finish_send_if_drained(request_id)
            return
        adapter = self._require_worker_adapter()
        aligned_end = min(end, len(token_ids)) // self.chunk_size * self.chunk_size
        start = min(
            self._submitted_tokens.get(
                request_id, self._stored_tokens.get(request_id, 0)
            ),
            aligned_end,
        )
        start = start // self.chunk_size * self.chunk_size
        if aligned_end <= start:
            self._finish_send_if_drained(request_id)
            return

        state_slot: int | None = None
        if self._state_group:
            if not self._free_state_store_slots:
                logger.debug(
                    "DeepInfer state snapshot pool is full; skipping store for %s",
                    request_id,
                )
                self._finish_send_if_drained(request_id)
                return
            state_slot = self._free_state_store_slots.popleft()
            state_staging = self._require_state_snapshot_pool()
            try:
                self._cache_engine.gpu_cache.select_prefix_state_to_store(
                    state_id,
                    aligned_end,
                    self._cache_engine.gpu_states,
                    state_staging[state_slot, 0],
                )
            except Exception:
                self._release_store_slot(state_slot)
                raise
        spec = DeepInferMPTransferSpec(
            token_ids=token_ids,
            block_ids=self._block_ids(
                block_ids, state_slot or 0, start=start, end=aligned_end
            ),
            start=start,
            end=aligned_end,
        )
        future = adapter.submit_store_request(request_id, spec)
        if future is None:
            self._release_store_slot(state_slot)
            self._finish_send_if_drained(request_id)
            return
        self._pending.setdefault(request_id, []).append(
            _PendingStore(start, aligned_end, state_slot, future)
        )
        self._submitted_tokens[request_id] = aligned_end

    def request_finished(self, request_id: str) -> None:
        """Finish role-specific work before DeepInfer releases the request."""
        if self.role == "scheduler":
            adapter = self._require_scheduler_adapter()
            self._scheduled_loads.pop(request_id, None)
            adapter.end_session(request_id)
            adapter.cleanup_lookup_result(request_id)
            return
        self._require_worker_adapter()
        # Request completion is not a transfer barrier. Snapshot slots and
        # DeepInfer's pinned cache blocks stay alive until polling observes the
        # corresponding store futures.
        self._finished_requests.add(request_id)
        if request_id not in self._pending and request_id not in self._active_sends:
            self._finished_requests.remove(request_id)
            self._stored_tokens.pop(request_id, None)
            self._submitted_tokens.pop(request_id, None)
            return
        self._finish_send_if_drained(request_id)

    def shutdown(self) -> None:
        """Drain role-specific work and close transport resources."""
        if self.role == "scheduler":
            self._require_scheduler_adapter().shutdown()
            return
        adapter = self._require_worker_adapter()
        for request_id, pending_load in list(self._pending_loads.items()):
            try:
                if pending_load.future is not None:
                    pending_load.future.result()
            except Exception:
                logger.exception(
                    "DeepInfer LMCache load failed during shutdown for %s",
                    request_id,
                )
            finally:
                self._release_load_slot(pending_load.state_slot)
                self._pending_loads.pop(request_id, None)
        for request_id, pending_stores in list(self._pending.items()):
            for pending_store in pending_stores:
                try:
                    if pending_store.future is not None:
                        pending_store.future.result(timeout=self._timeout)
                except Exception:
                    logger.exception(
                        "DeepInfer LMCache store failed during shutdown for %s",
                        request_id,
                    )
                finally:
                    self._release_store_slot(pending_store.state_slot)
            self._pending.pop(request_id, None)
        adapter.shutdown()
