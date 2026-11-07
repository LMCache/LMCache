# SPDX-License-Identifier: Apache-2.0
"""DeepInfer client adapter for LMCache multiprocess mode.

The MP server and transfer core stay engine-neutral.  This module only maps
DeepInfer's page-native CA/ICA views and full runtime state snapshots onto the
DeepInfer-owned multiprocess adapter and recurrent-state object-group semantics.
"""

# Future
from __future__ import annotations

# Standard
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
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


@dataclass
class _PendingStore:
    """One asynchronous store whose staging state must remain unchanged."""

    start: int
    end: int
    future: MessagingFuture[bool] | None


@dataclass
class _PendingLoad:
    """One asynchronous retrieve targeting request-owned cache locations."""

    state_id: int
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
    block_size: int,
) -> tuple[dict[str, torch.Tensor], list[EngineGroupInfo]]:
    """Collect locally owned CA/ICA pages without hard-coded layer maps."""
    caches: dict[str, torch.Tensor] = {}
    groups: list[EngineGroupInfo] = []
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
            layer_index = len(caches)
            caches[f"layer.{layer_id}.{component}"] = view
            groups.append(
                EngineGroupInfo(
                    engine_group_id=0,
                    layer_indices=(layer_index,),
                    tokens_per_block=block_size,
                )
            )
    if not caches:
        raise ValueError("DeepInfer exposed no locally owned CA/ICA cache pages")
    return caches, groups


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
        worker_id: int | None = None,
        cache_engine: Any | None = None,
    ) -> None:
        super().__init__(role)
        self._timeout = float(config.get("mq_timeout", DEFAULT_MQ_TIMEOUT))
        self._block_size = block_size
        server_url = str(_require_config(config, "server_url"))
        context = zmq.Context.instance()

        if self.role == "scheduler":
            self.scheduler_adapter = DeepInferMPSchedulerAdapter(
                server_url=server_url,
                context=context,
                model_name=model_name,
                block_size=block_size,
                parallel_config=DeepInferMPParallelConfig(
                    world_size=world_size,
                    worker_id=0,
                    tp_size=world_size,
                ),
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
        self.worker_adapter = DeepInferMPWorkerAdapter(
            server_url=server_url,
            context=context,
            model_name=model_name,
            block_size=block_size,
            parallel_config=DeepInferMPParallelConfig(
                world_size=world_size,
                worker_id=worker_id,
                tp_size=world_size,
            ),
            mq_timeout=self._timeout,
            heartbeat_interval=float(
                config.get("heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)
            ),
            transfer_mode=config.get("transfer_mode"),
        )
        self._chunk_size = self.worker_adapter.lmcache_tokens_per_chunk
        caches, groups = _collect_page_caches(
            cache_engine.gpu_cache.cache_list, block_size
        )

        state_size = int(cache_engine.gpu_cache.total_bytes_inference)
        self._state_staging: torch.Tensor | None = None
        self._state_group = state_size > 0
        if self._state_group:
            num_states = int(cache_engine.gpu_states.shape[0])
            self._state_staging = torch.empty(
                (num_states, 1, state_size),
                dtype=torch.uint8,
                device=cache_engine.gpu_states.device,
            )
            state_layer = len(caches)
            caches["runtime.state"] = self._state_staging
            groups.append(
                EngineGroupInfo(
                    engine_group_id=1,
                    layer_indices=(state_layer,),
                    tokens_per_block=self.chunk_size,
                    sw_size_tokens=self.chunk_size,
                    recurrent_state=True,
                )
            )
            logger.warning(
                "DeepInfer runtime-state snapshots require the LMCache MP "
                "server option --separate-object-groups"
            )

        self.worker_adapter.register_kv_caches(caches, engine_group_infos=groups)
        self._pending: dict[str, _PendingStore] = {}
        self._pending_loads: dict[str, _PendingLoad] = {}
        self._stored_tokens: dict[str, int] = {}

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
        """Release scheduler-side lookup ownership after worker completion."""
        adapter = self._require_scheduler_adapter()
        for request_id in output.finished_recving:
            scheduled = self._scheduled_loads.pop(request_id, None)
            if scheduled is None or request_id not in output.failed_recving:
                continue
            token_ids, num_external_tokens = scheduled
            adapter.free_lookup_locks(
                token_ids,
                0,
                num_external_tokens,
                request_id,
            )

    def _require_state_staging(self) -> torch.Tensor:
        """Return the registered state tensor or fail on invalid lifecycle use."""
        if self._state_staging is None:
            raise RuntimeError("DeepInfer runtime-state staging is not registered")
        return self._state_staging

    def _state_blocks(self, state_id: int, chunks: int) -> list[int]:
        """Map a sparse recurrent snapshot to only the range endpoint."""
        if not self._state_group:
            return []
        if state_id <= 0:
            raise ValueError(f"DeepInfer state id must be positive, got {state_id}")
        return [0] * max(0, chunks - 1) + [state_id]

    def _block_ids(
        self,
        block_ids: list[int],
        state_id: int,
        start: int,
        end: int,
    ) -> list[list[int]]:
        """Slice full DeepInfer block tables into LMCache engine groups."""
        main = block_ids[start // self._block_size : end // self._block_size]
        result = [main]
        if self._state_group:
            result.append(
                self._state_blocks(state_id, (end - start) // self.chunk_size)
            )
        return result

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
        spec = DeepInferMPTransferSpec(
            token_ids=token_ids,
            block_ids=self._block_ids(block_ids, state_id, start=0, end=hit_tokens),
            start=0,
            end=hit_tokens,
        )
        future = adapter.submit_retrieve_request(request_id, spec)
        self._pending_loads[request_id] = _PendingLoad(
            state_id=state_id,
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
                    state_staging = self._require_state_staging()
                    self._cache_engine.gpu_cache.load_full_state_to_cache(
                        pending.state_id,
                        pending.hit_tokens,
                        state_staging[pending.state_id, 0],
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
            del self._pending_loads[request_id]
        return KVConnectorTransferResults(
            finished_recving=finished,
            failed_recving=failed,
        )

    def _finish_pending(self, request_id: str) -> bool:
        """Resolve a prior store before reusing its per-state staging row."""
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return True
        success = pending.future is not None and bool(
            pending.future.result(timeout=self._timeout)
        )
        if success:
            self._stored_tokens[request_id] = pending.end
        else:
            self._stored_tokens[request_id] = pending.start
        return success

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
        adapter = self._require_worker_adapter()
        self._finish_pending(request_id)
        aligned_end = min(end, len(token_ids)) // self.chunk_size * self.chunk_size
        start = min(self._stored_tokens.get(request_id, 0), aligned_end)
        start = start // self.chunk_size * self.chunk_size
        if aligned_end <= start:
            return

        if self._state_group:
            state_staging = self._require_state_staging()
            self._cache_engine.gpu_cache.select_prefix_state_to_store(
                state_id,
                aligned_end,
                self._cache_engine.gpu_states,
                state_staging[state_id, 0],
            )
        spec = DeepInferMPTransferSpec(
            token_ids=token_ids,
            block_ids=self._block_ids(
                block_ids, state_id, start=start, end=aligned_end
            ),
            start=start,
            end=aligned_end,
        )
        future = adapter.submit_store_request(request_id, spec)
        self._pending[request_id] = _PendingStore(start, aligned_end, future)

    def request_finished(self, request_id: str) -> None:
        """Finish role-specific work before DeepInfer releases the request."""
        if self.role == "scheduler":
            adapter = self._require_scheduler_adapter()
            self._scheduled_loads.pop(request_id, None)
            adapter.end_session(request_id)
            adapter.cleanup_lookup_result(request_id)
            return
        self._require_worker_adapter()
        pending_load = self._pending_loads.pop(request_id, None)
        if pending_load is not None and pending_load.future is not None:
            pending_load.future.result()
        self._finish_pending(request_id)
        self._stored_tokens.pop(request_id, None)

    def shutdown(self) -> None:
        """Drain role-specific work and close transport resources."""
        if self.role == "scheduler":
            self._require_scheduler_adapter().shutdown()
            return
        adapter = self._require_worker_adapter()
        for request_id, pending in list(self._pending_loads.items()):
            try:
                if pending.future is not None:
                    pending.future.result()
            except Exception:
                logger.exception(
                    "DeepInfer LMCache load failed during shutdown for %s",
                    request_id,
                )
            finally:
                self._pending_loads.pop(request_id, None)
        for request_id in list(self._pending):
            try:
                self._finish_pending(request_id)
            except Exception:
                logger.exception(
                    "DeepInfer LMCache store failed during shutdown for %s",
                    request_id,
                )
        adapter.shutdown()
