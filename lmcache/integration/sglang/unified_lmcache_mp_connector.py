# SPDX-License-Identifier: Apache-2.0
"""LMCache multiprocess connector used by SGLang's unified radix cache.

This module deliberately talks to LMCache's engine-neutral MP protocol.  It
does not use LMCache's legacy SGLang integration and it never constructs an
in-process LMCache engine.  The registered SGLang GPU KV tensors remain owned
by SGLang; LMCache accesses them through device-memory and event IPC handles.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Optional
import logging
import threading
import uuid

# Third Party
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_DEFAULT_MQ_TIMEOUT_SECONDS = 300.0
_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0


class _ImmediateFuture:
    """Minimal future used to preserve TP operation order on local failure."""

    def __init__(self, result: bool) -> None:
        self._result = result

    def query(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> bool:
        del timeout
        return self._result

    def prepare(self, timeout: Optional[float] = None) -> bool:
        return self.result(timeout)

    def wait_on_stream(self, stream: Any, timeout: Optional[float] = None) -> bool:
        del stream
        return self.result(timeout)

    def retain_reference(self, value: object) -> None:
        del value


@dataclass
class LMCacheLookupOperation:
    request_id: str
    token_ids: list[int]
    local_hit_tokens: int
    cache_salt: str
    submission_future: Any = None
    completion_future: Any = None
    total_hit_tokens: Optional[int] = None
    locks_held: bool = False
    lock_start: int = 0


@dataclass(frozen=True)
class LMCacheKVGroup:
    """One SGLang KV address space exposed as one LMCache engine group.

    ``kv_tensors`` are registered as independent, single-plane byte-equivalent
    views.  This keeps SGLang's separately allocated K and V buffers zero-copy
    while giving every component a stable per-group block-id namespace.
    """

    name: str
    kv_tensors: tuple[torch.Tensor, ...]
    sliding_window_size: int = -1
    # Logical tokens covered by one engine block id. Attention groups use the
    # SGLang page size; a recurrent/Mamba group uses its checkpoint grid.
    tokens_per_block: int = 0
    # SGLang allocator slots covered by one block id. This is page_size for
    # attention and 1 for a Mamba checkpoint slot. Attention and MLA preserve
    # this as the explicit BS axis; recurrent state uses an opaque view.
    slots_per_block: int = 0
    # Number of source tensor rows making up one block, one value per tensor.
    # Usually this equals slots_per_block. Page-native sidecars such as the DSA
    # indexer and DeepSeek V4 compressed/state pools already store one complete
    # logical page in each row and therefore use 1.
    tensor_rows_per_block: tuple[int, ...] = ()
    recurrent_state: bool = False


@dataclass
class LMCacheLoadOperation:
    request_id: str
    token_ids: list[int]
    start: int
    end: int
    local_hit_tokens: int
    device_indices: torch.Tensor
    future: Any
    lookup: LMCacheLookupOperation
    result: Optional[bool] = None

    def query(self) -> bool:
        return self.result is not None or bool(self.future.query())


@dataclass
class LMCacheStoreOperation:
    request_id: str
    start: int
    end: int
    future: Any
    result: Optional[bool] = None

    def query(self) -> bool:
        return self.result is not None or bool(self.future.query())


class UnifiedLMCacheMPConnector:
    """Asynchronous, CUDA-IPC connector to a standalone LMCache server."""

    def __init__(
        self,
        *,
        config_file: Optional[str],
        model_name: str,
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup],
        pp_size: int = 1,
        pp_rank: int = 0,
        pp_group: Optional[dist.ProcessGroup] = None,
        page_size: int,
        kv_groups: list[LMCacheKVGroup],
        mla_enabled: bool = False,
    ) -> None:
        try:
            # Third Party
            import zmq

            # First Party
            from lmcache.v1.config import load_engine_config_with_overrides
            from lmcache.v1.multiprocess.mq import MessageQueueClient
        except ImportError as exc:
            raise ImportError(
                "LMCacheUnifiedRadixCache requires the `lmcache` package and "
                "a running LMCache multiprocess server."
            ) from exc

        if not kv_groups or any(not group.kv_tensors for group in kv_groups):
            raise ValueError("LMCache KV group registration cannot be empty")
        kv_tensors = [tensor for group in kv_groups for tensor in group.kv_tensors]
        if any(t.device.type != "cuda" for t in kv_tensors):
            raise NotImplementedError("LMCache MP currently requires CUDA KV tensors")
        if any(t.device != kv_tensors[0].device for t in kv_tensors):
            raise ValueError("All LMCache-registered KV tensors must share one device")
        if any(t.dim() < 2 for t in kv_tensors):
            raise NotImplementedError(
                "LMCache MP requires tensors with a leading block/slot axis"
            )
        if any(not tensor.is_contiguous() for tensor in kv_tensors):
            raise NotImplementedError(
                "LMCache MP currently requires contiguous SGLang NHD/MLA tensors"
            )

        resolved_groups: list[LMCacheKVGroup] = []
        for group in kv_groups:
            tokens_per_block = group.tokens_per_block or page_size
            slots_per_block = group.slots_per_block or page_size
            if tokens_per_block <= 0 or slots_per_block <= 0:
                raise ValueError(
                    f"LMCache group {group.name!r} has invalid block geometry: "
                    f"{tokens_per_block=}, {slots_per_block=}"
                )
            if tokens_per_block % slots_per_block:
                raise ValueError(
                    f"LMCache group {group.name!r} tokens_per_block "
                    f"{tokens_per_block} must be a multiple of slots_per_block "
                    f"{slots_per_block}"
                )
            tensor_rows_per_block = group.tensor_rows_per_block or (
                slots_per_block,
            ) * len(group.kv_tensors)
            if len(tensor_rows_per_block) != len(group.kv_tensors):
                raise ValueError(
                    f"LMCache group {group.name!r} has "
                    f"{len(group.kv_tensors)} tensors but "
                    f"{len(tensor_rows_per_block)} tensor row geometries"
                )
            for tensor, rows_per_block in zip(
                group.kv_tensors, tensor_rows_per_block, strict=True
            ):
                if rows_per_block <= 0 or tensor.shape[0] % rows_per_block:
                    raise ValueError(
                        f"LMCache group {group.name!r} tensor rows "
                        f"{tensor.shape[0]} are not divisible by "
                        f"tensor_rows_per_block={rows_per_block}"
                    )
            resolved_groups.append(
                LMCacheKVGroup(
                    name=group.name,
                    kv_tensors=group.kv_tensors,
                    sliding_window_size=group.sliding_window_size,
                    tokens_per_block=tokens_per_block,
                    slots_per_block=slots_per_block,
                    tensor_rows_per_block=tuple(tensor_rows_per_block),
                    recurrent_state=group.recurrent_state,
                )
            )

        self.mla_only = self._is_mla_only(mla_enabled, resolved_groups)

        # Preserve attention as [NB, BS, NH, HS] and MLA as [NB, BS, HS].
        # Each non-MLA tensor remains an independent K or V list entry; no K/V
        # regrouping is needed. Recurrent state retains the opaque block view.
        wire_groups: list[LMCacheKVGroup] = []
        for group in resolved_groups:
            wire_tensors = tuple(
                self._to_wire_block_tensor(
                    tensor,
                    rows_per_block,
                    preserve_head_geometry=(
                        not self.mla_only
                        and not group.recurrent_state
                        and tensor.dim() == 3
                    ),
                    preserve_mla_geometry=(
                        self.mla_only
                        and not group.recurrent_state
                        and tensor.dim() == 3
                    ),
                )
                for tensor, rows_per_block in zip(
                    group.kv_tensors,
                    group.tensor_rows_per_block,
                    strict=True,
                )
            )
            wire_block_counts = {tensor.shape[0] for tensor in wire_tensors}
            if len(wire_block_counts) != 1:
                raise ValueError(
                    f"LMCache group {group.name!r} tensors expose different "
                    f"block counts: {sorted(wire_block_counts)}"
                )
            wire_groups.append(
                LMCacheKVGroup(
                    name=group.name,
                    kv_tensors=wire_tensors,
                    sliding_window_size=group.sliding_window_size,
                    tokens_per_block=group.tokens_per_block,
                    slots_per_block=group.slots_per_block,
                    tensor_rows_per_block=(1,) * len(group.kv_tensors),
                    recurrent_state=group.recurrent_state,
                )
            )
        kv_tensors = [tensor for group in wire_groups for tensor in group.kv_tensors]

        config = load_engine_config_with_overrides(config_file_path=config_file)
        if not config.mp_host:  # type: ignore[attr-defined]
            raise ValueError(
                "LMCache MP config must define mp_host; pass "
                "--lmcache-config-file or LMCACHE_CONFIG_FILE"
            )
        host = str(config.mp_host)  # type: ignore[attr-defined]
        if "://" not in host:
            host = f"tcp://{host}"
        self.server_url = f"{host.rstrip(':')}:{int(config.mp_port)}"  # type: ignore[attr-defined]
        self._mq_timeout = float(
            config.get_extra_config_value(  # type: ignore[attr-defined]
                "lmcache.mp.mq_timeout", _DEFAULT_MQ_TIMEOUT_SECONDS
            )
        )
        self._heartbeat_interval = float(
            config.get_extra_config_value(  # type: ignore[attr-defined]
                "lmcache.mp.heartbeat_interval",
                _DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            )
        )
        if any(
            group.recurrent_state or group.sliding_window_size >= 0
            for group in wire_groups
        ):
            logger.warning(
                "LMCache sparse SWA/recurrent groups require the LMCache MP "
                "server to be started with --separate-object-groups."
            )

        self.model_name = model_name
        # Match vLLM's MLA-only parallel strategy. Replicated MLA collapses
        # the LMCache object identity across TP while retaining one distinct
        # piece per PP stage. The general path keeps one piece per TP x PP rank.
        (
            self.tp_size,
            self.tp_rank,
            self.pp_size,
            self.pp_rank,
            self.sglang_world_size,
            self.sglang_worker_id,
        ) = self._resolve_parallel_geometry(tp_size, tp_rank, pp_size, pp_rank)
        self.kv_world_size, self.kv_worker_id = self._resolve_kv_geometry(
            tp_size,
            tp_rank,
            pp_size,
            pp_rank,
            mla_only=self.mla_only,
        )
        self.num_kv_readers = self.tp_size if self.mla_only else 1
        self._is_kv_writer = not self.mla_only or self.tp_rank == 0
        self.tp_group = tp_group
        self.pp_group = pp_group
        if self.tp_size > 1:
            if self.tp_group is None:
                raise ValueError("LMCache TP>1 requires a CPU TP process group")
            group_size = dist.get_world_size(group=self.tp_group)
            if group_size != self.tp_size:
                raise ValueError(
                    "LMCache MP TP synchronization group has the wrong size: "
                    f"got {group_size=}, expected {self.tp_size}"
                )
        if self.pp_size > 1:
            if self.pp_group is None:
                raise ValueError("LMCache PP>1 requires a CPU PP process group")
            group_size = dist.get_world_size(group=self.pp_group)
            if group_size != self.pp_size:
                raise ValueError(
                    "LMCache MP PP synchronization group has the wrong size: "
                    f"got {group_size=}, expected {self.pp_size}"
                )
        self._lookup_leader = self.tp_rank == 0 and self.pp_rank == 0
        self.page_size = int(page_size)
        self.device = kv_tensors[0].device
        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        self._kv_caches = {f"kv_{i}": tensor for i, tensor in enumerate(kv_tensors)}
        self._kv_groups = tuple(wire_groups)
        (
            self._engine_group_info_specs,
            self._kernel_group_to_engine_group,
        ) = self._build_engine_group_info_specs()
        self._context = zmq.Context.instance()
        self._mq_client = MessageQueueClient(self.server_url, self._context)
        self._transfer_ctx: Any = None
        self._event_backend: Any = None
        self._registered = False
        self._closed = False
        self._lookups: dict[str, LMCacheLookupOperation] = {}
        self._active_sessions: set[str] = set()
        self._store_submitted_tokens: dict[str, int] = {}
        self._control_futures: list[Any] = []
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        self.chunk_size = self._get_chunk_size()
        if self.chunk_size <= 0 or any(
            self.chunk_size % group.tokens_per_block for group in self._kv_groups
        ):
            raise ValueError(
                f"LMCache chunk size {self.chunk_size} must be a positive "
                "multiple of every SGLang group tokens_per_block"
            )
        # LMCache-driven MP ignores this compatibility argument; per-group
        # block counts are derived from EngineGroupInfo.tokens_per_block.
        self.blocks_in_chunk = self.chunk_size
        self.register_kv_cache()
        self._start_heartbeat()
        logger.info("UnifiedLMCacheMPConnector initialized succeed.")

    @staticmethod
    def _resolve_parallel_geometry(
        tp_size: int, tp_rank: int, pp_size: int, pp_rank: int
    ) -> tuple[int, int, int, int, int, int]:
        tp_size = int(tp_size)
        tp_rank = int(tp_rank)
        pp_size = int(pp_size)
        pp_rank = int(pp_rank)
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"Invalid LMCache TP topology: {tp_size=}, {tp_rank=}")
        if pp_size <= 0 or not 0 <= pp_rank < pp_size:
            raise ValueError(f"Invalid LMCache PP topology: {pp_size=}, {pp_rank=}")
        return (
            tp_size,
            tp_rank,
            pp_size,
            pp_rank,
            tp_size * pp_size,
            pp_rank * tp_size + tp_rank,
        )

    @staticmethod
    def _is_mla_only(
        mla_enabled: bool,
        kv_groups: list[LMCacheKVGroup],
    ) -> bool:
        """Match vLLM's MLA-only optimization eligibility.

        Multiple paged-attention cache specs, such as FULL plus SWA, do not
        make a model hybrid for this purpose. Only recurrent Mamba/SSM/linear
        attention state prevents the TP-replicated MLA optimization.
        """
        return bool(mla_enabled) and not any(
            group.recurrent_state for group in kv_groups
        )

    @staticmethod
    def _resolve_kv_geometry(
        tp_size: int,
        tp_rank: int,
        pp_size: int,
        pp_rank: int,
        *,
        mla_only: bool,
    ) -> tuple[int, int]:
        """Return LMCache's KV-object world size and worker rank.

        The SGLang world remains TP x PP. Pure MLA collapses only its
        replicated TP dimension, retaining one distinct KV rank per PP stage.
        """
        _, _, _, _, world_size, worker_id = (
            UnifiedLMCacheMPConnector._resolve_parallel_geometry(
                tp_size, tp_rank, pp_size, pp_rank
            )
        )
        if mla_only:
            return int(pp_size), int(pp_rank)
        return world_size, worker_id

    @staticmethod
    def _to_wire_block_tensor(
        tensor: torch.Tensor,
        slots_per_block: int,
        *,
        preserve_head_geometry: bool = False,
        preserve_mla_geometry: bool = False,
    ) -> torch.Tensor:
        """Create a zero-copy block view for one SGLang component."""
        if tensor.shape[0] % slots_per_block:
            raise ValueError(
                f"Tensor rows {tensor.shape[0]} are not divisible by "
                f"slots_per_block={slots_per_block}"
            )
        num_blocks = tensor.shape[0] // slots_per_block
        if preserve_head_geometry:
            return tensor.view(num_blocks, slots_per_block, *tensor.shape[1:])
        if preserve_mla_geometry:
            return tensor.view(num_blocks, slots_per_block, -1)
        return tensor.view(num_blocks, 1, -1)

    def _build_engine_group_info_specs(
        self,
    ) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
        """Build kernel-group metadata while preserving component address spaces."""
        specs: list[dict[str, Any]] = []
        tensor_offset = 0
        for engine_group_id, group in enumerate(self._kv_groups):
            # LMCache creates one copy-kernel group per physical identity.
            buckets: dict[tuple[Any, ...], list[int]] = {}
            for local_idx, tensor in enumerate(group.kv_tensors):
                identity = (*tensor.shape[1:], tensor.dtype)
                buckets.setdefault(identity, []).append(tensor_offset + local_idx)
            for indices in buckets.values():
                specs.append(
                    {
                        "engine_group_id": engine_group_id,
                        "layer_indices": tuple(indices),
                        "tokens_per_block": group.tokens_per_block,
                        "sw_size_tokens": group.sliding_window_size,
                        "recurrent_state": group.recurrent_state,
                    }
                )
            tensor_offset += len(group.kv_tensors)
        return specs, tuple(spec["engine_group_id"] for spec in specs)

    @staticmethod
    def _send_request(mq_client: Any, request_type: Any, payloads: list[Any]):
        # First Party
        from lmcache.v1.multiprocess.protocol import get_response_class

        return mq_client.submit_request(
            request_type, payloads, get_response_class(request_type)
        )

    def _get_chunk_size(self) -> int:
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType

        return int(
            self._send_request(self._mq_client, RequestType.GET_CHUNK_SIZE, []).result(
                timeout=self._mq_timeout
            )
        )

    @property
    def is_lookup_leader(self) -> bool:
        return self._lookup_leader

    @property
    def is_kv_writer(self) -> bool:
        """Whether this rank sends STORE requests for its LMCache object."""
        return self._is_kv_writer

    @property
    def operation_timeout(self) -> float:
        return self._mq_timeout

    def _sync_leader_int(self, value: int) -> int:
        tensor = torch.tensor([value], dtype=torch.int64, device="cpu")
        self._parallel_all_reduce(tensor, dist.ReduceOp.MAX)
        return int(tensor.item())

    def _sync_success(self, success: bool) -> bool:
        tensor = torch.tensor([int(success)], dtype=torch.int32, device="cpu")
        self._parallel_all_reduce(tensor, dist.ReduceOp.MIN)
        return bool(tensor.item())

    def _parallel_all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp) -> None:
        """Reduce over the Cartesian TP x PP scheduler mesh.

        TP groups contain one pipeline stage; PP groups contain the same TP
        rank from every stage. Reducing over both dimensions gives every
        scheduler the same lookup/readiness/failure result without requiring
        an additional global process group.
        """
        if self.tp_size > 1:
            dist.all_reduce(tensor, op=op, group=self.tp_group)
        if self.pp_size > 1:
            dist.all_reduce(tensor, op=op, group=self.pp_group)

    def register_kv_cache(self) -> None:
        """Export the SGLang GPU tensors to the LMCache MP server once."""
        # First Party
        from lmcache.utils import EngineType
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo
        from lmcache.v1.multiprocess.transfer_context import create_transfer_context
        from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

        if self._registered:
            raise RuntimeError("LMCache KV tensors are already registered")
        self._event_backend = get_event_ipc_backend(self.device)
        self._event_backend.check_event_support(self.device)
        self._transfer_ctx = create_transfer_context(
            self._kv_caches, mode="lmcache_driven"
        )
        engine_group_infos = [
            EngineGroupInfo(**spec) for spec in self._engine_group_info_specs
        ]
        try:
            self._transfer_ctx.register(
                self.instance_id,
                self._kv_caches,
                self.model_name,
                self.kv_world_size,
                self.blocks_in_chunk,
                self._mq_client,
                self._mq_timeout,
                self._send_request,
                layout_hints={"kv_list_layout": "unified"},
                engine_group_infos=engine_group_infos,
                engine_type=EngineType.SGLANG,
            )
        except Exception:
            self._transfer_ctx.close()
            self._transfer_ctx = None
            raise
        self._registered = True

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return

        def heartbeat() -> None:
            # First Party
            from lmcache.v1.multiprocess.protocol import RequestType

            while not self._heartbeat_stop.wait(self._heartbeat_interval):
                try:
                    self._send_request(
                        self._mq_client, RequestType.PING, [self.instance_id]
                    ).result(timeout=self._heartbeat_interval)
                except Exception:
                    logger.warning("LMCache MP heartbeat failed", exc_info=True)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat, name="sglang-lmcache-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _new_event(self, stream: Any = None) -> Any:
        event = self._event_backend.create_event(self.device)
        if stream is None:
            stream = torch.get_device_module(self.device).current_stream()
        self._event_backend.record_event(event, stream)
        return event

    def _create_key(
        self,
        operation: LMCacheLookupOperation,
        *,
        start: int,
        end: int,
        worker_id: Optional[int],
    ) -> Any:
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.kv_world_size,
            worker_id=worker_id,
            # Sharded KV has one reader per object. TP-replicated MLA maps all
            # TP ranks in a PP stage to one object, so lookup must reserve one
            # read lock for every rank that retrieves that object.
            num_kv_readers=self.num_kv_readers,
            token_ids=tuple(operation.token_ids),
            start=start,
            end=end,
            request_id=operation.request_id,
            cache_salt=operation.cache_salt,
        )

    def submit_lookup(
        self,
        request_id: str,
        token_ids: list[int],
        *,
        local_hit_tokens: int,
        cache_salt: str,
    ) -> LMCacheLookupOperation:
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType

        operation = LMCacheLookupOperation(
            request_id=request_id,
            token_ids=list(token_ids),
            local_hit_tokens=int(local_hit_tokens),
            cache_salt=cache_salt,
        )
        self._lookups[request_id] = operation
        aligned_end = len(token_ids) // self.chunk_size * self.chunk_size
        if aligned_end == 0:
            operation.total_hit_tokens = 0
            return operation

        submitted = True
        if self.is_lookup_leader:
            try:
                key = self._create_key(
                    operation, start=0, end=aligned_end, worker_id=None
                )
                operation.submission_future = self._send_request(
                    self._mq_client,
                    RequestType.LOOKUP,
                    [key, self.kv_world_size],
                )
            except Exception:
                logger.exception("LMCache lookup submission failed for %s", request_id)
                submitted = False
        if self._sync_success(submitted):
            # This lookup has reached the server and will create/update its
            # Session. STORE tracks its own actual submissions separately.
            self._active_sessions.add(request_id)
        else:
            operation.total_hit_tokens = 0
        return operation

    def poll_lookup(self, operation: LMCacheLookupOperation) -> Optional[int]:
        """Return total hit tokens, or ``None`` while lookup+prefetch runs."""
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType

        if operation.total_hit_tokens is not None:
            return operation.total_hit_tokens
        result = -1
        if self.is_lookup_leader:
            try:
                if operation.submission_future is None:
                    result = 0
                elif not operation.submission_future.query():
                    result = -1
                elif operation.completion_future is None:
                    operation.submission_future.result(timeout=0)
                    operation.completion_future = self._send_request(
                        self._mq_client,
                        RequestType.QUERY_PREFETCH_STATUS,
                        [operation.request_id],
                    )
                    result = -1
                elif not operation.completion_future.query():
                    result = -1
                else:
                    matched_chunks = operation.completion_future.result(timeout=0)
                    # QUERY_PREFETCH_STATUS returns None while the server-side
                    # prefetch is still running. Retire this short-lived query
                    # future so a later scheduler pass submits another poll;
                    # unlike WAIT_PREFETCH_STATUS, no server worker remains
                    # blocked for the lifetime of the prefetch operation.
                    operation.completion_future = None
                    result = (
                        -1
                        if matched_chunks is None
                        else int(matched_chunks) * self.chunk_size
                    )
            except Exception:
                logger.exception(
                    "LMCache lookup completion failed for %s", operation.request_id
                )
                result = 0

        result = self._sync_leader_int(result)
        if result < 0:
            return None
        aligned_end = len(operation.token_ids) // self.chunk_size * self.chunk_size
        operation.total_hit_tokens = min(max(result, 0), aligned_end)
        operation.locks_held = operation.total_hit_tokens > 0
        previous_store_end = self._store_submitted_tokens.get(operation.request_id, 0)
        if operation.total_hit_tokens > previous_store_end:
            # The lookup proves this aligned prefix already exists in LMCache;
            # later stores for this request only need to cover its suffix.
            self._store_submitted_tokens[operation.request_id] = (
                operation.total_hit_tokens
            )
        return operation.total_hit_tokens

    def _slots_to_blocks(
        self,
        slots: torch.Tensor,
        *,
        slots_per_block: Optional[int] = None,
        allow_dummy_page: bool = False,
    ) -> list[int]:
        slots_per_block = slots_per_block or self.page_size
        if slots.numel() == 0:
            return []
        if slots.numel() % slots_per_block:
            raise ValueError("LMCache slots must contain complete SGLang pages")
        pages = (
            slots.detach()
            .to(dtype=torch.int64, device="cpu")
            .reshape(-1, slots_per_block)
        )
        starts = pages[:, 0]
        expected = starts[:, None] + torch.arange(slots_per_block, dtype=torch.int64)
        dummy_pages = torch.all(pages == 0, dim=1) if allow_dummy_page else None
        valid_pages = torch.all(pages == expected, dim=1)
        if dummy_pages is not None:
            valid_pages |= dummy_pages
        if torch.any(starts % slots_per_block) or not bool(torch.all(valid_pages)):
            raise ValueError("LMCache slots must be page-aligned contiguous pages")
        return (starts // slots_per_block).tolist()

    def _normalize_group_indices(
        self, device_indices: list[torch.Tensor] | torch.Tensor
    ) -> list[torch.Tensor]:
        if isinstance(device_indices, torch.Tensor):
            device_indices = [device_indices]
        if len(device_indices) != len(self._kv_groups):
            raise ValueError(
                f"Expected {len(self._kv_groups)} LMCache block-id groups, "
                f"got {len(device_indices)}"
            )
        for group, indices in zip(self._kv_groups, device_indices, strict=True):
            if indices.numel() % group.slots_per_block:
                raise ValueError(
                    f"LMCache group {group.name!r} indices do not contain "
                    "complete physical blocks"
                )
        return device_indices

    def _expand_engine_group_block_ids(
        self, engine_group_block_ids: list[list[int]]
    ) -> list[list[int]]:
        """Expand component block IDs to LMCache's physical kernel-group order."""
        return [
            list(engine_group_block_ids[engine_group_id])
            for engine_group_id in self._kernel_group_to_engine_group
        ]

    def _block_ids_for_transfer(
        self,
        device_indices: list[torch.Tensor] | torch.Tensor,
        *,
        allow_dummy_page: bool,
    ) -> list[list[int]]:
        per_engine_group = [
            self._slots_to_blocks(
                indices,
                slots_per_block=group.slots_per_block,
                allow_dummy_page=allow_dummy_page,
            )
            for group, indices in zip(
                self._kv_groups,
                self._normalize_group_indices(device_indices),
                strict=True,
            )
        ]
        return self._expand_engine_group_block_ids(per_engine_group)

    def _track_control_future(self, future: Any) -> None:
        # Keep fire-and-forget control RPCs alive and bound the local list.
        self._control_futures = [f for f in self._control_futures if not f.query()]
        self._control_futures.append(future)

    def _flush_control_futures(self) -> None:
        for future in self._control_futures:
            try:
                future.result(timeout=self._mq_timeout)
            except Exception:
                logger.warning(
                    "LMCache control RPC failed during shutdown", exc_info=True
                )
        self._control_futures.clear()

    def submit_load(
        self,
        operation: LMCacheLookupOperation,
        device_indices: list[torch.Tensor] | torch.Tensor,
        *,
        local_hit_tokens: int,
        owned_device_indices: Optional[torch.Tensor] = None,
        producer_stream: Any = None,
    ) -> LMCacheLoadOperation:
        if operation.total_hit_tokens is None or not operation.locks_held:
            raise RuntimeError("LMCache load requires a completed, locked lookup")
        total_hit = operation.total_hit_tokens
        start = local_hit_tokens // self.chunk_size * self.chunk_size
        prefix_pad = local_hit_tokens - start
        group_indices = self._normalize_group_indices(device_indices)
        fresh_tokens = total_hit - local_hit_tokens
        transfer_tokens = total_hit - start
        for group, indices in zip(self._kv_groups, group_indices, strict=True):
            group_covered_tokens = (
                int(indices.numel()) * group.tokens_per_block // group.slots_per_block
            )
            expected_tokens = transfer_tokens if group.recurrent_state else fresh_tokens
            if group_covered_tokens != expected_tokens:
                raise ValueError(
                    f"LMCache load group {group.name!r} indices cover "
                    f"{group_covered_tokens} tokens, expected {expected_tokens}"
                )
        if any(
            prefix_pad % group.tokens_per_block
            for group in self._kv_groups
            if not group.recurrent_state
        ):
            raise ValueError(
                "LMCache local hit must align to every group tokens_per_block"
            )
        fresh_blocks = [
            self._slots_to_blocks(
                indices,
                slots_per_block=group.slots_per_block,
                allow_dummy_page=True,
            )
            for group, indices in zip(self._kv_groups, group_indices, strict=True)
        ]
        engine_group_block_ids = [
            (
                blocks
                if group.recurrent_state
                else [0] * (prefix_pad // group.tokens_per_block) + blocks
            )
            for group, blocks in zip(self._kv_groups, fresh_blocks, strict=True)
        ]
        block_ids = self._expand_engine_group_block_ids(engine_group_block_ids)
        key = self._create_key(
            operation, start=start, end=total_hit, worker_id=self.kv_worker_id
        )
        if operation.lock_start != start:
            logger.warning(
                "LMCache lookup lock boundary %d does not "
                "match retrieve start %d for %s",
                operation.lock_start,
                start,
                operation.request_id,
            )
        try:
            event = (
                self._new_event()
                if producer_stream is None
                else self._new_event(producer_stream)
            )
            future = self._transfer_ctx.submit_retrieve(
                operation.request_id,
                key,
                self.instance_id,
                self._kv_caches,
                block_ids,
                event,
                self.blocks_in_chunk,
                skip_first_n_tokens=prefix_pad,
            )
        except Exception:
            # Every TP x PP rank must enqueue one operation in the same order. A
            # ready-false future lets completion consensus fail the operation
            # after successful peers finish their already-submitted retrieve.
            logger.exception(
                "LMCache retrieve submission failed for %s",
                operation.request_id,
            )
            future = _ImmediateFuture(False)
            event = None
        if event is not None:
            future.retain_reference(event)
        return LMCacheLoadOperation(
            request_id=operation.request_id,
            token_ids=operation.token_ids,
            start=start,
            end=total_hit,
            local_hit_tokens=local_hit_tokens,
            device_indices=(
                group_indices[0]
                if owned_device_indices is None
                else owned_device_indices
            ),
            future=future,
            lookup=operation,
        )

    def prepare_load_on_stream(
        self, operation: LMCacheLoadOperation, stream: Any
    ) -> bool:
        """Order a consumer stream after an asynchronous retrieve.

        Args:
            operation: Retrieve operation returned by :meth:`submit_load`.
            stream: SGLang forward stream that consumes the loaded KV cache.

        Returns:
            ``True`` when every TP/PP rank successfully submitted its retrieve;
            otherwise ``False``.

        Notes:
            Waiting for the raw MQ response imports the LMCache server's
            completion event. Device completion is then ordered with a stream
            wait, so the CPU does not wait for H2D. On a cross-rank failure the
            successful ranks synchronize their local work before the caller
            releases destination slots.
        """
        local_success = False
        prepared = False
        try:
            local_success = bool(operation.future.prepare(timeout=self._mq_timeout))
            prepared = True
            operation.future.wait_on_stream(stream, timeout=0)
        except Exception:
            logger.exception(
                "LMCache retrieve preparation failed for %s", operation.request_id
            )

        success = self._sync_success(local_success)
        if success:
            return True

        # Another rank may have failed after this rank successfully enqueued
        # H2D. Wait locally before SGLang returns these destination slots to its
        # allocator; otherwise LMCache could still be writing reused memory.
        if prepared:
            try:
                operation.future.result(timeout=self._mq_timeout)
            except Exception:
                logger.exception(
                    "Failed to drain LMCache retrieve for %s", operation.request_id
                )
        operation.lookup.locks_held = False
        operation.result = False
        self._cleanup_lookup_result(operation.lookup)
        return False

    def complete_load(
        self, operation: LMCacheLoadOperation, *, synchronize: bool = True
    ) -> bool:
        if operation.result is not None:
            return operation.result
        success = False
        try:
            success = bool(operation.future.result(timeout=0))
        except Exception:
            logger.exception("LMCache retrieve failed for %s", operation.request_id)
        if synchronize:
            success = self._sync_success(success)
        operation.lookup.locks_held = False
        operation.result = success
        self._cleanup_lookup_result(operation.lookup)
        return success

    def _store_group_blocks_are_valid(
        self, group: LMCacheKVGroup, blocks: list[int]
    ) -> bool:
        """Validate null blocks after applying LMCache's per-chunk SWA cut."""
        if group.recurrent_state:
            return True
        if group.sliding_window_size < 0:
            return 0 not in blocks

        blocks_per_chunk = self.chunk_size // group.tokens_per_block
        kept_blocks_per_chunk = (
            min(self.chunk_size, group.sliding_window_size) // group.tokens_per_block
        )
        if kept_blocks_per_chunk <= 0:
            return False

        for chunk_start in range(0, len(blocks), blocks_per_chunk):
            chunk = blocks[chunk_start : chunk_start + blocks_per_chunk]
            kept = chunk[-kept_blocks_per_chunk:]
            # An all-null historical SWA chunk is intentionally absent and the
            # separated LMCache object group will skip it. A partly missing
            # retained window would copy the null page together with real KV.
            if any(kept) and 0 in kept:
                return False
        return True

    def submit_store(
        self,
        request_id: str,
        token_ids: list[int],
        device_indices: list[torch.Tensor] | torch.Tensor,
        *,
        cache_salt: str,
    ) -> Optional[LMCacheStoreOperation]:
        aligned_end = len(token_ids) // self.chunk_size * self.chunk_size
        start = min(self._store_submitted_tokens.get(request_id, 0), aligned_end)
        start = start // self.chunk_size * self.chunk_size
        if aligned_end <= start:
            return None
        lookup = LMCacheLookupOperation(
            request_id=request_id,
            token_ids=list(token_ids),
            local_hit_tokens=0,
            cache_salt=cache_salt,
        )
        group_indices = self._normalize_group_indices(device_indices)
        for group, indices in zip(self._kv_groups, group_indices, strict=True):
            group_covered_tokens = (
                int(indices.numel()) * group.tokens_per_block // group.slots_per_block
            )
            if group_covered_tokens < aligned_end:
                raise ValueError(
                    f"LMCache store group {group.name!r} indices cover "
                    f"{group_covered_tokens} tokens, expected at least {aligned_end}"
                )
        engine_group_blocks = []
        for group, indices in zip(self._kv_groups, group_indices, strict=True):
            start_slot = start * group.slots_per_block // group.tokens_per_block
            end_slot = aligned_end * group.slots_per_block // group.tokens_per_block
            engine_group_blocks.append(
                self._slots_to_blocks(
                    indices[start_slot:end_slot],
                    slots_per_block=group.slots_per_block,
                    allow_dummy_page=True,
                )
            )
        # Slot/page zero is SGLang's padding sink. FULL attention must always
        # be complete. SWA may legitimately have all-null historical chunks;
        # with --separate-object-groups LMCache skips those chunks and stores
        # the independently complete FULL prefix.
        can_submit = all(
            self._store_group_blocks_are_valid(group, blocks)
            for group, blocks in zip(self._kv_groups, engine_group_blocks, strict=True)
        )
        if not self._sync_success(can_submit):
            if not can_submit:
                logger.debug(
                    "LMCache store deferred for %s: transfer range contains an "
                    "incomplete retained component page",
                    request_id,
                )
            return None
        submitted = False
        event = None
        if self.is_kv_writer:
            try:
                blocks = self._expand_engine_group_block_ids(engine_group_blocks)
                key = self._create_key(
                    lookup, start=start, end=aligned_end, worker_id=self.kv_worker_id
                )
                event = self._new_event()
                future = self._transfer_ctx.submit_store(
                    request_id,
                    key,
                    self.instance_id,
                    self._kv_caches,
                    blocks,
                    event,
                    self.blocks_in_chunk,
                )
                submitted = True
            except Exception:
                logger.exception("LMCache store submission failed for %s", request_id)
                future = _ImmediateFuture(False)
                event = None
        else:
            # Every scheduler rank must retain a pending operation and enter
            # complete_store() in identical order. The ready-success placeholder
            # lets non-writer MLA ranks participate in completion collectives
            # without issuing duplicate D2H/STORE work.
            future = _ImmediateFuture(True)
        # STORE also creates a server-side Session via resolve_obj_keys().
        # Track it only after an RPC was actually submitted. MAX is required:
        # one successful TP/PP worker is enough for the shared Session to exist.
        if self._sync_leader_int(int(submitted)) > 0:
            self._active_sessions.add(request_id)
        if event is not None:
            future.retain_reference(event)
        self._store_submitted_tokens[request_id] = aligned_end
        return LMCacheStoreOperation(request_id, start, aligned_end, future)

    def complete_store(
        self, operation: LMCacheStoreOperation, *, synchronize: bool = True
    ) -> bool:
        if operation.result is not None:
            return operation.result
        success = False
        try:
            success = bool(operation.future.result(timeout=0))
        except Exception:
            logger.exception("LMCache store failed for %s", operation.request_id)
        operation.result = self._sync_success(success) if synchronize else success
        if (
            not operation.result
            and self._store_submitted_tokens.get(operation.request_id) == operation.end
        ):
            # Allow a later chunk/final store to retry this failed tail.
            self._store_submitted_tokens[operation.request_id] = operation.start
        return operation.result

    def _cleanup_lookup_result(self, operation: LMCacheLookupOperation) -> None:
        """Forget client-side lookup state without releasing server read locks.

        A successful retrieve releases the locks it consumed from the LMCache
        server's H2D completion callback.  At that point only the local lookup
        bookkeeping must be removed.
        """
        if self._lookups.get(operation.request_id) is operation:
            self._lookups.pop(operation.request_id, None)

    def free_lookup_locks(self, request_id: str, start: int, end: int) -> None:
        """Release one lookup-locked prefix range before retrieve submission."""
        operation = self._lookups.get(request_id)
        if operation is None or not operation.locks_held:
            return
        if operation.total_hit_tokens is None:
            raise RuntimeError(f"LMCache lookup for {request_id} is not complete")
        if start != operation.lock_start:
            raise RuntimeError(
                f"LMCache lookup lock boundary {operation.lock_start} does not "
                f"match release start {start} for {request_id}"
            )
        end = min(end, operation.total_hit_tokens)
        if start >= end:
            return

        if self.is_lookup_leader:
            # First Party
            from lmcache.v1.multiprocess.protocol import RequestType

            key = self._create_key(operation, start=start, end=end, worker_id=None)
            self._track_control_future(
                self._send_request(
                    self._mq_client,
                    RequestType.FREE_LOOKUP_LOCKS,
                    [key, self.kv_world_size],
                )
            )
        operation.lock_start = end
        if end == operation.total_hit_tokens:
            operation.locks_held = False
            self._lookups.pop(request_id, None)

    def end_session(self, request_id: str) -> None:
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType

        # A local lookup object can represent an aligned-empty lookup for
        # which no LOOKUP RPC was sent. Only _active_sessions proves that a
        # LOOKUP or STORE reached the server and requires END_SESSION.
        was_active = request_id in self._active_sessions
        self._lookups.pop(request_id, None)
        if was_active and self.is_lookup_leader:
            self._track_control_future(
                self._send_request(
                    self._mq_client, RequestType.END_SESSION, [request_id]
                )
            )
        self._active_sessions.discard(request_id)

    def finish_request(self, request_id: str) -> None:
        self.end_session(request_id)
        self._store_submitted_tokens.pop(request_id, None)

    def end_all_sessions(self) -> None:
        for request_id in list(self._active_sessions):
            self.end_session(request_id)

    def clear(self) -> bool:
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType

        success = True
        try:
            if self.is_lookup_leader:
                self._send_request(self._mq_client, RequestType.CLEAR, []).result(
                    timeout=self._mq_timeout
                )
        except Exception:
            logger.exception("Failed to clear LMCache MP storage")
            success = False
        return self._sync_success(success)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # First Party
        from lmcache.v1.multiprocess.protocol import RequestType

        self.end_all_sessions()
        self._flush_control_futures()
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self._heartbeat_interval))
            self._heartbeat_thread = None
        if self._registered:
            try:
                self._send_request(
                    self._mq_client,
                    RequestType.UNREGISTER_KV_CACHE,
                    [self.instance_id],
                ).result(timeout=self._mq_timeout)
            except Exception:
                logger.warning("Failed to unregister LMCache KV tensors", exc_info=True)
            self._registered = False
        if self._transfer_ctx is not None:
            self._transfer_ctx.close()
            self._transfer_ctx = None
        self._mq_client.close()
