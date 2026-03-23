# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import os
import threading
import time
import uuid

# Third Party
from sglang.srt.configs.model_config import ModelConfig
import torch
import torch.distributed as dist
import zmq

# First Party
from lmcache.integration.sglang.sglang_adapter import (
    LoadMetadata,
    StoreMetadata,
    resolve_sglang_kv_pools,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MQ_TIMEOUT,
    HeartbeatThread,
    get_lmcache_chunk_size,
    send_lmcache_request,
)
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCacheRegistration,
)
from lmcache.v1.multiprocess.futures import CUDAMessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType

logger = init_logger(__name__)


def wrap_sglang_kv_caches(
    k_pool: list[torch.Tensor],
    v_pool: list[torch.Tensor],
) -> list[list[CudaIPCWrapper]]:
    return [
        [CudaIPCWrapper(tensor) for tensor in k_pool],
        [CudaIPCWrapper(tensor) for tensor in v_pool],
    ]


@dataclass
class _ActiveRetrieveState:
    request_id: str
    token_ids: list[int]
    offset: int
    matched_end: int
    block_ids: list[int]
    in_flight_layer: int
    future: CUDAMessagingFuture[bool] | None


class LMCacheMPLayerwiseConnector:
    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        page_size: int,
        host: str,
        port: int,
        k_pool: Optional[list[torch.Tensor]] = None,
        v_pool: Optional[list[torch.Tensor]] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        token_to_kv_pool_allocator: object | None = None,
        kvcache: object | None = None,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        k_pool, v_pool = resolve_sglang_kv_pools(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            kvcache=kvcache,
            k_pool=k_pool,
            v_pool=v_pool,
        )
        if not k_pool or not v_pool:
            raise ValueError("SGLang MP connector requires non-empty K/V pools")
        if len(k_pool) != len(v_pool):
            raise ValueError("K/V pool layer counts must match")
        if not k_pool[0].is_cuda:
            raise ValueError("SGLang MP connector requires CUDA KV caches")
        if tp_size > 1 and tp_group is None:
            raise ValueError("tp_group is required when tp_size > 1")

        self.tp_size = tp_size
        self.worker_id = rank
        self.page_size = page_size
        self.device = k_pool[0].device
        self.model_name = sgl_config.model_path
        self.num_layers = len(k_pool)
        self.tp_group = tp_group
        self.instance_id = os.getpid()
        self._mq_timeout = mq_timeout
        self._heartbeat_interval = heartbeat_interval
        self._active_retrieves: list[_ActiveRetrieveState] = []
        self._registered = False
        self._heartbeat: HeartbeatThread | None = None
        self._health_event = threading.Event()
        self._health_event.set()

        self.context = zmq.Context.instance()
        self.mq_client = MessageQueueClient(f"tcp://{host}:{port}", self.context)

        self._lmcache_chunk_size = get_lmcache_chunk_size(self.mq_client)
        if self._lmcache_chunk_size % self.page_size != 0:
            raise ValueError(
                "LMCache chunk size must be a multiple of SGLang page size, got "
                f"{self._lmcache_chunk_size} and {self.page_size}"
            )

        registration = KVCacheRegistration(
            instance_id=self.instance_id,
            model_name=self.model_name,
            world_size=self.tp_size,
            engine_type="sglang",
            block_size=self.page_size,
            kv_caches=wrap_sglang_kv_caches(k_pool, v_pool),
        )
        send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [registration],
        ).result(timeout=self._mq_timeout)
        self._registered = True
        self.start_heartbeat()

    def start_heartbeat(self) -> None:
        if self._heartbeat is not None:
            return
        self._heartbeat = HeartbeatThread(
            mq_client=self.mq_client,
            health_event=self._health_event,
            interval=self._heartbeat_interval,
        )
        self._heartbeat.start()

    @property
    def is_healthy(self) -> bool:
        return self._health_event.is_set()

    @torch.no_grad()
    def global_min_tokens(
        self,
        local_tokens: int,
        tp_group: dist.ProcessGroup | None,
        device: torch.device,
    ) -> int:
        if self.tp_size == 1:
            return local_tokens
        if tp_group is None:
            raise ValueError("tp_group is required when tp_size > 1")

        t = torch.tensor([local_tokens], dtype=torch.int32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=tp_group)
        return int(t.item())

    def create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        no_worker_id: bool = False,
    ) -> IPCCacheEngineKey:
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.tp_size,
            worker_id=None if no_worker_id else self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )

    def wait_for_lookup(self, job_id: int) -> int:
        deadline = time.monotonic() + self._mq_timeout
        while True:
            matched_chunks = send_lmcache_request(
                self.mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [job_id],
            ).result(timeout=self._mq_timeout)
            if matched_chunks is not None:
                return matched_chunks * self._lmcache_chunk_size
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for LMCache prefetch to finish")
            time.sleep(0.001)

    def slot_mapping_to_block_ids(self, slot_mapping: torch.Tensor) -> list[int]:
        if slot_mapping.numel() == 0:
            return []
        if slot_mapping.numel() % self.page_size != 0:
            raise ValueError(
                "Slot mapping length must be page-aligned for MP mode, got "
                f"{slot_mapping.numel()} and page_size={self.page_size}"
            )

        groups = (
            slot_mapping.detach()
            .to(dtype=torch.int64, device="cpu")
            .reshape(-1, self.page_size)
        )
        starts = groups[:, 0]
        if torch.any(starts % self.page_size != 0):
            raise ValueError("Slot mapping does not start on page boundaries")

        expected = starts[:, None] + torch.arange(self.page_size, dtype=torch.int64)
        if not torch.equal(groups, expected):
            raise ValueError("Slot mapping must cover full contiguous pages in MP mode")

        return (starts // self.page_size).tolist()

    def submit_retrieve(
        self,
        state: _ActiveRetrieveState,
        layer_id: int,
    ) -> CUDAMessagingFuture[bool]:
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                self.create_key(
                    state.token_ids,
                    start=state.offset,
                    end=state.matched_end,
                    request_id=state.request_id,
                ),
                self.instance_id,
                state.block_ids,
                event.ipc_handle(),
                0,
                layer_id,
                layer_id + 1,
            ],
        ).to_cuda_future(device=self.device)
        state.in_flight_layer = layer_id
        state.future = future
        return future

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> None:
        if start >= end or not self.is_healthy:
            return
        send_lmcache_request(
            self.mq_client,
            RequestType.FREE_LOOKUP_LOCKS,
            [
                self.create_key(
                    token_ids,
                    start=start,
                    end=end,
                    request_id=request_id,
                    no_worker_id=True,
                ),
                self.tp_size,
            ],
        )

    def end_session(self, request_id: str) -> None:
        if not self.is_healthy:
            return
        send_lmcache_request(self.mq_client, RequestType.END_SESSION, [request_id])

    def cleanup_retrieve_state(self, state: _ActiveRetrieveState) -> None:
        self.free_lookup_locks(
            state.token_ids, state.offset, state.matched_end, state.request_id
        )
        self.end_session(state.request_id)

    def chunk_size(self) -> int:
        return self._lmcache_chunk_size

    def start_load_kv(self, load_metadata: LoadMetadata) -> int:
        if not self.is_healthy:
            return 0

        aligned_end = (len(load_metadata.token_ids) // self._lmcache_chunk_size) * (
            self._lmcache_chunk_size
        )
        offset = load_metadata.offset
        if aligned_end == 0 or offset >= aligned_end:
            return 0
        if offset % self._lmcache_chunk_size != 0:
            raise ValueError(
                "LMCache MP mode requires chunk-aligned offsets, got "
                f"{offset} and chunk_size={self._lmcache_chunk_size}"
            )

        request_id = str(uuid.uuid4())
        lookup_key = self.create_key(
            load_metadata.token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            no_worker_id=True,
        )

        job_id = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [lookup_key, self.tp_size],
        ).result(timeout=self._mq_timeout)
        retrieve_token_num = self.wait_for_lookup(job_id)

        retrieve_token_num = self.global_min_tokens(
            retrieve_token_num, self.tp_group, self.device
        )

        if retrieve_token_num <= offset:
            self.free_lookup_locks(
                load_metadata.token_ids, 0, retrieve_token_num, request_id
            )
            self.end_session(request_id)
            return 0

        self.free_lookup_locks(load_metadata.token_ids, 0, offset, request_id)
        block_ids = self.slot_mapping_to_block_ids(
            load_metadata.slot_mapping[offset:retrieve_token_num]
        )

        state = _ActiveRetrieveState(
            request_id=request_id,
            token_ids=load_metadata.token_ids,
            offset=offset,
            matched_end=retrieve_token_num,
            block_ids=block_ids,
            in_flight_layer=0,
            future=None,
        )
        self.submit_retrieve(state, 0)
        self._active_retrieves.append(state)
        return retrieve_token_num - offset

    def load_kv_layerwise(self, layer_id: int) -> None:
        if not self._active_retrieves:
            return

        finished_indices: list[int] = []
        failures: list[Exception] = []
        for i, state in enumerate(self._active_retrieves):
            if state.in_flight_layer != layer_id:
                continue

            try:
                if state.future is None:
                    raise RuntimeError(
                        f"LMCache MP retrieve state is missing a future for "
                        f"request_id={state.request_id}"
                    )
                if not state.future.result(timeout=self._mq_timeout):
                    raise RuntimeError(
                        f"LMCache MP retrieve failed for request_id={state.request_id}"
                    )

                next_layer = layer_id + 1
                if next_layer < self.num_layers:
                    self.submit_retrieve(state, next_layer)
                else:
                    self.end_session(state.request_id)
                    finished_indices.append(i)
            except Exception as exc:
                self.cleanup_retrieve_state(state)
                finished_indices.append(i)
                failures.append(exc)

        for i in reversed(finished_indices):
            del self._active_retrieves[i]
        if failures:
            raise failures[0]

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        if not self.is_healthy:
            return

        aligned_end = (len(store_metadata.token_ids) // self._lmcache_chunk_size) * (
            self._lmcache_chunk_size
        )
        if aligned_end == 0:
            return

        request_id = str(uuid.uuid4())
        block_ids = self.slot_mapping_to_block_ids(
            store_metadata.kv_indices[:aligned_end]
        )
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        success = (
            send_lmcache_request(
                self.mq_client,
                RequestType.STORE,
                [
                    self.create_key(
                        store_metadata.token_ids,
                        start=0,
                        end=aligned_end,
                        request_id=request_id,
                    ),
                    self.instance_id,
                    block_ids,
                    event.ipc_handle(),
                ],
            )
            .to_cuda_future(device=self.device)
            .result(timeout=self._mq_timeout)
        )
        self.end_session(request_id)
        if not success:
            raise RuntimeError("LMCache MP store failed")

    def reset(self) -> None:
        while self._active_retrieves:
            state = self._active_retrieves.pop()
            self.cleanup_retrieve_state(state)

    def close(self) -> None:
        self.reset()
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        if self._registered:
            try:
                send_lmcache_request(
                    self.mq_client,
                    RequestType.UNREGISTER_KV_CACHE,
                    [self.instance_id],
                ).result(timeout=self._mq_timeout)
            except Exception:
                logger.warning("Failed to unregister SGLang MP KV cache", exc_info=True)
            self._registered = False
        self.mq_client.close()
