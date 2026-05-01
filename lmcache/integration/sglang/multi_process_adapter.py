# SPDX-License-Identifier: Apache-2.0
# Standard
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
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MQ_TIMEOUT,
    HeartbeatThread,
    get_lmcache_chunk_size,
    send_lmcache_request,
)
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType

logger = init_logger(__name__)


def _wrap_sglang_kv_caches(
    k_pool: list[torch.Tensor],
    v_pool: list[torch.Tensor],
) -> list[CudaIPCWrapper]:
    """Flatten SGLang's depth-2 ``[K_layers, V_layers]`` KV layout into a
    single flat ``list[CudaIPCWrapper]`` so it fits upstream's wire
    ``KVCache`` payload type. The ``layout_hints={"kv_layout":
    "SGLANG_MHA"}`` registration field tells the daemon to split this list
    back at its midpoint via :func:`reshape_flat_kv_for_engine` before
    format detection.
    """
    return [CudaIPCWrapper(tensor) for tensor in k_pool] + [
        CudaIPCWrapper(tensor) for tensor in v_pool
    ]


class LMCacheMPConnector:
    """SGLang LMCache MP connector — layerwise interface, single-shot wire.

    From SGLang's POV this connector is a drop-in for ``LMCacheLayerwise-
    Connector``: it exposes ``start_load_kv`` + ``load_kv_layerwise``,
    so SGLang's existing ``register_layer_transfer_counter`` hook keeps
    working unchanged. Internally:

    - ``start_load_kv`` issues **one** RETRIEVE that covers the full
      prefix and host-blocks until every layer has been transferred. By
      the time it returns, the GPU KV slots are populated.
    - ``load_kv_layerwise(layer_id)`` is a no-op — data is already there.

    The wire protocol stays at upstream's flat 5-arg RETRIEVE / 4-arg
    STORE, with no layer windowing. The daemon does the per-layer
    ``single_layer_kv_transfer_sgl`` dispatch internally based on the
    detected ``GPUKVFormat``.

    Mirrors the in-process `LMCacheConnector` from SGLang's perspective: a
    single blocking `load_kv` call drives the full retrieve and only returns
    once every layer has been written into the GPU KV cache. Internally the
    daemon transfers each layer sequentially because the SGLang MHA
    `GPUKVFormat` only supports the layerwise transfer kernel.
    """

    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        page_size: int,
        host: str,
        port: int,
        k_pool: list[torch.Tensor],
        v_pool: list[torch.Tensor],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
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

        # Upstream's REGISTER_KV_CACHE protocol takes flat positional args:
        # (instance_id, kv_cache, model_name, world_size, engine_type,
        # layout_hints). SGLang's natural KV layout is depth-2
        # ([K_layers, V_layers]); we flatten it on the wire to fit
        # ``KVCache = list[CudaIPCWrapper]`` and use the
        # ``kv_layout="SGLANG_MHA"`` hint so the daemon reconstructs the
        # depth-2 structure (via reshape_flat_kv_for_engine) before format
        # detection.
        send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self.instance_id,
                _wrap_sglang_kv_caches(k_pool, v_pool),
                self.model_name,
                self.tp_size,
                EngineType.SGLANG,
                {"kv_layout": "SGLANG_MHA", "block_size": self.page_size},
            ],
        ).result(timeout=self._mq_timeout)
        self._registered = True
        self._start_heartbeat()
        # State for the layerwise wrapper around the single-shot wire load.
        self._pending_load_event: torch.cuda.Event | None = None

    def _start_heartbeat(self) -> None:
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

    def chunk_size(self) -> int:
        return self._lmcache_chunk_size

    @torch.no_grad()
    def _global_min_tokens(self, local_tokens: int) -> int:
        if self.tp_size == 1:
            return local_tokens
        t = torch.tensor([local_tokens], dtype=torch.int32, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.tp_group)
        return int(t.item())

    def _create_key(
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

    def _slot_mapping_to_block_ids(self, slot_mapping: torch.Tensor) -> list[int]:
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

    def _wait_for_lookup(self, request_id: str) -> int:
        """Poll QUERY_PREFETCH_STATUS with the LOOKUP's request_id until the
        daemon reports a chunk count. Upstream switched LOOKUP to a fire-
        and-forget call and keys the prefetch job by request_id (a string);
        the result is the number of matched chunks once available.
        """
        deadline = time.monotonic() + self._mq_timeout
        while True:
            matched_chunks = send_lmcache_request(
                self.mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
            ).result(timeout=self._mq_timeout)
            if matched_chunks is not None:
                return matched_chunks * self._lmcache_chunk_size
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for LMCache prefetch to finish")
            time.sleep(0.001)

    def _free_lookup_locks(
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
                self._create_key(
                    token_ids,
                    start=start,
                    end=end,
                    request_id=request_id,
                    no_worker_id=True,
                ),
                self.tp_size,
            ],
        )

    def _end_session(self, request_id: str) -> None:
        if not self.is_healthy:
            return
        send_lmcache_request(self.mq_client, RequestType.END_SESSION, [request_id])

    def _submit_retrieve(
        self,
        request_id: str,
        token_ids: list[int],
        offset: int,
        matched_end: int,
        block_ids: list[int],
        skip_prefix_n_blocks: int = 0,
    ):
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        return send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                self._create_key(
                    token_ids,
                    start=offset,
                    end=matched_end,
                    request_id=request_id,
                ),
                self.instance_id,
                block_ids,
                event.ipc_handle(),
                skip_prefix_n_blocks,
            ],
        ).to_cuda_future(device=self.device)

    def start_load_kv(self, load_metadata: LoadMetadata) -> int:
        """Issue a single blocking RETRIEVE that covers the full prefix.

        Returns the number of tokens actually retrieved. By the time this
        method returns, the daemon has finished iterating
        ``single_layer_kv_transfer_sgl`` over every layer for this chunk,
        so the GPU KV slots are already populated. Subsequent
        ``load_kv_layerwise`` calls are no-ops.
        """
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
        lookup_key = self._create_key(
            load_metadata.token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            no_worker_id=True,
        )
        # Upstream LOOKUP is fire-and-forget: it submits a prefetch job
        # keyed by request_id and returns None. We then poll
        # QUERY_PREFETCH_STATUS with the same request_id.
        send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [lookup_key, self.tp_size],
        ).result(timeout=self._mq_timeout)
        retrieve_token_num = self._wait_for_lookup(request_id)
        retrieve_token_num = self._global_min_tokens(retrieve_token_num)

        if retrieve_token_num <= offset:
            self._free_lookup_locks(
                load_metadata.token_ids, 0, retrieve_token_num, request_id
            )
            self._end_session(request_id)
            return 0

        # ``slot_mapping[offset : offset + prefix_pad)`` is sentinel ``-1`` —
        # those tokens already live in the engine's radix tree and must not
        # be overwritten. We still RETRIEVE the full chunk-aligned range
        # (LMCache stores at chunk granularity), but tell the daemon to skip
        # the leading ``prefix_pad // page_size`` blocks. Real block_ids are
        # computed only from the freshly-allocated slot range; the skipped
        # blocks get harmless placeholder ids the kernel never dereferences.
        prefix_pad = load_metadata.prefix_pad
        if prefix_pad < 0 or prefix_pad % self.page_size != 0:
            raise ValueError(
                f"prefix_pad must be a non-negative multiple of page_size, got "
                f"prefix_pad={prefix_pad}, page_size={self.page_size}"
            )
        fresh_start = offset + prefix_pad
        prefix_pad_pages = prefix_pad // self.page_size

        self._free_lookup_locks(load_metadata.token_ids, 0, offset, request_id)
        if fresh_start >= retrieve_token_num:
            # LMCache's hit was entirely covered by what's already in radix.
            self._free_lookup_locks(
                load_metadata.token_ids, offset, retrieve_token_num, request_id
            )
            self._end_session(request_id)
            return retrieve_token_num - offset
        fresh_block_ids = self._slot_mapping_to_block_ids(
            load_metadata.slot_mapping[fresh_start:retrieve_token_num]
        )
        block_ids = [0] * prefix_pad_pages + fresh_block_ids

        try:
            future = self._submit_retrieve(
                request_id=request_id,
                token_ids=load_metadata.token_ids,
                offset=offset,
                matched_end=retrieve_token_num,
                block_ids=block_ids,
                skip_prefix_n_blocks=prefix_pad_pages,
            )
            if not future.result(timeout=self._mq_timeout):
                raise RuntimeError(
                    f"LMCache MP retrieve failed for request_id={request_id}"
                )
        finally:
            self._free_lookup_locks(
                load_metadata.token_ids, offset, retrieve_token_num, request_id
            )
            self._end_session(request_id)
        return retrieve_token_num - offset

    def load_kv_layerwise(self, layer_id: int) -> None:
        """No-op. ``start_load_kv`` already host-blocked until every
        layer was transferred, so by the time SGLang's per-layer hook
        fires the data is already in the GPU KV slots.
        """
        return

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        if not self.is_healthy:
            return

        aligned_end = (len(store_metadata.token_ids) // self._lmcache_chunk_size) * (
            self._lmcache_chunk_size
        )
        if aligned_end == 0:
            return

        request_id = str(uuid.uuid4())
        block_ids = self._slot_mapping_to_block_ids(
            store_metadata.kv_indices[:aligned_end]
        )
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        success = (
            send_lmcache_request(
                self.mq_client,
                RequestType.STORE,
                [
                    self._create_key(
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
        self._end_session(request_id)
        if not success:
            raise RuntimeError("LMCache MP store failed")

    def reset(self) -> None:
        pass

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
