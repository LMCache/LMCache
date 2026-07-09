# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Optional
import os
import threading

# Third Party
from sglang.srt.configs.model_config import ModelConfig
import torch
import torch.distributed as dist
import zmq

# First Party
from lmcache import torch_dev
from lmcache.integration.sglang.kv_cache_groups import (
    build_sglang_layout_hints,
    create_engine_group_infos_from_sglang,
)
from lmcache.integration.sglang.sglang_adapter import LoadMetadata, StoreMetadata
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MQ_TIMEOUT,
    HeartbeatThread,
    get_lmcache_chunk_size,
    send_lmcache_request,
)
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, expand_engine_block_ids
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

logger = init_logger(__name__)

# Extra seconds the WAIT_PREFETCH_STATUS response is allowed beyond the daemon's
# own blocking-wait budget, to cover the request/response round trip.
_WAIT_LOOKUP_RESPONSE_BUFFER_S = 5.0


def _wrap_sglang_kv_caches(
    kv_cache_pools: list[list[torch.Tensor]],
) -> KVCache:
    """Flatten SGLang's pool groups into a single flat ``KVCache`` so it
    fits upstream's wire ``KVCache`` payload type.  Each element of
    *kv_cache_pools* is a per-layer tensor list for one pool group
    (K, V, DSA-index, …).  The daemon's
    :func:`normalize_kv_and_discover_format` resolves the layout from
    ``EngineType.SGLANG`` plus ``tokens_per_block``.
    """
    wrapped: KVCache = []
    for pool in kv_cache_pools:
        wrapped.extend(CudaIPCWrapper(tensor) for tensor in pool)
    return wrapped


def _completed_future(result: bool) -> MessagingFuture[bool]:
    """Return an already-completed future resolving to ``result``.

    Used by :meth:`LMCacheMPConnector.store_kv_async` for the paths that
    perform no wire send, so every return value is a pollable future and
    callers never have to special-case ``None``. ``result`` carries the
    store outcome for that path: ``False`` when the connector is
    unhealthy (nothing was stored), ``True`` when there was simply no
    chunk-aligned range to store (a no-op success).

    Args:
        result: the success value the returned future resolves to.

    Returns:
        A ``MessagingFuture`` whose ``result()`` is immediately
        ``result``.
    """
    future: MessagingFuture[bool] = MessagingFuture()
    future.set_result(result)
    return future


@dataclass
class _PendingLookup:
    """Per-request_id state retained between ``lookup_kv`` and
    ``retrieve_kv``.

    Attributes:
        token_ids: tokens that LOOKUP was issued for.
        matched_token_num: number of chunk-aligned tokens the daemon
            reported as cached (return value of LOOKUP →
            QUERY_PREFETCH_STATUS).
        locks_held: True iff the daemon still holds the read locks
            reserved by this LOOKUP. RETRIEVE consumes them; explicit
            FREE_LOOKUP_LOCKS releases them.
    """

    token_ids: list[int]
    matched_token_num: int
    locks_held: bool


class LMCacheMPConnector:
    """SGLang LMCache multi-process connector.

    Talks to a standalone LMCache daemon over ZMQ.

    - ``lookup_kv``: fires LOOKUP. Daemon prefetches missing
      chunks L2→L1 (DRAM), keeps the read locks held, returns the
      matched-token count.
    - ``retrieve_kv``: fires RETRIEVE using the cached LOOKUP result.
      Daemon copies L1→GPU via ``multi_layer_block_kv_transfer``
      (single CUDA launch, all layers) and releases the read locks
      via ``finish_read_prefetched``.
    - ``release_pending``: frees the held locks when no RETRIEVE will
      follow (LMCache had nothing fresh beyond radix).
    - ``end_session``: per-request cleanup. Frees any still-held
      locks then sends END_SESSION so the daemon doesn't leak
      read-lock reservations.
    """

    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        page_size: int,
        host: str,
        port: int,
        kv_cache_pools: list[list[torch.Tensor]],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        self.tp_size = tp_size
        self.worker_id = rank
        self.page_size = page_size
        self.device = kv_cache_pools[0][0].device
        self.model_name = sgl_config.model_path
        self.num_layers = len(kv_cache_pools[0])
        self.tp_group = tp_group
        self.instance_id = os.getpid()
        self._mq_timeout = mq_timeout
        self._heartbeat_interval = heartbeat_interval
        self._registered = False
        self._heartbeat: HeartbeatThread | None = None
        self._health_event = threading.Event()
        self._health_event.set()
        self._pending_lookups: dict[str, _PendingLookup] = {}
        self._pending_lookups_lock = threading.Lock()

        self.context = zmq.Context.instance()
        self.mq_client = MessageQueueClient(f"tcp://{host}:{port}", self.context)

        self._lmcache_chunk_size = get_lmcache_chunk_size(self.mq_client)
        if self._lmcache_chunk_size % self.page_size != 0:
            raise ValueError(
                "LMCache chunk size must be a multiple of SGLang page size, got "
                f"{self._lmcache_chunk_size} and {self.page_size}"
            )

        # DSA classification is delegated to ``create_engine_group_infos_from_sglang``
        # (uses the same ``is_deepseek_dsa(hf_config)`` check as sglang's pool init).
        layout_hints = build_sglang_layout_hints(self.page_size)
        self.engine_group_infos: list[EngineGroupInfo] = (
            create_engine_group_infos_from_sglang(
                kv_cache_pools, self.page_size, sgl_config.hf_config
            )
        )

        # Upstream's REGISTER_KV_CACHE protocol takes flat positional args:
        # (instance_id, kv_cache, model_name, world_size, engine_type,
        # layout_hints, engine_group_infos).
        send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self.instance_id,
                _wrap_sglang_kv_caches(kv_cache_pools),
                self.model_name,
                self.tp_size,
                EngineType.SGLANG,
                layout_hints,
                self.engine_group_infos,
            ],
        ).result(timeout=self._mq_timeout)
        self._registered = True
        self._start_heartbeat()

    def _start_heartbeat(self) -> None:
        if self._heartbeat is not None:
            return
        self._heartbeat = HeartbeatThread(
            mq_client=self.mq_client,
            health_event=self._health_event,
            interval=self._heartbeat_interval,
            instance_id=self.instance_id,
        )
        self._heartbeat.start()

    @property
    def is_healthy(self) -> bool:
        return self._health_event.is_set()

    def chunk_size(self) -> int:
        return self._lmcache_chunk_size

    @torch.no_grad()
    def _broadcast_lookup_result(self, local_tokens: int) -> int:
        """Broadcast the lookup result from rank 0 to all TP ranks.

        Only rank 0 performs the daemon LOOKUP + WAIT_PREFETCH_STATUS
        flow (see :meth:`lookup_kv`). This method broadcasts the
        result so every rank sees the same matched-token count without
        independently querying the daemon.
        """
        if self.tp_size == 1:
            return local_tokens
        t = torch.tensor([local_tokens], dtype=torch.int32, device=self.device)
        dist.broadcast(t, src=0, group=self.tp_group)
        return int(t.item())

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        no_worker_id: bool = False,
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
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
        """Wait for the LOOKUP's prefetch to finish and return the matched bytes.

        Sends a single blocking WAIT_PREFETCH_STATUS request so the daemon
        blocks until the prefetch result is published (or its wait times out),
        instead of the client busy-polling QUERY_PREFETCH_STATUS. Upstream keys
        the prefetch job by request_id (a string); the result is the number of
        matched chunks once available.
        """
        # The daemon blocks up to ``self._mq_timeout`` for the result, so give
        # the response itself a little longer than that to cover the round trip.
        matched_chunks = send_lmcache_request(
            self.mq_client,
            RequestType.WAIT_PREFETCH_STATUS,
            [request_id, self._mq_timeout],
        ).result(timeout=self._mq_timeout + _WAIT_LOOKUP_RESPONSE_BUFFER_S)
        if matched_chunks is None:
            raise LMCacheTimeoutError(
                "Timed out waiting for LMCache prefetch to finish",
                session_id=request_id,
            )
        return matched_chunks * self._lmcache_chunk_size

    def _free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> None:
        """Release *this rank's* read locks in ``[start, end)``."""
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
                    no_worker_id=False,
                ),
                self.tp_size,
            ],
        )

    def lookup_kv(self, token_ids: list[int], request_id: str) -> int:
        """Phase 1 of the two-phase load — fires LOOKUP only.

        The daemon prefetches missing chunks L2 → L1 (DRAM), creates a
        session keyed by ``request_id`` with ``lookup_ipc_key`` set,
        and submits a prefetch task whose read locks stay held for the
        eventual ``retrieve_kv``. Does **not** copy KV to GPU.

        Idempotent across re-scheduling passes for the same
        ``request_id``: a prior pending LOOKUP for the same rid has
        its read locks released before the new LOOKUP fires, so locks
        don't accumulate.

        Returns the chunk-aligned matched-token count (0 if no
        chunk-aligned hit, including the ``aligned_end == 0`` short-
        prompt case).
        """
        if not self.is_healthy or not request_id:
            return 0

        # Release any stale locks from a prior LOOKUP for the same rid
        # (e.g., a rescheduling pass) before firing a new one.
        with self._pending_lookups_lock:
            stale = self._pending_lookups.pop(request_id, None)
        if stale is not None and stale.locks_held:
            self._free_lookup_locks(
                stale.token_ids, 0, stale.matched_token_num, request_id
            )

        aligned_end = (len(token_ids) // self._lmcache_chunk_size) * (
            self._lmcache_chunk_size
        )
        if aligned_end == 0:
            return 0  # too few tokens; no chunk-aligned range to LOOKUP

        # Only rank 0 talks to the daemon; the matched-token count is
        # broadcast to the followers.
        if self.worker_id == 0:
            lookup_key = self._create_key(
                token_ids,
                start=0,
                end=aligned_end,
                request_id=request_id,
                no_worker_id=True,
            )
            send_lmcache_request(
                self.mq_client,
                RequestType.LOOKUP,
                [lookup_key, self.tp_size],
            ).result(timeout=self._mq_timeout)
            matched = self._wait_for_lookup(request_id)
        else:
            matched = 0
        matched = self._broadcast_lookup_result(matched)

        # Daemon now holds read locks for the matched chunks. Record
        # state for the eventual retrieve_kv / release_pending /
        # end_session call. Locks are released by exactly one of those.
        with self._pending_lookups_lock:
            self._pending_lookups[request_id] = _PendingLookup(
                token_ids=list(token_ids),
                matched_token_num=matched,
                locks_held=matched > 0,
            )
        return matched

    def release_pending(self, request_id: str) -> None:
        """Free read locks acquired by ``lookup_kv`` when no ``retrieve_kv``
        will follow (LMCache's hit is covered by radix). The pending entry
        stays so ``end_session`` still sends END_SESSION.
        """
        with self._pending_lookups_lock:
            pending = self._pending_lookups.get(request_id)
            if pending is None or not pending.locks_held:
                return
            pending.locks_held = False
            token_ids = pending.token_ids
            matched = pending.matched_token_num
        if matched > 0:
            self._free_lookup_locks(token_ids, 0, matched, request_id)

    def end_session(self, request_id: str) -> None:
        """Tell the daemon we're done with this request_id.

        Single per-request cleanup hook — owned by the engine's
        request-finish path (e.g., :meth:`LMCRadixCache.cache_finished_req`),
        not bundled into ``store_kv``. Skipped (no wire send) for ids
        we never fired a LOOKUP for, so warmup and short-prompt
        requests don't trigger the daemon's "Session not found,
        skipping touch" warning. Frees any still-held read locks
        before sending END_SESSION (covers failure paths where
        retrieve_kv didn't consume the locks).
        """
        if not self.is_healthy:
            return
        with self._pending_lookups_lock:
            pending = self._pending_lookups.pop(request_id, None)
        if pending is None:
            return
        # Every rank frees *its own* still-held read locks. LOOKUP is
        # rank-0-only on the wire, but the daemon reserves an
        # independent lock per TP rank (worker_id=None expansion), so a
        # rank-0-only release would leak the other ranks' locks until
        # read-lock TTL expiry.
        if pending.locks_held and pending.matched_token_num > 0:
            self._free_lookup_locks(
                pending.token_ids, 0, pending.matched_token_num, request_id
            )
        # END_SESSION targets shared daemon-side session state, so only
        # rank 0 sends it.
        if self.worker_id == 0:
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
        event = torch_dev.Event(interprocess=True)
        event.record(torch_dev.current_stream())
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
                # RETRIEVE takes per-kernel-group block IDs (list[list[int]]).
                # sglang is non-hybrid, so we always feed ONE engine-side
                # block-id list; ``expand_engine_block_ids`` fans it out to
                # every kernel group according to ``engine_group_infos``:
                #   * MHA / MLA (empty engine_group_infos): returns [block_ids]
                #   * DSA       (two infos, both engine_group_id=0):
                #     returns [block_ids, block_ids] -- one copy per kernel
                #     group, both pointing at the same paged pages.
                expand_engine_block_ids(self.engine_group_infos, [block_ids]),
                event.ipc_handle(),
                skip_prefix_n_blocks,
            ],
        ).to_device_future(device=self.device)

    def retrieve_kv(self, load_metadata: LoadMetadata) -> int:
        """Phase 2 of the two-phase load — fires RETRIEVE only.

        Reuses the matched-token count cached by a prior ``lookup_kv``
        for the same ``request_id`` (no second LOOKUP wire call). The
        daemon's RETRIEVE handler copies L1 (DRAM) → GPU KV pool slots
        in a single ``multi_layer_block_kv_transfer`` launch and
        consumes the held read locks via ``finish_read_prefetched`` —
        we don't separately free them on the success path.

        Failure paths free the still-held trailing read locks
        explicitly to avoid leaking them in the daemon.

        Returns ``matched - offset`` (tokens covered by the chunks
        whose RETRIEVE was issued, equivalent to the legacy
        ``start_load_kv`` return). Caller subtracts ``prefix_pad`` to
        compute "newly added to radix".
        """
        if not self.is_healthy:
            return 0

        request_id = load_metadata.request_id
        with self._pending_lookups_lock:
            pending = self._pending_lookups.get(request_id)
        if pending is None or not pending.locks_held:
            raise RuntimeError(
                f"retrieve_kv called for {request_id} without a pending lookup_kv"
            )

        retrieve_token_num = pending.matched_token_num
        token_ids = pending.token_ids
        offset = load_metadata.offset

        # ``slot_mapping[offset : offset + prefix_pad)`` is sentinel ``-1`` —
        # those tokens already live in the engine's radix tree and must not
        # be overwritten. We still RETRIEVE the full chunk-aligned range
        # (LMCache stores at chunk granularity), but tell the daemon to skip
        # the leading ``prefix_pad // page_size`` blocks. Real block_ids are
        # computed only from the freshly-allocated slot range; the skipped
        # blocks get harmless placeholder ids the kernel never dereferences.
        prefix_pad = load_metadata.prefix_pad
        fresh_start = offset + prefix_pad
        prefix_pad_pages = prefix_pad // self.page_size

        # Free this rank's prefix read locks in ``[0, offset)``; the
        # daemon's RETRIEVE handler only consumes the trailing suffix
        # in ``[offset, retrieve_token_num)``.
        self._free_lookup_locks(token_ids, 0, offset, request_id)
        fresh_block_ids = self._slot_mapping_to_block_ids(
            load_metadata.slot_mapping[fresh_start:retrieve_token_num]
        )
        block_ids = [0] * prefix_pad_pages + fresh_block_ids

        # Successful RETRIEVE releases the trailing read locks via
        # ``finish_read_prefetched`` inside the daemon. The trailing
        # ``_free_lookup_locks`` is the failure path's cleanup — calling
        # it after a successful RETRIEVE would double-release and trigger
        # "finish read on non-read-locked key".
        retrieve_succeeded = False
        try:
            future = self._submit_retrieve(
                request_id=request_id,
                token_ids=token_ids,
                offset=offset,
                matched_end=retrieve_token_num,
                block_ids=block_ids,
                skip_prefix_n_blocks=prefix_pad_pages,
            )
            if not future.result(timeout=self._mq_timeout):
                raise RuntimeError(
                    f"LMCache MP retrieve failed for request_id={request_id}"
                )
            retrieve_succeeded = True
        finally:
            if not retrieve_succeeded:
                # Free this rank's suffix read locks that the daemon
                # would have consumed on success. Per-rank because
                # RETRIEVEs are independent across ranks.
                self._free_lookup_locks(
                    token_ids, offset, retrieve_token_num, request_id
                )
            with self._pending_lookups_lock:
                if request_id in self._pending_lookups:
                    self._pending_lookups[request_id].locks_held = False
        return retrieve_token_num - offset

    def store_kv_async(self, store_metadata: StoreMetadata) -> MessagingFuture[bool]:
        """Submit a STORE and return its completion future without waiting.

        Fires the STORE request for the chunk-aligned prefix of
        ``store_metadata`` and returns immediately with a future the
        caller can poll (``query`` / ``wait``) or block on (``result``)
        at a later, deferred checkpoint. The future resolves to a
        ``bool`` success flag once the daemon finishes copying the KV
        slots GPU → warehouse.

        The KV slots referenced by ``store_metadata`` must remain pinned
        (not evicted or reused) until the returned future reports done;
        the caller owns that lifetime. Paths that perform no wire send
        return an already-completed future so callers never special-case
        ``None``: an unhealthy connector resolves to ``False`` (nothing
        was stored), and no chunk-aligned range resolves to ``True`` (a
        no-op success).

        END_SESSION is owned by ``LMCRadixCache.cache_finished_req``
        (see :meth:`end_session`); it is not fired here.

        Args:
            store_metadata: tokens, request id, and KV slot indices for
                the finished request.

        Returns:
            A future resolving to ``True`` when the store completes
            successfully (or there was nothing to store), or ``False``
            on daemon-side failure or an unhealthy connector.
        """
        if not self.is_healthy:
            return _completed_future(False)

        aligned_end = (len(store_metadata.token_ids) // self._lmcache_chunk_size) * (
            self._lmcache_chunk_size
        )
        if aligned_end == 0:
            return _completed_future(True)

        request_id = store_metadata.request_id
        block_ids = self._slot_mapping_to_block_ids(
            store_metadata.kv_indices[:aligned_end]
        )
        event = torch_dev.Event(interprocess=True)
        event.record(torch_dev.current_stream())
        future = send_lmcache_request(
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
                # STORE takes per-kernel-group block IDs (list[list[int]]).
                # Same story as RETRIEVE: sglang is non-hybrid, so we
                # send ONE engine-side block-id list and let
                # ``expand_engine_block_ids`` broadcast it to each
                # kernel group (see ``_submit_retrieve``).
                expand_engine_block_ids(self.engine_group_infos, [block_ids]),
                event.ipc_handle(),
            ],
        ).to_cuda_future(device=self.device)
        # Keep the exporting CUDA event alive until the caller releases the
        # future. Since we return without blocking, the local ``event`` would
        # otherwise be garbage-collected immediately, destroying the underlying
        # CUDA event before the daemon imports its IPC handle and waits on it.
        # (Dynamic keepalive attribute; the future type doesn't declare it.)
        future._export_event = event  # type: ignore[attr-defined]
        return future

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        if not self.is_healthy:
            return

        aligned_end = (len(store_metadata.token_ids) // self._lmcache_chunk_size) * (
            self._lmcache_chunk_size
        )
        if aligned_end == 0:
            return

        request_id = store_metadata.request_id
        block_ids = self._slot_mapping_to_block_ids(
            store_metadata.kv_indices[:aligned_end]
        )
        event = torch_dev.Event(interprocess=True)
        event.record(torch_dev.current_stream())
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
                    # STORE takes per-kernel-group block IDs (list[list[int]]).
                    # Same story as RETRIEVE: sglang is non-hybrid, so we
                    # send ONE engine-side block-id list and let
                    # ``expand_engine_block_ids`` broadcast it to each
                    # kernel group (see ``_submit_retrieve``).
                    expand_engine_block_ids(self.engine_group_infos, [block_ids]),
                    event.ipc_handle(),
                ],
            )
            .to_device_future(device=self.device)
            .result(timeout=self._mq_timeout)
        )
        # END_SESSION is owned by ``LMCRadixCache.cache_finished_req`` so
        # it fires once per request, even when STORE early-returns or no
        # STORE was needed. See ``LMCacheMPConnector.end_session``.
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
