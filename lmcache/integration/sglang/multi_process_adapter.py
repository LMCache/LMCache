# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import os
import threading

# Third Party
import torch
import torch.distributed as dist
import zmq

# First Party
from lmcache import torch_dev
from lmcache.integration.sglang.sglang_adapter import (
    LoadMetadata,
    StoreMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MQ_TIMEOUT,
    HeartbeatThread,
    get_lmcache_chunk_size,
)
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform import get_device_spec
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.kv_wrap import wrap_one_kv_cache

if TYPE_CHECKING:
    # Third Party
    from sglang.srt.configs.model_config import ModelConfig

logger = init_logger(__name__)

# Extra seconds the WAIT_PREFETCH_STATUS response is allowed beyond the daemon's
# own blocking-wait budget, to cover the request/response round trip.
_WAIT_LOOKUP_RESPONSE_BUFFER_S = 5.0


def _validate_sglang_kv_pools(
    k_pool: list[torch.Tensor],
    v_pool: list[torch.Tensor],
) -> torch.device:
    """Validate SGLang's split MHA pools and return their shared device.

    Args:
        k_pool: Per-layer key-cache tensors.
        v_pool: Per-layer value-cache tensors.

    Returns:
        The device shared by every key and value tensor.

    Raises:
        ValueError: If either pool is empty, layer counts differ, or tensors
            span multiple devices.
    """
    if not k_pool or not v_pool:
        raise ValueError("SGLang MP registration requires non-empty K and V pools")
    if len(k_pool) != len(v_pool):
        raise ValueError("SGLang MP registration requires matching K and V layers")
    tensors = [*k_pool, *v_pool]
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("SGLang MP K and V pools must use one device")
    return device


def _wrap_sglang_kv_caches(
    k_pool: list[torch.Tensor],
    v_pool: list[torch.Tensor],
) -> KVCache:
    """Flatten SGLang's depth-2 ``[K_layers, V_layers]`` KV layout into a
    single flat ``KVCache`` so it fits upstream's wire
    ``KVCache`` payload type. The daemon's
    :func:`normalize_kv_and_discover_format` recognizes this shape from
    ``EngineType.SGLANG`` plus ``tokens_per_block`` and ``kv_list_layout``
    ``LayoutHints`` fields, then splits it back at its midpoint before format
    detection.

    Raises:
        ValueError: If the pools are empty, use different devices, or the
            selected platform cannot provide the complete handle-transfer
            path.
    """
    device = _validate_sglang_kv_pools(k_pool, v_pool)
    tensors = [*k_pool, *v_pool]
    device_spec = get_device_spec(device.type)
    if device_spec is None or not device_spec.is_handle_transfer_available():
        raise ValueError(
            "SGLang MP handle transfer is unavailable for device type "
            f"{device.type!r}: required memory IPC, event IPC, cache context, "
            "or block-transfer capabilities are missing"
        )
    return [wrap_one_kv_cache(tensor) for tensor in tensors]


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


class _SparseLeaseFuture(MessagingFuture):
    """Forward a sparse RPC result and clean up only after confirmation."""

    def __init__(self, future: MessagingFuture, cleanup, should_cleanup) -> None:
        super().__init__()
        self._future = future
        self._cleanup = cleanup
        self._should_cleanup = should_cleanup
        self._cleanup_lock = threading.Lock()
        self._cleaned = False

    def _observe_result(self, result) -> None:
        if not self._should_cleanup(result):
            return
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleanup()
            self._cleaned = True

    def result(self, timeout=None):
        result = self._future.result(timeout)
        self._observe_result(result)
        return result

    def wait(self, timeout=None) -> bool:
        if not self._future.wait(timeout):
            return False
        try:
            self._observe_result(self._future.result(timeout=0))
        except BaseException:
            # Preserve the underlying exception for the explicit result()
            # call and keep the cleanup record until it succeeds.
            pass
        return True

    def query(self) -> bool:
        if not self._future.query():
            return False
        try:
            self._observe_result(self._future.result(timeout=0))
        except BaseException:
            pass
        return True

    def retain_reference(self, value: object) -> None:
        self._future.retain_reference(value)


class LMCacheMPConnector:
    """SGLang LMCache multi-process connector.

    Talks to a standalone LMCache daemon through the configured transport.

    - ``lookup_kv``: fires LOOKUP. Daemon prefetches missing
      chunks L2→L1 (DRAM), keeps the read locks held, returns the
      matched-token count.
    - ``retrieve_kv``: fires RETRIEVE using the cached LOOKUP result.
      Daemon copies L1→GPU through the registered layer's single-layer
      transfer path and releases the read locks via the sparse completion
      callback.
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
        k_pool: list[torch.Tensor],
        v_pool: list[torch.Tensor],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        device = _validate_sglang_kv_pools(k_pool, v_pool)
        self.tp_size = tp_size
        self.worker_id = rank
        self.page_size = page_size
        self.device = device
        self._event_backend: EventIPCBackend = get_event_ipc_backend(device)
        self._event_backend.check_event_support(device)
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
        self._pending_lookups: dict[str, _PendingLookup] = {}
        self._pending_lookups_lock = threading.Lock()
        self._sparse_handles: set[tuple[str, int, int]] = set()
        self._sparse_handles_lock = threading.Lock()
        self._sparse_key_cache: dict[tuple, tuple[ObjectKey, ...]] = {}
        self._sparse_hash_cache: dict[tuple, tuple[bytes, ...]] = {}
        self._sparse_key_cache_lock = threading.Lock()

        self.context = zmq.Context.instance()
        server_url = f"{host}:{port}"
        self.req_client: RequestClient = RequestClientFactory.create(
            server_url,
            context=self.context,
        )

        self._lmcache_chunk_size = get_lmcache_chunk_size(self.req_client)
        if self._lmcache_chunk_size % self.page_size != 0:
            raise ValueError(
                "LMCache chunk size must be a multiple of SGLang page size, got "
                f"{self._lmcache_chunk_size} and {self.page_size}"
            )
        self._sparse_token_hasher = TokenHasher(
            chunk_size=self._lmcache_chunk_size,
            hash_algorithm="blake3",
        )

        # Upstream's REGISTER_KV_CACHE protocol takes flat positional args:
        # (instance_id, kv_cache, model_name, world_size, engine_type,
        # layout_hints, engine_group_infos). SGLang's natural KV layout is depth-2
        # ([K_layers, V_layers]); we flatten it on the wire to fit
        # ``KVCache = list[DeviceIPCWrapper]``. The daemon recognizes the
        # SGLang-MHA flat-of-2NL pattern from ``EngineType.SGLANG`` plus the
        # ``tokens_per_block`` and ``kv_list_layout`` hints, then un-flattens
        # and reshapes per layer.
        # SGLang is non-hybrid (a single KV cache group), so engine_group_infos is the
        # empty list -- which the server treats as one group spanning all layers
        # (matching the vLLM non-hybrid and TensorRT-LLM register paths).
        self.req_client.register_kv_cache(
            self.instance_id,
            _wrap_sglang_kv_caches(k_pool, v_pool),
            self.model_name,
            self.tp_size,
            EngineType.SGLANG,
            {"tokens_per_block": self.page_size, "kv_list_layout": "k_v"},
            [],
        ).result(timeout=self._mq_timeout)
        self._registered = True
        self._start_heartbeat()

    def _start_heartbeat(self) -> None:
        if self._heartbeat is not None:
            return
        self._heartbeat = HeartbeatThread(
            req_client=self.req_client,
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

    def _completed_sparse_future(self, result) -> MessagingFuture:
        future: MessagingFuture = MessagingFuture()
        future.set_result(result)
        return future

    def _sparse_key(self, request_id: str, generation: int, layer_id: int):
        return request_id, generation, layer_id

    def _forget_sparse_handle(
        self, request_id: str, generation: int, layer_id: int
    ) -> None:
        with self._sparse_handles_lock:
            self._sparse_handles.discard(
                self._sparse_key(request_id, generation, layer_id)
            )

    def _clear_sparse_key_cache(self, request_id: str | None = None) -> None:
        with self._sparse_key_cache_lock:
            if request_id is None:
                self._sparse_key_cache.clear()
                self._sparse_hash_cache.clear()
                return
            self._sparse_key_cache = {
                key: value
                for key, value in self._sparse_key_cache.items()
                if key[0] != request_id
            }
            self._sparse_hash_cache = {
                key: value
                for key, value in self._sparse_hash_cache.items()
                if key[0] != request_id
            }

    def create_sparse_object_keys(
        self,
        token_ids: list[int],
        chunk_indices: list[int],
        cache_salt: str | None = None,
        *,
        request_id: str | None = None,
        generation: int | None = None,
        layer_id: int | None = None,
    ) -> list[ObjectKey]:
        """Map complete token chunks to LMCache logical object keys.

        The server and this adapter must use the same rolling hash.  The
        multiprocess SGLang path therefore requires the default ``blake3``
        token hash; physical SGLang page IDs are deliberately not part of the
        returned keys.
        """
        indices = sorted({int(index) for index in chunk_indices})
        if not indices:
            return []
        if indices[0] < 0:
            raise ValueError("sparse chunk indices must be non-negative")
        end = (indices[-1] + 1) * self._lmcache_chunk_size
        if end > len(token_ids):
            return []

        normalized_salt = cache_salt or ""
        cache_key = None
        hash_cache_key = None
        if request_id is not None and generation is not None and layer_id is not None:
            cache_key = (
                request_id,
                int(generation),
                int(layer_id),
                len(token_ids),
                normalized_salt,
                tuple(indices),
            )
            hash_cache_key = (
                request_id,
                int(generation),
                len(token_ids),
                normalized_salt,
            )

        cache_lock = getattr(self, "_sparse_key_cache_lock", None)
        if cache_lock is None:
            cache_lock = threading.Lock()
            self._sparse_key_cache_lock = cache_lock
        with cache_lock:
            if cache_key is not None:
                cached_keys = getattr(self, "_sparse_key_cache", {}).get(cache_key)
                if cached_keys is not None:
                    return list(cached_keys)

            if hash_cache_key is not None:
                cached_hashes = getattr(self, "_sparse_hash_cache", {}).get(
                    hash_cache_key
                )
            else:
                cached_hashes = None

            required_hashes = end // self._lmcache_chunk_size
            if cached_hashes is not None and len(cached_hashes) >= required_hashes:
                hashes = list(cached_hashes[:required_hashes])
            else:
                hasher = getattr(self, "_sparse_token_hasher", None)
                if hasher is None:
                    hasher = TokenHasher(
                        chunk_size=self._lmcache_chunk_size,
                        hash_algorithm="blake3",
                    )
                hashes = hasher.compute_chunk_hashes(list(token_ids), end=end)
                if hash_cache_key is not None:
                    self._sparse_hash_cache[hash_cache_key] = tuple(hashes)

        kv_rank = ObjectKey.ComputeKVRank(
            world_size=self.tp_size,
            global_rank=self.worker_id,
            local_world_size=self.tp_size,
            local_rank=self.worker_id,
        )
        result = [
            ObjectKey(
                chunk_hash=hashes[index],
                model_name=self.model_name,
                kv_rank=kv_rank,
                object_group_id=0,
                cache_salt=normalized_salt,
            )
            for index in indices
        ]
        if cache_key is not None:
            with self._sparse_key_cache_lock:
                self._sparse_key_cache[cache_key] = tuple(result)
        return result

    def sparse_prefetch(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        keys: list[ObjectKey],
    ) -> MessagingFuture[bool]:
        """Submit a logical sparse prefetch and retain its server lease."""
        if (
            not self.is_healthy
            or not request_id
            or generation < 0
            or layer_id < 0
            or not keys
        ):
            return self._completed_sparse_future(False)
        with self._sparse_handles_lock:
            self._sparse_handles.add(self._sparse_key(request_id, generation, layer_id))
        future = self.req_client.sparse_prefetch(
            self.instance_id,
            request_id,
            generation,
            layer_id,
            keys,
        )
        return _SparseLeaseFuture(
            future,
            lambda: self._forget_sparse_handle(request_id, generation, layer_id),
            lambda result: result is False,
        )

    def sparse_query_prefetch(
        self, request_id: str, generation: int, layer_id: int
    ) -> MessagingFuture:
        """Query retained logical keys without releasing their lease."""
        if not self.is_healthy:
            return self._completed_sparse_future(None)
        return self.req_client.sparse_query_prefetch(
            self.instance_id, request_id, generation, layer_id
        )

    def sparse_wait_prefetch(
        self, request_id: str, generation: int, layer_id: int, timeout: float
    ) -> MessagingFuture:
        """Wait for retained logical keys without consuming their lease."""
        if not self.is_healthy or timeout < 0:
            return self._completed_sparse_future(None)
        return self.req_client.sparse_wait_prefetch(
            self.instance_id, request_id, generation, layer_id, timeout
        )

    def sparse_retrieve(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        keys: list[ObjectKey],
        block_ids: list[list[int]],
    ) -> MessagingFuture:
        """Load retained logical objects into the supplied GPU pages."""
        if not self.is_healthy:
            return self._completed_sparse_future((False, []))
        event = torch_dev.Event(interprocess=True)
        event.record(torch_dev.current_stream())
        raw_future = self.req_client.sparse_retrieve(
            self.instance_id,
            request_id,
            generation,
            layer_id,
            keys,
            block_ids,
            event.ipc_handle(),
        )
        future = raw_future.to_device_future(device=self.device)
        future.retain_reference(event)
        return future

    def sparse_cancel_prefetch(
        self, request_id: str, generation: int, layer_id: int
    ) -> MessagingFuture:
        """Cancel a logical sparse prefetch and release its lease."""
        if not self.is_healthy:
            return self._completed_sparse_future(False)
        future = self.req_client.sparse_cancel_prefetch(
            self.instance_id, request_id, generation, layer_id
        )
        return _SparseLeaseFuture(
            future,
            lambda: self._forget_sparse_handle(request_id, generation, layer_id),
            lambda result: result is True,
        )

    def sparse_release_prefetch(
        self, request_id: str, generation: int, layer_id: int
    ) -> MessagingFuture:
        """Release a logical sparse lease after the GPU consumer is done."""
        if not self.is_healthy:
            return self._completed_sparse_future(False)
        future = self.req_client.sparse_release_prefetch(
            self.instance_id, request_id, generation, layer_id
        )
        return _SparseLeaseFuture(
            future,
            lambda: self._forget_sparse_handle(request_id, generation, layer_id),
            lambda result: result is True,
        )

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
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.tp_size,
            # Each worker stores and reads only its own object (no MLA-style
            # sharing in this adapter yet).
            num_kv_readers=1,
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
        matched_chunks = self.req_client.wait_prefetch_status(
            request_id, self._mq_timeout
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
        if start >= end or not self.is_healthy:
            return
        self.req_client.free_lookup_locks(
            self._create_key(
                token_ids,
                start=start,
                end=end,
                request_id=request_id,
                no_worker_id=True,
            ),
            self.tp_size,
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

        # If a previous LOOKUP for this rid is still pending (e.g., a
        # rescheduling pass or a prior partial flow), release its locks
        # first so we don't accumulate read-lock reservations.
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

        lookup_key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            no_worker_id=True,
        )
        self.req_client.lookup(lookup_key, self.tp_size).result(
            timeout=self._mq_timeout
        )
        matched = self._wait_for_lookup(request_id)
        matched = self._global_min_tokens(matched)

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
        self._clear_sparse_key_cache(request_id)
        if not self.is_healthy:
            return
        with self._pending_lookups_lock:
            pending = self._pending_lookups.pop(request_id, None)
        if pending is None:
            return
        if pending.locks_held and pending.matched_token_num > 0:
            self._free_lookup_locks(
                pending.token_ids, 0, pending.matched_token_num, request_id
            )
        self.req_client.end_session(request_id)

    def _submit_retrieve(
        self,
        request_id: str,
        token_ids: list[int],
        offset: int,
        matched_end: int,
        block_ids: list[int],
        skip_prefix_n_blocks: int = 0,
    ) -> tuple[
        MessagingFuture[tuple[bytes, bool]],
        MessagingFuture[bool],
    ]:
        event = self._event_backend.create_event(self.device)
        self._event_backend.record_event(event, torch_dev.current_stream())
        raw_future: MessagingFuture[tuple[bytes, bool]] = self.req_client.retrieve(
            self._create_key(
                token_ids,
                start=offset,
                end=matched_end,
                request_id=request_id,
            ),
            self.instance_id,
            # RETRIEVE takes per-group block IDs (list[list[int]]); SGLang is
            # non-hybrid, so wrap the flat list as a single group.
            [block_ids],
            self._event_backend.export_event(event, self.device),
            skip_prefix_n_blocks,
        )
        raw_future.retain_reference(event)
        future = raw_future.to_device_future(
            device=self.device,
            event_backend=self._event_backend,
        )
        # The daemon imports this IPC event after the request crosses the wire.
        # Keep ownership on the raw future so timeout callers cannot drop the
        # exported event before the transport finishes the request.
        return raw_future, future

    def retrieve_kv(self, load_metadata: LoadMetadata) -> int:
        """Phase 2 of the two-phase load — fires RETRIEVE only.

        Reuses the matched-token count cached by a prior ``lookup_kv``
        for the same ``request_id`` (no second LOOKUP wire call). The
        daemon's RETRIEVE handler copies L1 (DRAM) → GPU KV pool slots
        in a single ``multi_layer_block_kv_transfer`` launch and
        consumes the held read locks via ``finish_read_prefetched`` —
        we don't separately free them on the success path.

        Failure paths free the still-held trailing read locks explicitly,
        except when an event-free terminal response confirms that the daemon
        already released this worker's share.

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

        self._free_lookup_locks(token_ids, 0, offset, request_id)
        fresh_block_ids = self._slot_mapping_to_block_ids(
            load_metadata.slot_mapping[fresh_start:retrieve_token_num]
        )
        block_ids = [0] * prefix_pad_pages + fresh_block_ids

        # Successful RETRIEVE releases the trailing read locks via
        # ``finish_read_prefetched`` inside the daemon. The trailing
        # ``_free_lookup_locks`` is the fallback failure cleanup — calling it
        # after a successful RETRIEVE or an event-free terminal failure would
        # double-release locks already consumed by the daemon.
        retrieve_succeeded = False
        server_released_locks = False
        try:
            raw_future, future = self._submit_retrieve(
                request_id=request_id,
                token_ids=token_ids,
                offset=offset,
                matched_end=retrieve_token_num,
                block_ids=block_ids,
                skip_prefix_n_blocks=prefix_pad_pages,
            )
            if not future.result(timeout=self._mq_timeout):
                event_handle, _ = raw_future.result(timeout=0)
                # An event-free False is the missing-registration response.
                # Its server-side path already released this worker's share;
                # sending FREE_LOOKUP_LOCKS here would release every rank again.
                server_released_locks = not event_handle
                raise RuntimeError(
                    f"LMCache MP retrieve failed for request_id={request_id}"
                )
            retrieve_succeeded = True
        finally:
            if not retrieve_succeeded and not server_released_locks:
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
        event = self._event_backend.create_event(self.device)
        self._event_backend.record_event(event, torch_dev.current_stream())
        raw_future = self.req_client.store(
            self._create_key(
                store_metadata.token_ids,
                start=0,
                end=aligned_end,
                request_id=request_id,
            ),
            self.instance_id,
            # STORE takes per-group block IDs (list[list[int]]); SGLang is
            # non-hybrid, so wrap the flat list as a single group.
            [block_ids],
            self._event_backend.export_event(event, self.device),
        )
        raw_future.retain_reference(event)
        future = raw_future.to_device_future(
            device=self.device,
            event_backend=self._event_backend,
        )
        # Keep ownership on the raw future so timeout callers cannot drop the
        # exported event before the transport finishes the request.
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
        event = self._event_backend.create_event(self.device)
        self._event_backend.record_event(event, torch_dev.current_stream())
        raw_future = self.req_client.store(
            self._create_key(
                store_metadata.token_ids,
                start=0,
                end=aligned_end,
                request_id=request_id,
            ),
            self.instance_id,
            # STORE takes per-group block IDs (list[list[int]]); SGLang is
            # non-hybrid, so wrap the flat list as a single group.
            [block_ids],
            self._event_backend.export_event(event, self.device),
        )
        # The daemon imports this IPC event after the request crosses the wire.
        # Keep ownership on the raw future so timeout callers cannot drop the
        # exported event before the transport finishes the request.
        raw_future.retain_reference(event)
        future = raw_future.to_device_future(
            device=self.device,
            event_backend=self._event_backend,
        )
        success = future.result(timeout=self._mq_timeout)
        # END_SESSION is owned by ``LMCRadixCache.cache_finished_req`` so
        # it fires once per request, even when STORE early-returns or no
        # STORE was needed. See ``LMCacheMPConnector.end_session``.
        if not success:
            raise RuntimeError("LMCache MP store failed")

    def reset(self) -> None:
        self._clear_sparse_key_cache()

    def close(self) -> None:
        self.reset()
        with self._sparse_handles_lock:
            sparse_handles = list(self._sparse_handles)
        cleanup_confirmed = True
        for request_id, generation, layer_id in sparse_handles:
            try:
                released = self.req_client.sparse_cancel_prefetch(
                    self.instance_id,
                    request_id,
                    generation,
                    layer_id,
                ).result(timeout=self._mq_timeout)
                if released:
                    self._forget_sparse_handle(request_id, generation, layer_id)
                else:
                    cleanup_confirmed = False
                    logger.warning(
                        "LMCache sparse cleanup was not confirmed during close: "
                        "request_id=%s generation=%d layer=%d",
                        request_id,
                        generation,
                        layer_id,
                    )
            except Exception:
                cleanup_confirmed = False
                logger.warning(
                    "Failed to cancel an SGLang sparse prefetch during close",
                    exc_info=True,
                )
        if not cleanup_confirmed:
            logger.error(
                "Keeping LMCache connector open because sparse cleanup was not "
                "confirmed; retry close after the connection recovers"
            )
            return
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        if self._registered:
            try:
                self.req_client.unregister_kv_cache(self.instance_id).result(
                    timeout=self._mq_timeout
                )
            except Exception:
                logger.warning("Failed to unregister SGLang MP KV cache", exc_info=True)
            self._registered = False
        self.req_client.close()
