# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from dataclasses import dataclass
from typing import Any, Callable
import os
import threading

# Third Party
import torch
import zmq

# First Party
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.utils import EngineType, _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary

logger = init_logger(__name__)

# Timeout (seconds) for blocking MQ requests: initial chunk-size query,
# KV cache registration/unregistration, and other synchronous operations.
DEFAULT_MQ_TIMEOUT: float = 300.0
# Interval (seconds) between periodic heartbeat pings to the server.
DEFAULT_HEARTBEAT_INTERVAL: float = 10.0


def wrap_kv_caches(kv_caches: dict[str, torch.Tensor]) -> KVCache:
    # Emit a per-layer (name, shape, dtype) summary so the operator can
    # verify the exact layer set & tensor geometry being shipped to the
    # LMCache server, then the low-noise count of handles being wrapped.
    kept_summary = [
        (name, tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in kv_caches.items()
    ]
    logger.debug(
        "KV cache transfer keeping %d layer(s) (name, shape, dtype):\n%s",
        len(kept_summary),
        "\n".join(
            f"  [{i}] {name}  shape={shape}  dtype={dtype}"
            for i, (name, shape, dtype) in enumerate(kept_summary)
        ),
    )
    logger.info("Wrapping %d KV cache tensors for IPC", len(kv_caches))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result(timeout=DEFAULT_MQ_TIMEOUT)
    return chunk_size


def send_ping(
    mq_client: MessageQueueClient,
    timeout: float,
) -> bool:
    """Send a PING request and return the result.

    Returns:
        True if server is healthy, False on timeout or error.
    """
    try:
        future = send_lmcache_request(mq_client, RequestType.PING, [])
        return future.result(timeout=timeout)
    except TimeoutError:
        return False
    except Exception:
        logger.debug("Ping failed with exception", exc_info=True)
        return False


@dataclass
class ParallelStrategy:
    use_mla: bool
    """Whether to use the MLA."""

    kv_world_size: int
    """
    The kv world size, kv_world_size may not be equal to the actual_world_size,
    in the case of mla, it will 'exclude' the effect of TP, the value is
    calculated by `extract_world_size_and_kv_rank` in `lmcache_mp_connector.py`.
    """

    kv_worker_id: int
    """
    The kv worker id of the sub-process, kv_worker_id may not be equal to the
    actual_worker_id, in the case of mla, it will 'exclude' the effect of TP,
    the value is calculated by `extract_world_size_and_kv_rank` in
    `lmcache_mp_connector.py`.
    """

    actual_world_size: int
    """The actual world size."""

    actual_worker_id: int
    """The actual worker id of the sub-process."""

    tp_size: int
    """The tensor parallel size."""

    pp_size: int
    """The pipeline parallel size."""

    kv_world_size_per_node: int
    """The kv world size per node."""

    tp_size_per_node: int
    """The tensor parallel size per node."""

    n_servers: int
    """The number of lmcache servers."""


def _normalize_adapter_init_args(
    vllm_block_size: int,
    parallel_strategy: ParallelStrategy | int,
    legacy_block_size: int | None,
    mq_timeout: float,
) -> tuple[int, ParallelStrategy, float]:
    """Normalize adapter constructor args from old and new vLLM connectors.

    Args:
        vllm_block_size: The vLLM block size for the current connector API, or
            the legacy KV world size when ``parallel_strategy`` is an int.
        parallel_strategy: The current ``ParallelStrategy`` object, or the
            legacy KV worker id from older vLLM MP connectors.
        legacy_block_size: The legacy vLLM block size passed positionally by
            older vLLM MP connectors.
        mq_timeout: Timeout in seconds for synchronous message queue requests.

    Returns:
        A tuple of normalized ``(vllm_block_size, parallel_strategy,
        mq_timeout)``.

    Raises:
        TypeError: If the connector argument shape is not supported.
    """
    if isinstance(parallel_strategy, ParallelStrategy):
        return vllm_block_size, parallel_strategy, mq_timeout
    if not isinstance(parallel_strategy, int) or legacy_block_size is None:
        raise TypeError(
            "parallel_strategy must be ParallelStrategy, or legacy "
            "(kv_world_size, kv_worker_id, block_size) arguments"
        )

    kv_world_size = int(vllm_block_size)
    kv_worker_id = int(parallel_strategy)
    strategy = ParallelStrategy(
        use_mla=False,
        kv_world_size=kv_world_size,
        kv_worker_id=kv_worker_id,
        actual_world_size=kv_world_size,
        actual_worker_id=kv_worker_id,
        tp_size=kv_world_size,
        pp_size=1,
        kv_world_size_per_node=kv_world_size,
        tp_size_per_node=kv_world_size,
        n_servers=1,
    )
    return int(legacy_block_size), strategy, mq_timeout


class HeartbeatThread(PeriodicThread):
    """Periodically checks server health via PING.

    Manages a threading.Event that adapters use to gate operations.
    When unhealthy, the adapter enters degraded mode; if the server
    recovers, the adapter automatically resumes normal operation.
    """

    def __init__(
        self,
        mq_client: MessageQueueClient,
        health_event: threading.Event,
        interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        """
        Args:
            mq_client: The message queue client used to send PING requests.
            health_event: A threading.Event shared with the adapter.
                Set when the server is healthy, cleared when unhealthy.
                Adapters check this event to decide whether to proceed
                with operations or enter degraded mode.
            interval: Seconds between heartbeat pings and ping timeout.
        """
        super().__init__(
            name="lmcache-heartbeat",
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._mq_client = mq_client
        self._health_event = health_event
        self._interval = interval

        # Optional callback invoked on the unhealthy->healthy edge,
        # before the health event is set. See register_recover_callback.
        def noop() -> bool:
            return True

        self._recover_callback: Callable[[], bool] = noop

    def register_recover_callback(self, callback: Callable[[], bool]) -> None:
        """Register a callback fired on the unhealthy->healthy transition.

        The callback runs **before** the health event is set. It must
        return ``True`` on success (event will be set) or ``False`` on
        failure (event will stay cleared, and the next heartbeat will
        invoke the callback again on the next successful PING).

        The callback function should NEVER raise exceptions.

        Intended for setup work that must complete before downstream
        callers observe the recovery — for example, re-registering KV
        caches with a server that just restarted.

        Should be called before :meth:`start`. Only one callback is
        supported; a second call replaces the first.

        Args:
            callback: Zero-arg callable returning a success bool.
        """
        self._recover_callback = callback

    def _execute(self) -> ThreadRunSummary:
        was_healthy = self._health_event.is_set()
        healthy = send_ping(self._mq_client, timeout=self._interval)
        need_trigger_recover = (
            healthy and not was_healthy and self._recover_callback is not None
        )

        # Try to call recover callback
        if need_trigger_recover:
            logger.warning(
                "LMCache server is healthy again, triggering recovery callback"
            )
            # If the callback fails, it should not become healthy
            healthy = self._recover_callback()

        if healthy:
            self._health_event.set()
            if not was_healthy:
                logger.warning(
                    "LMCache server is healthy again — resuming normal operation"
                )
        else:
            self._health_event.clear()
            if was_healthy:
                logger.warning("LMCache server is unhealthy — entering degraded mode")

        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )


@dataclass
class LoadStoreOp:
    token_ids: list[int]
    """Token IDs for the load/store operation"""

    block_ids: list[int]
    """Block ids for the load/store operation"""

    start: int = 0
    """Start token index"""

    end: int = 0
    """End token index"""

    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""

    def __len__(self) -> int:
        return len(self.block_ids)


StoreResult = bool
RetrieveResult = bool
LookupResult = int


class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_urls: list[str],
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy | int,
        legacy_block_size: int | None = None,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        """
        Args:
            server_urls: The servers URL for the LMCache message queue
            context: The ZMQ context
            model_name: The model name used for LMCache keys
            vllm_block_size: The block size used in vLLM
            parallel_strategy:
                The parallel strategy, which includes `use_mla`,
                `kv_world_size`, `kv_worker_id` and so on. Older vLLM
                connectors pass the KV worker id here.
            legacy_block_size: The vLLM block size passed positionally by
                older vLLM connectors.
            mq_timeout: Timeout in seconds for message queue requests.
            heartbeat_interval: Interval in seconds between heartbeat pings.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        assert len(server_urls) >= 1, "At least one server url required"
        self._server_urls: list[str] = list(server_urls)
        self.mq_clients: dict[str, MessageQueueClient] = {
            url: MessageQueueClient(url, context) for url in self._server_urls
        }
        self._mq_timeout = mq_timeout

        # Lookup state tracking (multi-server aware, N>=1):
        # - _pending_lookups: request_ids submitted but not yet resolved
        #   (resolved == results from ALL servers have been merged).
        # - _finished_lookup_results: aggregated chunk count keyed by
        #   request_id, computed as the min hit across servers so a chunk
        #   only counts as "hit" when every server has it. Cached so that
        #   repeated calls to check_lookup_result return the same value
        #   even after the servers have popped the job (exactly-once).
        # - _per_server_hits: per-server raw hit counts keyed by request_id,
        #   kept for debugging and consistency reporting.
        #
        # Single-server compatibility: when len(server_urls) == 1, min() over
        # a single value is a no-op, so the aggregated result is identical to
        # the legacy single-server behavior -- this class is a drop-in
        # replacement for the original single-server lookup.

        # One worker thread per server so all lookups can be fired off at once.
        # Standard
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=len(self._server_urls),
            thread_name_prefix="lmcache-mp-lookup",
        )
        self._pending_lookups: set[str] = set()
        self._finished_lookup_results: dict[str, int] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self._per_server_hits: dict[str, dict[str, int]] = {}

        # Fetch chunk_size from every server and verify they all agree.
        chunk_sizes: dict[str, int] = {}
        for url, client in self.mq_clients.items():
            try:
                chunk_sizes[url] = get_lmcache_chunk_size(client)
            except TimeoutError:
                for c in self.mq_clients.values():
                    c.close()
                raise ConnectionError(
                    f"LMCache server {url} did not respond within {mq_timeout}s"
                ) from None

        # All servers must share chunk_size, otherwise the min() aggregation
        # over per-server hits would mix different granularities.
        unique_sizes = set(chunk_sizes.values())
        assert len(unique_sizes) == 1, (
            f"All LMCache servers must share the same chunk_size, got {chunk_sizes}"
        )
        self.chunk_size = unique_sizes.pop()

        # chunk_size must align to vLLM block_size; relied on by lookup / load.
        assert self.chunk_size % vllm_block_size == 0, (
            f"chunk_size ({self.chunk_size}) must be a multiple of "
            f"vllm_block_size ({vllm_block_size})"
        )
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

        # Health state: one Event per server. The adapter is considered healthy
        # only if ALL per-server events are set (any unhealthy server taints
        # the whole adapter, matching the min() semantics used for lookups).
        self._health_events: dict[str, threading.Event] = {}
        for url in self._server_urls:
            ev = threading.Event()
            ev.set()  # start optimistic; heartbeat will clear on failure
            self._health_events[url] = ev

        # Heartbeats: one thread per server so a slow/dead node cannot block
        # the others. Threads are NOT created here -- they are lazily started
        # on the first lookup (by then vLLM is fully ready).
        self._heartbeat_interval = heartbeat_interval
        self._heartbeats: dict[str, HeartbeatThread] = {}
        self._heartbeat_lock = threading.Lock()

    @property
    def world_size(self) -> int:
        if not self.parallel_strategy.use_mla and self.parallel_strategy.n_servers > 1:
            return self.parallel_strategy.kv_world_size_per_node
        return self.parallel_strategy.kv_world_size

    @property
    def tp_size(self) -> int:
        if self.parallel_strategy.n_servers > 1:
            return self.parallel_strategy.tp_size_per_node
        return self.parallel_strategy.tp_size

    @property
    def is_healthy(self) -> bool:
        """Whether all the LMCache server is healthy."""
        return all(ev.is_set() for ev in self._health_events.values())

    def healthy_urls(self) -> list[str]:
        return [u for u, ev in self._health_events.items() if ev.is_set()]

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start heartbeat threads (one per server) on first use.
        Safe to call concurrently; threads are only created once thanks
        to the lock + membership check on the ``self._heartbeats`` dict.
        """
        # Fast path: threads already started for every server.
        if self._heartbeats:
            return
        with self._heartbeat_lock:
            if self._heartbeats:
                return
            for url, client in self.mq_clients.items():
                hb = HeartbeatThread(
                    mq_client=client,
                    health_event=self._health_events[url],
                    interval=self._heartbeat_interval,
                )
                hb.start()
                self._heartbeats[url] = hb

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
    ):
        """
        Submit a new lookup request to LMCache if there is no ongoing request.

        Sends a LOOKUP request to the server and blocks until a prefetch
        job ID is returned.  The actual prefetch result can then be polled
        via ``check_lookup_result``.

        Args:
            request_id: The ID of the lookup request. The same ID indicates it's
                from the same request
            token_ids: Token IDs to lookup from LMCache
            cache_salt: Per-user isolation salt. Requests with different
                cache_salt values produce separate cache entries.

        Returns:
            None

        Notes:
            This function will have a side-effect: submitting a look up request to
            LMCache, which will essentially 'lock' the KV cache chunks in the LMCache
            for later retrieve operations.
            In the meantime, this function will record the lookup request, and the
            status of the look up request can be checked by `check_lookup_result`.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            logger.warning(
                "Skip LOOKUP for req=%s because not all servers are healthy: %s",
                request_id,
                {u: ev.is_set() for u, ev in self._health_events.items()},
            )
            return

        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()

        # One task per server.
        def _submit_one(url: str) -> tuple[str, bool]:
            client = self.mq_clients[url]
            try:
                fut = send_lmcache_request(
                    client,
                    RequestType.LOOKUP,
                    [key, self.tp_size],
                )
                fut.result(timeout=self._mq_timeout)
                return url, True
            except TimeoutError:
                logger.warning(
                    "LOOKUP to %s timed out after %ss; marking unhealthy.",
                    url,
                    self._mq_timeout,
                )
                self._health_events[url].clear()
                return url, False
            except Exception as e:
                logger.error("LOOKUP to %s failed: %s", url, e, exc_info=True)
                self._health_events[url].clear()
                return url, False

        # Fan out in parallel; total latency ~= slowest server, not sum.
        results = list(self._executor.map(_submit_one, self._server_urls))

        # Only track as pending when every server accepted the job.
        if all(ok for _, ok in results):
            self._pending_lookups.add(request_id)
        else:
            failed = [u for u, ok in results if not ok]
            logger.error(
                "[req=%s] LOOKUP failed on servers %s -- fall back to no-hit",
                request_id,
                failed,
            )

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        Check the result of a previously submitted lookup request.

        Sends a QUERY_PREFETCH_STATUS request to the server and blocks
        until the server responds.  Returns the matched token count
        when the prefetch is complete, or None if still in progress.

        Args:
            request_id: The ID of the lookup request submitted in
                `maybe_submit_lookup_request`

        Returns:
            An integer representing the total number of tokens matched
            in LMCache (prefix matching), or
            None if the lookup request is not finished yet.
        """
        if request_id not in self._pending_lookups:
            # No job — either unhealthy at submit time or already cleaned up.
            # If we have a cached result, return it to handle repeated calls.
            return self._finished_lookup_results.get(request_id, 0)

        if not self.is_healthy:
            # Server went down — give up on this lookup
            return 0

        if request_id in self._finished_lookup_results:
            # Return cached result if the job is already finished
            return self._finished_lookup_results[request_id]

        def _query_one(url: str) -> tuple[str, int | None]:
            client = self.mq_clients[url]
            try:
                r = send_lmcache_request(
                    client,
                    RequestType.QUERY_PREFETCH_STATUS,
                    [request_id],
                ).result(timeout=self._mq_timeout)
                return url, r
            except TimeoutError:
                logger.warning(
                    "QUERY_PREFETCH_STATUS to %s timed out. Marking unhealthy.",
                    url,
                )
                self._health_events[url].clear()
                return url, 0
            except Exception as e:
                logger.error(
                    "QUERY_PREFETCH_STATUS to %s failed: %s", url, e, exc_info=True
                )
                self._health_events[url].clear()
                return url, 0

        results = list(self._executor.map(_query_one, self._server_urls))

        per_server: dict[str, int] = {}
        for url, r in results:
            if r is None:
                return None
            per_server[url] = int(r)

        min_chunks = min(per_server.values())
        max_chunks = max(per_server.values())
        if min_chunks != max_chunks:
            logger.warning(
                "[req=%s] LMCache hit mismatch across servers: %s → take min=%d",
                request_id,
                per_server,
                min_chunks,
            )
        self._per_server_hits[request_id] = per_server
        token_count = min_chunks * self.chunk_size
        self._finished_lookup_results[request_id] = token_count
        self._pending_lookups.discard(request_id)
        return token_count

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup state for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookup_results.pop(request_id, None)
        self._per_server_hits.pop(request_id, None)

    def shutdown(self) -> None:
        """Shutdown the scheduler adapter and its resources."""
        self._executor.shutdown(wait=True)
        for client in self.mq_clients.values():
            client.close()
        with self._heartbeat_lock:
            for hb in self._heartbeats.values():
                hb.stop()

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> None:
        """Release read locks acquired during lookup without a full retrieve.

        Use this when some chunks matched by lookup overlap with blocks that
        vLLM has already computed, so they will never be retrieved.  Calling
        this prevents those chunks from holding read locks until TTL expiry.

        Or use this when a request is cancelled or aborted after lookup but
        before retrieve to avoid holding read locks until TTL expiry.

        When ``start`` or ``end`` is not aligned to the chunk size, the
        entire chunk containing start boundary is freed but not end boundary.
        It is caller's responsibility to properly align the boundaries.

        Args:
            token_ids: Token IDs for the key (same as used in lookup).
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
        """
        per_server = self._per_server_hits.get(request_id)
        if per_server is None:
            targets = {
                url: (end - start) // self.chunk_size for url in self._server_urls
            }
        else:
            targets = dict(per_server)

        for url, hit_chunks in targets.items():
            if hit_chunks <= 0:
                continue
            per_server_end = start + hit_chunks * self.chunk_size
            per_server_end = min(per_server_end, len(token_ids))

            key = self._create_key(
                token_ids=token_ids,
                start=start,
                end=per_server_end,
                request_id=request_id,
                cache_salt=cache_salt,
            ).no_worker_id_version()

            client = self.mq_clients.get(url)
            if client is None:
                continue
            try:
                send_lmcache_request(
                    client,
                    RequestType.FREE_LOOKUP_LOCKS,
                    [key, self.tp_size],
                )
            except Exception as e:
                logger.warning(
                    "[req=%s] FREE_LOOKUP_LOCKS to %s failed: %s "
                    "(rely on server-side GC for any residual lock)",
                    request_id,
                    url,
                    e,
                )

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        if not self.is_healthy:
            return

        for url in self.healthy_urls():
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.END_SESSION,
                [request_id],
            )

    def report_block_allocations(
        self,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Report vLLM GPU block allocation deltas to LMCache server.

        Fire-and-forget: does not wait for a response. If the server
        is unhealthy the report is silently dropped.

        Args:
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        if not self.is_healthy or not records:
            return

        for url in self.healthy_urls():
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.REPORT_BLOCK_ALLOCATION,
                [os.getpid(), self.model_name, records],
            )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

        Returns:
            IPCCacheEngineKey: The constructed key.
        """
        # NOTE: for the scheduler adapter, we don't have a worker id,
        # so we set it to None in the key.
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )


class LMCacheMPWorkerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy | int,
        legacy_block_size: int | None = None,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        """Initialize the worker adapter for current or legacy vLLM callers.

        Args:
            server_url: The server URL for the LMCache message queue.
            context: The ZMQ context.
            model_name: The model name used for LMCache keys.
            vllm_block_size: The block size used in vLLM, or legacy KV world
                size when ``parallel_strategy`` is an int.
            parallel_strategy: Current ``ParallelStrategy`` metadata, or the
                legacy KV worker id from older vLLM connectors.
            legacy_block_size: The vLLM block size passed positionally by
                older vLLM connectors.
            mq_timeout: Timeout in seconds for message queue requests.
            heartbeat_interval: Interval in seconds between heartbeat pings.

        Raises:
            TypeError: If the connector argument shape is unsupported.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        self.mq_client = MessageQueueClient(server_url, context)
        self._mq_timeout = mq_timeout

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        # request_id -> (future, block_ids)
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}

        # Block IDs that failed due to retrieve timeout
        self.error_block_ids: set[int] = set()

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()
        # Request IDs already returned as finished_sending to the scheduler.
        # Prevents re-reporting the same ID after drain clears tracking sets.
        self._returned_finished: set[str] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        try:
            chunk_size = get_lmcache_chunk_size(self.mq_client)
        except TimeoutError:
            self.mq_client.close()
            raise ConnectionError(
                f"LMCache server did not respond within {mq_timeout}s. "
                "Is the server running?"
            ) from None
        assert chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = chunk_size // vllm_block_size
        # Retain the vLLM logical block size so we can ship it to the
        # LMCache server in ``register_kv_caches`` — the server uses it
        # (as ``layout_hints["inference_engine_logical_block_size"]``)
        # to derive per-group compression ratios when some KV layer
        # groups compress multiple logical tokens into a single physical
        # slot (``shape_desc.bs <
        # inference_engine_logical_block_size``).
        self.vllm_logical_block_size = vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first store or retrieve
        # request, by which time vLLM is fully ready (model loaded,
        # KV caches allocated, warmup & CUDA graph capture done).
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()

        # request telemetry, used for prefill-decode disagg
        # TODO: pass down the configuration via vLLM connector config
        # instead of env var
        self.request_telemetry = RequestTelemetryFactory.create(
            telemetry_type=os.getenv("LMCACHE_REQUEST_TELEMETRY_TYPE", "noop"),
            config={
                "endpoint": os.getenv(
                    "LMCACHE_REQUEST_TELEMETRY_ENDPOINT",
                    "http://localhost:5768/api/v1/telemetry",
                ),
            },
        )

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy.

        Reflects the most recent heartbeat result. KV cache
        re-registration on the unhealthy->healthy transition is handled
        by the heartbeat thread itself via ``register_recover_callback``,
        so this property only reads the shared event.
        """
        return self._health_event.is_set()

    @property
    def world_size(self) -> int:
        if not self.parallel_strategy.use_mla and self.parallel_strategy.n_servers > 1:
            return self.parallel_strategy.kv_world_size_per_node
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        if not self.parallel_strategy.use_mla and self.parallel_strategy.n_servers > 1:
            return (
                self.parallel_strategy.actual_worker_id
                % self.parallel_strategy.kv_world_size_per_node
            )
        return self.parallel_strategy.kv_worker_id

    @property
    def use_mla(self) -> bool:
        """Whether to use MLA."""
        return self.parallel_strategy.use_mla

    @property
    def is_first_rank_of_pp_group(self) -> bool:
        """Is the first rank of the pipeline parallel group (TP-group local rank 0).

        In multi-server MLA deployments, this only identifies the global
        rank-0 worker.  Use ``is_first_rank_of_node`` for per-node STORE gating.
        """
        return (
            self.parallel_strategy.actual_worker_id % self.parallel_strategy.tp_size
            == 0
        )

    @property
    def is_first_rank_of_node(self) -> bool:
        """Whether this worker is the first rank on its node.

        In multi-server MLA deployments each node runs one LMCache server.
        Only the first rank on each node needs to STORE the KV cache, since
        all ranks on the same node hold identical KV data under MLA.
        For single-server deployments this degenerates to is_first_rank_of_pp_group.
        """
        n_servers = self.parallel_strategy.n_servers
        if n_servers <= 1:
            return self.is_first_rank_of_pp_group
        ranks_per_node = self.parallel_strategy.actual_world_size // n_servers
        return self.parallel_strategy.actual_worker_id % ranks_per_node == 0

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register the kv caches with LMCache server.

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.

        Raises:
            ConnectionError: if the server does not respond within
                mq_timeout.
        """
        logger.info("Registering kv caches")
        self.kv_caches = kv_caches
        self._send_register_kv_caches_request(kv_caches)

    def _send_register_kv_caches_request(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Submit a REGISTER_KV_CACHE request and wait for the response.

        Shared by the public ``register_kv_caches`` entry point and the
        recovery path inside ``is_healthy``.

        Args:
            kv_caches: The KV cache dict to register.

        Raises:
            ConnectionError: if the server does not respond within
                mq_timeout.
        """
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints

        layout_hints = vllm_layout_hints()
        layout_hints["inference_engine_logical_block_size"] = (
            self.vllm_logical_block_size
        )
        future = send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self.instance_id,
                wrap_kv_caches(kv_caches),
                self.model_name,
                self.world_size,
                EngineType.VLLM,
                layout_hints,
            ],
        )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to "
                "register_kv_caches within "
                f"{self._mq_timeout}s. Is the server running?"
            ) from None

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            self._heartbeat = HeartbeatThread(
                mq_client=self.mq_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
            )
            self._heartbeat.register_recover_callback(
                self._reregister_kv_caches_callback
            )
            self._heartbeat.start()

    def _reregister_kv_caches_callback(self) -> bool:
        """Heartbeat recover callback: re-register KV caches after the
        server returns. Runs on the heartbeat thread, before the health
        event is set.

        Returns:
            ``True`` if there is nothing to re-register or registration
            succeeds; ``False`` on registration failure (the heartbeat
            will keep the health event cleared and retry on the next
            successful PING).
        """
        if not self.kv_caches:
            # Nothing was registered yet (server flapped before the
            # very first register_kv_caches). Treat as success so the
            # health event can be set.
            return True

        try:
            self._send_register_kv_caches_request(self.kv_caches)
            logger.warning("Finished re-registering KV caches after server recovery")
        except ConnectionError:
            logger.exception(
                "Failed to re-register KV caches after server recovery; "
                "will retry on next heartbeat"
            )
            return False
        except Exception:
            logger.exception(
                "Unexpected error during KV cache re-registration; "
                "will retry on next heartbeat"
            )
            return False
        return True

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: Any,
        cache_salt: str = "",
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salt: Per-user isolation salt.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [key, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.store_futures[request_id] = future

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: Any,
        cache_salt: str = "",
    ):
        """
        Submit a KV cache retrieve request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salt: Per-user isolation salt.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            self.error_block_ids.update(op.block_ids)
            return

        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                key,
                self.instance_id,
                op.block_ids,
                event.ipc_handle(),
                op.skip_first_n_tokens,
            ],
        ).to_cuda_future()
        self.retrieve_futures[request_id] = (future, list(op.block_ids))

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: Any,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be the same as
                request_ids.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        for request_id, op, salt in zip(request_ids, ops, cache_salts, strict=False):
            self.submit_store_request(request_id, op, event, cache_salt=salt)

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: Any,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be same as
                request_ids.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        for request_id, op, salt in zip(request_ids, ops, cache_salts, strict=False):
            self.submit_retrieve_request(request_id, op, event, cache_salt=salt)

    def _process_finished_stores(
        self,
        finished_req_ids_from_lmcache: set[str],
        finished_req_ids_from_engine: set[str],
    ) -> set[str]:
        """Merge LMCache-side and engine-side finished store info."""
        self.finished_stores.update(finished_req_ids_from_lmcache)
        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if req_id in self._returned_finished:
                continue
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)
        ret_stores.update(self._update_and_get_finished_store())
        self._returned_finished.update(ret_stores)
        return ret_stores

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests.

        Args:
            finished_req_ids_from_engine: the set of request ids that are
                reported as finished from the vLLM engine side.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids.

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        # If unhealthy, drain all pending futures immediately
        if not self.is_healthy:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                self.error_block_ids.update(r_block_ids)
            self.store_futures.clear()
            self.retrieve_futures.clear()

            ret_stores = self._process_finished_stores(
                finished_stores, finished_req_ids_from_engine
            )
            # A request may have a pending retrieve AND appear in
            # finished_req_ids_from_engine (it ran without loading KV after
            # the server died).  The scheduler processes finished_recving
            # first and deletes the request, so we must not also report it
            # in finished_sending.
            ret_stores -= finished_retrieves
            return ret_stores, finished_retrieves

        finished_stores = set()
        finished_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result()
            finished_stores.add(request_id)

            if not s_result:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, _) in self.retrieve_futures.items():
            if not r_future.query():
                continue

            r_result = r_future.result()
            finished_retrieves.add(request_id)

            if not r_result:
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)

        # Update the internal states
        ret_stores = self._process_finished_stores(
            finished_stores, finished_req_ids_from_engine
        )

        # the invocation of `get_finished` means that
        # these requests' KV caches are already fully stored.
        # or the requests normally ends without any store.
        if ret_stores:
            self.request_telemetry.on_request_store_finished(
                request_ids_set=ret_stores,
                model_name=self.model_name,
                world_size=self.world_size,
                kv_rank=self.worker_id,
            )

        return ret_stores, finished_retrieves

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Returns the block IDs that failed due to retrieve timeout,
        then clears the internal set.
        """
        errors = self.error_block_ids.copy()
        self.error_block_ids.clear()
        return errors

    def shutdown(self):
        """
        Shutdown the LMCache MP worker adapter
        """
        logger.info("Unregistering kv caches")
        try:
            send_lmcache_request(
                self.mq_client,
                RequestType.UNREGISTER_KV_CACHE,
                [self.instance_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LMCache server did not respond to unregister within %ss. "
                "Proceeding with shutdown.",
                self._mq_timeout,
            )

        self.mq_client.close()
        self.request_telemetry.close()

    # Helper functions
    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

        Returns:
            IPCCacheEngineKey: The constructed key.
        """
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
