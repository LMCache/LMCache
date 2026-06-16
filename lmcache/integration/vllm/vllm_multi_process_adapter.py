# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, NoReturn, Protocol
import enum
import os
import threading
import uuid

# Third Party
import torch
import zmq

# First Party
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.integration.vllm.utils import vllm_layout_hints
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    expand_engine_block_ids,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context import (
    EngineDrivenTransferContext,
    TransferContext,
    create_transfer_context,
)
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary
from lmcache.v1.platform import _registry as platform_registry

logger = init_logger(__name__)


class ExtraConfigDefault(enum.Enum):
    """Centralized default values for extra_config keys.

    Each member's *name* is the key used in the extra_config dict,
    and its *value* is the default.
    """

    # Timeout (seconds) for blocking MQ requests: initial
    # chunk-size query, KV cache registration/unregistration,
    # and other synchronous operations.
    mq_timeout = 300.0
    # Interval (seconds) between periodic heartbeat pings
    # to the server.
    heartbeat_interval = 10.0
    # Routing mode for ``create_transfer_context``: ``auto`` keeps the
    # historical CUDA -> lmcache_driven / others -> engine_driven dispatch;
    # ``lmcache_driven`` forces the IPC / SHM zero-copy path where the
    # LMCache server pulls data via device handles;
    # ``engine_driven`` forces the worker-side gather/scatter copy path.
    # Mirrors the ``LMCACHE_MP_TRANSFER_MODE`` env var; this extra_config
    # key wins when both are set.
    mp_transfer_mode = "auto"


# Backward-compatible aliases for the legacy `lmcache_mp_connector_0180`
# entry point, which still passes these as positional/keyword args.
DEFAULT_MQ_TIMEOUT: float = ExtraConfigDefault.mq_timeout.value
DEFAULT_HEARTBEAT_INTERVAL: float = ExtraConfigDefault.heartbeat_interval.value

_EXTRA_CONFIG_KEY_PREFIX = "lmcache.mp."

# Floor (seconds) of the MP server's worker reap timeout. It only covers the
# default 10 s heartbeat interval (3 x 10 s); the adapter warns at startup
# when 3 x heartbeat_interval exceeds it (server timeout must be raised too).
_SERVER_REAP_TIMEOUT_FLOOR_SECONDS: float = 30.0


def _resolve_extra_config(
    extra_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve extra_config against :class:`ExtraConfigDefault`.

    Keys in *extra_config* are expected to carry the
    ``lmcache.mp.`` prefix (e.g. ``lmcache.mp.mq_timeout``).
    The prefix is stripped before matching against
    :class:`ExtraConfigDefault` members.

    All ``lmcache.mp.*`` entries are logged.  Entries whose
    value differs from the default are additionally marked as
    *overridden*.

    Args:
        extra_config: User-supplied config dict (may be *None*).

    Returns:
        A dict keyed by :pyattr:`ExtraConfigDefault` member names.
    """
    stripped: dict[str, Any] = {}
    if extra_config is not None:
        for k, v in extra_config.items():
            if k.startswith(_EXTRA_CONFIG_KEY_PREFIX):
                short = k[len(_EXTRA_CONFIG_KEY_PREFIX) :]
                stripped[short] = v

    resolved: dict[str, Any] = {}
    for item in ExtraConfigDefault:
        default = item.value
        raw = stripped.get(item.name)
        value = type(default)(raw) if raw is not None else default
        if value != default:
            logger.info(
                "%s%s = %s (overridden, default: %s)",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
                default,
            )
        else:
            logger.info(
                "%s%s = %s",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
            )
        resolved[item.name] = value
    return resolved


class _IpcEvent(Protocol):
    def ipc_handle(self) -> Any: ...


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
    # Per-iteration resource management: if wrapping the N-th tensor
    # raises, ``shm_unlink`` whatever earlier iterations already
    # registered with POSIX SHM so the named segments do not outlive
    # the failed batch. CUDA wrappers do not own a named segment and
    # are skipped via the duck-typed ``shm_name`` check.
    wrappers: KVCache = []
    try:
        for tensor in kv_caches.values():
            wrappers.append(wrap_one_kv_cache(tensor))
    except BaseException:
        _release_partial_kv_wrappers(wrappers)
        raise
    return wrappers


def _release_partial_kv_wrappers(wrappers: list[Any]) -> None:
    """Best-effort unlink of SHM segments owned by partially built wrappers.

    Used by :func:`wrap_kv_caches` to roll back a half-finished batch
    when a later iteration raises. Only POSIX-SHM-backed wrappers carry
    a ``shm_name`` attribute, so other wrapper kinds (e.g. CUDA-IPC)
    are silently skipped.
    """
    # First Party
    from lmcache.v1.multiprocess.posix_shm import shm_unlink

    for w in wrappers:
        name = getattr(w, "shm_name", None)
        if name is None:
            continue
        try:
            shm_unlink(name)
        except Exception:  # pragma: no cover - best effort
            logger.debug("shm_unlink failed during rollback", exc_info=True)


def wrap_one_kv_cache(tensor: torch.Tensor) -> Any:
    """Dispatch by ``tensor.device.type`` via the platform registry.

    Concrete factories self-register at import time (CUDA in
    ``lmcache.v1.platform.cuda``, CPU SHM in
    ``lmcache.v1.platform.cpu``), so this call site stays free of
    if/elif chains and new accelerators plug in by registering a
    sibling factory.
    """
    return platform_registry.get_kv_wrapper_factory(tensor.device.type)(tensor)


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
    timeout: float = DEFAULT_MQ_TIMEOUT,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        timeout: Timeout in seconds for the blocking request.

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    lmcache_tokens_per_chunk = future.result(timeout=timeout)
    return lmcache_tokens_per_chunk


def _raise_server_unreachable(server_url: str, timeout: float) -> NoReturn:
    """Raise a verbose ConnectionError when the LMCache MP server is
    unreachable.

    The message intentionally spells out the most common cause (the
    standalone ``lmcache server`` process is not running), the URL that
    was being dialed, and the exact command to start it -- so that users
    landing here via ``vllm serve --kv-offloading-backend lmcache`` are
    not left guessing.
    """
    hint = (
        "Cannot reach the LMCache MP server at "
        f"'{server_url}' within {timeout}s.\n"
        "This usually means the standalone LMCache server is not "
        "running, or it is listening on a different host/port.\n"
        "To start one locally with the default port (5555):\n"
        "    lmcache server --l1-size-gb 20 --eviction-policy LRU\n"
        "To target a different host/port, override via "
        "kv_connector_extra_config (lmcache.mp.host / lmcache.mp.port), "
        "e.g.:\n"
        '    --kv-transfer-config \'{"kv_connector":'
        '"LMCacheMPConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config":{"lmcache.mp.host":'
        '"tcp://localhost","lmcache.mp.port":5555}}\'\n'
        "See https://docs.lmcache.ai/mp/quickstart.html for details."
    )
    logger.warning(hint)
    raise ConnectionError(hint) from None


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

    vllm_world_size: int
    """Number of workers managed by one vLLM scheduler (TP × PP; excludes DP).

    Mirrors ``vllm.parallel_config.world_size``.
    """

    vllm_worker_id: int
    """This worker's rank within its scheduler group."""

    tp_size: int
    """The tensor parallel size."""

    pp_size: int
    """The pipeline parallel size."""

    @property
    def kv_world_size(self) -> int:
        """Number of pieces a single token chunk's KV cache is split into
        on the LMCache server storage."""
        if self.use_mla:
            return self.vllm_world_size // self.tp_size
        return self.vllm_world_size

    @property
    def kv_worker_id(self) -> int:
        """Index of the piece of a single token chunk's KV cache
        that the current worker is responsible for,
        in ``[0, kv_world_size)``."""
        if self.use_mla:
            return self.vllm_worker_id // self.tp_size
        return self.vllm_worker_id

    @property
    def is_kv_writer(self) -> bool:
        """Whether this rank is responsible for storing KV."""
        if not self.use_mla:
            return True
        return self.vllm_worker_id % self.tp_size == 0


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
        vllm_world_size=kv_world_size,
        vllm_worker_id=kv_worker_id,
        tp_size=kv_world_size,
        pp_size=1,
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
        """Run one heartbeat cycle: ping, recover callback, event update.

        A cycle that observes a stop request returns without firing the
        callback or touching the event — a straggler success after
        UNREGISTER must not re-register a ghost context.
        """
        was_healthy = self._health_event.is_set()
        healthy = send_ping(self._mq_client, timeout=self._interval)

        if self.stop_requested:
            return ThreadRunSummary(
                success=True,
                message="stop requested; skipping health update",
            )

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

    block_ids: list[list[int]]
    """Block IDs for the load/store operation, indexed by engine KV cache
    group (one inner list per engine group). Worker submit paths expand
    this to LMCache KV group order before sending requests to the server.
    """

    start: int = 0
    """Start token index"""

    end: int = 0
    """End token index"""

    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""

    @property
    def flat_block_ids(self) -> list[int]:
        """Return all block IDs flattened for group-blind error paths.

        Handles both the normal ``list[list[int]]`` format and the
        IPC-flattened ``list[int]`` format that vLLM v0.19.0 produces when
        ``SchedulerOutput`` serializes single-element nested lists across
        process boundaries (e.g. ``[[20, 21]]`` → ``[20, 21]``).
        Returns an empty list when ``block_ids`` is empty.
        """
        if not self.block_ids:
            return []
        # Defend against IPC serialization flattening [[20, 21, …]] → [20, 21, …]
        if isinstance(self.block_ids[0], int):
            return list(self.block_ids)
        return [
            block_id
            for group_block_ids in self.block_ids
            for block_id in group_block_ids
        ]


StoreResult = bool
RetrieveResult = bool
LookupResult = int


class LMCacheMPSchedulerAdapter:
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
        extra_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            server_url: The server URL for the LMCache message queue
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
                Ignored when ``extra_config`` is provided.
            heartbeat_interval: Interval in seconds between heartbeat pings.
                Ignored when ``extra_config`` is provided.
            extra_config: Optional dict with keys starting with
                ``lmcache.mp.`` (e.g., ``lmcache.mp.mq_timeout``). When
                provided, it overrides ``mq_timeout`` / ``heartbeat_interval``.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        if extra_config is not None:
            cfg = _resolve_extra_config(extra_config)
            mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
        self.mq_client = MessageQueueClient(server_url, context)
        self._mq_timeout = mq_timeout

        # Lookup state tracking:
        # - _pending_lookups: request_ids submitted but not yet resolved
        # - _finished_lookup_results: cached chunk count keyed by request_id,
        #   so that repeated calls to check_lookup_result return the same value
        #   even after the server has already popped the job (exactly-once).
        self._pending_lookups: set[str] = set()
        self._finished_lookup_results: dict[str, int] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        try:
            self.lmcache_tokens_per_chunk = get_lmcache_chunk_size(
                self.mq_client, timeout=self._mq_timeout
            )
        except TimeoutError:
            self.mq_client.close()
            _raise_server_unreachable(server_url, self._mq_timeout)
        assert self.lmcache_tokens_per_chunk % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = self.lmcache_tokens_per_chunk // vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first lookup
        # request, by which time vLLM is fully ready.
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def tp_size(self) -> int:
        """The tensor parallel size."""
        return self.parallel_strategy.tp_size

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy."""
        return self._health_event.is_set()

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
            self._heartbeat.start()

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
            return

        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (
            len(token_ids) // self.lmcache_tokens_per_chunk
        ) * self.lmcache_tokens_per_chunk

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()

        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [key, self.tp_size],
        )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LOOKUP request timed out after %ss. Marking server as unhealthy.",
                self._mq_timeout,
            )
            self._health_event.clear()
            return
        self._pending_lookups.add(request_id)

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

        try:
            result = send_lmcache_request(
                self.mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "QUERY_PREFETCH_STATUS timed out after %ss. "
                "Marking server as unhealthy.",
                self._mq_timeout,
            )
            self._health_event.clear()
            return 0

        if result is None:
            return None

        token_count = result * self.lmcache_tokens_per_chunk
        self._finished_lookup_results[request_id] = token_count
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
        if not self.is_healthy:
            return

        key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()
        send_lmcache_request(
            self.mq_client,
            RequestType.FREE_LOOKUP_LOCKS,
            [key, self.tp_size],
        )

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        if not self.is_healthy:
            return

        send_lmcache_request(
            self.mq_client,
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

        send_lmcache_request(
            self.mq_client,
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
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

        Returns:
            IPCCacheServerKey: The constructed key.
        """
        # NOTE: for the scheduler adapter, we don't have a worker id,
        # so we set it to None in the key.
        return IPCCacheServerKey(
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
        extra_config: dict[str, Any] | None = None,
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
                Ignored when ``extra_config`` is provided.
            heartbeat_interval: Interval in seconds between heartbeat pings.
                Ignored when ``extra_config`` is provided.
            extra_config: Optional dict with keys starting with
                ``lmcache.mp.`` (e.g., ``lmcache.mp.mq_timeout``). When
                provided, it overrides ``mq_timeout`` / ``heartbeat_interval``.

        Raises:
            TypeError: If the connector argument shape is unsupported.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        if extra_config is not None:
            cfg = _resolve_extra_config(extra_config)
            mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
            # Only treat ``mp_transfer_mode`` as an explicit override when
            # the user actually set it in extra_config; otherwise leave it
            # as ``None`` so ``create_transfer_context`` can still consult
            # the ``LMCACHE_MP_TRANSFER_MODE`` env var.
            mp_mode_key = (
                _EXTRA_CONFIG_KEY_PREFIX + ExtraConfigDefault.mp_transfer_mode.name
            )
            if mp_mode_key in extra_config:
                self._mp_transfer_mode = cfg[ExtraConfigDefault.mp_transfer_mode.name]
            else:
                self._mp_transfer_mode = None
        else:
            self._mp_transfer_mode = None
        self.mq_client = MessageQueueClient(server_url, context)
        self._mq_timeout = mq_timeout

        # Instance id for GPU worker. uuid4-derived (OS entropy) rather
        # than os.getpid() to avoid collision in containerized deployments.
        # Masked to 63 bits to stay signed-int64-safe for any msgpack peer.
        self.instance_id: int = uuid.uuid4().int & ((1 << 63) - 1)
        logger.info(
            "LMCache MP worker adapter created with instance_id=%d",
            self.instance_id,
        )

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.engine_group_infos: list[EngineGroupInfo] = []

        # Transport context for transfer operations.
        self.transfer_ctx: TransferContext | None = None

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        # request_id -> (future, block_ids)
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}
        # The IPC handle is not enough by itself; CUDA needs the exporting
        # event object to stay alive until the consumer is done with it.
        self.store_events: dict[str, _IpcEvent] = {}
        self.retrieve_events: dict[str, _IpcEvent] = {}

        # Block IDs that failed due to retrieve timeout
        self.error_block_ids: set[int] = set()

        # Retrieve request ids dropped by the unhealthy early-return of
        # submit_retrieve_request. get_finished must still report each id
        # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS.
        self._dropped_retrieves: set[str] = set()

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
            lmcache_tokens_per_chunk = get_lmcache_chunk_size(
                self.mq_client, timeout=self._mq_timeout
            )
        except TimeoutError:
            self.mq_client.close()
            _raise_server_unreachable(server_url, self._mq_timeout)
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        assert lmcache_tokens_per_chunk % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = lmcache_tokens_per_chunk // vllm_block_size

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
        if 3 * heartbeat_interval > _SERVER_REAP_TIMEOUT_FLOOR_SECONDS:
            logger.warning(
                "lmcache.mp.heartbeat_interval is %.1fs, so 3 x "
                "heartbeat_interval (%.1fs) exceeds the MP server's "
                "default worker reap timeout floor (%.1fs). Raise the "
                "server's worker reap timeout to at least 3 x the "
                "heartbeat interval, or the server may reap this "
                "worker between heartbeats.",
                heartbeat_interval,
                3 * heartbeat_interval,
                _SERVER_REAP_TIMEOUT_FLOOR_SECONDS,
            )

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
        """Get the kv world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """Get the kv worker id."""
        return self.parallel_strategy.kv_worker_id

    @property
    def is_kv_writer(self) -> bool:
        """Whether this worker is responsible for storing KV."""
        return self.parallel_strategy.is_kv_writer

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """
        Register the kv caches with LMCache server.

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            ConnectionError: if the server does not respond within
                mq_timeout.
            ValueError: if the LMCache chunk size is not a multiple of an
                engine group's ``tokens_per_block`` (chunk boundaries would
                not align with that group's paged-chunk boundaries).
        """
        logger.info("Registering kv caches")
        for info in engine_group_infos:
            if (
                info.tokens_per_block > 0
                and self.lmcache_tokens_per_chunk % info.tokens_per_block
            ):
                raise ValueError(
                    f"LMCache chunk size {self.lmcache_tokens_per_chunk} must be a "
                    f"multiple of engine group {info.engine_group_id} "
                    f"tokens_per_block {info.tokens_per_block}"
                )
        self.kv_caches = kv_caches
        self.engine_group_infos = list(engine_group_infos)
        self._send_register_kv_caches_request(kv_caches)

    def _block_ids_per_group(self, op: LoadStoreOp) -> list[list[int]]:
        return expand_engine_block_ids(self.engine_group_infos, op.block_ids)

    def _send_register_kv_caches_request(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Submit a REGISTER_KV_CACHE request and wait for the response.

        Shared by the public ``register_kv_caches`` entry point and the
        heartbeat recovery path (``_reregister_kv_caches_callback``).

        Args:
            kv_caches: The KV cache dict to register.

        Raises:
            ConnectionError: if the server does not respond within
                mq_timeout.
        """
        self.kv_caches = kv_caches
        transfer_ctx = create_transfer_context(kv_caches, mode=self._mp_transfer_mode)
        layout_hints = vllm_layout_hints()
        self.transfer_ctx = transfer_ctx
        try:
            # Register on the local, not self.transfer_ctx: a concurrent
            # shutdown() may null self.transfer_ctx between publish and this
            # call. The local is always non-None.
            transfer_ctx.register(
                self.instance_id,
                kv_caches,
                self.model_name,
                self.world_size,
                self.blocks_in_chunk,
                self.mq_client,
                self._mq_timeout,
                send_request=send_lmcache_request,
                layout_hints=layout_hints,
                engine_group_infos=self.engine_group_infos,
            )
        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to "
                "register_kv_caches within "
                f"{self._mq_timeout}s. Is the server running?"
            ) from None

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first store/retrieve.

        The heartbeat starts healthy (the event was set at construction). A
        live worker pings every interval, refreshing its server-side
        ``last_seen``, so it is never reaped while alive -- no re-registration
        is needed at startup, and the first store/retrieve is not gated. The
        recover callback still re-registers on a genuine unhealthy->healthy
        edge (server restart) and on freeze-gap recovery.
        """
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            heartbeat = HeartbeatThread(
                mq_client=self.mq_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
            )
            heartbeat.register_recover_callback(self._reregister_kv_caches_callback)
            heartbeat.start()
            self._heartbeat = heartbeat

    def _heartbeat_stop_requested(self) -> bool:
        """Whether a created heartbeat thread has a stop requested.

        Returns:
            ``True`` only if a heartbeat exists and its stop was requested.
        """
        heartbeat = self._heartbeat
        return heartbeat is not None and heartbeat.stop_requested

    def _reregister_kv_caches_callback(self) -> bool:
        """Heartbeat recover callback: re-register KV caches after the
        server returns. Runs on the heartbeat thread, before the health
        event is set.

        Returns:
            ``True`` if nothing needs re-registering or registration
            succeeds; ``False`` on failure or a requested heartbeat stop
            (event stays cleared; retried on the next successful PING).
        """
        if not self.kv_caches:
            # Nothing was registered yet (server flapped before the
            # very first register_kv_caches). Treat as success so the
            # health event can be set.
            return True

        # Skip the rebuild if a shutdown already requested the heartbeat stop.
        if self._heartbeat_stop_requested():
            logger.info("Heartbeat stop requested; skipping KV cache re-registration")
            return False

        try:
            self._send_register_kv_caches_request(self.kv_caches)
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
        logger.warning("Finished re-registering KV caches after server recovery")
        return True

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent,
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
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting store requests."
            )
        future = self.transfer_ctx.submit_store(
            request_id,
            key,
            self.instance_id,
            self.kv_caches,
            self._block_ids_per_group(op),
            event,
            self.blocks_in_chunk,
        )
        self.store_futures[request_id] = future
        self.store_events[request_id] = event

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent,
        cache_salt: str = "",
    ) -> None:
        """
        Submit a KV cache retrieve request to LMCache

        When the server is unhealthy the request is not submitted: blocks
        are flagged via ``error_block_ids`` (vLLM recomputes) and the id is
        recorded so ``get_finished`` still reports it exactly once.

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salt: Per-user isolation salt.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            self.error_block_ids.update(op.flat_block_ids)
            self._dropped_retrieves.add(request_id)
            return

        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting retrieve requests."
            )
        future = self.transfer_ctx.submit_retrieve(
            request_id,
            key,
            self.instance_id,
            self.kv_caches,
            self._block_ids_per_group(op),
            event,
            self.blocks_in_chunk,
            skip_first_n_tokens=op.skip_first_n_tokens,
        )
        self.retrieve_futures[request_id] = (future, op.flat_block_ids)
        self.retrieve_events[request_id] = event

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: _IpcEvent,
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
        event: _IpcEvent,
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
            - The second set contains the finished retrieve request ids,
                including retrieves dropped at submit time while unhealthy
                (reported exactly once; blocks already in error_block_ids).

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
            self.store_events.clear()
            self.retrieve_events.clear()

            # Retrieves dropped at submit time still must be reported,
            # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS.
            # Swap-drain (not update-then-clear): a concurrent
            # submit_retrieve_request add lands in the old set (reported now)
            # or the fresh set (reported next call), never lost.
            dropped = self._dropped_retrieves
            self._dropped_retrieves = set()
            finished_retrieves.update(dropped)

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
            self.store_events.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)
            self.retrieve_events.pop(request_id, None)

        # Retrieves dropped while unhealthy still must be reported,
        # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS. No
        # finished_sending dedup is needed (unlike the unhealthy branch): a
        # dropped retrieve's request is parked in WAITING_FOR_REMOTE_KVS until
        # this report, so it cannot also be engine-finished in the same call.
        # Swap-drain so a concurrent submit_retrieve_request add is never lost.
        dropped = self._dropped_retrieves
        self._dropped_retrieves = set()
        finished_retrieves.update(dropped)

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

    def shutdown(self) -> None:
        """
        Shutdown the LMCache MP worker adapter.

        Stops the heartbeat (if started) before UNREGISTER: no new ping
        on the closing mq_client, and a straggler in-flight cycle cannot
        re-register or flip the health event after unregistration.
        """
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                self._heartbeat.stop()

        logger.info("Unregistering kv caches")
        try:
            unregister_type = (
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
                if isinstance(self.transfer_ctx, EngineDrivenTransferContext)
                else RequestType.UNREGISTER_KV_CACHE
            )
            send_lmcache_request(
                self.mq_client,
                unregister_type,
                [self.instance_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LMCache server did not respond to unregister within %ss. "
                "Proceeding with shutdown.",
                self._mq_timeout,
            )

        if self.transfer_ctx is not None:
            self.transfer_ctx.close()
            self.transfer_ctx = None

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
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

        Returns:
            IPCCacheServerKey: The constructed key.
        """
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
