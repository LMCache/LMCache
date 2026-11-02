# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import lcm
from typing import Any, Callable, Protocol, cast
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import (
    GroupLayout,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.inline import EngineDrivenContextInline
from lmcache.v1.multiprocess.transfer_mode import MPTransferMode
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import (
    get_device_spec,
    resolve_kv_wrapper_factory,
    synchronize_device,
)
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.kv_wrap import wrap_kv_caches
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` defers to the device spec's declared
# default (:meth:`DeviceSpec.default_mp_transfer_mode`).
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


# Helper functions
def _supports_async_primitives() -> bool:
    """Probe whether the worker device supports the async store primitives.

    The async engine-driven store path needs a stream, an event exposing
    ``record``/``synchronize``/``wait``, and pinned (page-locked) host memory.
    When any of these is unavailable (e.g. a CPU-only backend), the factory
    falls back to the synchronous :class:`EngineDrivenTransferContext`. This
    dispatch is internal and capability-based; there is no user-facing
    async/sync flag.

    Returns:
        True if all required async primitives are available, else False.
    """
    if not hasattr(torch_dev, "Stream") or not hasattr(torch_dev, "Event"):
        return False
    # CPU-only stub exposes Stream/Event but has no real async capability.
    if hasattr(torch_dev, "is_available") and not torch_dev.is_available():
        return False
    try:
        stream = torch_dev.Stream()
        event = torch_dev.Event()
    except Exception:
        return False
    for attr in ("record", "synchronize", "wait"):
        if not callable(getattr(event, attr, None)):
            del stream, event
            return False
    del stream, event
    try:
        probe = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        del probe
    except (RuntimeError, TypeError):
        return False
    return True


def _build_engine_driven_context(
    instance_id: int,
    req_client: RequestClient,
) -> "TransferContext":
    """Build the engine-driven context, async when device-capable else sync.

    Routes the ``ENGINE_DRIVEN`` and AUTO branches through a single capability
    check. ``AsyncEngineDrivenTransferContext`` is imported lazily to avoid an
    import cycle and to keep the synchronous path free of stream/event
    dependencies.

    Returns:
        ``AsyncEngineDrivenTransferContext`` when async primitives are
        available, otherwise ``EngineDrivenTransferContext``.
    """
    if _supports_async_primitives():
        # First Party
        from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
            AsyncEngineDrivenTransferContext,
        )

        logger.info("Using AsyncEngineDrivenTransferContext for store path")
        return AsyncEngineDrivenTransferContext(instance_id, req_client)

    logger.info("Using EngineDrivenTransferContext (sync) for store path")
    return EngineDrivenTransferContext(instance_id, req_client)


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode %r (valid: %s)" % (raw, valid)
        ) from exc


def _build_lmcache_driven_context(
    device_type: str,
    instance_id: int,
    req_client: RequestClient,
) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        resolve_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' instead." % device_type
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_lmcache_driven_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' instead." % device_type
        )
    return LMCacheDrivenTransferContext(instance_id, req_client)


class IPCEvent(Protocol):
    """Protocol for device events used by transport operations."""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (async ordering primitive)."""


@dataclass
class _GroupState:
    """Worker-side per-LMCache-group transfer state (multi-group registration).

    Attributes:
        layer_names: KV cache dict keys belonging to this group, in group
            layer order — selects the gather/scatter tensor subset.
        engine_kv_format: Detected KV format for this group's tensors.
        blocks_in_chunk: Paged blocks of THIS group per LMCache chunk
            (``chunk_tokens / tokens_per_block``).
        blocks_per_window: Paged blocks of THIS group actually stored per
            chunk. Equals ``blocks_in_chunk`` for full-attention groups; for a
            sliding-window group it is ``window_tokens / tokens_per_block``, so
            only the trailing ``blocks_per_window`` blocks of each chunk are
            gathered / scattered.
        layout_desc: Chunk layout for this group's objects.
        window_chunks: Trailing chunks this group needs for a cache hit
            (``-1`` = full attention). A retrieve returns only these chunks
            for the group, mirroring the server's prefetch window.
    """

    layer_names: list[str]
    engine_kv_format: "lmcache_native.EngineKVFormat"
    blocks_in_chunk: int
    blocks_per_window: int
    layout_desc: MemoryLayoutDesc
    window_chunks: int = -1


# Allocates ``count`` CPU gather targets of one shape and dtype.
_StagingAlloc = Callable[[torch.Size, torch.dtype, int], list[torch.Tensor]]


@dataclass
class _GroupStorePlan:
    """One group's part of a multi-group store.

    Attributes:
        gid: LMCache group index.
        chunk_indices: Chunk positions this group stores (all-null chunks left
            out), in increasing order.
        out: Gather targets parallel to ``chunk_indices`` (SHM slots or pinned
            staging), or ``None`` to let gather allocate fresh CPU tensors.
    """

    gid: int
    chunk_indices: list[int]
    out: "list[torch.Tensor] | None"


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _get_kv_device(kv_caches: dict[str, torch.Tensor]) -> torch.device:
    """Return the device shared by a non-empty KV-cache mapping.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name.

    Returns:
        The device of the first KV-cache tensor.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    if not kv_caches:
        raise ValueError("Transfer requires at least one KV cache")
    return get_device(next(iter(kv_caches.values())))


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. Device-handle paths
    return event-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Bind this context to a single worker and request client.

        Args:
            instance_id: Worker process instance identifier used by all
                context-owned transport requests.
            req_client: Transport-neutral client used by this context. The
                adapter retains ownership and closes it after the context.
        """
        self._instance_id = instance_id
        self._req_client = req_client
        self._registration_request_sent = False
        self._closed = False
        self._lifecycle_lock = threading.Lock()

    def _submit_registration(
        self, submit: Callable[[], MessagingFuture[Any]]
    ) -> MessagingFuture[Any]:
        """Submit REGISTER before allowing a concurrent UNREGISTER.

        If shutdown won the race before the request was sent, registration is
        rejected. Once submission begins, unregistration remains available even
        if waiting for the registration response times out.
        """
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Transfer context is closed.")
            self._registration_request_sent = True
            return submit()

    def _submit_unregistration(
        self, submit: Callable[[], MessagingFuture[Any]]
    ) -> MessagingFuture[Any] | None:
        """Submit UNREGISTER only after this context sent REGISTER."""
        with self._lifecycle_lock:
            if not self._registration_request_sent:
                return None
            return submit()

    def _is_closed(self) -> bool:
        """Return whether adapter shutdown closed this context."""
        with self._lifecycle_lock:
            return self._closed

    def _mark_closed(self) -> None:
        """Prevent future registration requests from a racing caller."""
        with self._lifecycle_lock:
            self._closed = True

    @abstractmethod
    def register(
        self,
        _kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        lmcache_chunk_tokens: int | None = None,
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of engine blocks per LMCache chunk.
            mq_timeout: Timeout in seconds for synchronous request wait.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            engine_type: Serving engine that produced the caches. Transfer
                contexts use it to normalize engine-specific KV layouts.
                Adapters should pass their own :class:`EngineType`; the
                default is :attr:`EngineType.VLLM` for backwards compatibility.
            lmcache_chunk_tokens: Tokens in one LMCache chunk. Engine-driven
                multi-group registration sizes every group from it; without
                it the chunk is derived from ``blocks_in_chunk`` and this
                worker's own groups, which is wrong when a pipeline stage does
                not hold the group with the largest block. Adapters that know
                the chunk size should pass it.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    @abstractmethod
    def unregister(self) -> MessagingFuture[Any] | None:
        """Start unregistering this context's server-side KV-cache state.

        Concrete contexts select the unregister RPC that matches the protocol
        used by :meth:`register`. The returned future lets the caller apply
        its own lifecycle timeout policy before :meth:`close` releases local
        state.

        Returns:
            A future that resolves when the server acknowledges unregistration,
            or ``None`` when no registration request was sent.

        """

    def register_q(
        self,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register the paged Q ring with the server under the same worker
        instance ID but different model_name (model_name##query).

        Args:
            q_caches: Worker Q cache tensors keyed by layer name.
            model_name: Model name used by cache keys (model_name##query).
            world_size: KV world size.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.
            mq_timeout: Timeout in seconds for synchronous request wait.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (now only lmcache-driven).
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """
        raise NotImplementedError(
            "Q ring registration is not supported by this transfer context"
        )

    @abstractmethod
    def create_recorded_event(self) -> IPCEvent | None:
        """Create the event needed to order the next transfer.

        Returns:
            A recorded device event when the transfer context needs stream
            ordering, or ``None`` when the context orders transfers
            synchronously without an event.

        Raises:
            RuntimeError: If the context has not been registered or cannot
                create the event required by its transfer protocol.
        """

    def submit_q_store(
        self,
        request_id: str,
        key: Any,
        q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a Q ring store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key for the Q store range (query-specific model_name).
            q_caches: Q ring tensors keyed by layer name.
            block_ids: Q ring block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (only the lmcache-driven path does).
            RuntimeError: If register_q() was not called first.
        """
        raise NotImplementedError(
            "Q ring store is not supported by this transfer context"
        )

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: Engine block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object, or ``None`` when the concrete
                context does not require one.
            blocks_in_chunk: Number of engine blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: Engine block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object, or ``None`` when the concrete
                context does not require one.
            blocks_in_chunk: Number of engine blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""

    @abstractmethod
    def flush_inflight_stores(self) -> None:
        """Synchronize any in-flight gather operations.

        Subclasses must implement this method. Contexts with no deferred
        operations should implement it as a no-op. Async contexts that
        defer GPU->CPU gather work must block until all in-flight stores
        have completed, so that vLLM cannot overwrite paged KV blocks
        before they are read.
        """


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (accelerator IPC,
    or SHM wrappers for CPU with IPC-like semantics) and the LMCache server
    performs direct device-side data transfer.
    """

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Initialize a handle-path context bound to one worker.

        Args:
            instance_id: Worker process instance identifier.
            req_client: Transport client for this worker.
        """
        super().__init__(instance_id, req_client)
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None
        self._mq_timeout: float = 0.0
        self._inflight_stores: list[MessagingFuture] = []
        self._inflight_lock = threading.Lock()

    @staticmethod
    def _store_settled(future: MessagingFuture) -> bool:
        """Whether the server is done with this store's engine KV blocks.

        ``query()`` raises when the store's RPC failed, so a failed store is
        reported as settled: it is no longer reading the blocks, and its error
        is surfaced by the request path that owns it rather than here.

        Args:
            future: A store future returned by ``submit_store``.

        Returns:
            True if the store completed or failed, False if still in flight.
        """
        try:
            return future.query()
        except Exception:
            logger.debug("Treating a failed store as settled", exc_info=True)
            return True

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        lmcache_chunk_tokens: int | None = None,
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            mq_timeout: Timeout for the registration response.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata.
            engine_type: Serving engine that produced the caches.

        Raises:
            RuntimeError: If event IPC is unsupported for the KV-cache device.
            ValueError: If ``kv_caches`` is empty.
        """
        device = _get_kv_device(kv_caches)
        event_backend = get_event_ipc_backend(device)
        event_backend.check_event_support(device)

        future = self._submit_registration(
            lambda: self._req_client.register_kv_cache(
                self._instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                engine_type,
                layout_hints or {},
                list(engine_group_infos),
            )
        )
        future.result(timeout=mq_timeout)
        if self._is_closed():
            return
        self._device = device
        self._event_backend = event_backend
        self._mq_timeout = mq_timeout

    def create_recorded_event(self) -> IPCEvent:
        """Create and record an exportable event for handle-based transfer.

        Returns:
            An interprocess-capable event recorded on the current stream.

        Raises:
            RuntimeError: If :meth:`register` has not completed.
        """
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before creating transfer events."
            )
        event = self._event_backend.create_event(self._device)
        self._event_backend.record_event(event, torch_dev.current_stream())
        return cast(IPCEvent, event)

    def unregister(self) -> MessagingFuture[Any] | None:
        """Start handle-path unregistration for this worker instance.

        Returns:
            A future for the server's unregister acknowledgement.

        """
        return self._submit_unregistration(
            lambda: self._req_client.unregister_kv_cache(self._instance_id)
        )

    def register_q(
        self,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        future = self._req_client.register_q_cache(
            self._instance_id,
            wrap_kv_caches(q_caches),
            model_name,
            world_size,
            EngineType.VLLM,
            layout_hints or {},
            list(engine_group_infos),
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a handle-based store ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the store range.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders reads of the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        if event is None:
            raise RuntimeError("LMCache-driven transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        future = self._req_client.store(
            key, self._instance_id, block_ids, event_ipc_handle
        ).to_device_future(
            device=self._device,
            event_backend=self._event_backend,
        )
        with self._inflight_lock:
            self._inflight_stores = [
                f for f in self._inflight_stores if not self._store_settled(f)
            ]
            self._inflight_stores.append(future)
        return future

    def submit_q_store(
        self,
        _request_id: str,
        key: Any,
        _q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_q_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._req_client.store_q(
            key, self._instance_id, block_ids, event_ipc_handle
        ).to_device_future(
            device=self._device,
            event_backend=self._event_backend,
        )

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a handle-based retrieve ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the retrieve range.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders writes to the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).
            skip_first_n_tokens: Initial tokens the server must not overwrite.

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        if event is None:
            raise RuntimeError("LMCache-driven transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._req_client.retrieve(
            key,
            self._instance_id,
            block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        ).to_device_future(
            device=self._device,
            event_backend=self._event_backend,
        )

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        self._mark_closed()
        self._device = None
        self._event_backend = None

    def flush_inflight_stores(self) -> None:
        """Block until the server has finished reading the engine KV blocks.

        In this mode the server copies the blocks on its own stream after the
        forward pass that produced them, and the engine never waits for that
        copy.  When the scheduler preempts a request it frees those blocks
        immediately and may hand them to another request in the same step,
        whose forward pass would overwrite blocks the server is still reading
        and commit the wrong KV under the preempted request's keys.

        A timeout is logged rather than raised, so a slow server degrades to a
        possibly stale store instead of a crashed engine.
        """
        with self._inflight_lock:
            pending = [f for f in self._inflight_stores if not self._store_settled(f)]
            self._inflight_stores = []
        for future in pending:
            try:
                if not future.wait(timeout=self._mq_timeout):
                    logger.warning(
                        "A store did not finish within %.1fs; its KV blocks may "
                        "be overwritten while the server is still reading them",
                        self._mq_timeout,
                    )
            except Exception:
                logger.exception("Failed waiting for an in-flight store")


class EngineDrivenTransferContext(TransferContext):
    """Transfer context for worker-managed KV copies.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    """

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Initialize an engine-driven context bound to one worker.

        Args:
            instance_id: Worker process instance identifier.
            req_client: Transport client for this worker.
        """
        super().__init__(instance_id, req_client)
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._engine_type = EngineType.VLLM
        # Multi-group worker state; empty means single-group (stage: filled
        # by register() for hybrid-KV models).
        self._group_states: list = []
        # Engine block ID the server treats as absent KV (from registration).
        self._null_block_id = 0
        # Tokens in one LMCache chunk (multi-group registration).
        self._chunk_tokens = 0

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/inline context created by ``register``.

        Raises:
            RuntimeError: If accessed before ``register`` has run.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "EngineDrivenTransferContext is not registered, call register() first."
            )
        return self._engine_driven_context

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        lmcache_chunk_tokens: int | None = None,
    ) -> None:
        """Register KV caches with the engine-driven context server.

        ``engine_type`` selects the KV layout normalization used during
        registration and subsequent gather/scatter operations.

        With multiple ``engine_group_infos`` or a single compressed group, the layers
        are partitioned by :class:`KVLayerGroupsManager` and each group is
        described by its own :class:`GroupLayout` and gathered/scattered with
        that group's block-id list (uniform coverage: every group stores and
        retrieves every chunk; sliding-window groups keep only the trailing
        window of each chunk). Works over both transports: SHM gathers into
        server-reserved slots, inline sends a typed group-major envelope.
        Uncompressed single-group registration keeps the legacy path (see
        ``_single_group_block_ids``). Compressed groups size their buffers in
        physical slots while block counts and windows remain in logical tokens.
        """
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(
            kv_caches,
            layout_hints=layout_hints,
            engine_type=engine_type,
        )
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format
        self._engine_type = engine_type

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1

        group_layouts: list[GroupLayout] = []
        group_states: list[_GroupState] = []
        if len(engine_group_infos) > 1 or (
            engine_group_infos and engine_group_infos[0].tokens_per_block > block_size
        ):
            # One LMCache chunk must cover a whole number of blocks of EVERY
            # group. Prefer the chunk size the adapter knows: blocks_in_chunk
            # is counted in the engine's global block size, and a pipeline
            # stage that does not hold the group with the largest block would
            # otherwise reconstruct a smaller chunk from its own groups' lcm
            # (0 = unreported, treated as the detected engine block size).
            local_lcm = lcm(
                *(group.tokens_per_block or block_size for group in engine_group_infos)
            )
            if lmcache_chunk_tokens is None:
                chunk_tokens = blocks_in_chunk * local_lcm
            else:
                if lmcache_chunk_tokens <= 0 or lmcache_chunk_tokens % local_lcm:
                    raise ValueError(
                        f"LMCache chunk of {lmcache_chunk_tokens} tokens is not "
                        "a positive multiple of this worker's group block "
                        f"sizes (lcm {local_lcm})"
                    )
                chunk_tokens = lmcache_chunk_tokens
            self._chunk_tokens = chunk_tokens
            layer_names = list(kv_caches)
            normalized_kv, per_layer_formats = normalize_and_discover_per_layer_formats(
                list(kv_caches.values()),
                [group.layer_indices for group in engine_group_infos],
                engine_type,
                layout_hints=layout_hints,
            )
            # The manager rejects a chunk/window that is not a whole multiple
            # of some group's block size, so no divisibility checks here.
            manager = KVLayerGroupsManager(
                normalized_kv,
                per_layer_formats,
                engine_group_infos,
                lmcache_tokens_per_chunk=chunk_tokens,
            )
            for gid in range(manager.num_kernel_groups):
                kernel_group = manager.kernel_groups[gid]
                assert kernel_group.engine_kv_format is not None
                tokens_per_block = (
                    kernel_group.tokens_per_block or kernel_group.slots_per_block
                )
                # Sliding-window groups store only the trailing window of each
                # chunk; for full attention the window is the whole chunk and
                # everything below collapses to full-coverage behaviour.
                window_tokens = manager.get_subchunk_sw_size_tokens(gid)
                window_slots = kernel_group.calculate_slots(window_tokens)
                # Trailing chunks this group needs for a hit; the server sizes
                # lookup/prefetch and the retrieve by it (-1 = full prefix).
                window_chunks = manager.get_kernel_group_window_chunks(gid)
                group_dtype_str = str(kernel_group.dtype).removeprefix("torch.")
                group_shape = (
                    torch.Size(
                        [
                            kernel_group.num_layers,
                            window_slots,
                            kernel_group.hidden_dim_size,
                        ]
                    )
                    if kernel_group.shape_desc.kv_size == 1
                    else torch.Size(
                        [
                            2,
                            kernel_group.num_layers,
                            window_slots,
                            kernel_group.hidden_dim_size,
                        ]
                    )
                )
                group_layouts.append(
                    GroupLayout(
                        num_layers=kernel_group.num_layers,
                        hidden_dim_size=kernel_group.hidden_dim_size,
                        dtype_str=group_dtype_str,
                        tokens_per_block=tokens_per_block,
                        window_tokens=window_tokens,
                        kv_size=kernel_group.shape_desc.kv_size,
                        window_chunks=max(window_chunks, 0),
                        recurrent=kernel_group.recurrent_state,
                        num_physical_slots=window_slots,
                    )
                )
                group_states.append(
                    _GroupState(
                        layer_names=[
                            layer_names[i] for i in kernel_group.layer_indices
                        ],
                        engine_kv_format=kernel_group.engine_kv_format,
                        blocks_in_chunk=chunk_tokens // tokens_per_block,
                        blocks_per_window=window_tokens // tokens_per_block,
                        layout_desc=MemoryLayoutDesc(
                            shapes=[group_shape], dtypes=[kernel_group.dtype]
                        ),
                        window_chunks=window_chunks,
                    )
                )
            # Group 0's layout doubles as the legacy top-level layout so
            # single-group readers of the payload keep working.
            layout_desc = group_states[0].layout_desc
            num_physical_slots = chunk_tokens
        else:
            shape = (
                torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
                if use_mla_flag
                else torch.Size(
                    [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
                )
            )
            dtype = getattr(torch, dtype_str)
            layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
            num_physical_slots = blocks_in_chunk * block_size
        self._group_states = group_states

        future = self._submit_registration(
            lambda: self._req_client.register_kv_cache_engine_driven_context(
                RegisterEngineDrivenContextPayload(
                    instance_id=self._instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    num_physical_slots=num_physical_slots,
                    group_layouts=group_layouts,
                )
            )
        )
        response = future.result(timeout=mq_timeout)
        if self._is_closed():
            return
        shm_name = ""
        pool_size = 0
        null_block_id = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size
            null_block_id = response.null_block_id
        self._null_block_id = null_block_id

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
            group_layouts=[state.layout_desc for state in group_states] or None,
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            self._req_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "inline"
        logger.info(
            "Worker engine-driven context registered "
            "(instance_id=%d, mode=%s, groups=%d)",
            self._instance_id,
            supported_transfer_mode,
            max(1, len(self._group_states)),
        )

    def unregister(self) -> MessagingFuture[Any] | None:
        """Start engine-driven unregistration for this worker instance.

        Returns:
            A future for the server's unregister acknowledgement.

        """
        return self._submit_unregistration(
            lambda: self._req_client.unregister_kv_cache_engine_driven_context(
                self._instance_id
            )
        )

    def create_recorded_event(self) -> IPCEvent | None:
        """Return no event for the synchronous engine-driven transfer path.

        Returns:
            ``None`` because store and retrieve synchronize the KV tensor device
            before accessing or releasing KV-cache buffers.

        Raises:
            RuntimeError: If :meth:`register` has not completed.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before creating transfer events."
            )
        return None

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent | None,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        if self._group_states:
            return self._submit_store_multigroup(key, kv_caches, block_ids)

        device = _get_kv_device(kv_caches)
        synchronize_device(device)
        result = self._engine_driven_context.prepare_store(key, self._instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        cpu_chunks = gather_paged_kv_to_cpu(
            kv_caches,
            _single_group_block_ids(block_ids),
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            engine_kv_format=self._engine_kv_format,
            out=out_buffers,
            chunk_indices=chunk_indices,
            engine_type=self._engine_type,
        )
        # Gather issues async device->CPU copies on BOTH transports: into the
        # SHM slots when out_buffers is given, otherwise into fresh buffers that
        # commit_store serializes immediately. Either way the copies must be
        # complete first, so this is unconditional -- guarding it on out_buffers
        # left the inline path serializing a buffer still being written.
        synchronize_device(device)
        ok = self._engine_driven_context.commit_store(
            key, self._instance_id, cpu_chunks
        )

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        if self._group_states:
            return self._submit_retrieve_multigroup(
                key, kv_caches, block_ids, event, skip_first_n_tokens
            )

        device = _get_kv_device(kv_caches)
        full_block_ids = _single_group_block_ids(block_ids)
        if len(full_block_ids) % blocks_in_chunk != 0:
            raise ValueError(
                "Engine block count must be divisible by blocks_in_chunk: "
                f"blocks={len(full_block_ids)}, blocks_in_chunk={blocks_in_chunk}"
            )
        expected_count = len(full_block_ids) // blocks_in_chunk
        src_buffers = self._engine_driven_context.prepare_retrieve(
            key, self._instance_id, expected_count
        )
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                if event is not None:
                    event.wait()
                scatter_cpu_to_paged_kv(
                    kv_caches,
                    full_block_ids,
                    src_buffers,
                    blocks_in_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    engine_kv_format=self._engine_kv_format,
                    engine_type=self._engine_type,
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            synchronize_device(device)
        self._engine_driven_context.commit_retrieve(key, self._instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def _group_slots(
        self,
        tensors: list[torch.Tensor],
        per_slot_group_ids: list[int],
        gid: int,
        chunk_indices: "list[int] | None" = None,
    ) -> "tuple[list[torch.Tensor], list[int] | None]":
        """Select group ``gid``'s slot tensors (and chunk indices) in order."""
        idxs = [i for i, g in enumerate(per_slot_group_ids) if g == gid]
        picked = [tensors[i] for i in idxs]
        if chunk_indices is None:
            return picked, None
        return picked, [chunk_indices[i] for i in idxs]

    def _null_chunk_indices(self, block_ids: list[list[int]]) -> list[list[int]]:
        """Per group, the chunks whose engine blocks are all the null block.

        Sliding-window groups lose their out-of-window blocks and align-mode
        recurrent (Mamba/linear-attention) groups keep only the block holding
        the last state, so their earlier chunks are entirely the null block.
        Those chunks carry no valid KV and must never be stored: object keys
        are prefix hashes, so a later shorter-prefix hit would restore them.
        Mirrors ``all_null_chunk_masks`` on the lmcache-driven path.
        """
        null_id = self._null_block_id
        skipped: list[list[int]] = []
        for gid, state in enumerate(self._group_states):
            bpc = state.blocks_in_chunk
            ids = block_ids[gid]
            skipped.append(
                [
                    c
                    for c in range(len(ids) // bpc)
                    if all(b == null_id for b in ids[c * bpc : (c + 1) * bpc])
                ]
            )
        return skipped

    def _check_group_block_ids(self, block_ids: list[list[int]]) -> None:
        """Require one block-id list per registered group."""
        if len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {len(block_ids)} block-id lists for "
                f"{len(self._group_states)} registered groups"
            )

    def _prepare_multigroup_store(
        self,
        ctx: "EngineDrivenContext",
        key: Any,
        block_ids: list[list[int]],
        alloc: "_StagingAlloc | None" = None,
    ) -> "list[_GroupStorePlan] | None":
        """Decide what each group stores and where it gathers to.

        All-null chunks are left out on both transports. Over SHM the server
        reserves slots for the rest (and none for skipped chunks) and they are
        the gather targets. Inline has no slots: ``alloc``, when given,
        provides each group's gather targets (the async path's pinned
        staging), otherwise gather allocates fresh CPU tensors.

        Returns:
            One plan per group with chunks to store (an empty list when
            nothing needs storing), or ``None`` on a malformed SHM response.
        """
        skipped = self._null_chunk_indices(block_ids)
        if not isinstance(ctx, EngineDrivenContextInline):
            result = ctx.prepare_store_grouped(
                key,
                self._instance_id,
                tuple(tuple(g) for g in skipped) if any(skipped) else (),
            )
            if result is None:
                return None
            tensors, chunk_indices, group_ids = result
            plans: list[_GroupStorePlan] = []
            for gid in range(len(self._group_states)):
                out_g, chunks_g = self._group_slots(
                    tensors, group_ids, gid, chunk_indices
                )
                if out_g:
                    assert chunks_g is not None
                    plans.append(_GroupStorePlan(gid, chunks_g, out_g))
            return plans

        plans = []
        for gid, state in enumerate(self._group_states):
            skip = set(skipped[gid])
            num_chunks = len(block_ids[gid]) // state.blocks_in_chunk
            indices = [c for c in range(num_chunks) if c not in skip]
            out = (
                alloc(
                    state.layout_desc.shapes[0],
                    state.layout_desc.dtypes[0],
                    len(indices),
                )
                if alloc is not None and indices
                else None
            )
            plans.append(_GroupStorePlan(gid, indices, out))
        if not any(plan.chunk_indices for plan in plans):
            return []
        # Handshake only: the inline strategy reserves nothing at prepare.
        ctx.prepare_store(key, self._instance_id)
        return plans

    def _gather_multigroup(
        self,
        plans: "list[_GroupStorePlan]",
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
    ) -> list[list[torch.Tensor]]:
        """Enqueue each planned group's device->CPU copies; one list per plan."""
        gathered: list[list[torch.Tensor]] = []
        for plan in plans:
            state = self._group_states[plan.gid]
            if not plan.chunk_indices:
                gathered.append([])
                continue
            gathered.append(
                gather_paged_kv_to_cpu(
                    {name: kv_caches[name] for name in state.layer_names},
                    block_ids[plan.gid],
                    state.blocks_in_chunk,
                    layout_hints=self._layout_hints,
                    engine_kv_format=state.engine_kv_format,
                    out=plan.out,
                    chunk_indices=plan.chunk_indices,
                    blocks_per_window=state.blocks_per_window,
                    engine_type=self._engine_type,
                )
            )
        return gathered

    def _commit_multigroup(
        self,
        ctx: "EngineDrivenContext",
        key: Any,
        plans: "list[_GroupStorePlan]",
        gathered: list[list[torch.Tensor]],
    ) -> bool:
        """Commit a gathered multi-group store; the copies must be complete."""
        if not isinstance(ctx, EngineDrivenContextInline):
            return ctx.commit_store(key, self._instance_id, [])
        return ctx.commit_store(
            key,
            self._instance_id,
            [
                (plan.chunk_indices, chunks)
                for plan, chunks in zip(plans, gathered, strict=True)
            ],
        )

    def _store_multigroup(
        self,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
    ) -> bool:
        """Run a multi-group store to completion on the calling thread."""
        ctx = self._engine_driven_context
        assert ctx is not None
        self._check_group_block_ids(block_ids)
        device = _get_kv_device(kv_caches)
        synchronize_device(device)
        plans = self._prepare_multigroup_store(ctx, key, block_ids)
        if plans is None:
            return False
        if not plans:
            return True
        gathered = self._gather_multigroup(plans, kv_caches, block_ids)
        # Gather issues async device->CPU copies (into SHM slots, or into
        # buffers inline commit serializes at once): complete them first.
        synchronize_device(device)
        return self._commit_multigroup(ctx, key, plans, gathered)

    def _submit_store_multigroup(
        self,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
    ) -> MessagingFuture:
        """Synchronous multi-group store; resolves before returning."""
        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(self._store_multigroup(key, kv_caches, block_ids))
        return future

    def _group_windows(self, block_ids: list[list[int]]) -> list[tuple[int, int]]:
        """Per group ``(first_chunk, num_chunks)`` a retrieve serves.

        A group with a window of ``w`` chunks gets only its last ``w`` chunks,
        matching the server's prefetch window; full-attention groups get all.
        """
        windows: list[tuple[int, int]] = []
        for gid, state in enumerate(self._group_states):
            if len(block_ids[gid]) % state.blocks_in_chunk != 0:
                raise ValueError(
                    "Engine block count must be divisible by blocks_in_chunk: "
                    f"group={gid}, blocks={len(block_ids[gid])}, "
                    f"blocks_in_chunk={state.blocks_in_chunk}"
                )
            num_chunks = len(block_ids[gid]) // state.blocks_in_chunk
            window = state.window_chunks
            first = 0 if window < 0 else max(0, num_chunks - window)
            windows.append((first, num_chunks))
        return windows

    def _scatter_group(
        self,
        kv_caches: dict[str, torch.Tensor],
        gid: int,
        group_block_ids: list[int],
        chunks: list[torch.Tensor],
        first_chunk: int,
        skip_first_n_tokens: int,
    ) -> None:
        """Scatter one group's chunks starting at chunk ``first_chunk``.

        The block ids are cut at that chunk boundary, and the already-computed
        token prefix is shifted by the dropped chunks.
        """
        state = self._group_states[gid]
        scatter_cpu_to_paged_kv(
            {name: kv_caches[name] for name in state.layer_names},
            group_block_ids[first_chunk * state.blocks_in_chunk :],
            chunks,
            state.blocks_in_chunk,
            skip_first_n_tokens=max(
                0, skip_first_n_tokens - first_chunk * self._chunk_tokens
            ),
            layout_hints=self._layout_hints,
            engine_kv_format=state.engine_kv_format,
            blocks_per_window=state.blocks_per_window,
            engine_type=self._engine_type,
            tokens_per_block=self._chunk_tokens // state.blocks_in_chunk,
        )

    def _submit_retrieve_multigroup_inline(
        self,
        ctx: "EngineDrivenContextInline",
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        skip_first_n_tokens: int,
    ) -> MessagingFuture:
        """Retrieve over inline: scatter each group's in-window chunks.

        Each group's payload is validated against that group's registered
        layout, and its chunk positions must be exactly the group's window.
        """
        future: MessagingFuture[bool] = MessagingFuture()
        device = _get_kv_device(kv_caches)
        windows = self._group_windows(block_ids)
        expected = [
            (state.layout_desc.shapes[0], state.layout_desc.dtypes[0], num_chunks)
            for state, (_, num_chunks) in zip(self._group_states, windows, strict=True)
        ]
        groups = ctx.prepare_retrieve_multigroup(key, self._instance_id, expected)
        if groups is not None and (
            len(groups) != len(self._group_states)
            or any(
                indices != list(range(first, num_chunks))
                for (indices, _), (first, num_chunks) in zip(
                    groups, windows, strict=True
                )
            )
        ):
            logger.error(
                "inline retrieve returned chunks that do not match the "
                "registered group windows (groups=%d)",
                len(groups),
            )
            groups = None
        ok = groups is not None
        if groups is not None:
            try:
                if event is not None:
                    event.wait()
                for gid, ((_, chunks), (first, _)) in enumerate(
                    zip(groups, windows, strict=True)
                ):
                    self._scatter_group(
                        kv_caches,
                        gid,
                        block_ids[gid],
                        chunks,
                        first,
                        skip_first_n_tokens,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            synchronize_device(device)
        ctx.commit_retrieve(key, self._instance_id)
        future.set_result(ok)
        return future

    def _submit_retrieve_multigroup(
        self,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        skip_first_n_tokens: int,
    ) -> MessagingFuture:
        """Retrieve: scatter each group's in-window chunks from its slots."""
        ctx = self._engine_driven_context
        assert ctx is not None
        future: MessagingFuture[bool] = MessagingFuture()
        self._check_group_block_ids(block_ids)
        if isinstance(ctx, EngineDrivenContextInline):
            return self._submit_retrieve_multigroup_inline(
                ctx, key, kv_caches, block_ids, event, skip_first_n_tokens
            )
        device = _get_kv_device(kv_caches)
        windows = self._group_windows(block_ids)
        result = ctx.prepare_retrieve_grouped(key, self._instance_id)
        ok = result is not None
        if result is not None:
            tensors, group_ids, chunk_indices = result
            try:
                per_group = [
                    self._group_slots(tensors, group_ids, gid, chunk_indices)
                    for gid in range(len(self._group_states))
                ]
                for (_, got), (first, num_chunks) in zip(
                    per_group, windows, strict=True
                ):
                    if got != list(range(first, num_chunks)):
                        raise ValueError(
                            "SHM retrieve chunks do not match the registered "
                            f"group window: expected {first}..{num_chunks - 1}, "
                            f"got {got}"
                        )
                if event is not None:
                    event.wait()
                for gid, ((src_g, _), (first, _)) in enumerate(
                    zip(per_group, windows, strict=True)
                ):
                    self._scatter_group(
                        kv_caches,
                        gid,
                        block_ids[gid],
                        src_g,
                        first,
                        skip_first_n_tokens,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # Complete device writes before the server may reuse the slots.
            synchronize_device(device)
        ctx.commit_retrieve(key, self._instance_id)
        future.set_result(ok)
        return future

    def close(self) -> None:
        self._mark_closed()
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def flush_inflight_stores(self) -> None:
        pass


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    *,
    instance_id: int,
    req_client: RequestClient,
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.
    ``auto`` resolves to the device spec's declared default
    (:meth:`DeviceSpec.default_mp_transfer_mode`).

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        instance_id: Worker process instance identifier bound to the context.
        req_client: Transport client bound to the context. The caller retains
            ownership and must close it after the context.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO` (the device-declared default).
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {get_device(v).type for v in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    if resolved_mode is MPTransferMode.AUTO:
        device_spec = get_device_spec(device_type)
        resolved_mode = (
            device_spec.default_mp_transfer_mode()
            if device_spec is not None
            else MPTransferMode.ENGINE_DRIVEN
        )
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type, instance_id, req_client)
    return _build_engine_driven_context(instance_id, req_client)
