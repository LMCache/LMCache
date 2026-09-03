# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import lcm
from typing import Any, Protocol, cast
import os

# Third Party
import torch

# First Party
import lmcache.lmcache_native as lmcache_native

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
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import get_device_spec, resolve_kv_wrapper_factory
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.kv_wrap import wrap_kv_caches

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
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


def _build_engine_driven_context() -> "TransferContext":
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
        return AsyncEngineDrivenTransferContext()

    logger.info("Using EngineDrivenTransferContext (sync) for store path")
    return EngineDrivenTransferContext()


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


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


def _build_lmcache_driven_context(device_type: str) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        resolve_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext()


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
    """

    layer_names: list[str]
    engine_kv_format: "lmcache_native.EngineKVFormat"
    blocks_in_chunk: int
    blocks_per_window: int
    layout_desc: MemoryLayoutDesc


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
        raise ValueError("LMCache-driven transfer requires at least one KV cache")
    return get_device(next(iter(kv_caches.values())))


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. Device-handle paths
    return event-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        req_client: RequestClient,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            req_client: Transport-neutral client used for server requests.
            mq_timeout: Timeout in seconds for synchronous request wait.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            engine_type: Serving engine that produced the caches. Only
                consumed by the handle path; adapters should pass their
                own :class:`EngineType` so this transport stays engine-
                neutral. Defaults to :attr:`EngineType.VLLM` for
                backwards compatibility.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        req_client: RequestClient,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register the paged Q ring with the server under the same worker
        instance_id but different model_name (model_name##query).

        Args:
            instance_id: Worker process instance identifier.
            q_caches: Worker Q cache tensors keyed by layer name.
            model_name: Model name used by cache keys (model_name##query).
            world_size: KV world size.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.
            req_client: Transport-neutral client used for server requests.
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
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a Q ring store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key for the Q store range (query-specific model_name).
            instance_id: Worker process instance identifier (shared with KV).
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
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object, or ``None`` when the concrete
                context does not require one.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

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
        instance_id: int,
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
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object, or ``None`` when the concrete
                context does not require one.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
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

    def __init__(self) -> None:
        self._req_client: RequestClient | None = None
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        req_client: RequestClient,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            req_client: Transport-neutral client used for server requests.
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

        self._req_client = req_client
        future = req_client.register_kv_cache(
            instance_id,
            wrap_kv_caches(kv_caches),
            model_name,
            world_size,
            engine_type,
            layout_hints,
            list(engine_group_infos),
        )
        future.result(timeout=mq_timeout)
        self._device = device
        self._event_backend = event_backend

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

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        req_client: RequestClient,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        self._req_client = req_client
        future = req_client.register_q_cache(
            instance_id,
            wrap_kv_caches(q_caches),
            model_name,
            world_size,
            EngineType.VLLM,
            layout_hints,
            list(engine_group_infos),
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a handle-based store ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the store range.
            instance_id: Worker process instance identifier.
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
        if (
            self._req_client is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        if event is None:
            raise RuntimeError("LMCache-driven transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._req_client.store(
            key, instance_id, block_ids, event_ipc_handle
        ).to_device_future(device=self._device)

    def submit_q_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if (
            self._req_client is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_q_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._req_client.store_q(
            key, instance_id, block_ids, event_ipc_handle
        ).to_device_future(device=self._device)

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
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
            instance_id: Worker process instance identifier.
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
        if (
            self._req_client is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        if event is None:
            raise RuntimeError("LMCache-driven transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._req_client.retrieve(
            key,
            instance_id,
            block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        ).to_device_future(device=self._device)

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        self._req_client = None
        self._device = None
        self._event_backend = None

    def flush_inflight_stores(self) -> None:
        pass


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        # Multi-group worker state; empty means single-group (stage: filled
        # by register() for hybrid-KV models).
        self._group_states: list = []

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/pickle context created by ``register``.

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
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        req_client: RequestClient,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the non-GPU context server.

        With multiple ``engine_group_infos`` (hybrid-KV models), the layers
        are partitioned by :class:`KVLayerGroupsManager` and each group is
        described by its own :class:`GroupLayout` and gathered/scattered with
        that group's block-id list (uniform coverage: every group stores and
        retrieves every chunk; sliding-window groups keep only the trailing
        window of each chunk). Works over both transports: SHM gathers into
        server-reserved slots, pickle serializes a group-major payload.
        Single-group registration keeps the legacy path (see
        ``_single_group_block_ids``).
        """
        # TODO: per-group compression (tokens_per_block > the detected slot
        # count, e.g. DeepSeek V4 indexer pages) is validated by the manager
        # but not yet threaded through gather/scatter buffer sizing.
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1

        group_layouts: list[GroupLayout] = []
        group_states: list[_GroupState] = []
        if len(engine_group_infos) > 1:
            # The engine schedules block ids in units of every group's block
            # size at once, so one LMCache chunk must cover a whole number of
            # blocks of EVERY group: chunk tokens are ``blocks_in_chunk``
            # multiples of the lcm of the group block sizes (0 = unreported,
            # treated as the detected engine block size).
            chunk_tokens = blocks_in_chunk * lcm(
                *(group.tokens_per_block or block_size for group in engine_group_infos)
            )
            layer_names = list(kv_caches)
            normalized_kv, per_layer_formats = normalize_and_discover_per_layer_formats(
                kv_caches,
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
                group_dtype_str = str(kernel_group.dtype).removeprefix("torch.")
                group_shape = (
                    torch.Size(
                        [
                            kernel_group.num_layers,
                            window_tokens,
                            kernel_group.hidden_dim_size,
                        ]
                    )
                    if kernel_group.shape_desc.kv_size == 1
                    else torch.Size(
                        [
                            2,
                            kernel_group.num_layers,
                            window_tokens,
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

        future = req_client.register_kv_cache_engine_driven_context(
            RegisterEngineDrivenContextPayload(
                instance_id=instance_id,
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
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
            group_layouts=[state.layout_desc for state in group_states] or None,
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            req_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered "
            "(instance_id=%d, mode=%s, groups=%d)",
            instance_id,
            supported_transfer_mode,
            max(1, len(self._group_states)),
        )

    def create_recorded_event(self) -> IPCEvent | None:
        """Return no event for the synchronous engine-driven transfer path.

        Returns:
            ``None`` because store and retrieve synchronize the active device
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
        instance_id: int,
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
            return self._submit_store_multigroup(key, instance_id, kv_caches, block_ids)

        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, instance_id)
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
        )
        # Gather issues async device->CPU copies on BOTH transports: into the
        # SHM slots when out_buffers is given, otherwise into fresh buffers that
        # commit_store serializes immediately. Either way the copies must be
        # complete first, so this is unconditional -- guarding it on out_buffers
        # left the pickle path serializing a buffer still being written.
        torch_dev.synchronize()
        ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent | None,
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
                key, instance_id, kv_caches, block_ids, skip_first_n_tokens
            )

        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                scatter_cpu_to_paged_kv(
                    kv_caches,
                    _single_group_block_ids(block_ids),
                    src_buffers,
                    blocks_in_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    engine_kv_format=self._engine_kv_format,
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

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

    def _submit_store_multigroup(
        self,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
    ) -> MessagingFuture:
        """Uniform-coverage store: gather every group's chunks into its slots."""
        ctx = self._engine_driven_context
        assert ctx is not None
        future: MessagingFuture[bool] = MessagingFuture()
        if len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {len(block_ids)} block-id lists for "
                f"{len(self._group_states)} registered groups"
            )
        if isinstance(ctx, EngineDrivenContextPickle):
            return self._submit_store_multigroup_pickle(
                ctx, key, instance_id, kv_caches, block_ids
            )
        torch_dev.synchronize()
        result = ctx.prepare_store_grouped(key, instance_id)
        if result is None:
            future.set_result(False)
            return future
        tensors, chunk_indices, group_ids = result
        if not tensors:
            future.set_result(True)
            return future
        for gid, state in enumerate(self._group_states):
            out_g, chunks_g = self._group_slots(tensors, group_ids, gid, chunk_indices)
            if not out_g:
                continue
            gather_paged_kv_to_cpu(
                {name: kv_caches[name] for name in state.layer_names},
                block_ids[gid],
                state.blocks_in_chunk,
                layout_hints=self._layout_hints,
                engine_kv_format=state.engine_kv_format,
                out=out_g,
                chunk_indices=chunks_g,
                blocks_per_window=state.blocks_per_window,
            )
        # SHM writes are async device->CPU copies; complete them before commit.
        torch_dev.synchronize()
        ok = ctx.commit_store(key, instance_id, [])
        future.set_result(ok)
        return future

    def _submit_store_multigroup_pickle(
        self,
        ctx: "EngineDrivenContextPickle",
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
    ) -> MessagingFuture:
        """Uniform-coverage store over pickle: group-major serialized payload.

        There are no server-reserved slots in pickle mode, so each group's
        chunks are gathered into fresh CPU tensors and the whole
        ``chunks[group][chunk]`` list is pickled in one COMMIT_STORE payload.
        """
        future: MessagingFuture[bool] = MessagingFuture()
        torch_dev.synchronize()
        # Handshake only: the pickle strategy reserves nothing at prepare.
        ctx.prepare_store(key, instance_id)
        group_chunks: list[list[torch.Tensor]] = []
        for gid, state in enumerate(self._group_states):
            group_chunks.append(
                gather_paged_kv_to_cpu(
                    {name: kv_caches[name] for name in state.layer_names},
                    block_ids[gid],
                    state.blocks_in_chunk,
                    layout_hints=self._layout_hints,
                    engine_kv_format=state.engine_kv_format,
                    blocks_per_window=state.blocks_per_window,
                )
            )
        # Gather issues async device->CPU copies; commit_store serializes the
        # buffers immediately, so the copies must be complete first.
        torch_dev.synchronize()
        ok = ctx.commit_store(key, instance_id, group_chunks)
        future.set_result(ok)
        return future

    def _submit_retrieve_multigroup_pickle(
        self,
        ctx: "EngineDrivenContextPickle",
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        skip_first_n_tokens: int,
    ) -> MessagingFuture:
        """Uniform-coverage retrieve over pickle: scatter a group-major payload."""
        future: MessagingFuture[bool] = MessagingFuture()
        group_chunks = ctx.prepare_retrieve_multigroup(key, instance_id)
        if group_chunks is not None and len(group_chunks) != len(self._group_states):
            logger.error(
                "pickle retrieve returned %d groups for %d registered groups",
                len(group_chunks),
                len(self._group_states),
            )
            group_chunks = None
        ok = group_chunks is not None
        if group_chunks is not None:
            try:
                for gid, state in enumerate(self._group_states):
                    scatter_cpu_to_paged_kv(
                        {name: kv_caches[name] for name in state.layer_names},
                        block_ids[gid],
                        group_chunks[gid],
                        state.blocks_in_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        engine_kv_format=state.engine_kv_format,
                        blocks_per_window=state.blocks_per_window,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            torch_dev.synchronize()
        ctx.commit_retrieve(key, instance_id)
        future.set_result(ok)
        return future

    def _submit_retrieve_multigroup(
        self,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        skip_first_n_tokens: int,
    ) -> MessagingFuture:
        """Uniform-coverage retrieve: scatter every group's chunks from its slots."""
        ctx = self._engine_driven_context
        assert ctx is not None
        future: MessagingFuture[bool] = MessagingFuture()
        if len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {len(block_ids)} block-id lists for "
                f"{len(self._group_states)} registered groups"
            )
        if isinstance(ctx, EngineDrivenContextPickle):
            return self._submit_retrieve_multigroup_pickle(
                ctx, key, instance_id, kv_caches, block_ids, skip_first_n_tokens
            )
        result = ctx.prepare_retrieve_grouped(key, instance_id)
        ok = result is not None
        if result is not None:
            tensors, group_ids = result
            try:
                for gid, state in enumerate(self._group_states):
                    src_g, _ = self._group_slots(tensors, group_ids, gid)
                    scatter_cpu_to_paged_kv(
                        {name: kv_caches[name] for name in state.layer_names},
                        block_ids[gid],
                        src_g,
                        state.blocks_in_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        engine_kv_format=state.engine_kv_format,
                        blocks_per_window=state.blocks_per_window,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # Complete device writes before the server may reuse the slots.
            torch_dev.synchronize()
        ctx.commit_retrieve(key, instance_id)
        future.set_result(ok)
        return future

    def close(self) -> None:
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def flush_inflight_stores(self) -> None:
        pass


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
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
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return _build_engine_driven_context()
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext()
    return _build_engine_driven_context()
