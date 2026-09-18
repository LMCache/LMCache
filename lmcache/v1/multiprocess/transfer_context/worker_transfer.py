# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Protocol, cast
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints, get_device
from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
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
    When any of these is unavailable, the factory falls back to the
    synchronous :class:`EngineDrivenTransferContext`.

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
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext(instance_id, req_client)


class IPCEvent(Protocol):
    """Protocol for device events used by transport operations."""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (async ordering primitive)."""


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _select_kv_caches_by_index(
    kv_caches: dict[str, torch.Tensor],
    layer_indices: frozenset[int],
) -> dict[str, torch.Tensor]:
    """Return the subset of ``kv_caches`` at positions in ``layer_indices``.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name, in
            registration order.
        layer_indices: Registered KV tensor positions to keep. Empty selects
            every layer (the single-group fallback).

    Returns:
        Only the layers at positions in ``layer_indices``, or ``kv_caches``
        itself when ``layer_indices`` is empty.
    """
    if not layer_indices:
        return kv_caches
    return {
        name: tensor
        for idx, (name, tensor) in enumerate(kv_caches.items())
        if idx in layer_indices
    }


def _kv_caches_for_group(
    kv_caches: dict[str, torch.Tensor],
    group_info: EngineGroupInfo | None,
) -> dict[str, torch.Tensor]:
    """Return the subset of ``kv_caches`` registered for one LMCache group.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name, in
            registration order.
        group_info: The LMCache group to filter for, or ``None`` for the
            single-group (non-hybrid) fallback, which returns ``kv_caches``
            unchanged.

    Returns:
        A dict containing only the layers in ``group_info.layer_indices``,
        or all of ``kv_caches`` when ``group_info`` is ``None``.
    """
    if group_info is None:
        return kv_caches
    return _select_kv_caches_by_index(kv_caches, frozenset(group_info.layer_indices))


def _blocks_per_chunk_for_group(
    group_info: EngineGroupInfo | None,
    default_blocks_in_chunk: int,
    default_block_size: int,
) -> int:
    """Return one LMCache chunk's block count for a group's own block size.

    Args:
        group_info: The LMCache group, or ``None`` for the single-group
            fallback, which returns ``default_blocks_in_chunk`` unchanged.
        default_blocks_in_chunk: Blocks per LMCache chunk for the default
            (single-group) block size.
        default_block_size: Tokens per paged block for the default
            (single-group) layout, used to recover the LMCache chunk size in
            tokens (``default_blocks_in_chunk * default_block_size``).

    Returns:
        ``default_blocks_in_chunk`` when ``group_info`` is ``None`` or does
        not report ``tokens_per_block``; otherwise the number of this
        group's own paged blocks needed to cover one LMCache chunk.

    Raises:
        ValueError: If the LMCache chunk size in tokens is not a multiple of
            the group's ``tokens_per_block``.
    """
    if group_info is None or group_info.tokens_per_block <= 0:
        return default_blocks_in_chunk
    chunk_size_tokens = default_blocks_in_chunk * default_block_size
    if chunk_size_tokens % group_info.tokens_per_block != 0:
        raise ValueError(
            f"LMCache chunk size {chunk_size_tokens} must be a multiple of "
            f"group tokens_per_block {group_info.tokens_per_block}"
        )
    return chunk_size_tokens // group_info.tokens_per_block


def _group_chunk_shape(
    group_info: EngineGroupInfo | None,
    default_layout_desc: MemoryLayoutDesc,
    default_num_layers: int,
    default_hidden_dim_size: int,
    group_kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None,
) -> torch.Size:
    """Return one chunk's tensor shape for a group's own layer count and
    hidden dimension.

    Non-recurrent groups also get their own recomputed ``hidden_dim_size``,
    since hybrid groups can pack K/V differently and reusing the default
    group's width can mis-size this group's SHM chunk buffer. Recurrent
    groups skip that recompute: their tensor is a synthetic addressing view
    whose split has no real hidden-dim semantics to recompute from.

    Args:
        group_info: The LMCache group, or ``None`` for the single-group
            fallback, which returns ``default_layout_desc.shapes[0]`` unchanged.
        default_layout_desc: The default (single-group) layout descriptor
            computed at ``register()`` time.
        default_num_layers: The default (single-group) layer count that
            ``default_layout_desc.shapes[0]`` was built from.
        default_hidden_dim_size: The default (single-group) hidden dimension
            that ``default_layout_desc.shapes[0]`` was built from.
        group_kv_caches: This group's own KV-cache tensor subset.
        layout_hints: Optional engine layout hints.

    Returns:
        This group's chunk shape, with its own layer count and (for
        non-recurrent groups) hidden dimension substituted in.
    """
    default_shape = default_layout_desc.shapes[0]
    if group_info is None or not group_info.layer_indices:
        return default_shape
    num_layers = len(group_info.layer_indices)
    if group_info.recurrent_state:
        if num_layers == default_num_layers:
            return default_shape
        layer_dim = 0 if len(default_shape) == 3 else 1
        dims = list(default_shape)
        dims[layer_dim] = num_layers
        return torch.Size(dims)
    _, _, hidden_dim_size, _, _, _ = compute_kv_layout(
        group_kv_caches, layout_hints=layout_hints
    )
    if num_layers == default_num_layers and hidden_dim_size == default_hidden_dim_size:
        return default_shape
    layer_dim = 0 if len(default_shape) == 3 else 1
    dims = list(default_shape)
    dims[layer_dim] = num_layers
    dims[-1] = hidden_dim_size
    return torch.Size(dims)


@dataclass(frozen=True)
class GroupTransferPlan:
    """One LMCache group's registration-time constants for gather / scatter.

    Resolved once at ``register`` time instead of recomputed per transfer,
    which was ``O(num_groups * num_layers^2)`` plus a format-detection call
    per group. KV tensors are deliberately not cached here, since retrieve
    may scatter into a different mapping than the one registered --
    :meth:`select_kv_caches` applies the cached layer selection to whatever
    mapping the caller passes.

    Attributes:
        group_info: The LMCache group this plan describes, or ``None`` for
            the single-group (non-hybrid) fallback.
        layer_indices: Registered KV tensor positions this group owns, as a
            set for O(1) membership. Empty selects every layer (the
            single-group fallback).
        blocks_per_chunk: Paged blocks of this group's own block size needed
            to cover one LMCache chunk.
        chunk_shape: One chunk's tensor shape for this group's layer count.
        engine_kv_format: This group's pre-detected KV format, passed to
            gather / scatter so they skip re-detection. ``None`` when it
            could not be resolved up front, which makes them detect it per
            transfer as before.
    """

    group_info: EngineGroupInfo | None
    layer_indices: frozenset[int]
    blocks_per_chunk: int
    chunk_shape: torch.Size
    engine_kv_format: Any

    def select_kv_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return this group's subset of ``kv_caches``.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name, in the
                same registration order the plan was built from. Retrieve
                may pass a different mapping than ``register`` did, so the
                selection is applied to the argument rather than to a cached
                subset.

        Returns:
            Only the layers this group owns, or ``kv_caches`` itself for the
            single-group fallback.
        """
        return _select_kv_caches_by_index(kv_caches, self.layer_indices)


@dataclass(frozen=True)
class GroupChunkSelection:
    """One LMCache group's slice of a flat, group-major chunk sequence.

    Rebases the server's flat, group-major ``chunk_indices`` (and, on the SHM
    path, ``out_buffers``) to be relative to one group.

    Attributes:
        num_group_chunks: Total chunks this group spans, selected or not.
            Advances the caller's group offset even when nothing is selected.
        chunk_indices: This group's selected chunk indices, rebased to be
            group-local. ``None`` when the server selected nothing explicitly
            (store every chunk of every group).
        out_indices: Positions in the flat ``out_buffers`` / output sequence
            that ``chunk_indices`` correspond to, in the same order.
    """

    num_group_chunks: int
    chunk_indices: list[int] | None
    out_indices: list[int]

    @property
    def is_empty(self) -> bool:
        """Whether this group has no chunks to gather and can be skipped."""
        return self.chunk_indices is not None and not self.chunk_indices


def null_chunk_mask_from_groups(
    transfer_groups: Sequence[tuple["GroupTransferPlan", Any, list[int]]],
) -> tuple[tuple[bool, ...], ...]:
    """Compute each group's per-chunk null-block mask for a store.

    vLLM nulls out-of-window block-table entries (id ``0``) for recurrent
    and sliding-window groups; those chunks carry no valid KV and must not
    be stored. Every group is checked the same way, so full-attention
    groups (never nulled) just yield an all-``False`` mask.

    Args:
        transfer_groups: This store's per-group plans paired with their flat
            block-id lists, as yielded by
            :meth:`EngineDrivenTransferContext.iter_transfer_groups`.

    Returns:
        One tuple of per-chunk flags per group, in the same (protocol)
        order as ``transfer_groups``.
    """
    mask: list[tuple[bool, ...]] = []
    for plan, _group_kv_caches, group_block_ids in transfer_groups:
        bpc = plan.blocks_per_chunk
        num_chunks = len(group_block_ids) // bpc
        mask.append(
            tuple(
                not any(group_block_ids[i * bpc : (i + 1) * bpc])
                for i in range(num_chunks)
            )
        )
    return tuple(mask)


def _select_group_chunks(
    flat_chunk_indices: list[int] | None,
    group_offset: int,
    num_group_chunks: int,
) -> GroupChunkSelection:
    """Rebase a flat, group-major chunk selection onto one group.

    Args:
        flat_chunk_indices: The server's chunk selection, flat over every
            group's chunks group-major, or ``None`` to select all of them.
        group_offset: Number of chunks belonging to earlier groups, i.e. the
            flat index at which this group's chunks start.
        num_group_chunks: Number of chunks this group spans.

    Returns:
        The selection restricted and rebased to this group.
    """
    if flat_chunk_indices is None:
        return GroupChunkSelection(
            num_group_chunks=num_group_chunks,
            chunk_indices=None,
            out_indices=list(range(num_group_chunks)),
        )
    group_end = group_offset + num_group_chunks
    selected = [
        (out_idx, chunk_idx - group_offset)
        for out_idx, chunk_idx in enumerate(flat_chunk_indices)
        if group_offset <= chunk_idx < group_end
    ]
    return GroupChunkSelection(
        num_group_chunks=num_group_chunks,
        chunk_indices=[chunk_idx for _, chunk_idx in selected],
        out_indices=[out_idx for out_idx, _ in selected],
    )


def _select_live_scatter_chunks(
    group_block_ids: list[int],
    blocks_per_chunk: int,
    group_mask: tuple[bool, ...] | None,
) -> tuple[list[int], int]:
    """Drop a group's null-masked chunks from its block IDs before scatter.

    Args:
        group_block_ids: This group's flat block IDs for the request.
        blocks_per_chunk: Paged blocks of this group's own block size needed
            to cover one LMCache chunk.
        group_mask: This group's null-chunk mask (``True`` = drop), or falsy
            (``None`` or empty) to keep every block ID.

    Returns:
        ``(scatter_block_ids, num_live_chunks)``: the block IDs with masked
        chunks' blocks removed, and the resulting chunk count -- both needed
        to line up positionally with the server's already-shortened
        ``group_chunks`` (it never queried the masked chunks' keys).
    """
    num_group_chunks = len(group_block_ids) // blocks_per_chunk
    if not group_mask:
        return group_block_ids, num_group_chunks
    scatter_block_ids = [
        block_id
        for idx, block_id in enumerate(group_block_ids)
        if not (
            idx // blocks_per_chunk < len(group_mask)
            and group_mask[idx // blocks_per_chunk]
        )
    ]
    return scatter_block_ids, num_group_chunks - sum(group_mask)


def _detect_group_kv_format(
    group_kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None,
) -> Any:
    """Detect one group's KV format, or return ``None`` if detection fails.

    Hybrid models may mix per-group KV formats, so each group's format is
    discovered from its own tensor subset rather than reusing the
    model-wide value.

    Args:
        group_kv_caches: This group's KV-cache subset, keyed by layer name.
        layout_hints: Optional engine layout hints.

    Returns:
        The group's ``EngineKVFormat``, or ``None`` when it cannot be
        detected up front.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format

    if not group_kv_caches:
        return None
    try:
        engine_kv_format, _normalized = normalize_kv_and_discover_format(
            list(group_kv_caches.values()), EngineType.VLLM, layout_hints=layout_hints
        )
    except (ValueError, RuntimeError, TypeError, IndexError):
        logger.warning(
            "Could not pre-detect the KV format for a group of %d layer(s); "
            "gather / scatter will detect it per transfer.",
            len(group_kv_caches),
            exc_info=True,
        )
        return None
    return engine_kv_format


def _build_group_transfer_plans(
    engine_group_infos: Sequence[EngineGroupInfo],
    kv_caches: dict[str, torch.Tensor],
    blocks_in_chunk: int,
    block_size: int,
    layout_desc: MemoryLayoutDesc,
    num_layers: int,
    hidden_dim_size: int,
    layout_hints: LayoutHints | None,
) -> list[GroupTransferPlan]:
    """Precompute every LMCache group's transfer constants once.

    See :class:`GroupTransferPlan` for why these are cached rather than
    derived per transfer.

    Args:
        engine_group_infos: The worker's registered LMCache groups, in
            protocol order. Empty builds a single collapsed plan covering all
            of ``kv_caches``.
        kv_caches: Worker KV-cache tensors keyed by layer name, in
            registration order.
        blocks_in_chunk: Blocks per LMCache chunk for the default
            (single-group) block size.
        block_size: Tokens per paged block for the default (single-group)
            layout.
        layout_desc: The default (single-group) layout descriptor.
        num_layers: The default (single-group) layer count that
            ``layout_desc`` was built from.
        hidden_dim_size: The default (single-group) hidden dimension that
            ``layout_desc`` was built from.
        layout_hints: Optional engine layout hints.

    Returns:
        One plan per group, in protocol order. A single-element list holding
        the ``group_info=None`` fallback when ``engine_group_infos`` is empty.

    Raises:
        ValueError: If a group's ``tokens_per_block`` does not divide the
            LMCache chunk size (see :func:`_blocks_per_chunk_for_group`).
    """
    none_group: list[EngineGroupInfo | None] = [None]
    group_infos: Sequence[EngineGroupInfo | None] = engine_group_infos or none_group
    plans: list[GroupTransferPlan] = []
    for group_info in group_infos:
        group_kv_caches = _kv_caches_for_group(kv_caches, group_info)
        plans.append(
            GroupTransferPlan(
                group_info=group_info,
                layer_indices=frozenset(
                    group_info.layer_indices if group_info is not None else ()
                ),
                blocks_per_chunk=_blocks_per_chunk_for_group(
                    group_info, blocks_in_chunk, block_size
                ),
                chunk_shape=_group_chunk_shape(
                    group_info,
                    layout_desc,
                    num_layers,
                    hidden_dim_size,
                    group_kv_caches,
                    layout_hints,
                ),
                engine_kv_format=_detect_group_kv_format(group_kv_caches, layout_hints),
            )
        )
    return plans


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
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
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

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Initialize a handle-path context bound to one worker.

        Args:
            instance_id: Worker process instance identifier.
            req_client: Transport client for this worker.
        """
        super().__init__(instance_id, req_client)
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None

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
                layout_hints,
                list(engine_group_infos),
            )
        )
        future.result(timeout=mq_timeout)
        if self._is_closed():
            return
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
            layout_hints,
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
        return self._req_client.store(
            key, self._instance_id, block_ids, event_ipc_handle
        ).to_device_future(
            device=self._device,
            event_backend=self._event_backend,
        )

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
        pass


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

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
        self._engine_group_infos: list[EngineGroupInfo] = []
        self._block_size: int = 0
        self._num_layers: int = 0
        # Registration-time per-group constants; see GroupTransferPlan.
        self._group_plans: list[GroupTransferPlan] = []

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

    def iter_transfer_groups(
        self,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        blocks_in_chunk: int,
    ) -> Iterator[tuple[GroupTransferPlan, dict[str, torch.Tensor], list[int]]]:
        """Yield this context's registered LMCache groups for one transfer.

        Pairs each cached :class:`GroupTransferPlan` with that group's slice
        of ``kv_caches`` and its block IDs for this request. The plans were
        built once by :meth:`register`, so the only per-call work is the
        block-id validation and applying each group's layer selection.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name, in the
                same registration order. Retrieve may pass a different
                mapping than ``register`` did (it scatters into destination
                tensors), so the group subsets are taken from this argument
                rather than cached.
            block_ids: Engine block IDs indexed by LMCache group id.
            blocks_in_chunk: Blocks per LMCache chunk for the default
                (single-group) block size. Accepted for interface stability;
                the per-group value comes from the cached plans.

        Yields:
            For each group, in protocol order: its cached plan, its KV-cache
            subset, and its flat block-id list.

        Raises:
            RuntimeError: If this is a single-group registration and
                ``block_ids`` does not carry exactly one group.
            ValueError: If this is a multi-group registration and
                ``block_ids`` does not carry exactly one entry per group.
        """
        if not self._group_plans:
            # No cached plans: the context was set up without going through
            # register(). Derive the single-group fallback on the fly so the
            # transfer still works, exactly as it did before the plan cache.
            yield (
                GroupTransferPlan(
                    group_info=None,
                    layer_indices=frozenset(),
                    blocks_per_chunk=blocks_in_chunk,
                    chunk_shape=self.engine_driven_context.layout_desc.shapes[0],
                    engine_kv_format=self._engine_kv_format,
                ),
                kv_caches,
                _single_group_block_ids(block_ids),
            )
            return
        if self._engine_group_infos:
            # Multi-group registration: one block-id entry per group.
            if len(block_ids) != len(self._group_plans):
                raise ValueError(
                    f"Expected {len(self._group_plans)} block-id groups, "
                    f"got {len(block_ids)}"
                )
            for plan, group_block_ids in zip(self._group_plans, block_ids, strict=True):
                yield plan, plan.select_kv_caches(kv_caches), group_block_ids
            return
        # Single-group registration: reject a hybrid block-id payload the
        # same way the pre-cache path did.
        plan = self._group_plans[0]
        yield plan, plan.select_kv_caches(kv_caches), _single_group_block_ids(block_ids)

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
    ) -> None:
        """Register KV caches with the non-GPU context server.

        ``engine_group_infos`` splits worker-side gather / scatter by LMCache
        group at store / retrieve time; an empty sequence registers a single
        non-hybrid group. Each group's transfer constants are resolved once
        here into :class:`GroupTransferPlan` entries and reused by every
        subsequent transfer. ``engine_type`` is accepted to satisfy the base
        interface but is unused: only the handle path consumes it.
        """
        del engine_type  # unused on the engine-driven path
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
        self._engine_group_infos = list(engine_group_infos)
        self._block_size = block_size
        self._num_layers = num_layers

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1
        shape = (
            torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
            if use_mla_flag
            else torch.Size(
                [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
            )
        )
        dtype = getattr(torch, dtype_str)
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])

        # Precompute each group's tensor subset, blocks-per-chunk, chunk
        # shape and KV format once, so store / retrieve stay O(num_groups).
        self._group_plans = _build_group_transfer_plans(
            engine_group_infos,
            kv_caches,
            blocks_in_chunk,
            block_size,
            layout_desc,
            num_layers,
            hidden_dim_size,
            layout_hints,
        )
        # chunk_shape already carries this group's own hidden_dim_size, so
        # the server can size each group's SHM buffer instead of reusing
        # the shared one.
        group_hidden_dim_sizes = (
            [int(plan.chunk_shape[-1]) for plan in self._group_plans]
            if engine_group_infos
            else None
        )
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
                    num_physical_slots=blocks_in_chunk * block_size,
                    engine_group_infos=list(engine_group_infos),
                    group_hidden_dim_sizes=group_hidden_dim_sizes,
                )
            )
        )
        response = future.result(timeout=mq_timeout)
        if self._is_closed():
            return
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            self._req_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s, "
            "num_groups=%d, kv_formats=%s)",
            self._instance_id,
            supported_transfer_mode,
            len(self._group_plans),
            [str(plan.engine_kv_format) for plan in self._group_plans],
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

    def _has_maskable_group(self) -> bool:
        """Return whether any registered group can produce a null chunk.

        Recurrent-state and sliding-window groups can; full-attention
        groups never do. Gates the null-chunk-mask computation so an
        all-full-attention store/retrieve skips it.
        """
        return any(
            plan.group_info is not None
            and (plan.group_info.recurrent_state or plan.group_info.sw_size_tokens >= 1)
            for plan in self._group_plans
        )

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

        transfer_groups = list(
            self.iter_transfer_groups(kv_caches, block_ids, blocks_in_chunk)
        )
        if self._has_maskable_group():
            key = replace(
                key, null_chunk_mask=null_chunk_mask_from_groups(transfer_groups)
            )

        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, self._instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        cpu_chunks: list[torch.Tensor] = []
        # ``out_buffers`` / ``chunk_indices`` (when present) are flat,
        # group-major over the whole multi-group chunk sequence; each
        # group's own range is sliced out before gathering it.
        group_offset = 0
        for plan, group_kv_caches, group_block_ids in transfer_groups:
            selection = _select_group_chunks(
                chunk_indices,
                group_offset,
                len(group_block_ids) // plan.blocks_per_chunk,
            )
            group_offset += selection.num_group_chunks
            if selection.is_empty:
                continue
            if chunk_indices is None:
                # No selection: server wants every chunk, already in group order.
                group_out_buffers = out_buffers
            else:
                group_out_buffers = (
                    [out_buffers[out_idx] for out_idx in selection.out_indices]
                    if out_buffers is not None
                    else None
                )
            cpu_chunks.extend(
                gather_paged_kv_to_cpu(
                    group_kv_caches,
                    group_block_ids,
                    plan.blocks_per_chunk,
                    layout_hints=self._layout_hints,
                    # This group's own pre-detected format (see GroupTransferPlan).
                    engine_kv_format=plan.engine_kv_format,
                    out=group_out_buffers,
                    chunk_indices=selection.chunk_indices,
                )
            )
        # Gather issues async device->CPU copies on BOTH transports: into the
        # SHM slots when out_buffers is given, otherwise into fresh buffers that
        # commit_store serializes immediately. Either way the copies must be
        # complete first, so this is unconditional -- guarding it on out_buffers
        # left the pickle path serializing a buffer still being written.
        torch_dev.synchronize()
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
        _event: IPCEvent | None,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        transfer_groups = list(
            self.iter_transfer_groups(kv_caches, block_ids, blocks_in_chunk)
        )
        null_chunk_mask = (
            null_chunk_mask_from_groups(transfer_groups)
            if self._has_maskable_group()
            else None
        )
        if null_chunk_mask is not None:
            key = replace(key, null_chunk_mask=null_chunk_mask)

        src_buffers = self._engine_driven_context.prepare_retrieve(
            key, self._instance_id
        )
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                # ``src_buffers`` is flat, group-major, minus any null
                # chunks the server excluded -- those were never stored, so
                # their positions are dropped rather than filled with data.
                group_offset = 0
                for group_id, (plan, group_kv_caches, group_block_ids) in enumerate(
                    transfer_groups
                ):
                    # Out-of-range or absent mask: treat as "nothing masked".
                    group_mask = (
                        null_chunk_mask[group_id]
                        if null_chunk_mask is not None
                        and group_id < len(null_chunk_mask)
                        else None
                    )
                    scatter_block_ids, num_live_chunks = _select_live_scatter_chunks(
                        group_block_ids, plan.blocks_per_chunk, group_mask
                    )
                    group_chunks = src_buffers[
                        group_offset : group_offset + num_live_chunks
                    ]
                    scatter_cpu_to_paged_kv(
                        group_kv_caches,
                        scatter_block_ids,
                        group_chunks,
                        plan.blocks_per_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        # See submit_store(): the group's own format, cached
                        # at register() instead of re-detected per retrieve.
                        engine_kv_format=plan.engine_kv_format,
                    )
                    group_offset += num_live_chunks
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, self._instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
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

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        instance_id: Worker process instance identifier bound to the context.
        req_client: Transport client bound to the context. The caller retains
            ownership and must close it after the context.
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
        return _build_lmcache_driven_context(device_type, instance_id, req_client)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return _build_engine_driven_context(instance_id, req_client)
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext(instance_id, req_client)
    return _build_engine_driven_context(instance_id, req_client)
