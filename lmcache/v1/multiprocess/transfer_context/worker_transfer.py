# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Callable, Protocol, TypeGuard
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    KernelGroupTransferMetadataWire,
    KVTransferMetadataWire,
    ObjectGroupTransferMetadataWire,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    EngineDrivenStorePreparation,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupPlan,
    KernelGroupTransferMetadata,
    KVTransferMetadata,
    TransferPlan,
    TransferPlanDirection,
    build_engine_driven_object_group_layout_desc,
    build_transfer_plan_from_kernel_group_block_ids,
    export_kv_transfer_metadata,
)
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


def _kv_transfer_metadata_to_wire(
    transfer_metadata: KVTransferMetadata,
) -> KVTransferMetadataWire:
    """Convert a :class:`~lmcache.v1.multiprocess.transfer_plan.KVTransferMetadata`
    to its msgspec wire DTO.

    Replaces pickle-based serialization with a structured, msgspec-compatible
    representation that survives cross-process transmission without requiring
    pickle hooks.

    Args:
        transfer_metadata: Immutable transfer metadata snapshot to convert.

    Returns:
        A :class:`KVTransferMetadataWire` with all non-primitive fields reduced
        to primitive types (``dtype`` → ``dtype_str``, ``engine_kv_format`` →
        ``engine_kv_format_int``).
    """
    kernel_groups_wire = [
        KernelGroupTransferMetadataWire(
            kernel_group_id=kg.kernel_group_id,
            engine_group_id=kg.engine_group_id,
            layer_indices=list(kg.layer_indices),
            blocks_per_chunk=kg.blocks_per_chunk,
            blocks_per_window=kg.blocks_per_window,
            slots_per_chunk_in_window=kg.slots_per_chunk_in_window,
            kv_size=kg.kv_size,
            num_layers=kg.num_layers,
            hidden_dim_size=kg.hidden_dim_size,
            slots_per_block=kg.slots_per_block,
            tokens_per_block=kg.tokens_per_block,
            dtype_str=str(kg.dtype).removeprefix("torch."),
            engine_kv_format_int=int(kg.engine_kv_format),
        )
        for kg in transfer_metadata.kernel_groups
    ]
    object_groups_wire = [
        ObjectGroupTransferMetadataWire(
            object_group_id=og.object_group_id,
            kernel_group_ids=list(og.kernel_group_ids),
            sw_size_chunks=og.sw_size_chunks,
        )
        for og in transfer_metadata.object_groups
    ]
    return KVTransferMetadataWire(
        num_chunks_in_sw=list(transfer_metadata.num_chunks_in_sw),
        tokens_per_chunk=transfer_metadata.tokens_per_chunk,
        kernel_groups=kernel_groups_wire,
        object_groups=object_groups_wire,
    )


def _build_multi_group_wire_fields(
    kv_caches: dict[str, torch.Tensor],
    engine_group_infos: Sequence[EngineGroupInfo],
    blocks_in_chunk: int,
    block_size: int,
    layout_hints: "LayoutHints | None",
) -> tuple[
    list[EngineGroupInfo],
    list[list[list[int]]],
    list[list[str]],
    list[int],
    list[MemoryLayoutDesc],
    AttnWindowDesc,
    KVTransferMetadata | None,
]:
    """Build wire-format multi-group fields for the registration payload.

    When ``engine_group_infos`` is empty (legacy single-group mode) returns
    empty lists and the default full-attention descriptor so the payload is
    backward-compatible.  When non-empty, constructs a
    :class:`~lmcache.v1.kv_layer_groups.KVLayerGroupsManager` and exports
    shared transfer metadata via :func:`export_kv_transfer_metadata`.

    Args:
        kv_caches: Per-layer KV tensor mapping keyed by layer name.
        engine_group_infos: Engine KV-group metadata. Non-empty triggers the
            multi-group metadata export path.
        blocks_in_chunk: Number of engine blocks per LMCache chunk.
        block_size: Detected tokens per paged block (from ``compute_kv_layout``).
        layout_hints: Optional engine layout hints passed to format detection.

    Returns:
        A tuple of::

            (engine_group_infos,
             object_group_layout_shapes,
             object_group_layout_dtype_strs,
             num_chunks_in_sw,
             object_group_layout_descs,
             attn_desc,
             transfer_metadata)

        ``object_group_layout_shapes[g][k]`` is the flat integer list for
        kernel group ``k`` in object group ``g``.
        ``object_group_layout_dtype_strs[g][k]`` is the dtype string for the
        same entry.  ``transfer_metadata`` is a full immutable snapshot of
        kernel-group geometry, engine-group mapping, and object-group metadata
        for use by downstream transfer planning; ``None`` in legacy mode.
    """
    # First Party
    from lmcache.v1.distributed.api import DEFAULT_ATTN_WINDOW_DESC

    if not engine_group_infos:
        return [], [], [], [], [], DEFAULT_ATTN_WINDOW_DESC, None

    # Import KVLayerGroupsManager and format discovery lazily to avoid
    # introducing a hard dependency on the GPU connector at module load time.
    # First Party
    from lmcache.v1.gpu_connector.utils import normalize_and_discover_per_layer_formats
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
    from lmcache.v1.multiprocess.group_view import engine_group_layer_indices

    tensors = list(kv_caches.values())
    kv_caches_norm, engine_kv_formats = normalize_and_discover_per_layer_formats(
        tensors,
        engine_group_layer_indices(engine_group_infos),
        EngineType.VLLM,
        layout_hints=layout_hints,
    )
    tokens_per_chunk = blocks_in_chunk * block_size
    manager = KVLayerGroupsManager(
        kv_caches_norm,
        engine_kv_formats,
        engine_group_infos=list(engine_group_infos),
        lmcache_tokens_per_chunk=tokens_per_chunk,
    )
    transfer_metadata = export_kv_transfer_metadata(manager, tokens_per_chunk)

    num_object_groups = len(transfer_metadata.object_groups)
    object_group_layout_descs: list[MemoryLayoutDesc] = [
        build_engine_driven_object_group_layout_desc(
            transfer_metadata, tokens_per_chunk, og_id
        )
        for og_id in range(num_object_groups)
    ]
    wire_shapes: list[list[list[int]]] = [
        [list(s) for s in desc.shapes] for desc in object_group_layout_descs
    ]
    wire_dtype_strs: list[list[str]] = [
        [str(dt).removeprefix("torch.") for dt in desc.dtypes]
        for desc in object_group_layout_descs
    ]
    wire_num_chunks_in_sw = list(transfer_metadata.num_chunks_in_sw)
    attn_desc = transfer_metadata.build_attn_desc()

    return (
        list(engine_group_infos),
        wire_shapes,
        wire_dtype_strs,
        wire_num_chunks_in_sw,
        object_group_layout_descs,
        attn_desc,
        transfer_metadata,
    )


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


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _kernel_group_kv_caches(
    kv_caches: dict[str, torch.Tensor],
    layer_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Select one kernel group's KV tensors in declared layer order."""
    layer_items = list(kv_caches.items())
    try:
        return {layer_items[index][0]: layer_items[index][1] for index in layer_indices}
    except IndexError as exc:
        raise ValueError(
            f"kernel group references layer outside {len(layer_items)} KV tensors"
        ) from exc


def _plan_engine_driven_request(
    transfer_metadata: KVTransferMetadata,
    block_ids_by_kernel_group: list[list[int]],
    direction: TransferPlanDirection,
    skip_first_n_tokens: int = 0,
) -> TransferPlan:
    """Build the shared logical plan for one engine-driven request.

    Args:
        transfer_metadata: Immutable registered kernel and object-group metadata.
        block_ids_by_kernel_group: Request block IDs in protocol kernel-group
            order.
        direction: Whether the worker gathers for store or scatters for
            retrieve.
        skip_first_n_tokens: Retrieve prefix to preserve without overwriting.

    Returns:
        The ordered, transport-independent work plan to bind to worker buffers.
    """
    return build_transfer_plan_from_kernel_group_block_ids(
        transfer_metadata,
        block_ids_by_kernel_group,
        direction,
        skip_first_n_tokens,
    )


def _kernel_group_metadata_for_plan(
    transfer_metadata: KVTransferMetadata,
    kernel_group_plan: KernelGroupPlan,
) -> KernelGroupTransferMetadata:
    """Return metadata for a plan group after checking its stable identity."""
    kernel_group_id = kernel_group_plan.kernel_group_id
    if kernel_group_id < 0 or kernel_group_id >= len(transfer_metadata.kernel_groups):
        raise ValueError(f"invalid kernel_group_id {kernel_group_id} in transfer plan")
    kernel_group = transfer_metadata.kernel_groups[kernel_group_id]
    if (
        kernel_group.kernel_group_id != kernel_group_id
        or kernel_group.engine_group_id != kernel_group_plan.engine_group_id
    ):
        raise ValueError(
            f"transfer plan kernel group {kernel_group_id} does not match metadata"
        )
    return kernel_group


def _gather_multi_group_pickle_payload(
    kv_caches: dict[str, torch.Tensor],
    transfer_metadata: KVTransferMetadata,
    block_ids_by_engine_group: list[list[int]],
    layout_hints: LayoutHints | None,
) -> list[list[list[torch.Tensor]]]:
    """Compatibility wrapper that plans and gathers a pickle payload."""
    transfer_plan = _plan_engine_driven_request(
        transfer_metadata,
        block_ids_by_engine_group,
        TransferPlanDirection.STORE,
    )
    return _gather_multi_group_pickle_payload_for_plan(
        kv_caches, transfer_metadata, transfer_plan, layout_hints
    )


def _gather_multi_group_pickle_payload_for_plan(
    kv_caches: dict[str, torch.Tensor],
    transfer_metadata: KVTransferMetadata,
    transfer_plan: TransferPlan,
    layout_hints: LayoutHints | None,
) -> list[list[list[torch.Tensor]]]:
    """Bind a store plan to CPU pickle payload tensors.

    Args:
        kv_caches: Worker KV caches keyed by layer name.
        transfer_metadata: Immutable registration metadata.
        transfer_plan: Shared store plan whose object and kernel order defines
            the payload.
        layout_hints: Optional engine-provided KV layout metadata.

    Returns:
        CPU tensors ordered by object group, plan chunk, and kernel group.
    """
    payload: list[list[list[torch.Tensor]]] = []
    for object_group_plan in transfer_plan.object_groups:
        gathered_by_kernel_group: list[list[torch.Tensor]] = []
        for kernel_group_plan in object_group_plan.kernel_groups:
            kernel_group = _kernel_group_metadata_for_plan(
                transfer_metadata, kernel_group_plan
            )
            gathered_by_kernel_group.append(
                gather_paged_kv_to_cpu(
                    _kernel_group_kv_caches(kv_caches, kernel_group.layer_indices),
                    list(kernel_group_plan.block_ids),
                    kernel_group_plan.blocks_per_window,
                    layout_hints=layout_hints,
                    engine_kv_format=kernel_group.engine_kv_format,
                )
            )
        payload.append(
            [
                [
                    gathered_by_kernel_group[tensor_idx][chunk_idx]
                    for tensor_idx in range(len(object_group_plan.kernel_groups))
                ]
                for chunk_idx in range(len(object_group_plan.chunk_indices))
            ]
        )
    return payload


def _gather_multi_group_shm_payload(
    kv_caches: dict[str, torch.Tensor],
    transfer_metadata: KVTransferMetadata,
    block_ids_by_engine_group: list[list[int]],
    out: list[list[list[torch.Tensor]]],
    chunk_indices_by_object_group: list[list[int]],
    layout_hints: LayoutHints | None,
) -> list[list[list[torch.Tensor]]]:
    """Compatibility wrapper that plans and gathers into SHM slots."""
    transfer_plan = _plan_engine_driven_request(
        transfer_metadata,
        block_ids_by_engine_group,
        TransferPlanDirection.STORE,
    )
    return _gather_multi_group_shm_payload_for_plan(
        kv_caches,
        transfer_metadata,
        transfer_plan,
        out,
        chunk_indices_by_object_group,
        layout_hints,
    )


def _gather_multi_group_shm_payload_for_plan(
    kv_caches: dict[str, torch.Tensor],
    transfer_metadata: KVTransferMetadata,
    transfer_plan: TransferPlan,
    out: list[list[list[torch.Tensor]]],
    chunk_indices_by_object_group: list[list[int]],
    layout_hints: LayoutHints | None,
) -> list[list[list[torch.Tensor]]]:
    """Bind a store plan to sparse shared-memory slot buffers.

    Args:
        kv_caches: Worker KV caches keyed by layer name.
        transfer_metadata: Immutable registration metadata.
        transfer_plan: Shared store plan defining source chunk and tensor order.
        out: SHM-backed tensors in object-group, sparse-chunk, kernel order.
        chunk_indices_by_object_group: Original source chunk positions for
            every sparse SHM slot.
        layout_hints: Optional engine-provided KV layout metadata.

    Returns:
        The same nested ``out`` list after in-place gather.

    Raises:
        ValueError: If server-provided slots cannot bind to the shared plan.
    """
    if len(out) != len(transfer_plan.object_groups) or len(
        chunk_indices_by_object_group
    ) != len(transfer_plan.object_groups):
        raise ValueError(
            "SHM slot plan object-group count does not match transfer plan"
        )

    for object_group_plan, group_out, chunk_indices in zip(
        transfer_plan.object_groups,
        out,
        chunk_indices_by_object_group,
        strict=True,
    ):
        object_group_id = object_group_plan.object_group_id
        if len(group_out) != len(chunk_indices):
            raise ValueError(
                f"SHM object group {object_group_id} has {len(group_out)} slot "
                f"chunks but {len(chunk_indices)} chunk indices"
            )
        if chunk_indices != sorted(set(chunk_indices)):
            raise ValueError(
                f"SHM object group {object_group_id} has invalid chunk ordering"
            )
        plan_positions = {
            chunk_index: position
            for position, chunk_index in enumerate(object_group_plan.chunk_indices)
        }
        try:
            planned_positions = [
                plan_positions[chunk_index] for chunk_index in chunk_indices
            ]
        except KeyError as exc:
            raise ValueError(
                f"SHM object group {object_group_id} references a chunk "
                "outside the transfer plan"
            ) from exc
        if any(
            len(chunk) != len(object_group_plan.kernel_groups) for chunk in group_out
        ):
            raise ValueError(
                f"SHM object group {object_group_id} has malformed "
                "kernel-group slot ordering"
            )
        for tensor_idx, kernel_group_plan in enumerate(object_group_plan.kernel_groups):
            kernel_group = _kernel_group_metadata_for_plan(
                transfer_metadata, kernel_group_plan
            )
            gather_paged_kv_to_cpu(
                _kernel_group_kv_caches(kv_caches, kernel_group.layer_indices),
                list(kernel_group_plan.block_ids),
                kernel_group_plan.blocks_per_window,
                layout_hints=layout_hints,
                engine_kv_format=kernel_group.engine_kv_format,
                out=[chunk[tensor_idx] for chunk in group_out],
                chunk_indices=planned_positions,
            )
    return out


def _is_legacy_shm_store_preparation(
    preparation: object,
) -> TypeGuard[tuple[list[torch.Tensor], list[int]]]:
    """Return whether a SHM preparation contains legacy flat slot buffers."""
    if not isinstance(preparation, tuple) or len(preparation) != 2:
        return False
    slot_buffers, chunk_indices = preparation
    return (
        isinstance(slot_buffers, list)
        and all(isinstance(buffer, torch.Tensor) for buffer in slot_buffers)
        and isinstance(chunk_indices, list)
        and all(isinstance(chunk_index, int) for chunk_index in chunk_indices)
    )


def _is_multi_group_shm_store_preparation(
    preparation: object,
) -> TypeGuard[tuple[list[list[list[torch.Tensor]]], list[list[int]]]]:
    """Return whether a SHM preparation contains ordered multi-group slots."""
    if not isinstance(preparation, tuple) or len(preparation) != 2:
        return False
    grouped_slot_buffers, grouped_chunk_indices = preparation
    return (
        isinstance(grouped_slot_buffers, list)
        and all(
            isinstance(group_slots, list)
            and all(
                isinstance(slot_buffers, list)
                and all(isinstance(buffer, torch.Tensor) for buffer in slot_buffers)
                for slot_buffers in group_slots
            )
            for group_slots in grouped_slot_buffers
        )
        and isinstance(grouped_chunk_indices, list)
        and all(
            isinstance(group_indices, list)
            and all(isinstance(chunk_index, int) for chunk_index in group_indices)
            for group_indices in grouped_chunk_indices
        )
    )


def _scatter_multi_group_pickle_payload(
    kv_caches: dict[str, torch.Tensor],
    transfer_metadata: KVTransferMetadata,
    block_ids_by_engine_group: list[list[int]],
    payload: list[list[list[torch.Tensor]]],
    skip_first_n_tokens: int,
    layout_hints: LayoutHints | None,
) -> None:
    """Compatibility wrapper that plans and scatters a pickle payload."""
    transfer_plan = _plan_engine_driven_request(
        transfer_metadata,
        block_ids_by_engine_group,
        TransferPlanDirection.RETRIEVE,
        skip_first_n_tokens,
    )
    _scatter_multi_group_pickle_payload_for_plan(
        kv_caches,
        transfer_metadata,
        transfer_plan,
        payload,
        layout_hints,
    )


def _scatter_multi_group_pickle_payload_for_plan(
    kv_caches: dict[str, torch.Tensor],
    transfer_metadata: KVTransferMetadata,
    transfer_plan: TransferPlan,
    payload: list[list[list[torch.Tensor]]],
    layout_hints: LayoutHints | None,
) -> None:
    """Bind a retrieve plan to an ordered pickle or SHM payload.

    Args:
        kv_caches: Worker KV caches keyed by layer name.
        transfer_metadata: Immutable registration metadata.
        transfer_plan: Shared retrieve plan defining block, object, and prefix
            skip order.
        payload: CPU tensors in object-group, plan chunk, and kernel-group
            order.
        layout_hints: Optional engine-provided KV layout metadata.

    Raises:
        ValueError: If the payload does not exactly bind to the shared plan.
    """
    if len(payload) != len(transfer_plan.object_groups):
        raise ValueError(
            "pickle payload object-group count does not match transfer plan"
        )

    for object_group_plan, object_group_payload in zip(
        transfer_plan.object_groups, payload, strict=True
    ):
        object_group_id = object_group_plan.object_group_id
        if len(object_group_payload) != len(object_group_plan.chunk_indices):
            raise ValueError(
                f"object group {object_group_id} has "
                f"{len(object_group_payload)} chunks; expected "
                f"{len(object_group_plan.chunk_indices)}"
            )
        if any(
            len(chunk) != len(object_group_plan.kernel_groups)
            for chunk in object_group_payload
        ):
            raise ValueError(
                f"object group {object_group_id} has malformed "
                "kernel-group payload ordering"
            )
        for tensor_idx, kernel_group_plan in enumerate(object_group_plan.kernel_groups):
            kernel_group = _kernel_group_metadata_for_plan(
                transfer_metadata, kernel_group_plan
            )
            chunks = [chunk[tensor_idx] for chunk in object_group_payload]
            scatter_cpu_to_paged_kv(
                _kernel_group_kv_caches(kv_caches, kernel_group.layer_indices),
                list(kernel_group_plan.block_ids),
                chunks,
                kernel_group_plan.blocks_per_window,
                skip_first_n_tokens=(
                    kernel_group_plan.skip_first_n_blocks
                    * kernel_group.tokens_per_block
                ),
                layout_hints=layout_hints,
                engine_kv_format=kernel_group.engine_kv_format,
            )


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
    return next(iter(kv_caches.values())).device


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
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
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
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
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
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
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
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
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
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
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
        event: IPCEvent,
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
            event: Synchronization event object.
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
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
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
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
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

        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                engine_type,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)
        self._device = device
        self._event_backend = event_backend

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        # First Party
        from lmcache.integration.vllm.vllm_multi_process_adapter import (  # type: ignore[attr-defined]
            wrap_kv_caches,
        )

        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_Q_CACHE,
            [
                instance_id,
                wrap_kv_caches(q_caches),
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
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
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event_ipc_handle],
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
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_q_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE_Q,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
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
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
        ).to_device_future(device=self._device)

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        self._mq_client = None
        self._send_request = None
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
        self._transfer_metadata: KVTransferMetadata | None = None

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
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the non-GPU context server.

        Legacy dense registrations retain their single-group layout. Hybrid
        registrations export immutable transfer metadata so the worker and
        server can bind the same shared logical transfer plan.
        """
        # TODO: per-group compression (EngineGroupInfo.tokens_per_block vs
        # the tensor-detected slot count, e.g. DeepSeek V4) is only handled
        # on the CUDA path. The non-CUDA path is yet to be implemented.
        # Hybrid KV groups can use different packed widths. The legacy flat
        # layout is only a compatibility fallback once multi-group metadata is
        # present, so discover it from one representative tensor instead of
        # asking the vLLM detector to reshape heterogeneous tensors together.
        layout_source = kv_caches
        tensor_layouts = {
            (tuple(tensor.shape), tensor.dtype) for tensor in kv_caches.values()
        }
        if len(tensor_layouts) > 1:
            first_layer, first_tensor = next(iter(kv_caches.items()))
            layout_source = {first_layer: first_tensor}

        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(layout_source, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format

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

        # Step 3: when engine_group_infos is provided, export shared
        # transfer metadata so the server receives the full multi-group
        # registration information.
        (
            wire_engine_group_infos,
            wire_obj_shapes,
            wire_obj_dtype_strs,
            wire_num_chunks_in_sw,
            object_group_layout_descs,
            attn_desc,
            transfer_metadata,
        ) = _build_multi_group_wire_fields(
            kv_caches,
            engine_group_infos,
            blocks_in_chunk,
            block_size,
            layout_hints,
        )

        # Preserve legacy registration for dense layouts. Hybrid layouts carry
        # their complete immutable metadata to the server.
        if len(object_group_layout_descs) == 1:
            wire_engine_group_infos = []
            wire_obj_shapes = []
            wire_obj_dtype_strs = []
            wire_num_chunks_in_sw = []
            object_group_layout_descs = []
            attn_desc = DEFAULT_ATTN_WINDOW_DESC
            transfer_metadata = None

        # Convert KVTransferMetadata to a structured msgspec wire DTO so the
        # server can reconstruct it without pickle.
        transfer_metadata_wire = (
            _kv_transfer_metadata_to_wire(transfer_metadata)
            if transfer_metadata is not None
            else None
        )

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    engine_group_infos=wire_engine_group_infos,
                    object_group_layout_shapes=wire_obj_shapes,
                    object_group_layout_dtype_strs=wire_obj_dtype_strs,
                    num_chunks_in_sw=wire_num_chunks_in_sw,
                    transfer_metadata_wire=transfer_metadata_wire,
                )
            ],
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
            object_group_layout_descs=object_group_layout_descs,
            attn_desc=attn_desc,
            transfer_metadata=transfer_metadata,
        )
        self._transfer_metadata = transfer_metadata
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s)",
            instance_id,
            supported_transfer_mode,
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        transfer_metadata = self._transfer_metadata
        transfer_plan = (
            _plan_engine_driven_request(
                transfer_metadata,
                block_ids,
                TransferPlanDirection.STORE,
            )
            if transfer_metadata is not None
            else None
        )
        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, instance_id)
        if transfer_metadata is None:
            if result is not None and not _is_legacy_shm_store_preparation(result):
                self._abort_shm_store(key, instance_id, result)
                raise ValueError("legacy SHM store returned an invalid slot plan")
            out_buffers, chunk_indices = result if result is not None else (None, None)
            if chunk_indices is not None and len(chunk_indices) == 0:
                # All chunks already in cache — nothing to gather or commit.
                future: MessagingFuture[bool] = MessagingFuture()
                future.set_result(True)
                return future
            try:
                cpu_chunks: list[torch.Tensor] | list[list[list[torch.Tensor]]] = (
                    gather_paged_kv_to_cpu(
                        kv_caches,
                        _single_group_block_ids(block_ids),
                        blocks_in_chunk,
                        layout_hints=self._layout_hints,
                        engine_kv_format=self._engine_kv_format,
                        out=out_buffers,
                        chunk_indices=chunk_indices,
                    )
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                self._abort_shm_store(key, instance_id, result)
                raise
        else:
            if transfer_plan is None:
                raise RuntimeError("multi-group store has no transfer plan")
            if result is None:
                cpu_chunks = _gather_multi_group_pickle_payload_for_plan(
                    kv_caches,
                    transfer_metadata,
                    transfer_plan,
                    self._layout_hints,
                )
                used_shm = False
            else:
                if not _is_multi_group_shm_store_preparation(result):
                    self._abort_shm_store(key, instance_id, result)
                    raise ValueError(
                        "multi-group SHM store returned an invalid slot plan"
                    )
                grouped_out_buffers, grouped_chunk_indices = result
                if all(
                    len(group_indices) == 0 for group_indices in grouped_chunk_indices
                ):
                    # Every object group was already cached.
                    future = MessagingFuture()
                    future.set_result(True)
                    return future
                try:
                    cpu_chunks = _gather_multi_group_shm_payload_for_plan(
                        kv_caches,
                        transfer_metadata,
                        transfer_plan,
                        grouped_out_buffers,
                        grouped_chunk_indices,
                        self._layout_hints,
                    )
                except (RuntimeError, ValueError, TypeError, IndexError):
                    self._abort_shm_store(key, instance_id, result)
                    raise
                used_shm = True
        if transfer_metadata is None:
            used_shm = out_buffers is not None
        if used_shm:
            # SHM path uses async device->CPU copies; complete them before commit.
            try:
                torch_dev.synchronize()
            except RuntimeError:
                self._abort_shm_store(key, instance_id, result)
                raise
        ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)
        if not ok:
            self._abort_shm_store(key, instance_id, result)

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
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        transfer_metadata = self._transfer_metadata
        transfer_plan = (
            _plan_engine_driven_request(
                transfer_metadata,
                block_ids,
                TransferPlanDirection.RETRIEVE,
                skip_first_n_tokens,
            )
            if transfer_metadata is not None
            else None
        )
        src_buffers = self._engine_driven_context.prepare_retrieve(
            key,
            instance_id,
            skip_first_n_tokens,
            transfer_plan,
        )
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                if transfer_metadata is None:
                    scatter_cpu_to_paged_kv(
                        kv_caches,
                        _single_group_block_ids(block_ids),
                        src_buffers,
                        blocks_in_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        engine_kv_format=self._engine_kv_format,
                    )
                else:
                    if transfer_plan is None:
                        raise RuntimeError("multi-group retrieve has no transfer plan")
                    if not isinstance(src_buffers, list) or (
                        src_buffers and not isinstance(src_buffers[0], list)
                    ):
                        raise ValueError(
                            "multi-group retrieve returned a legacy flat payload"
                        )
                    _scatter_multi_group_pickle_payload_for_plan(
                        kv_caches,
                        transfer_metadata,
                        transfer_plan,
                        src_buffers,
                        self._layout_hints,
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

    def close(self) -> None:
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None
        self._transfer_metadata = None

    def flush_inflight_stores(self) -> None:
        pass

    def _abort_shm_store(
        self,
        key: Any,
        instance_id: int,
        preparation: EngineDrivenStorePreparation | None,
    ) -> None:
        """Abort a prepared SHM store while preserving the original failure.

        Args:
            key: Cache key for the failed store.
            instance_id: Worker instance identifier.
            preparation: Result returned from ``prepare_store``.
        """
        if preparation is None or self._engine_driven_context is None:
            return
        if not self._engine_driven_context.abort_store(key, instance_id):
            logger.error(
                "Failed to abort prepared SHM store for instance_id=%d",
                instance_id,
            )


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
    device_types = {tensor.device.type for tensor in kv_caches.values()}
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
