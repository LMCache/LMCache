# SPDX-License-Identifier: Apache-2.0
"""Path-agnostic helpers for multiprocess transfer planning."""

# Standard
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import TYPE_CHECKING, Generator, Mapping, Sequence, TypeVar

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops

ItemT = TypeVar("ItemT")


@dataclass(frozen=True)
class KernelGroupTransferMetadata:
    """Immutable transfer metadata for one kernel group."""

    kernel_group_id: int
    """Kernel-group index in deterministic manager/kernel-order."""
    engine_group_id: int
    """Engine group ID that provides block IDs for this kernel group."""
    layer_indices: tuple[int, ...]
    """Layer indices in transfer/kernel order for this group."""
    blocks_per_chunk: int
    """Total blocks in one LMCache chunk for this kernel group."""
    blocks_per_window: int
    """Retained blocks per chunk after subchunk-window downsampling."""
    slots_per_chunk_in_window: int
    """Per-chunk transfer slots in the retained subchunk window."""
    kv_size: int
    """KV-plane size (1 for MLA/key-only style, 2 for K/V style)."""
    num_layers: int
    """Number of layers covered by this kernel group."""
    hidden_dim_size: int
    """Hidden dimension width per slot in this group."""
    slots_per_block: int
    """Physical slots represented by one engine block for this group."""
    tokens_per_block: int
    """Logical tokens represented by one engine block for this group."""
    dtype: torch.dtype
    """Torch dtype for this group's tensors."""
    engine_kv_format: "lmc_ops.EngineKVFormat"
    """Engine KV format required by copy-kernel execution."""

    @property
    def compress_ratio(self) -> int:
        """Logical tokens represented by one physical slot for this group."""
        return self.tokens_per_block // self.slots_per_block


@dataclass(frozen=True)
class ObjectGroupTransferMetadata:
    """Immutable transfer metadata for one object group."""

    object_group_id: int
    """Object-group index in deterministic manager/object-group order."""
    kernel_group_ids: tuple[int, ...]
    """Kernel-group indices packed into this object group's memory object."""
    sw_size_chunks: int
    """Cross-chunk attention window in chunks; ``-1`` means full attention."""


@dataclass(frozen=True)
class KVTransferMetadata:
    """Immutable transfer metadata for all object/kernel groups."""

    num_chunks_in_sw: tuple[int, ...]
    """Attention-window chunk counts in object-group order."""
    tokens_per_chunk: int
    """LMCache chunk size in logical tokens used to derive per-group geometry."""
    kernel_groups: tuple[KernelGroupTransferMetadata, ...]
    """Kernel-group metadata in deterministic kernel-group index order."""
    object_groups: tuple[ObjectGroupTransferMetadata, ...]
    """Object-group metadata in deterministic object-group index order."""

    def build_attn_desc(self) -> AttnWindowDesc:
        """Build a defensive AttnWindowDesc copy for external APIs."""
        return AttnWindowDesc(num_chunks_in_sw=list(self.num_chunks_in_sw))


class TransferPlanDirection(str, Enum):
    """Logical direction for a planned KV transfer."""

    STORE = "store"
    """Copy paged engine KV into LMCache objects."""
    RETRIEVE = "retrieve"
    """Copy LMCache objects into paged engine KV."""


@dataclass(frozen=True)
class KernelGroupPlan:
    """Logical transfer work for one kernel group in one object group."""

    kernel_group_id: int
    """Kernel-group index in the enclosing object's declared order."""
    engine_group_id: int
    """Engine group that supplied ``block_ids``."""
    block_ids: tuple[int, ...]
    """Selected and downsampled block IDs in enclosing ``chunk_indices`` order."""
    blocks_per_chunk: int
    """Total source blocks in one LMCache chunk."""
    blocks_per_window: int
    """Transferred blocks retained from each LMCache chunk."""
    skip_first_n_blocks: int
    """Blocks to skip in the first planned chunk after window downsampling."""


@dataclass(frozen=True)
class ObjectGroupPlan:
    """Logical transfer work for one ordered LMCache object group."""

    object_group_id: int
    """Object-group index in deterministic object-group order."""
    chunk_indices: tuple[int, ...]
    """Original chunk indices included in this group's transfer, in order."""
    skip_first_n_tokens: int
    """Token prefix to skip within the first planned chunk."""
    kernel_groups: tuple[KernelGroupPlan, ...]
    """Kernel-group work in the object's declared kernel-group order."""


@dataclass(frozen=True)
class TransferPlan:
    """Transport- and device-independent logical KV transfer schedule."""

    direction: TransferPlanDirection
    """Logical transfer direction."""
    num_chunks: int
    """Number of source chunks before attention-window or prefix skipping."""
    object_groups: tuple[ObjectGroupPlan, ...]
    """Object-group operations in deterministic object-group order."""


def build_transfer_plan(
    transfer_metadata: KVTransferMetadata,
    block_ids_by_engine_group: Mapping[int, Sequence[int]],
    num_chunks: int,
    direction: TransferPlanDirection,
    skip_first_n_tokens: int = 0,
) -> TransferPlan:
    """Build a logical transfer schedule from immutable KV metadata.

    The returned plan contains no device pointers, transport buffers, storage
    keys, locks, or serialized data.  It expands each kernel group's engine
    block IDs, applies subchunk-window downsampling, skips stale objects on
    retrieve, and preserves object/kernel ordering from ``transfer_metadata``.
    Executors bind the plan to their own device or transport resources.

    Args:
        transfer_metadata: Immutable kernel- and object-group geometry.
        block_ids_by_engine_group: Source block IDs keyed by engine group ID.
            Each referenced engine group must contain at least
            ``num_chunks * blocks_per_chunk`` IDs for every kernel group using it.
        num_chunks: Number of source LMCache chunks to plan.
        direction: Whether this plan stores or retrieves KV.
        skip_first_n_tokens: Retrieve prefix to preserve without overwriting.
            Non-block-aligned values are rounded down at each kernel group's
            block geometry, matching the existing transfer behavior.

    Returns:
        An immutable, ordered logical transfer plan.

    Raises:
        ValueError: If the metadata ordering is invalid, an engine group is
            missing or has insufficient IDs, or a count is negative.
    """
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")
    if skip_first_n_tokens < 0:
        raise ValueError("skip_first_n_tokens must be non-negative")
    if transfer_metadata.tokens_per_chunk < 1:
        raise ValueError("transfer_metadata.tokens_per_chunk must be at least one")

    kernel_groups_by_id = _validate_transfer_metadata_order(transfer_metadata)
    retrieve_prefix_tokens = (
        skip_first_n_tokens if direction == TransferPlanDirection.RETRIEVE else 0
    )
    object_group_plans: list[ObjectGroupPlan] = []
    full_chunk_indices = tuple(range(num_chunks))
    for object_group in transfer_metadata.object_groups:
        object_group_skip = compute_num_objects_to_skip(
            object_group.sw_size_chunks,
            num_chunks,
            is_retrieve=direction == TransferPlanDirection.RETRIEVE,
        )
        prefix_chunk_skip = min(
            num_chunks,
            retrieve_prefix_tokens // transfer_metadata.tokens_per_chunk,
        )
        first_chunk_idx = max(object_group_skip, prefix_chunk_skip)
        chunk_indices = full_chunk_indices[first_chunk_idx:]
        skip_tokens_in_first_chunk = max(
            0,
            retrieve_prefix_tokens
            - first_chunk_idx * transfer_metadata.tokens_per_chunk,
        )

        kernel_group_plans: list[KernelGroupPlan] = []
        for kernel_group_id in object_group.kernel_group_ids:
            kernel_group = kernel_groups_by_id[kernel_group_id]
            engine_block_ids = block_ids_by_engine_group.get(
                kernel_group.engine_group_id
            )
            if engine_block_ids is None:
                raise ValueError(
                    f"missing block IDs for engine group {kernel_group.engine_group_id}"
                )
            required_block_count = num_chunks * kernel_group.blocks_per_chunk
            if len(engine_block_ids) < required_block_count:
                raise ValueError(
                    f"engine group {kernel_group.engine_group_id} has "
                    f"{len(engine_block_ids)} block IDs, but kernel group "
                    f"{kernel_group_id} requires {required_block_count}"
                )

            selected_block_ids = select_block_ids_for_window(
                engine_block_ids[:required_block_count],
                kernel_group.blocks_per_chunk,
                kernel_group.blocks_per_window,
            )
            planned_block_ids = tuple(
                block_id
                for chunk_idx in chunk_indices
                for block_id in selected_block_ids[
                    chunk_idx * kernel_group.blocks_per_window : (chunk_idx + 1)
                    * kernel_group.blocks_per_window
                ]
            )
            skipped_source_blocks = (
                skip_tokens_in_first_chunk * kernel_group.blocks_per_chunk
            ) // transfer_metadata.tokens_per_chunk
            kernel_group_plans.append(
                KernelGroupPlan(
                    kernel_group_id=kernel_group_id,
                    engine_group_id=kernel_group.engine_group_id,
                    block_ids=planned_block_ids,
                    blocks_per_chunk=kernel_group.blocks_per_chunk,
                    blocks_per_window=kernel_group.blocks_per_window,
                    skip_first_n_blocks=recalculate_blocks_to_skip(
                        kernel_group.blocks_per_chunk,
                        kernel_group.blocks_per_window,
                        skipped_source_blocks,
                    ),
                )
            )

        object_group_plans.append(
            ObjectGroupPlan(
                object_group_id=object_group.object_group_id,
                chunk_indices=chunk_indices,
                skip_first_n_tokens=skip_tokens_in_first_chunk,
                kernel_groups=tuple(kernel_group_plans),
            )
        )

    return TransferPlan(
        direction=direction,
        num_chunks=num_chunks,
        object_groups=tuple(object_group_plans),
    )


def map_kernel_group_block_ids_to_engine_groups(
    transfer_metadata: KVTransferMetadata,
    block_ids_by_kernel_group: Sequence[Sequence[int]],
) -> dict[int, Sequence[int]]:
    """Map request block IDs from kernel-group to engine-group order.

    Serving-engine requests retain one block-ID sequence for every kernel
    group, while :func:`build_transfer_plan` consumes one sequence for each
    distinct engine block address space. Kernel groups sharing an engine group
    must therefore provide identical sequences.

    Args:
        transfer_metadata: Immutable metadata mapping kernel groups to engine
            groups.
        block_ids_by_kernel_group: Request block-ID sequences in kernel-group
            order.

    Returns:
        Block IDs keyed by engine-group ID.

    Raises:
        ValueError: If the request group count differs from the metadata or
            repeated kernel groups for one engine group disagree.
    """
    if len(block_ids_by_kernel_group) != len(transfer_metadata.kernel_groups):
        raise ValueError(
            "block ID group count does not match transfer metadata: "
            f"got {len(block_ids_by_kernel_group)}, expected "
            f"{len(transfer_metadata.kernel_groups)}"
        )

    result: dict[int, Sequence[int]] = {}
    for kernel_group, group_block_ids in zip(
        transfer_metadata.kernel_groups,
        block_ids_by_kernel_group,
        strict=True,
    ):
        existing = result.get(kernel_group.engine_group_id)
        if existing is not None and tuple(existing) != tuple(group_block_ids):
            raise ValueError(
                "conflicting block IDs for engine group "
                f"{kernel_group.engine_group_id} from repeated kernel groups"
            )
        result[kernel_group.engine_group_id] = group_block_ids
    return result


def infer_num_chunks(
    transfer_metadata: KVTransferMetadata,
    block_ids_by_engine_group: Mapping[int, Sequence[int]],
) -> int:
    """Infer and validate a common complete chunk count from request block IDs.

    This helper is intentionally part of the logical planner boundary. It
    validates the complete-chunk and common-range requirements before callers
    bind a plan to device buffers, storage objects, or transports.

    Args:
        transfer_metadata: Immutable kernel-group geometry.
        block_ids_by_engine_group: Request block IDs keyed by engine-group ID.

    Returns:
        The common number of complete LMCache chunks represented by all kernel
        groups, or zero when the metadata contains no kernel groups.

    Raises:
        ValueError: If a referenced engine group is missing, an ID sequence
            ends mid-chunk, or kernel groups cover different chunk counts.
    """
    num_chunks: int | None = None
    for kernel_group in transfer_metadata.kernel_groups:
        engine_block_ids = block_ids_by_engine_group.get(kernel_group.engine_group_id)
        if engine_block_ids is None:
            raise ValueError(
                f"missing block IDs for engine group {kernel_group.engine_group_id}"
            )
        if len(engine_block_ids) % kernel_group.blocks_per_chunk != 0:
            raise ValueError(
                f"engine group {kernel_group.engine_group_id} has "
                f"{len(engine_block_ids)} block IDs, which does not form "
                "complete LMCache chunks"
            )
        group_num_chunks = len(engine_block_ids) // kernel_group.blocks_per_chunk
        if num_chunks is None:
            num_chunks = group_num_chunks
        elif group_num_chunks != num_chunks:
            raise ValueError("kernel groups cover different numbers of LMCache chunks")
    return num_chunks or 0


def build_transfer_plan_from_kernel_group_block_ids(
    transfer_metadata: KVTransferMetadata,
    block_ids_by_kernel_group: Sequence[Sequence[int]],
    direction: TransferPlanDirection,
    skip_first_n_tokens: int = 0,
) -> TransferPlan:
    """Build a transfer plan directly from protocol-order request block IDs.

    The helper is a compatibility adapter for engines whose requests carry
    block IDs in kernel-group order. It normalizes repeated engine groups and
    derives the common complete chunk count before calling
    :func:`build_transfer_plan`.

    Args:
        transfer_metadata: Immutable transfer metadata snapshot.
        block_ids_by_kernel_group: Request block IDs in kernel-group order.
        direction: Whether this plan stores or retrieves KV.
        skip_first_n_tokens: Retrieve prefix to preserve without overwriting.

    Returns:
        An immutable, ordered logical transfer plan.

    Raises:
        ValueError: If request block IDs cannot be normalized into one
            complete logical transfer range.
    """
    engine_group_block_ids = map_kernel_group_block_ids_to_engine_groups(
        transfer_metadata, block_ids_by_kernel_group
    )
    return build_transfer_plan(
        transfer_metadata,
        engine_group_block_ids,
        infer_num_chunks(transfer_metadata, engine_group_block_ids),
        direction,
        skip_first_n_tokens,
    )


def build_transfer_plan_without_block_ids(
    transfer_metadata: KVTransferMetadata,
    num_chunks: int,
    direction: TransferPlanDirection,
    skip_first_n_tokens: int = 0,
) -> TransferPlan:
    """Build a logical schedule for executors that do not access engine blocks.

    Storage and transport executors bind object keys, locks, and buffers to
    the same logical object-group schedule as the worker copy executor, but
    never read or write the plan's block IDs. This helper supplies placeholder
    IDs solely to satisfy the planner's block-ID contract; callers must not
    use those IDs as engine resources.

    Args:
        transfer_metadata: Immutable transfer metadata snapshot.
        num_chunks: Number of source LMCache chunks.
        direction: Whether this plan stores or retrieves KV.
        skip_first_n_tokens: Retrieve prefix to preserve without overwriting.

    Returns:
        An immutable, ordered logical transfer plan.

    Raises:
        ValueError: If ``num_chunks`` is negative or the transfer metadata is
            invalid.
    """
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")
    required_blocks_by_engine_group: dict[int, int] = {}
    for kernel_group in transfer_metadata.kernel_groups:
        required_blocks_by_engine_group[kernel_group.engine_group_id] = max(
            required_blocks_by_engine_group.get(kernel_group.engine_group_id, 0),
            num_chunks * kernel_group.blocks_per_chunk,
        )
    return build_transfer_plan(
        transfer_metadata,
        {
            engine_group_id: tuple(range(block_count))
            for engine_group_id, block_count in required_blocks_by_engine_group.items()
        },
        num_chunks,
        direction,
        skip_first_n_tokens,
    )


def _validate_transfer_metadata_order(
    transfer_metadata: KVTransferMetadata,
) -> dict[int, KernelGroupTransferMetadata]:
    """Validate deterministic metadata ordering and return groups by ID."""
    kernel_groups_by_id: dict[int, KernelGroupTransferMetadata] = {}
    for kernel_group_id, kernel_group in enumerate(transfer_metadata.kernel_groups):
        if kernel_group.kernel_group_id != kernel_group_id:
            raise ValueError(
                "transfer_metadata.kernel_groups ordering does not match "
                "kernel_group_id"
            )
        kernel_groups_by_id[kernel_group_id] = kernel_group

    for object_group_id, object_group in enumerate(transfer_metadata.object_groups):
        if object_group.object_group_id != object_group_id:
            raise ValueError(
                "transfer_metadata.object_groups ordering does not match "
                "object_group_id"
            )
        for kernel_group_id in object_group.kernel_group_ids:
            if kernel_group_id not in kernel_groups_by_id:
                raise ValueError(
                    f"object group {object_group_id} references invalid "
                    f"kernel group {kernel_group_id}"
                )
    return kernel_groups_by_id


def export_kv_transfer_metadata(
    manager: KVLayerGroupsManager,
    tokens_per_chunk: int,
) -> KVTransferMetadata:
    """Export a path-agnostic immutable transfer metadata snapshot.

    Args:
        manager: The KV layer-groups manager.
        tokens_per_chunk: LMCache chunk size in logical tokens.

    Returns:
        An immutable transfer metadata snapshot.

    Raises:
        ValueError: If metadata is inconsistent or invalid.
    """
    if tokens_per_chunk < 1:
        raise ValueError(
            f"tokens_per_chunk must be at least one, got {tokens_per_chunk}"
        )

    attn_desc = manager.get_attn_desc()
    num_chunks_in_sw = tuple(attn_desc.num_chunks_in_sw)
    kernel_groups: list[KernelGroupTransferMetadata] = []
    for kernel_group_id, group in enumerate(manager.kernel_groups):
        if group.engine_kv_format is None:
            raise ValueError(
                f"kernel group {kernel_group_id} has no engine_kv_format (got None)"
            )

        subchunk_sw_size_tokens = manager.get_subchunk_sw_size_tokens(kernel_group_id)
        if subchunk_sw_size_tokens < 1:
            raise ValueError(
                f"kernel group {kernel_group_id} has invalid subchunk window "
                f"{subchunk_sw_size_tokens}"
            )
        tokens_per_window = min(tokens_per_chunk, subchunk_sw_size_tokens)
        blocks_per_chunk = manager.calculate_num_blocks(
            kernel_group_id, tokens_per_chunk
        )
        blocks_per_window = manager.calculate_num_blocks(
            kernel_group_id, tokens_per_window
        )
        slots_per_chunk_in_window = manager.get_slots_per_chunk_in_sw(kernel_group_id)
        if blocks_per_chunk < 1:
            raise ValueError(
                f"kernel group {kernel_group_id} has invalid blocks_per_chunk "
                f"{blocks_per_chunk}"
            )
        if blocks_per_window < 1 or blocks_per_window > blocks_per_chunk:
            raise ValueError(
                f"kernel group {kernel_group_id} has invalid blocks_per_window "
                f"{blocks_per_window} for blocks_per_chunk {blocks_per_chunk}"
            )
        if slots_per_chunk_in_window < 1:
            raise ValueError(
                f"kernel group {kernel_group_id} has invalid slots_per_chunk_in_window "
                f"{slots_per_chunk_in_window}"
            )
        if group.tokens_per_block < 1:
            raise ValueError(
                f"kernel group {kernel_group_id} has invalid tokens_per_block "
                f"{group.tokens_per_block}"
            )
        if group.slots_per_block < 1:
            raise ValueError(
                f"kernel group {kernel_group_id} has invalid slots_per_block "
                f"{group.slots_per_block}"
            )
        if group.tokens_per_block % group.slots_per_block != 0:
            raise ValueError(
                f"kernel group {kernel_group_id} has non-integral compress ratio "
                f"{group.tokens_per_block}/{group.slots_per_block}"
            )

        kernel_groups.append(
            KernelGroupTransferMetadata(
                kernel_group_id=kernel_group_id,
                engine_group_id=group.engine_group_idx,
                layer_indices=tuple(group.layer_indices),
                blocks_per_chunk=blocks_per_chunk,
                blocks_per_window=blocks_per_window,
                slots_per_chunk_in_window=slots_per_chunk_in_window,
                kv_size=group.shape_desc.kv_size,
                num_layers=group.num_layers,
                hidden_dim_size=group.hidden_dim_size,
                slots_per_block=group.slots_per_block,
                tokens_per_block=group.tokens_per_block,
                dtype=group.dtype,
                engine_kv_format=group.engine_kv_format,
            )
        )

    if len(num_chunks_in_sw) != len(manager.object_groups):
        raise ValueError(
            "attention-window metadata length does not match object-group count"
        )

    num_kernel_groups = len(kernel_groups)
    object_groups: list[ObjectGroupTransferMetadata] = []
    for object_group_id, object_group in enumerate(manager.object_groups):
        sw_size_chunks = num_chunks_in_sw[object_group_id]
        if sw_size_chunks == 0 or sw_size_chunks < -1:
            raise ValueError(
                f"object group {object_group_id} has invalid sw_size_chunks "
                f"{sw_size_chunks}"
            )

        kernel_group_ids = tuple(object_group.kernel_group_indices)
        if not kernel_group_ids:
            raise ValueError(f"object group {object_group_id} has no kernel groups")
        for kernel_group_id in kernel_group_ids:
            if kernel_group_id < 0 or kernel_group_id >= num_kernel_groups:
                raise ValueError(
                    f"object group {object_group_id} references invalid "
                    f"kernel group {kernel_group_id}"
                )

        object_groups.append(
            ObjectGroupTransferMetadata(
                object_group_id=object_group_id,
                kernel_group_ids=kernel_group_ids,
                sw_size_chunks=sw_size_chunks,
            )
        )

    return KVTransferMetadata(
        num_chunks_in_sw=num_chunks_in_sw,
        tokens_per_chunk=tokens_per_chunk,
        kernel_groups=tuple(kernel_groups),
        object_groups=tuple(object_groups),
    )


def build_kernel_group_layout(
    transfer_metadata: KVTransferMetadata,
    num_tokens: int,
    kernel_group_id: int,
) -> tuple[torch.Size, torch.dtype]:
    """Build one kernel group's ``(shape, dtype)`` layout for a token count.

    Args:
        transfer_metadata: Immutable transfer metadata snapshot.
        num_tokens: Number of logical tokens.
        kernel_group_id: Kernel group index.

    Returns:
        A tuple of ``(shape, dtype)`` for that kernel group.

    Raises:
        ValueError: If arguments are invalid or token alignment is invalid.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if kernel_group_id < 0 or kernel_group_id >= len(transfer_metadata.kernel_groups):
        raise ValueError(f"invalid kernel_group_id {kernel_group_id}")

    group = transfer_metadata.kernel_groups[kernel_group_id]
    if group.kernel_group_id != kernel_group_id:
        raise ValueError(
            "transfer_metadata.kernel_groups ordering does not match kernel_group_id"
        )

    if num_tokens % transfer_metadata.tokens_per_chunk != 0:
        raise ValueError(
            f"num_tokens ({num_tokens}) must be a multiple of tokens_per_chunk "
            f"({transfer_metadata.tokens_per_chunk})"
        )

    num_chunks = num_tokens // transfer_metadata.tokens_per_chunk
    num_slots = group.slots_per_chunk_in_window * num_chunks
    shape = torch.Size(
        (group.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
    )
    return shape, group.dtype


def build_object_group_layout_desc(
    transfer_metadata: KVTransferMetadata,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Build one object group's MemoryLayoutDesc in kernel-group layout order.

    Args:
        transfer_metadata: Immutable transfer metadata snapshot.
        num_tokens: Number of logical tokens.
        object_group_id: Object group index.

    Returns:
        A MemoryLayoutDesc for one object group.

    Raises:
        ValueError: If inputs are invalid.
    """
    if object_group_id < 0 or object_group_id >= len(transfer_metadata.object_groups):
        raise ValueError(f"invalid object_group_id {object_group_id}")

    object_group = transfer_metadata.object_groups[object_group_id]
    if object_group.object_group_id != object_group_id:
        raise ValueError(
            "transfer_metadata.object_groups ordering does not match object_group_id"
        )

    if not object_group.kernel_group_ids:
        raise ValueError(f"object group {object_group_id} has no kernel groups")

    shapes_and_dtypes = [
        build_kernel_group_layout(transfer_metadata, num_tokens, kernel_group_id)
        for kernel_group_id in object_group.kernel_group_ids
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=True)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


def build_engine_driven_object_group_layout_desc(
    transfer_metadata: KVTransferMetadata,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Build an Engine-driven pickle/SHM payload layout for one object group.

    Engine-driven transfer serializes single-plane (MLA or fused K/V) chunks
    without a leading singleton KV-plane dimension, unlike the native
    LMCache-driven buffer layout.

    Args:
        transfer_metadata: Immutable transfer metadata snapshot.
        num_tokens: Number of logical tokens.
        object_group_id: Object group index.

    Returns:
        A MemoryLayoutDesc matching the tensors returned by
        ``gather_paged_kv_to_cpu`` for this object group.
    """
    native_layout = build_object_group_layout_desc(
        transfer_metadata, num_tokens, object_group_id
    )
    object_group = transfer_metadata.object_groups[object_group_id]
    shapes = [
        shape[1:]
        if transfer_metadata.kernel_groups[kernel_group_id].kv_size == 1
        else shape
        for shape, kernel_group_id in zip(
            native_layout.shapes, object_group.kernel_group_ids, strict=True
        )
    ]
    return MemoryLayoutDesc(shapes=shapes, dtypes=native_layout.dtypes)


def has_sufficient_block_ids(
    block_ids: Sequence[Sequence[int]],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
) -> bool:
    """Return whether every group has enough block IDs for all chunks.

    Args:
        block_ids: Per-group block-ID sequences.
        blocks_per_chunk: Per-group number of blocks in one chunk.
        num_chunks: Number of chunks that must be covered.

    Returns:
        True if each group has at least ``num_chunks * blocks_per_chunk[group]``
        block IDs.

    Raises:
        ValueError: If ``num_chunks`` is negative, if any entry in
            ``blocks_per_chunk`` is less than 1, or if the two per-group
            sequences have different lengths.
    """
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")
    if any(group_blocks < 1 for group_blocks in blocks_per_chunk):
        raise ValueError("blocks_per_chunk entries must be at least one")
    return all(
        len(group_block_ids) >= num_chunks * group_blocks
        for group_block_ids, group_blocks in zip(
            block_ids, blocks_per_chunk, strict=True
        )
    )


def select_block_ids_for_window(
    block_ids: Sequence[int],
    total_blocks_per_chunk: int,
    keep_blocks_per_chunk: int,
) -> list[int]:
    """Select the trailing per-chunk block IDs required by an attention window.

    Args:
        block_ids: Block IDs for one kernel group across all chunks.
        total_blocks_per_chunk: Total number of blocks in one LMCache chunk.
        keep_blocks_per_chunk: Number of trailing blocks to keep per chunk.

    Returns:
        A new list containing the selected block IDs.

    Raises:
        ValueError: If block geometry is invalid or if ``block_ids`` does not
            contain complete chunks.
    """
    if total_blocks_per_chunk < 1:
        raise ValueError("total_blocks_per_chunk must be at least one")
    if keep_blocks_per_chunk < 1:
        raise ValueError("keep_blocks_per_chunk must be at least one")
    if keep_blocks_per_chunk > total_blocks_per_chunk:
        raise ValueError(
            "keep_blocks_per_chunk must be less than or equal to total_blocks_per_chunk"
        )
    if len(block_ids) % total_blocks_per_chunk != 0:
        raise ValueError("len(block_ids) must be a multiple of total_blocks_per_chunk")

    selected_block_ids: list[int] = []
    for start_idx in range(0, len(block_ids), total_blocks_per_chunk):
        chunk_block_ids = block_ids[start_idx : start_idx + total_blocks_per_chunk]
        selected_block_ids.extend(chunk_block_ids[-keep_blocks_per_chunk:])
    return selected_block_ids


def downsample_block_ids(
    block_ids: Sequence[Sequence[int]],
    blocks_per_chunk: Sequence[int],
    blocks_per_window: Sequence[int],
) -> list[list[int]]:
    """Downsample block IDs for each kernel group based on its keep window.

    Args:
        block_ids: Per-group block-ID sequences.
        blocks_per_chunk: Per-group total blocks per chunk.
        blocks_per_window: Per-group trailing blocks to keep per chunk.

    Returns:
        A new per-group block-ID list in the original group ordering.

    Raises:
        ValueError: If per-group sequence lengths differ, or group geometry is
            invalid for any group.
    """
    return [
        select_block_ids_for_window(group_block_ids, total_blocks, keep_blocks)
        for group_block_ids, total_blocks, keep_blocks in zip(
            block_ids, blocks_per_chunk, blocks_per_window, strict=True
        )
    ]


def compute_num_objects_to_skip(
    sw_size_chunks: int,
    num_objects: int,
    is_retrieve: bool,
) -> int:
    """Compute how many leading objects should be skipped for transfer.

    Args:
        sw_size_chunks: Attention-window size in chunks for the object group.
            ``-1`` means full attention and ``>=1`` means sliding window.
        num_objects: Number of objects in the transfer list.
        is_retrieve: Whether the transfer direction is retrieve (H2D).

    Returns:
        Number of leading objects to skip.

    Raises:
        ValueError: If ``sw_size_chunks`` is 0 or less than -1, or if
            ``num_objects`` is negative.
    """
    if sw_size_chunks != -1 and sw_size_chunks < 1:
        raise ValueError("sw_size_chunks must be -1 (full) or at least one")
    if num_objects < 0:
        raise ValueError("num_objects must be non-negative")
    if sw_size_chunks == -1:
        return 0
    if not is_retrieve:
        return 0
    return max(0, num_objects - sw_size_chunks)


def batched_iteration_with_skip(
    sequence: Sequence[ItemT],
    batch_size: int,
    skip_count: int,
) -> Generator[tuple[int, tuple[ItemT, ...]], None, None]:
    """Iterate over a sequence in batches after skipping a leading prefix.

    Args:
        sequence: The sequence to iterate over.
        batch_size: Number of items per yielded batch.
        skip_count: Number of leading items to skip.

    Yields:
        Tuples ``(batch_start_idx, batch)`` where ``batch_start_idx`` is in the
        original sequence coordinate space and ``batch`` is a tuple of values.

    Raises:
        ValueError: If ``batch_size`` is less than 1 or ``skip_count`` is
            negative.

    Note:
        If ``skip_count`` exceeds ``len(sequence)``, the iterator is exhausted
        and no batches are yielded.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    if skip_count < 0:
        raise ValueError("skip_count must be non-negative")

    seq_iter = iter(sequence)
    for _ in range(skip_count):
        next(seq_iter, None)

    batch_start_idx = skip_count
    while batch := tuple(islice(seq_iter, batch_size)):
        yield batch_start_idx, batch
        batch_start_idx += len(batch)


def recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Map chunk-space skip blocks into downsampled-window block space.

    Args:
        blocks_per_chunk: Total blocks in one chunk.
        blocks_per_window: Retained trailing blocks in one chunk.
        blocks_to_skip: Blocks to skip in full-chunk coordinates.

    Returns:
        The skip count in downsampled-window coordinates.

    Raises:
        ValueError: If geometry is invalid or ``blocks_to_skip`` is negative.
    """
    if blocks_per_chunk < 1:
        raise ValueError("blocks_per_chunk must be at least one")
    if blocks_per_window < 1:
        raise ValueError("blocks_per_window must be at least one")
    if blocks_per_window > blocks_per_chunk:
        raise ValueError(
            "blocks_per_window must be less than or equal to blocks_per_chunk"
        )
    if blocks_to_skip < 0:
        raise ValueError("blocks_to_skip must be non-negative")

    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    # For the partial tail chunk, drop the discarded prefix
    # (blocks_per_chunk - blocks_per_window) and keep only overlap in the
    # retained trailing window coordinate space.
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)
