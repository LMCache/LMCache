# SPDX-License-Identifier: Apache-2.0
"""Planning for the layer-major KV staging layout.

The layout is a property of the *object*, not of either transfer mode: the
per-chunk module stores objects in it and the layer-wise module retrieves
them from it, so both have to agree byte for byte. Keeping the planning here
lets each of them depend on the layout rather than on each other.

Layer-major orders an object by model depth -- depth 0's caches, then depth
1's, and so on -- instead of grouping each kernel group's layers together.
That is what makes a range of consecutive depths a single contiguous copy,
at the cost of scattering any one kernel group through the object. A launch
still covers several of a group's layers at once, because the layers of a
group are usually evenly spread and a constant stride is enough to reach
them; see :func:`uniform_stride_runs`.

One thing the layout is *not* recorded in is ``MemoryLayoutDesc``, which
names an object's contents as one entry per kernel group -- that reads as a
kernel-group-major concatenation. It is accurate today only because every
consumer uses the descriptor to size or allocate a buffer, which is order
independent.

TODO: teach ``MemoryLayoutDesc`` to carry the layout (and the per-layer
stride) before any consumer reshapes an object by those shapes -- a serde
reconstructing per-kernel-group tensors, or a remote backend
re-materialising them. Such a consumer would silently misread a layer-major
object.
"""

# Standard
from typing import Any, Callable, Sequence

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Whether staging buffers may auto-select the layer-major KV layout.
#
# The layout is a property of the *deployment*, not of a single transfer: the
# layer-wise path loads objects that the per-chunk path stored, so both must
# agree. Only a node started with ``--layerwise-batch > 0`` benefits from
# layer-major, so the layer-wise transfer module opts in and every per-chunk
# deployment keeps the kernel-group-major layout it has always used.
_LAYER_MAJOR_STAGING_ENABLED = False


def set_layer_major_staging_enabled(enabled: bool) -> None:
    """Opt this process in or out of the layer-major staging layout.

    Must be called before any cache context is created: the layout is baked
    into the staging buffer's offset map at construction time.
    """
    global _LAYER_MAJOR_STAGING_ENABLED
    _LAYER_MAJOR_STAGING_ENABLED = enabled


def layer_major_staging_enabled() -> bool:
    """Whether staging buffers may auto-select the layer-major layout."""
    return _LAYER_MAJOR_STAGING_ENABLED


def layer_major_placement(
    kv_groups_manager: Any,
    object_group_idx: int,
    size_for_kernel_group: Callable[[int], int],
) -> list[tuple[int, int, int]] | None:
    """Order one object group's per-layer slices by model depth.

    Returns ``[(kernel_group_idx, local_layer_idx, size_in_bytes)]`` in
    layer-major order, or ``None`` when the staging gate is off -- that
    is, for every per-chunk deployment, which keeps the legacy
    kernel-group-major order untouched.

    With the gate on, every geometry the layer-wise path accepts gets a
    placement, including the single-cache-per-layer models whose depth
    order happens to equal kernel-group-major order.  Those produce the
    same bytes either way, and placing them here too is what lets the
    layer-wise path carry one layout instead of two.

    Raises ``ValueError`` for a geometry the layer-wise path cannot
    address safely, so an unsupported registration is refused up front
    rather than transferred to a wrong offset.

    Args:
        kv_groups_manager: The registered geometry.
        object_group_idx: Which object group to place.
        size_for_kernel_group: Byte size of one kernel group's staging
            region, as the staging buffer computes it.
    """
    if not layer_major_staging_enabled():
        return None

    groups = kv_groups_manager.kernel_groups
    kernel_group_indices = kv_groups_manager.object_groups[
        object_group_idx
    ].kernel_group_indices

    entries: list[tuple[int, int, int, int]] = []
    for kernel_group_idx in kernel_group_indices:
        group = groups[kernel_group_idx]
        # Registration ordinals are a usable stand-in only when each model
        # layer registers a single cache -- precisely the case in which
        # this function returns None anyway.
        depths = group.model_depths or group.layer_indices
        num_layers = group.num_layers
        size = size_for_kernel_group(kernel_group_idx)
        if num_layers <= 0 or size % num_layers != 0:
            raise ValueError(
                f"Layer-wise transfer needs a whole number of staging "
                f"bytes per layer, but kernel group {kernel_group_idx} "
                f"stages {size} bytes across {num_layers} layers."
            )
        # A layer is transferred on its own, which relies on the kernel's
        # engine-side offset being independent of the layer axis.  That
        # holds for the per-layer formats, where the paged pointer array
        # selects the layer, but not for the fused cross-layer tensor:
        # it supplies a single base pointer and bakes an absolute
        # ``layer_idx`` into the engine offset, so a per-layer launch
        # would read whichever layer the rebased index landed on.  The
        # engines that produce this format already load every layer in
        # one kernel through their own adapter, so refuse the pairing
        # here rather than address their tensor wrongly.
        if lmcache_native.is_cross_layer(group.engine_kv_format):
            raise ValueError(
                f"Layer-wise transfer does not support the fused "
                f"cross-layer KV format of kernel group "
                f"{kernel_group_idx}: its engine tensor is addressed by "
                f"an absolute layer index, so one layer cannot be "
                f"transferred on its own. Run this model without "
                f"--layerwise-batch."
            )
        if len(depths) != num_layers:
            raise ValueError(
                f"Layer-wise transfer needs one model depth per layer, "
                f"but kernel group {kernel_group_idx} reports "
                f"{len(depths)} depths for {num_layers} layers."
            )
        per_layer = size // num_layers
        for local, depth in enumerate(depths):
            entries.append((depth, kernel_group_idx, local, per_layer))

    entries.sort()
    return [
        (kernel_group_idx, local, per_layer)
        for _depth, kernel_group_idx, local, per_layer in entries
    ]


def placement_keeps_kernel_groups_contiguous(
    placement: Sequence[tuple[int, int, int]],
) -> bool:
    """Whether a placement leaves every kernel group one contiguous run.

    Depth order and kernel-group-major order coincide when each model
    layer owns its caches in a single kernel group -- the common case.
    The staged bytes are then identical to the legacy layout, so callers
    that address a whole kernel group at once stay valid and the buffer
    can still record a kernel-group-only offset for them.  A model whose
    layers span several kernel groups interleaves those groups by depth,
    and no contiguous per-group range exists to hand out.

    Args:
        placement: Per-layer slices in depth order, as
            :func:`layer_major_placement` returns them.
    """
    seen: set[int] = set()
    current = -1
    next_local = 0
    for kernel_group_idx, local, _size in placement:
        if kernel_group_idx != current:
            if kernel_group_idx in seen:
                return False
            seen.add(kernel_group_idx)
            current = kernel_group_idx
            next_local = 0
        if local != next_local:
            return False
        next_local += 1
    return True


def resolve_layer_major_placements(
    kv_groups_manager: Any,
    size_for_kernel_group: Callable[[int], int],
) -> tuple[dict[int, list[tuple[int, int, int]] | None], bool, bool]:
    """Decide the staging layout of every object group, once.

    The layout follows from the registered geometry and is identical for
    every batch slot, so a staging buffer resolves it up front and then only
    consults the result.

    Returns the per-object-group placement -- ``None`` meaning the legacy
    kernel-group-major order -- whether the layer-major layout is in use,
    and whether it still leaves every kernel group contiguous.  The last
    is what tells whole-kernel-group callers such as blend whether they
    can address the buffer at all.

    Args:
        kv_groups_manager: The registered geometry.
        size_for_kernel_group: Byte size of one kernel group's staging
            region, as the staging buffer computes it.
    """
    placements: dict[int, list[tuple[int, int, int]] | None] = {
        object_group_idx: layer_major_placement(
            kv_groups_manager, object_group_idx, size_for_kernel_group
        )
        for object_group_idx in range(kv_groups_manager.num_object_groups)
    }
    layer_major = any(placement is not None for placement in placements.values())
    kernel_groups_contiguous = all(
        placement is None or placement_keeps_kernel_groups_contiguous(placement)
        for placement in placements.values()
    )

    if layer_major:
        logger.info(
            "Layer-major KV staging layout selected for object group(s) %s; "
            "kernel groups remain contiguous: %s",
            sorted(
                gid for gid, placement in placements.items() if placement is not None
            ),
            kernel_groups_contiguous,
        )

    return placements, layer_major, kernel_groups_contiguous


def carve_layer_major_object_group(
    placement: Sequence[tuple[int, int, int]],
    batch_idx: int,
    offset: int,
    layer_offsets: dict[tuple[int, int, int], tuple[int, int]],
) -> tuple[int, int]:
    """Lay one object group out layer-major, recording each layer's slice.

    One layer's caches are adjacent here, so a run of consecutive layers is a
    single contiguous byte range.  A kernel group is no longer contiguous,
    which is why the caller records nothing in its kernel-group-only map.

    Args:
        placement: Per-layer slices in depth order, as
            :func:`layer_major_placement` returns them.
        batch_idx: Which batch slot is being carved.
        offset: Byte offset the object group starts at.
        layer_offsets: Map to record ``(batch_idx, kernel_group_idx,
            local_layer_idx) -> (byte offset, size in bytes)`` into.

    Returns the offset just past the group, and the group's size in bytes.
    """
    object_group_size = 0
    for kernel_group_idx, local, per_layer in placement:
        layer_offsets[(batch_idx, kernel_group_idx, local)] = (offset, per_layer)
        offset += per_layer
        object_group_size += per_layer
    return offset, object_group_size


def layer_slices_from_kernel_groups(
    kv_groups_manager: Any,
    kernel_group_offsets: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Subdivide each whole-kernel-group slice into its per-layer slices.

    A group that occupies one contiguous run keeps its layers together and
    evenly sized, so one layer's slice is a plain subdivision of the group's.
    Only the layer-wise path addresses a single layer, so this lives with the
    layout rather than in the staging buffer.

    Per-chunk deployments reach this with every group. Layer-wise ones reach it
    only with the groups the placement left contiguous, which it records
    precisely so this subdivision stays valid; a group interleaved by model
    depth has no whole-group entry to subdivide, and is addressed one layer at
    a time from the placement instead.

    Args:
        kv_groups_manager: The registered geometry.
        kernel_group_offsets: ``(batch_idx, kernel_group_idx) -> (byte
            offset, size in bytes)`` for the groups that occupy one
            contiguous run.
    """
    layer_offsets: dict[tuple[int, int, int], tuple[int, int]] = {}
    for (batch_idx, kernel_group_idx), (
        offset,
        size,
    ) in kernel_group_offsets.items():
        num_layers = kv_groups_manager.kernel_groups[kernel_group_idx].num_layers
        if num_layers <= 0 or size % num_layers != 0:
            continue
        per_layer = size // num_layers
        for local in range(num_layers):
            layer_offsets[(batch_idx, kernel_group_idx, local)] = (
                offset + local * per_layer,
                per_layer,
            )
    return layer_offsets


def uniform_stride_runs(offsets: Sequence[int]) -> list[tuple[int, int, int]]:
    """Split a kernel group's layers into maximal constant-stride runs.

    One scatter launch walks the object with a single layer stride, so layers
    can be merged into one launch only while consecutive offsets differ by
    the same amount. Under the layer-major layout that holds for a kernel
    group whose layers are evenly spread through the model, and breaks
    exactly where the spread changes -- a group that also owns the very first
    layer, say.

    Args:
        offsets: Byte offsets of the layers, in the order they are laid out.

    Returns:
        ``(start_index, length, stride_bytes)`` per run; ``stride_bytes`` is 0
        for a single-layer run, where no stride is applied.
    """
    runs: list[tuple[int, int, int]] = []
    i = 0
    n = len(offsets)
    while i < n:
        j = i + 1
        stride = 0
        while j < n:
            delta = offsets[j] - offsets[j - 1]
            if j == i + 1:
                stride = delta
            elif delta != stride:
                break
            j += 1
        runs.append((i, j - i, stride if j - i > 1 else 0))
        i = j
    return runs


def build_run_shape_desc(shape_desc: Any, *, nl: int, stride_bytes: int) -> Any:
    """Clone ``shape_desc`` for one constant-stride run of ``nl`` layers.

    Two fields differ from the kernel-group-major descriptor. ``nl`` covers
    only this run rather than the whole group, and ``layer_stride_elems``
    tells the kernel how far apart the run's layers actually sit, instead of
    letting it assume the group owns everything between them.

    ``kv_interleaved`` is forced on because layer-major stores each layer's
    K and V next to each other -- that adjacency is what makes a layer a
    single slice, and hence what makes a depth range contiguous.
    """
    if stride_bytes % shape_desc.element_size != 0:
        raise ValueError(
            f"Layer stride {stride_bytes} B is not a whole number of "
            f"{shape_desc.element_size}-byte elements"
        )

    run_desc = device_ops.PageBufferShapeDesc()
    run_desc.kv_size = shape_desc.kv_size
    run_desc.nl = nl
    run_desc.nb = shape_desc.nb
    run_desc.bs = shape_desc.bs
    run_desc.nh = shape_desc.nh
    run_desc.hs = shape_desc.hs
    run_desc.element_size = shape_desc.element_size
    run_desc.block_stride_elems = shape_desc.block_stride_elems
    run_desc.kv_interleaved = True
    run_desc.layer_stride_elems = stride_bytes // shape_desc.element_size
    return run_desc


def build_layer_major_specs(
    cache_context: BaseCacheContext,
    object_group_id: int,
    block_ids_gpu: list[torch.Tensor],
    kernel_group_ids: Sequence[int],
    max_batch_size: int,
) -> tuple[list[Any], list[tuple[int, int]]]:
    """Build the specs that transfer a whole object in layer-major order.

    Used by the per-chunk path, which stages an entire object at once and so
    can merge each kernel group down to one spec per constant-stride run --
    typically one, or two for a group that also owns a leading layer.

    Returns:
        ``(kernel_group_specs, launch_units)`` where a launch unit is a
        ``(spec_index, kernel_group_id)`` pair.
    """
    groups = cache_context.kv_layer_groups_manager.kernel_groups

    kernel_group_specs: list[Any] = []
    launch_units: list[tuple[int, int]] = []

    for kernel_group_id in kernel_group_ids:
        num_layers = groups[kernel_group_id].num_layers
        offsets = [
            cache_context.get_layer_offset_in_object(
                object_group_id, kernel_group_id, local_layer_idx
            )
            for local_layer_idx in range(num_layers)
        ]
        shape_desc = cache_context.get_shape_desc(kernel_group_id)
        paged_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        block_ids_tensor = block_ids_gpu[kernel_group_id]

        for first_local, run_len, stride_bytes in uniform_stride_runs(offsets):
            # A run is consecutive in local index, so the pointer array can
            # be sliced rather than rebuilt: within a kernel group the local
            # index rises with model depth, and layer-major orders by depth.
            # The slice is a view of a tensor the cache context holds for its
            # whole life, so the address stays valid without a keepalive.
            run_paged_ptrs = paged_ptrs[first_local : first_local + run_len]

            staging_ptrs = [
                cache_context.get_temp_layer_buffer(
                    slot, kernel_group_id, first_local
                ).data_ptr()
                for slot in range(max_batch_size)
            ]

            launch_units.append((len(kernel_group_specs), kernel_group_id))
            kernel_group_specs.append(
                device_ops.KernelGroupSpec(
                    run_paged_ptrs.data_ptr(),
                    staging_ptrs,
                    build_run_shape_desc(
                        shape_desc, nl=run_len, stride_bytes=stride_bytes
                    ),
                    cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                    cache_context.get_engine_kv_format(kernel_group_id),
                    block_ids_tensor.data_ptr(),
                    block_ids_tensor.numel(),
                )
            )

    return kernel_group_specs, launch_units
