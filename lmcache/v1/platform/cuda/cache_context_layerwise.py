# SPDX-License-Identifier: Apache-2.0
"""Layer-wise KV staging for the CUDA cache context.

Everything in this module is reached only when the layer-wise staging gate
is on, so a per-chunk deployment keeps the kernel-group-major layout in
:mod:`lmcache.v1.platform.cuda.cache_context` exactly as it was.  The split
follows the one already made for the layer-wise connector, adapter and
transfer module: a separate file that extends the legacy path instead of
threading a branch through it.
"""

# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.modules.layer_major_plan import (
    carve_layer_major_object_group,
    layer_slices_from_kernel_groups,
    resolve_layer_major_placements,
)
from lmcache.v1.platform.cuda.cache_context import GPUCacheContext, _TempGPUBuffer


class _TempLayerMajorGPUBuffer(_TempGPUBuffer):
    """Stages each object group in model-depth order.

    Built only when the layer-wise staging gate is on, so the per-chunk
    deployment never reaches this class and its base keeps the
    kernel-group-major carve it has always had.  The carve is re-run here
    rather than parameterised in the base so that the legacy ordering stays
    one unbroken block of code with no layer-wise branch threaded through
    it -- the same reason the ZMQ client carries a separate streaming
    submit instead of a flag on the per-chunk one.

    Re-carving after ``super().__init__`` is safe because the buffer's size
    comes from :meth:`_get_size_for_single_batch`, which depends on the
    registered geometry and not on the order the slices are laid out in:
    both layouts place the same slices in the same number of bytes.
    """

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        super().__init__(
            kv_layer_groups_manager,
            lmcache_tokens_per_chunk,
            device,
            max_batch_size,
        )
        # (batch_idx, kernel_group_idx, local_layer_idx) -> (byte offset of
        # that one layer's slice, size in bytes).  Only this layout addresses
        # a single layer, so the map and the flags below live here instead of
        # on the base class, which never fills them.
        self._offset_map_layer: dict[tuple[int, int, int], tuple[int, int]] = {}
        self._layer_major = False
        self._kernel_groups_contiguous = True

        self._recarve_in_depth_order()

    def _recarve_in_depth_order(self) -> None:
        """Replaces the base's kernel-group-major offsets with depth order.

        Object groups the layout module declines to place keep the base's
        ordering, so a deployment can mix the two.
        """
        (
            self._layer_placements,
            self._layer_major,
            self._kernel_groups_contiguous,
        ) = resolve_layer_major_placements(
            self._kv_groups_manager, self._get_size_for_kernel_group
        )

        self._offset_map.clear()
        self._offset_map_kernel_group_only.clear()
        self._offset_map_object_group_only.clear()
        self._offset_map_layer.clear()

        offset = 0
        for batch_idx in range(self._max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_size = 0
                object_group_start_offset = offset
                placement = self._layer_placements[object_group_idx]

                if placement is None:
                    for kernel_group_idx in self._kv_groups_manager.object_groups[
                        object_group_idx
                    ].kernel_group_indices:
                        size = self._get_size_for_kernel_group(kernel_group_idx)
                        self._offset_map[
                            (batch_idx, object_group_idx, kernel_group_idx)
                        ] = (offset, size)
                        self._offset_map_kernel_group_only[
                            (batch_idx, kernel_group_idx)
                        ] = (offset, size)

                        offset += size
                        object_group_size += size
                else:
                    offset, object_group_size = carve_layer_major_object_group(
                        placement, batch_idx, offset, self._offset_map_layer
                    )
                    if self._kernel_groups_contiguous:
                        # Depth order coincided with kernel-group-major
                        # order, so each group is still one run of bytes.
                        # Record it so whole-kernel-group callers keep
                        # working on the layer-wise path too.
                        for kernel_group_idx in self._kv_groups_manager.object_groups[
                            object_group_idx
                        ].kernel_group_indices:
                            size = self._get_size_for_kernel_group(kernel_group_idx)
                            first = self._offset_map_layer[
                                (batch_idx, kernel_group_idx, 0)
                            ][0]
                            self._offset_map[
                                (batch_idx, object_group_idx, kernel_group_idx)
                            ] = (first, size)
                            self._offset_map_kernel_group_only[
                                (batch_idx, kernel_group_idx)
                            ] = (first, size)

                self._offset_map_object_group_only[(batch_idx, object_group_idx)] = (
                    object_group_start_offset,
                    object_group_size,
                )

        self._offset_map_layer.update(
            layer_slices_from_kernel_groups(
                self._kv_groups_manager, self._offset_map_kernel_group_only
            )
        )

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Refuses whole-group addressing when the groups are interleaved.

        Depth order and kernel-group-major order coincide whenever every
        model layer owns its caches in a single kernel group, and then the
        inherited lookup is exactly right.  Only a model whose layers span
        several kernel groups leaves no per-group range to hand back.
        """
        if not self._kernel_groups_contiguous:
            raise ValueError(
                "A kernel group is not contiguous for a model whose layers "
                "span several kernel groups; use get_temp_layer_buffer() "
                "to address one layer"
            )

        return super().get_temp_kernel_group_buffer(batch_idx, kernel_group_idx)

    @property
    def layer_major(self) -> bool:
        """Whether the object groups are staged in model-depth order.

        True for every layer-wise deployment and false for every
        per-chunk one: the layout follows from the staging gate, not
        from the model.  It says nothing about whether a kernel group is
        still contiguous -- see :attr:`kernel_groups_contiguous`.
        """
        return self._layer_major

    @property
    def kernel_groups_contiguous(self) -> bool:
        """Whether each kernel group is one contiguous run of bytes.

        Always true under the kernel-group-major layout, and true under
        the layer-major layout as well whenever every model layer owns
        its caches in a single kernel group -- then depth order and
        kernel-group-major order coincide.  Only a model whose layers
        span several kernel groups interleaves them, leaving no
        per-group range for callers that address a whole group.
        """
        return self._kernel_groups_contiguous

    def get_temp_layer_buffer(
        self, batch_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> torch.Tensor:
        """Returns the temp GPU buffer holding a single layer's caches.

        Valid under both layouts, which is what lets callers stay layout
        agnostic.  The returned view keeps the kernel group's shape with the
        layer axis pinned to length one.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.
            local_layer_idx: Position of the layer within that kernel group,
                i.e. an index into its ``layer_indices`` -- not a model depth.

        Raises:
            ValueError: If any index is out of range.
        """
        key = (batch_idx, kernel_group_idx, local_layer_idx)
        if key not in self._offset_map_layer:
            raise ValueError(
                f"Invalid batch_idx {batch_idx}, kernel_group_idx "
                f"{kernel_group_idx} or local_layer_idx {local_layer_idx}"
            )

        offset, size = self._offset_map_layer[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        layer_shape = torch.Size((shape[0], 1, *shape[2:]))
        return self._temp_buffer[offset : offset + size].view(dtype).view(layer_shape)

    def get_layer_offset_in_object(
        self, object_group_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> int:
        """Byte offset of one layer's caches from the start of its object.

        Layout-agnostic: it reads the same per-layer map both layouts fill,
        so callers do not have to know which one was selected.  Relative to
        the object group because that is what an LMCache memory object holds.
        """
        entry = self._offset_map_layer.get((0, kernel_group_idx, local_layer_idx))
        object_group = self._offset_map_object_group_only.get((0, object_group_idx))
        if entry is None or object_group is None:
            raise ValueError(
                "No staging offset for object_group="
                f"{object_group_idx}, kernel_group={kernel_group_idx}, "
                f"local_layer={local_layer_idx}"
            )
        return entry[0] - object_group[0]


class GPULayerwiseCacheContext(GPUCacheContext):
    """CUDA cache context whose staging buffer is carved in depth order.

    Selected by the platform factory when the layer-wise staging gate is on.
    Every request type other than the per-layer accessors below behaves
    exactly as it does on the per-chunk context, which this class inherits
    unchanged.
    """

    @property
    def _layer_buffer(self) -> _TempLayerMajorGPUBuffer:
        """The staging buffer, narrowed to the layout this context builds."""
        return cast(_TempLayerMajorGPUBuffer, self._temp_buffer)

    def _make_temp_buffer(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int,
        full_sw_kv: bool,
    ) -> _TempGPUBuffer:
        """Builds a depth-ordered staging buffer, refusing blend if it cannot
        serve it.

        CacheBlend addresses the staging buffer one kernel group at a time
        and takes data pointers into those contiguous regions, which a model
        whose layers span several kernel groups cannot offer: its groups are
        interleaved by depth.  ``full_sw_kv`` is set from ``engine_type ==
        "blend"`` when the server context is built, so it is the blend signal
        available here.  Refuse the combination up front rather than let
        blend read misaddressed bytes.
        """
        temp_buffer = _TempLayerMajorGPUBuffer(
            kv_layer_groups_manager=kv_layer_groups_manager,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=device,
            max_batch_size=max_batch_size,
        )
        if full_sw_kv and not temp_buffer.kernel_groups_contiguous:
            raise ValueError(
                "Blending is not supported for models whose layers span "
                "several kernel groups: their KV staging interleaves the "
                "groups by model depth, leaving no contiguous "
                "per-kernel-group region for blend to address. Disable "
                "blending for this model."
            )

        return temp_buffer

    @property
    def layer_major(self) -> bool:
        """Whether the staging buffer uses the layer-major layout."""
        return self._layer_buffer.layer_major

    @property
    def kernel_groups_contiguous(self) -> bool:
        """Whether each kernel group is one contiguous staging range."""
        return self._layer_buffer.kernel_groups_contiguous

    def get_temp_layer_buffer(
        self, batch_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> torch.Tensor:
        """Returns the temporary GPU buffer holding one layer's caches.

        Valid under both layouts; see
        :meth:`_TempGPUBuffer.get_temp_layer_buffer`.
        """
        return self._layer_buffer.get_temp_layer_buffer(
            batch_idx, kernel_group_idx, local_layer_idx
        )

    def get_layer_offset_in_object(
        self, object_group_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> int:
        """Byte offset of one layer's caches within its object group."""
        return self._layer_buffer.get_layer_offset_in_object(
            object_group_idx, kernel_group_idx, local_layer_idx
        )
