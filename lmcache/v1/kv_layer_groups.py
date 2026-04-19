# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


@dataclass
class KVLayerGroupInfo:
    """Identity + kernel-facing shape descriptor for a group of KV layers.

    ``shape_desc`` holds all dimensional fields the transfer kernels need.
    ``dtype`` is kept separately because ``PageBufferShapeDesc.element_size``
    cannot distinguish dtypes with equal byte width (e.g. bfloat16 vs float16).
    """

    layer_names: list[str]
    """Layer names, ordered by layer index."""
    layer_indices: list[int]
    """0-based layer indices in this group."""
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    """Kernel-facing shape descriptor shared by every layer in the group."""
    dtype: torch.dtype
    """Torch dtype of the KV cache tensors for this group."""

    _layer_indices_set: set[int] = field(init=False, repr=False)
    _layer_names_set: set[str] = field(init=False, repr=False)

    def __post_init__(self):
        self._layer_indices_set = set(self.layer_indices)
        self._layer_names_set = set(self.layer_names)

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        sd = self.shape_desc
        return (
            f"KVLayerGroupInfo(layers={len(self.layer_names)}, "
            f"indices={indices_repr}, "
            f"shape_desc=(kv={sd.kv_size}, nl={sd.nl}, nb={sd.nb}, "
            f"bs={sd.bs}, nh={sd.nh}, hs={sd.hs}, "
            f"element_size={sd.element_size}), dtype={self.dtype})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in this group."""
        return len(self.layer_names)

    @property
    def hidden_dim_size(self) -> int:
        """Hidden dimension size (``num_heads * head_size``)."""
        return self.shape_desc.nh * self.shape_desc.hs

    def contains_layer(self, layer_idx: int) -> bool:
        """Return True if *layer_idx* is in this group."""
        return layer_idx in self._layer_indices_set

    def contains_layer_name(self, layer_name: str) -> bool:
        """Return True if *layer_name* is in this group."""
        return layer_name in self._layer_names_set


class KVLayerGroupsManager:
    """Owns the per-group :class:`PageBufferShapeDesc` objects and the
    topology (``num_blocks``, ``block_size``, ``gpu_kv_format``) shared by
    every group.

    Layout parsing is delegated to :mod:`lmcache.v1.gpu_connector.utils`;
    this class only drives the grouping and look-up.
    """

    def __init__(
        self,
        kv_caches: Any,
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_blocks: int,
        block_size: int,
        layer_names: Optional[list[str]] = None,
    ) -> None:
        """Partition layers into groups with matching kernel-facing shape.

        Layers sharing both the ``(kv_size, num_heads, head_size)`` signature
        and dtype end up in the same group.

        Args:
            kv_caches: KV cache structure accepted by
                :func:`discover_gpu_kv_format`.
            gpu_kv_format: Format returned by :func:`discover_gpu_kv_format`.
            num_blocks: Number of paged blocks.
            block_size: Tokens per block.
            layer_names: Optional layer names indexed by layer position;
                synthetic ``["0", "1", ...]`` are used if omitted.

        Raises:
            ValueError: If *layer_names* length disagrees with the number
                of layers reported by :func:`get_num_layers`.
        """
        # Import here to break a circular import via
        # lmcache.v1.gpu_connector.__init__ → metadata → kv_layer_groups.
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_layer_dtype,
            get_layer_shape_signature,
            get_num_layers,
            make_page_buffer_shape_desc,
        )

        self.gpu_kv_format: "lmc_ops.GPUKVFormat" = gpu_kv_format
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        self.kv_layer_groups: list[KVLayerGroupInfo] = []

        num_layers = get_num_layers(kv_caches, gpu_kv_format)
        if num_layers == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        if layer_names is None:
            layer_names = [str(i) for i in range(num_layers)]
        elif len(layer_names) != num_layers:
            raise ValueError(
                f"layer_names has {len(layer_names)} entries but "
                f"get_num_layers returned {num_layers}"
            )

        groups_dict: dict[
            tuple[tuple[int, ...], torch.dtype], list[tuple[str, int]]
        ] = defaultdict(list)
        for idx in range(num_layers):
            sig = get_layer_shape_signature(kv_caches, gpu_kv_format, idx)
            dt = get_layer_dtype(kv_caches, gpu_kv_format, idx)
            groups_dict[(sig, dt)].append((layer_names[idx], idx))

        sorted_keys = sorted(groups_dict.keys(), key=lambda k: groups_dict[k][0][1])

        for key in sorted_keys:
            members = groups_dict[key]
            names = [name for name, _ in members]
            indices = [idx for _, idx in members]
            _, dt = key
            shape_desc = make_page_buffer_shape_desc(
                kv_caches,
                gpu_kv_format,
                layer_idx=indices[0],
                num_layers_in_group=len(indices),
                num_blocks=num_blocks,
                block_size=block_size,
            )
            self.kv_layer_groups.append(
                KVLayerGroupInfo(
                    layer_names=names,
                    layer_indices=indices,
                    shape_desc=shape_desc,
                    dtype=dt,
                )
            )

        logger.info("KV layer groups: %s", self.kv_layer_groups)

    @classmethod
    def from_layer_groups(
        cls,
        kv_layer_groups: list[KVLayerGroupInfo],
        gpu_kv_format: Optional["lmc_ops.GPUKVFormat"] = None,
        num_blocks: int = 0,
        block_size: int = 0,
    ) -> "KVLayerGroupsManager":
        """Construct from pre-built groups, bypassing the grouping pass.

        Intended for test fixtures and callers that already hold
        :class:`KVLayerGroupInfo` instances.

        Args:
            kv_layer_groups: Pre-built groups.
            gpu_kv_format: Optional cached format.
            num_blocks: Optional cached number of paged blocks.
            block_size: Optional cached tokens per block.

        Returns:
            A manager populated with the given groups.
        """
        instance = cls.__new__(cls)
        instance.kv_layer_groups = list(kv_layer_groups)
        instance.gpu_kv_format = gpu_kv_format
        instance.num_blocks = num_blocks
        instance.block_size = block_size
        return instance

    @property
    def num_groups(self) -> int:
        """Number of KV layer groups."""
        return len(self.kv_layer_groups)

    def get_group_by_layer_idx(self, layer_idx: int) -> Optional[KVLayerGroupInfo]:
        """Return the group containing *layer_idx*, or ``None`` if absent."""
        for group in self.kv_layer_groups:
            if group.contains_layer(layer_idx):
                return group
        return None

    def get_group_by_layer_name(self, layer_name: str) -> Optional[KVLayerGroupInfo]:
        """Return the group containing *layer_name*, or ``None`` if absent."""
        for group in self.kv_layer_groups:
            if group.contains_layer_name(layer_name):
                return group
        return None

    def get_layer_dtype(self, layer_idx: int) -> Optional[torch.dtype]:
        """Return the dtype for *layer_idx*, or ``None`` if not found."""
        group = self.get_group_by_layer_idx(layer_idx)
        return group.dtype if group else None

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Return the :class:`PageBufferShapeDesc` for *group_idx*.

        Raises:
            IndexError: If *group_idx* is out of range.
        """
        return self.kv_layer_groups[group_idx].shape_desc
