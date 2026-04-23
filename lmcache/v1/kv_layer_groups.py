# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache

logger = init_logger(__name__)


@dataclass
class KVLayerGroupInfo:
    """Identity + kernel-facing shape descriptor for a group of KV layers.

    ``shape_desc`` holds all dimensional fields the transfer kernels need.
    ``dtype`` is kept separately because ``PageBufferShapeDesc.element_size``
    cannot distinguish dtypes with equal byte width (e.g. bfloat16 vs float16).
    """

    layer_indices: list[int]
    """0-based layer indices in this group."""
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    """Kernel-facing shape descriptor shared by every layer in the group."""
    dtype: torch.dtype
    """Torch dtype of the KV cache tensors for this group."""

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        sd = self.shape_desc
        return (
            f"KVLayerGroupInfo(layers={len(self.layer_indices)}, "
            f"indices={indices_repr}, "
            f"shape_desc=(kv={sd.kv_size}, nl={sd.nl}, nb={sd.nb}, "
            f"bs={sd.bs}, nh={sd.nh}, hs={sd.hs}, "
            f"element_size={sd.element_size}), dtype={self.dtype})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in this group."""
        return len(self.layer_indices)

    @property
    def hidden_dim_size(self) -> int:
        """Hidden dimension size (``num_heads * head_size``)."""
        return self.shape_desc.nh * self.shape_desc.hs


class KVLayerGroupsManager:
    """Owns the per-group :class:`PageBufferShapeDesc` objects.

    Layout parsing is delegated to :mod:`lmcache.v1.gpu_connector.utils`;
    this class only drives the grouping and look-up.
    """

    def __init__(
        self,
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_blocks: int,
        block_size: int,
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
        """
        # Import here to break a circular import via
        # lmcache.v1.gpu_connector.__init__ → metadata → kv_layer_groups.
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_dtype,
            get_head_size,
            get_num_heads,
            get_num_layers,
            is_mla,
            make_page_buffer_shape_desc,
        )

        self.kv_layer_groups: list[KVLayerGroupInfo] = []

        num_layers = get_num_layers(kv_caches, gpu_kv_format)
        if num_layers == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        # Grouping key: two layers are kernel-equivalent iff they share
        # (kv_size, num_heads, head_size, dtype).
        mla = is_mla(gpu_kv_format)
        kv_size = 1 if mla else 2
        groups_dict: dict[tuple[int, int, int, torch.dtype], list[int]] = defaultdict(
            list
        )
        for idx in range(num_layers):
            nh = 1 if mla else get_num_heads(kv_caches, gpu_kv_format, idx)
            hs = get_head_size(kv_caches, gpu_kv_format, idx)
            dt = get_dtype(kv_caches, gpu_kv_format, idx)
            groups_dict[(kv_size, nh, hs, dt)].append(idx)

        sorted_keys = sorted(groups_dict.keys(), key=lambda k: groups_dict[k][0])

        for key in sorted_keys:
            indices = groups_dict[key]
            _, _, _, dt = key
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
                    layer_indices=indices,
                    shape_desc=shape_desc,
                    dtype=dt,
                )
            )

        logger.info("KV layer groups: %s", self.kv_layer_groups)

    @property
    def num_groups(self) -> int:
        """Number of KV layer groups."""
        return len(self.kv_layer_groups)

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Return the :class:`PageBufferShapeDesc` for *group_idx*.

        Raises:
            IndexError: If *group_idx* is out of range.
        """
        return self.kv_layer_groups[group_idx].shape_desc
