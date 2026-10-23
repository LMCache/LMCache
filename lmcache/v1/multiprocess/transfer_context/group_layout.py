# SPDX-License-Identifier: Apache-2.0
"""Object-group layout shared by both sides of multi-group engine-driven transfer.

Hybrid models (e.g. full attention plus Mamba state) register several KV
groups. The LMCache-driven path splits them into object groups through
:class:`KVLayerGroupsManager`; the engine-driven path reuses the same manager
so both paths store identical objects:

- The worker builds the layout from its real KV tensors.
- The server builds it from ``torch.device("meta")`` tensors carrying the
  worker's shapes, strides and dtypes, plus its own object-group flags.

Both sides then agree, per object group, on the memory object layout and on
which chunks of a request are transferred. Only the trailing window of
``num_chunks_in_sw`` chunks is stored or retrieved for a windowed group, the
same chunks the LMCache-driven retrieve reads. The memory object of an object
group concatenates one ``[kv_size, num_layers, slots, hidden]`` region per
member kernel group, in member order.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    engine_group_layer_indices,
)
from lmcache.v1.multiprocess.object_group_transfer import recalculate_blocks_to_skip
import lmcache.lmcache_native as lmcache_native


@dataclass(frozen=True)
class KernelGroupTransfer:
    """Transfer geometry of one kernel group for one LMCache chunk.

    Attributes:
        layer_indices: Registered KV tensor indices of the group.
        engine_kv_format: Engine KV format shared by the group's layers.
        slots_per_block: Physical slots in one engine block.
        tokens_per_block: Logical tokens covered by one engine block.
        blocks_per_chunk: Engine blocks covering one LMCache chunk.
        blocks_per_window: Trailing blocks of each chunk that are transferred
            (equal to ``blocks_per_chunk`` unless the window is sub-chunk).
        region_shape: ``[kv_size, num_layers, slots, hidden]`` region of the
            group inside one memory object.
        dtype: Element type of the region.
    """

    layer_indices: tuple[int, ...]
    engine_kv_format: "lmcache_native.EngineKVFormat"
    slots_per_block: int
    tokens_per_block: int
    blocks_per_chunk: int
    blocks_per_window: int
    region_shape: torch.Size
    dtype: torch.dtype

    @property
    def region_nbytes(self) -> int:
        """Byte size of the group's region in one memory object."""
        return self.region_shape.numel() * self.dtype.itemsize

    @property
    def chunk_shape(self) -> torch.Size:
        """Shape of one chunk as produced by :func:`gather_paged_kv_to_cpu`."""
        kv_size, num_layers, slots, hidden = self.region_shape
        if kv_size == 1:
            return torch.Size([num_layers, slots, hidden])
        return self.region_shape

    def num_blocks(self, num_tokens: int) -> int:
        """Return the whole engine blocks within ``num_tokens`` logical tokens."""
        return num_tokens // self.tokens_per_block


@dataclass(frozen=True)
class ObjectGroupTransfer:
    """Transfer layout of one object group.

    Attributes:
        kernel_group_indices: Member kernel groups, in object layout order.
        num_chunks_in_sw: Trailing chunks transferred per request; ``-1``
            transfers every chunk (full attention).
        layout_desc: Memory layout of one object of the group.
    """

    kernel_group_indices: tuple[int, ...]
    num_chunks_in_sw: int
    layout_desc: MemoryLayoutDesc

    @property
    def nbytes(self) -> int:
        """Byte size of one memory object of the group."""
        return sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(
                self.layout_desc.shapes, self.layout_desc.dtypes, strict=True
            )
        )

    def first_chunk(self, num_chunks: int) -> int:
        """Return the first transferred chunk of a ``num_chunks`` request."""
        if self.num_chunks_in_sw < 0:
            return 0
        return max(0, num_chunks - self.num_chunks_in_sw)


class EngineDrivenGroupLayout:
    """Kernel and object groups of a multi-group engine-driven registration.

    Args:
        kv_tensors: Registered KV tensors in layer order. Only shapes,
            strides and dtypes are read, so ``meta`` tensors are accepted.
        engine_group_infos: One entry per kernel group, in protocol order.
        engine_type: Serving engine that produced the KV tensors.
        layout_hints: Engine layout hints used for format detection.
        chunk_size: LMCache tokens per chunk.
        separate_object_groups: Server ``--separate-object-groups`` flag.
        full_sw_kv: Server ``--full-sw-kv`` flag.

    Raises:
        ValueError: If an engine group info carries a connector-private extra
            tag; engine-driven transfer has no aux-group support.
    """

    def __init__(
        self,
        kv_tensors: Sequence[torch.Tensor],
        engine_group_infos: Sequence[EngineGroupInfo],
        engine_type: EngineType,
        layout_hints: LayoutHints | None,
        chunk_size: int,
        separate_object_groups: bool,
        full_sw_kv: bool,
    ) -> None:
        if any(info.extra_object_group_tag for info in engine_group_infos):
            raise ValueError(
                "engine-driven transfer does not support extra object groups"
            )
        normalized, engine_kv_formats = normalize_and_discover_per_layer_formats(
            list(kv_tensors),
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        manager = KVLayerGroupsManager(
            list(normalized),
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=chunk_size,
            separate_object_groups=separate_object_groups,
        )
        if full_sw_kv:
            manager.enable_full_sw_kv()

        self._chunk_size = chunk_size
        self._kernel_groups: list[KernelGroupTransfer] = []
        for idx, group in enumerate(manager.kernel_groups):
            if group.engine_kv_format is None:
                raise ValueError(f"kernel group {idx} has no engine_kv_format")
            window_tokens = min(chunk_size, manager.get_subchunk_sw_size_tokens(idx))
            self._kernel_groups.append(
                KernelGroupTransfer(
                    layer_indices=tuple(group.layer_indices),
                    engine_kv_format=group.engine_kv_format,
                    slots_per_block=group.slots_per_block,
                    tokens_per_block=group.tokens_per_block,
                    blocks_per_chunk=manager.calculate_num_blocks(idx, chunk_size),
                    blocks_per_window=manager.calculate_num_blocks(idx, window_tokens),
                    region_shape=torch.Size(
                        (
                            group.shape_desc.kv_size,
                            group.num_layers,
                            manager.get_slots_per_chunk_in_sw(idx),
                            group.hidden_dim_size,
                        )
                    ),
                    dtype=group.dtype,
                )
            )

        self._attn_desc = manager.get_attn_desc()
        self._object_groups = [
            ObjectGroupTransfer(
                kernel_group_indices=tuple(group.kernel_group_indices),
                num_chunks_in_sw=window,
                layout_desc=MemoryLayoutDesc(
                    shapes=[
                        self._kernel_groups[k].region_shape
                        for k in group.kernel_group_indices
                    ],
                    dtypes=[
                        self._kernel_groups[k].dtype for k in group.kernel_group_indices
                    ],
                ),
            )
            for group, window in zip(
                manager.object_groups, self._attn_desc.num_chunks_in_sw, strict=True
            )
        ]

    @property
    def chunk_size(self) -> int:
        """LMCache tokens per chunk."""
        return self._chunk_size

    @property
    def kernel_groups(self) -> list[KernelGroupTransfer]:
        """Kernel groups, in protocol (block-id list) order."""
        return self._kernel_groups

    @property
    def object_groups(self) -> list[ObjectGroupTransfer]:
        """Object groups, in object group id order."""
        return self._object_groups

    @property
    def attn_desc(self) -> AttnWindowDesc:
        """Cross-chunk attention windows of the object groups."""
        return self._attn_desc

    def num_chunks(self, block_ids: list[list[int]]) -> int:
        """Return the chunk count of a request from its per-group block IDs.

        Args:
            block_ids: Engine block IDs, one list per kernel group.

        Returns:
            Number of LMCache chunks the block IDs cover.

        Raises:
            ValueError: If the list count does not match the kernel groups, or
                the lists cover different numbers of whole chunks.
        """
        if len(block_ids) != len(self._kernel_groups):
            raise ValueError(
                f"Got {len(block_ids)} block-id lists for "
                f"{len(self._kernel_groups)} kernel groups"
            )
        counts = set()
        for ids, group in zip(block_ids, self._kernel_groups, strict=True):
            if len(ids) % group.blocks_per_chunk != 0:
                raise ValueError(
                    f"Block count {len(ids)} is not a multiple of "
                    f"blocks_per_chunk={group.blocks_per_chunk}"
                )
            counts.add(len(ids) // group.blocks_per_chunk)
        if len(counts) != 1:
            raise ValueError(f"Kernel groups cover different chunk counts: {counts}")
        return counts.pop()

    def window_block_ids(
        self, kernel_group_idx: int, block_ids: list[int], first_chunk: int
    ) -> list[int]:
        """Return the transferred block IDs of chunks from ``first_chunk`` on.

        Keeps the trailing ``blocks_per_window`` blocks of each chunk, as the
        LMCache-driven ``downsample_and_stage_block_ids`` does.

        Args:
            kernel_group_idx: Kernel group the block IDs belong to.
            block_ids: The group's block IDs for the whole request.
            first_chunk: First chunk to keep.

        Returns:
            ``blocks_per_window`` block IDs per kept chunk, in chunk order.
        """
        group = self._kernel_groups[kernel_group_idx]
        bpc = group.blocks_per_chunk
        drop = bpc - group.blocks_per_window
        kept: list[int] = []
        for start in range(first_chunk * bpc, len(block_ids), bpc):
            kept.extend(block_ids[start + drop : start + bpc])
        return kept

    def scatter_skip_slots(
        self, kernel_group_idx: int, first_chunk: int, skip_first_n_tokens: int
    ) -> int:
        """Return the physical slots to leave unwritten when scattering.

        ``skip_first_n_tokens`` counts logical tokens from the request start;
        the result counts physical slots from the first transferred chunk, as
        :func:`scatter_cpu_to_paged_kv` expects for this group.

        Args:
            kernel_group_idx: Kernel group being scattered.
            first_chunk: First transferred chunk of the group.
            skip_first_n_tokens: Logical prefix tokens the engine already has.

        Returns:
            Physical slots to skip, a whole number of blocks.
        """
        group = self._kernel_groups[kernel_group_idx]
        skip_tokens = max(0, skip_first_n_tokens - first_chunk * self._chunk_size)
        skip_blocks = recalculate_blocks_to_skip(
            group.blocks_per_chunk,
            group.blocks_per_window,
            group.num_blocks(skip_tokens),
        )
        return skip_blocks * group.slots_per_block

    def region_views(
        self, object_group_idx: int, buffer: torch.Tensor
    ) -> list[torch.Tensor]:
        """Split one object's byte buffer into per-kernel-group chunk views.

        Args:
            object_group_idx: Object group the buffer belongs to.
            buffer: 1-D ``uint8`` buffer of exactly one object.

        Returns:
            One view per member kernel group, shaped like a gathered chunk.

        Raises:
            ValueError: If the buffer size does not match the object layout.
        """
        object_group = self._object_groups[object_group_idx]
        if buffer.dtype != torch.uint8 or buffer.numel() != object_group.nbytes:
            raise ValueError(
                f"Object group {object_group_idx} expects a uint8 buffer of "
                f"{object_group.nbytes} bytes, got {buffer.dtype} "
                f"with {buffer.numel()} elements"
            )
        views: list[torch.Tensor] = []
        offset = 0
        for k in object_group.kernel_group_indices:
            group = self._kernel_groups[k]
            end = offset + group.region_nbytes
            views.append(buffer[offset:end].view(group.dtype).view(group.chunk_shape))
            offset = end
        return views
