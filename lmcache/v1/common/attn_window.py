# SPDX-License-Identifier: Apache-2.0
"""Per-object-group attention windows, independent of either endpoint."""

# Standard
from dataclasses import dataclass
from typing import Literal, get_args

GroupKind = Literal["attention", "recurrent", "aux"]
"""Object-group kind label: attention KV, recurrent state pages, or a
connector-private aux group. Derived server-side from
``EngineGroupInfo.extra_object_group_tag``; never sent on the wire."""


@dataclass(frozen=True)
class AttnWindowDesc:
    """Per-object-group cross-chunk attention windows, in LMCache chunks.

    ``num_chunks_in_sw[g]`` is the number of trailing prefix chunks that must
    be present for object group ``g`` to serve a cache hit. ``-1`` means full
    attention (the whole prefix); ``w >= 1`` is a sliding window of ``w``
    chunks.
    """

    num_chunks_in_sw: list[int]

    world_size: int = 1
    """Number of kv_rank shards per chunk: the TP world size for
    head-sharded models, pipeline stages times DCP size for MLA."""

    group_kinds: tuple[GroupKind, ...] = ()
    """Optional per-group kind labels parallel to ``num_chunks_in_sw``.
    Empty when the producer predates kinds (treat every group as
    attention)."""

    _VALID_GROUP_KINDS = frozenset(get_args(GroupKind))

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(
                f"AttnWindowDesc: world_size must be >= 1, got {self.world_size}"
            )
        for w in self.num_chunks_in_sw:
            if w == 0 or w < -1:
                raise ValueError(
                    "AttnWindowDesc: each window must be -1 (full attention) "
                    f"or >= 1 chunk, got {w}"
                )
        if self.group_kinds:
            if len(self.group_kinds) != len(self.num_chunks_in_sw):
                raise ValueError(
                    f"AttnWindowDesc: group_kinds has {len(self.group_kinds)} "
                    f"entries but num_chunks_in_sw has "
                    f"{len(self.num_chunks_in_sw)}"
                )
            bad = set(self.group_kinds) - self._VALID_GROUP_KINDS
            if bad:
                raise ValueError(f"AttnWindowDesc: unknown group kinds {bad!r}")

    @property
    def num_object_groups(self) -> int:
        """Number of object groups this descriptor covers."""
        return len(self.num_chunks_in_sw)

    def is_full_attention(self, object_group_idx: int) -> bool:
        """Whether the object group depends on the entire prefix.

        Args:
            object_group_idx: 0-based object group index.

        Returns:
            True if the group attends to the whole prefix, False if it uses a
            bounded sliding window.
        """
        return self.num_chunks_in_sw[object_group_idx] < 0


DEFAULT_ATTN_WINDOW_DESC = AttnWindowDesc(num_chunks_in_sw=[-1])
"""A single full-attention object group; the default when no per-object-group
windows are supplied."""


def recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Re-calculate the number of blocks to skip for a batch of chunks based
    on the blocks per chunk and blocks per sliding window WHEN the window
    size is smaller than the lmcache chunk size.

    Args:
        blocks_per_chunk: The total number of blocks in one chunk for the
            current group.
        blocks_per_window: The number of blocks in the sliding window
            for the current group. Should be less than or equal to
            blocks_per_chunk.
        blocks_to_skip: The number of blocks to skip.

    Returns:
        The re-calculated number of blocks to skip for the current batch of
        chunks.
    """
    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)
