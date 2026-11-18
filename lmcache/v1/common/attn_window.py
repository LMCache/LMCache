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
