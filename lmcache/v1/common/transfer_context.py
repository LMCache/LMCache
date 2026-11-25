# SPDX-License-Identifier: Apache-2.0
"""Engine-driven transfer metadata shared by workers and the server."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any

# First Party
from lmcache.v1.common.api import MemoryLayoutDesc


@dataclass
class EngineDrivenContextMetadata:
    """Non-GPU context layout metadata for non-CUDA workers.

    Attributes:
        layout_desc: Memory layout descriptor used to interpret chunk payloads
            (group 0's layout for multi-group workers).
        block_size: Number of tokens per paged block.
        use_mla: Whether the worker KV format is MLA.
        group_layouts: Per-LMCache-group layouts for multi-group (hybrid-KV)
            workers, in protocol group order. ``None`` means single-group.
        group_windows: Per-group trailing-chunk windows parallel to
            ``group_layouts`` (``-1`` = full attention). A retrieve serves
            only each group's in-window suffix. ``None`` means every group is
            full attention.
    """

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool
    group_layouts: "list[MemoryLayoutDesc] | None" = None
    group_windows: "list[int] | None" = None


@dataclass(frozen=True)
class ShmSlotDescriptor:
    """Describe one tensor slot in the shared-memory pool.

    Args:
        offset: Byte offset into the shared-memory pool.
        length: Byte length of the slot.
        shape: Logical tensor shape to view at the slot.
        dtype: Torch dtype attribute name, such as ``"bfloat16"``.
    """

    offset: int
    length: int
    shape: list[int]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the slot descriptor into the MQ context schema.

        Returns:
            Dict payload shared between the server and worker for one SHM slot.
        """
        return {
            "offset": self.offset,
            "length": self.length,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShmSlotDescriptor":
        """Parse a slot descriptor from the MQ context schema.

        Args:
            d: Mapping containing ``offset``, ``length``, ``shape``, and
                ``dtype`` fields.

        Returns:
            Parsed immutable slot descriptor.

        Raises:
            KeyError: If any required field is missing.
            TypeError: If ``shape`` cannot be converted with ``list(...)``.
            ValueError: If numeric fields cannot be coerced to integers.
        """
        return cls(
            offset=int(d["offset"]),
            length=int(d["length"]),
            shape=list(d["shape"]),
            dtype=str(d["dtype"]),
        )
