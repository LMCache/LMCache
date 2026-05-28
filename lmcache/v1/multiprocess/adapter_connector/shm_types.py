# SPDX-License-Identifier: Apache-2.0
"""Shared schema types for multiprocess shared-memory transport."""

# Standard
from dataclasses import dataclass
from typing import Any


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
