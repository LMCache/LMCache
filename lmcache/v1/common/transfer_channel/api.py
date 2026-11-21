# SPDX-License-Identifier: Apache-2.0
"""
Data types exposed by the transfer channel abstraction.
"""

# Standard
from dataclasses import asdict, dataclass, field
import hashlib
import json


@dataclass(frozen=True)
class CxlArenaDescriptor:
    """Identify one owner's slab and its current incarnation.

    Args:
        pool_id: Deployment-assigned identity of the shared pool.
        offset: Byte offset of the slab header in the pool device.
        size: Usable payload bytes following the header.
        alignment: Header length and allocation alignment in bytes.
        session_id: Fresh identity each time the owner opens the slab.

    Raises:
        ValueError: If the identity or aligned range is invalid.

    The first ``alignment`` bytes contain an identity fingerprint. Readers
    verify it through their own mapping before using peer payload pointers.
    """

    pool_id: str
    offset: int
    size: int
    alignment: int
    session_id: str

    def __post_init__(self) -> None:
        if not self.pool_id or not self.session_id:
            raise ValueError("CXL pool and session identities must be non-empty")
        if self.alignment < 4096 or self.alignment & (self.alignment - 1):
            raise ValueError("CXL alignment must be a power of two >= 4096")
        if (
            self.offset < 0
            or self.offset % self.alignment
            or self.size <= 0
            or self.size % self.alignment
        ):
            raise ValueError("CXL slab offset and size must be aligned")

    def to_json(self) -> str:
        """Return a canonical JSON string for coordinator registration."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "CxlArenaDescriptor":
        """Decode registration metadata; raise ValueError for invalid fields.

        Args:
            value: JSON produced by :meth:`to_json`.

        Returns:
            The validated slab descriptor.
        """
        try:
            fields = json.loads(value)
            if not isinstance(fields, dict):
                raise ValueError("CXL arena metadata must be an object")
            for name in ("offset", "size", "alignment"):
                if type(fields.get(name)) is not int:
                    raise ValueError(f"CXL {name} must be an integer")
            for name in ("pool_id", "session_id"):
                if not isinstance(fields.get(name), str):
                    raise ValueError(f"CXL {name} must be a string")
            return cls(**fields)
        except TypeError as exc:
            raise ValueError("Invalid CXL arena metadata") from exc

    def fingerprint(self) -> bytes:
        """Return the 32-byte identity expected in the mapped slab header."""
        return hashlib.sha256(self.to_json().encode()).digest()


@dataclass(frozen=True)
class TransferChannelAddress:
    """Locate one object within a registered or mapped memory region.

    A single address corresponds to a single memory object.
    """

    offset: int
    """ The starting position of the memory object: a byte offset into the
    registered region, or into the payload of ``cxl_arena``. Negative for a
    miss. """

    size: int
    """ The size (in bytes) of the memory object. """

    cxl_arena: CxlArenaDescriptor | None = None
    """ The shared CXL slab ``offset`` is relative to; None when the transfer
    client's registered region scopes the address. """

    def is_valid(self) -> bool:
        """Whether the address is valid (non-negative offset and size)."""
        return self.offset >= 0 and self.size > 0


@dataclass
class TransferChannelTransferResult:
    """Result of querying a submitted read or write task."""

    finished: bool
    """ Whether the transfer has reached a terminal state (done or errored). """

    succeeded_mask: list[bool] = field(default_factory=list)
    """ Per-object success flags, aligned with the submitted addresses. Empty while
    the transfer is still in flight. """

    def is_finished(self) -> bool:
        """Whether the transfer reached a terminal state (done or errored)."""
        return self.finished
