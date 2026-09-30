# SPDX-License-Identifier: Apache-2.0
"""
Region-relative addresses shared by copying and borrowing peer adapters.
"""

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.internal_api import CxlArenaDescriptor


@dataclass(frozen=True)
class TransferChannelAddress:
    """Locate one object within a registered or mapped memory region.

    Args:
        offset: Byte offset from the region's payload base; negative for a miss.
        size: Object length in bytes.
        cxl_arena: Shared-slab identity for CXL, otherwise the transfer client's
            registered region supplies the address scope.
        cxl_ttl_seconds: Owner read TTL, exposed as ``read_ttl_seconds``.
            Zero means unspecified for legacy copying peers.

    Use the neutral ``MemoryRegionAddress`` alias in peer APIs. The class and
    serialized field names remain stable for existing ZMQ/gRPC clients.
    """

    offset: int
    """ The offset (in bytes) against the L1 base address. """

    size: int
    """ The size (in bytes) of the memory object. """

    cxl_arena: CxlArenaDescriptor | None = None
    """When present, offset is relative to this shared CXL slab's payload."""

    cxl_ttl_seconds: int = 0
    """Owner read-lock TTL; borrowers bound views from lookup submission time."""

    @property
    def read_ttl_seconds(self) -> int:
        """Return the owner read TTL, or zero when the peer did not provide it."""
        return self.cxl_ttl_seconds

    def is_valid(self) -> bool:
        """Whether the address is valid (non-negative offset and size)."""
        return self.offset >= 0 and self.size > 0


# Preserve the serialized class identity while sharing one address type.
MemoryRegionAddress = TransferChannelAddress


@dataclass
class TransferChannelReadResult:
    """Result of querying a submitted read task."""

    finished: bool
    """ Whether the transfer has reached a terminal state (done or errored). """

    succeeded_mask: list[bool] = field(default_factory=list)
    """ Per-object success flags, aligned with the submitted addresses. Empty while
    the transfer is still in flight. """

    def is_finished(self) -> bool:
        """Whether the transfer reached a terminal state (done or errored)."""
        return self.finished
