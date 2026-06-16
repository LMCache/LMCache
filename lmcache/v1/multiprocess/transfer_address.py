# SPDX-License-Identifier: Apache-2.0
"""Placeholder transfer-channel address type for the P2P protocol.

NOTE: This is a temporary definition. The real ``TransferChannelAddress``
is owned by #3712; this module exists only so the P2P protocol definitions
and the P2P controller share a single type without a circular import
(``protocols/`` must not import ``modules/p2p_controller.py``). Replace this
module once #3712 is merged.
"""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class TransferChannelAddress:
    """Location of an object within the local transfer channel.

    Args:
        offset: Byte offset of the object inside the transfer channel.
            A negative value marks the address as invalid (object not found).
        size: Byte length of the object.
    """

    offset: int
    size: int

    def is_valid(self) -> bool:
        """Return whether this address points at a real object."""
        return self.offset >= 0
