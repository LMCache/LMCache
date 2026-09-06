# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List

# Third Party
import msgspec


class OffloadMsg(msgspec.Struct):
    """Message for Offloading"""

    hashes: List[int]
    slot_mapping: List[int]
    offsets: List[int]

    def describe(self) -> str:
        return (
            f"OffloadMsg(hashes={self.hashes}, "
            f"slot_mapping={self.slot_mapping}, "
            f"offsets={self.offsets})"
        )


class OffloadRetMsg(msgspec.Struct):
    """Outcome of processing an offload request.

    ``success`` is false if the request is invalid or the engine call raises.
    True means the engine call returned normally, not that data was persisted.
    The engine may skip a store or complete it asynchronously.
    """

    success: bool

    def describe(self) -> str:
        return f"OffloadRetMsg(success={self.success})"
