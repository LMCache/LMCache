# SPDX-License-Identifier: Apache-2.0
"""Typed results used by the DAX-Coordinated L1 client."""

# Standard
from enum import Enum
from typing import NamedTuple

# First Party
from lmcache.v1.memory_management import MemoryObj


class DaxCoordinatedL1RawResult(str, Enum):
    """Stable Python names for native DAX-Coordinated L1 operation results."""

    SUCCESS = "SUCCESS"
    NOT_FOUND = "NOT_FOUND"
    TARGET_INVALIDATING = "TARGET_INVALIDATING"
    WRITER_BUSY = "WRITER_BUSY"
    NO_FREE_BUCKET = "NO_FREE_BUCKET"
    NO_LOCAL_PAYLOAD_SLOT = "NO_LOCAL_PAYLOAD_SLOT"
    GENERATION_MISMATCH = "GENERATION_MISMATCH"
    ACTIVE_READER = "ACTIVE_READER"
    OWNER_MISMATCH = "OWNER_MISMATCH"
    INVALID_STATE = "INVALID_STATE"
    CORRUPT_FORWARD_BACKREF = "CORRUPT_FORWARD_BACKREF"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class DevDaxReservationResult(NamedTuple):
    """One client operation outcome and its optional memory view."""

    result: DaxCoordinatedL1RawResult
    memory_obj: MemoryObj | None = None
