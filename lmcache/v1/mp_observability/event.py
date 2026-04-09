# SPDX-License-Identifier: Apache-2.0

"""Unified event model for the MP observability system."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """All observable event types in the MP system.

    Naming convention: ``<COMPONENT>_<OPERATION>`` or
    ``<COMPONENT>_<OPERATION>_<PHASE>`` for start/end pairs.
    """

    # L1 Manager events
    L1_READ_RESERVED = "l1.read.reserved"
    L1_READ_FINISHED = "l1.read.finished"
    L1_WRITE_RESERVED = "l1.write.reserved"
    L1_WRITE_FINISHED = "l1.write.finished"
    L1_WRITE_FINISHED_AND_READ_RESERVED = "l1.write_finished_and_read_reserved"
    L1_KEYS_EVICTED = "l1.keys.evicted"

    # StorageManager events
    SM_READ_PREFETCHED = "sm.read.prefetched"
    SM_READ_PREFETCHED_FINISHED = "sm.read.prefetched_finished"
    SM_WRITE_RESERVED = "sm.write.reserved"
    SM_WRITE_FINISHED = "sm.write.finished"

    # L2 Store Controller events
    L2_STORE_SUBMITTED = "l2.store.submitted"
    L2_STORE_COMPLETED = "l2.store.completed"

    # L2 Prefetch Controller events
    L2_PREFETCH_LOOKUP_SUBMITTED = "l2.prefetch.lookup.submitted"
    L2_PREFETCH_LOOKUP_COMPLETED = "l2.prefetch.lookup.completed"
    L2_PREFETCH_LOAD_SUBMITTED = "l2.prefetch.load.submitted"
    L2_PREFETCH_LOAD_COMPLETED = "l2.prefetch.load.completed"

    # MP Server request-level events (start/end pairs)
    MP_STORE_START = "mp.store.start"
    MP_STORE_END = "mp.store.end"
    MP_RETRIEVE_START = "mp.retrieve.start"
    MP_RETRIEVE_END = "mp.retrieve.end"
    MP_LOOKUP_PREFETCH_START = "mp.lookup_prefetch.start"
    MP_LOOKUP_PREFETCH_END = "mp.lookup_prefetch.end"

    # vLLM block allocation events
    MP_VLLM_BLOCK_ALLOCATION = "mp.vllm.block_allocation"


@dataclass
class Event:
    """A single observable event in the MP system.

    Attributes:
        event_type: The type of event.
        timestamp: Wall-clock time (``time.time()``) stamped by
            ``EventBus.publish()`` at the moment it is called — not when the
            drain thread processes the event.  For CUDA host-callback events
            this captures GPU-accurate timing.
        metadata: Flat key-value payload.  Contents depend on ``event_type``;
            see the metadata contracts in DESIGN.md Section 2.7.
        session_id: Caller-provided ID for correlating start/end pairs.
    """

    event_type: EventType
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
