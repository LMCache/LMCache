# SPDX-License-Identifier: Apache-2.0
"""
Base types and classes for the multiprocess protocol system.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Optional
import enum


class HandlerType(enum.Enum):
    """
    Defines how a protocol handler should be executed.

    - SYNC: Handler runs directly in the main loop (fast, non-blocking operations)
    - BLOCKING: Handler may block, run in a thread pool (I/O, slow operations)
    - NON_BLOCKING: Not supported yet (for future async handlers)
    """

    SYNC = enum.auto()
    BLOCKING = enum.auto()
    NON_BLOCKING = enum.auto()


class RequestType(enum.Enum):
    """
    Enum of all available request types in the protocol system.

    When adding a new request type:
    1. Add the enum member here
    2. Add the protocol definition in the appropriate protocols/*.py file
    3. The validation system will ensure they stay in sync

    Organized by category:
    - Engine operations: Core KV cache operations
    - Controller operations: Cache management and configuration
    - Debug operations: Testing and monitoring
    """

    # Engine operations
    REGISTER_KV_CACHE = enum.auto()
    UNREGISTER_KV_CACHE = enum.auto()
    STORE = enum.auto()
    RETRIEVE = enum.auto()
    LOOKUP = enum.auto()
    QUERY_PREFETCH_STATUS = enum.auto()
    QUERY_PREFETCH_LOOKUP_HITS = enum.auto()
    FREE_LOOKUP_LOCKS = enum.auto()
    END_SESSION = enum.auto()
    REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT = enum.auto()
    UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT = enum.auto()
    PREPARE_STORE = enum.auto()
    COMMIT_STORE = enum.auto()
    PREPARE_RETRIEVE = enum.auto()
    COMMIT_RETRIEVE = enum.auto()

    # Controller operations
    CLEAR = enum.auto()
    GET_CHUNK_SIZE = enum.auto()
    PING = enum.auto()

    # Observability operations
    REPORT_BLOCK_ALLOCATION = enum.auto()

    # Debug operations
    NOOP = enum.auto()

    # Blend operations
    CB_REGISTER_KV_CACHE = enum.auto()
    CB_UNREGISTER_KV_CACHE = enum.auto()
    CB_STORE_PRE_COMPUTED = enum.auto()
    CB_LOOKUP_PRE_COMPUTED = enum.auto()
    CB_RETRIEVE_PRE_COMPUTED = enum.auto()
    CB_STORE_FINAL = enum.auto()

    # Blend V2 operations (use CBMatchResult instead of list[tuple[int, int]])
    CB_LOOKUP_PRE_COMPUTED_V2 = enum.auto()
    CB_RETRIEVE_PRE_COMPUTED_V2 = enum.auto()

    # Blend V3 — paged-aware CB.
    CB_REGISTER_ROPE_V3 = enum.auto()
    CB_UNREGISTER_ROPE_V3 = enum.auto()
    CB_RETRIEVE_PRE_COMPUTED_V3 = enum.auto()
    CB_UNIFIED_LOOKUP = enum.auto()

    # ---------------------------------------------------------------
    # Multi-group engine-driven transfer.
    #
    # IMPORTANT: RequestType is msgspec-encoded BY VALUE on the wire.
    # New members must always be appended at the END of this enum;
    # inserting them in the middle renumbers every following member
    # and silently breaks wire compatibility between client and
    # server builds of different versions.
    # ---------------------------------------------------------------
    # Multi-group engine-driven store: one message per group to avoid
    # the 4 GiB msgspec msgpack bin limit when groups are large.
    # Payload: [key, instance_id, group_idx, cpu_data]
    COMMIT_STORE_GROUP = enum.auto()
    # Delta-store: like COMMIT_STORE_GROUP, but the worker already knows
    # that the first ``skip_count`` chunks for this group are in L2
    # (from the prior ``lookup``). The server writes only the chunks
    # provided in ``cpu_data`` at offset ``skip_count``.
    # Payload: [key, instance_id, group_idx, skip_count, cpu_data]
    COMMIT_STORE_GROUP_DELTA = enum.auto()
    # Multi-group engine-driven retrieve: one message per group, so the
    # response for a single group stays under the 4 GiB msgspec msgpack
    # bin limit (mirror of COMMIT_STORE_GROUP on the retrieve side) and
    # the server only materializes one group at a time.
    # Payload: [key, instance_id, group_idx]
    PREPARE_RETRIEVE_GROUP = enum.auto()


@dataclass
class ProtocolDefinition:
    """
    Defines the structure and behavior of a protocol request.

    Attributes:
        payload_classes: List of expected payload types in order
        response_class: Expected response type, or None if no response
        handler_type: How the handler should be executed (SYNC/BLOCKING/NON_BLOCKING)
    """

    payload_classes: list[Any]
    response_class: Optional[Any]
    handler_type: HandlerType
