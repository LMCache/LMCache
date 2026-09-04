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

    Values are wire identifiers: clients and servers from different builds
    exchange them (msgspec encodes the enum by value), so the mapping is
    APPEND-ONLY. Never renumber, reuse, or remove a value -- removing a member
    with ``auto()`` renumbered every later request type twice already (#4758,
    and the py3.10 alias placement in #4897). Deprecated members keep their
    value forever; new members take the next unused integer at the end of
    their category or at the end of the enum.
    ``tests/v1/multiprocess/test_protocols.py`` pins this table.

    When adding a new request type:
    1. Add the enum member here with the next unused explicit value
    2. Add the protocol definition in the appropriate protocols/*.py file
    3. Add the pinned value to the frozen table in test_protocols.py
    4. The validation system will ensure definitions stay in sync

    Organized by category:
    - Engine operations: Core KV cache operations
    - Controller operations: Cache management and configuration
    - Debug operations: Testing and monitoring
    """

    # Engine operations
    REGISTER_KV_CACHE = 1
    UNREGISTER_KV_CACHE = 2
    REGISTER_Q_CACHE = 3
    UNREGISTER_Q_CACHE = 4
    STORE_Q = 5
    STORE = 6
    RETRIEVE = 7
    LOOKUP = 8
    QUERY_PREFETCH_STATUS = 9
    WAIT_PREFETCH_STATUS = 10
    QUERY_PREFETCH_LOOKUP_HITS = 11
    FREE_LOOKUP_LOCKS = 12
    END_SESSION = 13
    REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT = 14
    UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT = 15
    PREPARE_STORE = 16
    COMMIT_STORE = 17
    PREPARE_RETRIEVE = 18
    COMMIT_RETRIEVE = 19

    # Controller operations
    CLEAR = 20
    GET_CHUNK_SIZE = 21
    PING = 22

    # Observability operations
    REPORT_BLOCK_ALLOCATION = 23

    # Debug operations
    NOOP = 24

    # Blend operations (paged-aware; KV cache registration rides the
    # standard REGISTER_KV_CACHE). A payload-shape change means a new request
    # name -- the blend plugin dispatches on these.
    CB_REGISTER_ROPE = 25
    CB_UNREGISTER_ROPE = 26
    CB_RETRIEVE_PRE_COMPUTED = 27
    CB_UNIFIED_LOOKUP = 28

    # P2P operations
    P2P_LOOKUP_AND_LOCK = 29
    P2P_QUERY_LOOKUP_RESULTS = 30
    P2P_UNLOCK_OBJECTS = 31

    # Experimental transfer intermediate tensor
    GET_EXPERIMENTAL = 32

    # Blend protocol handshake (protocols/blend.py: BLEND_PROTOCOL_VERSION).
    # Only sent when the client enables cb.handshake (default off): the MQ
    # server has no error-reply channel and dies on an unknown request type,
    # so probing an older server is destructive.
    CB_PROTOCOL_HANDSHAKE = 33

    # Deprecated aliases: same value as the canonical member, excluded from
    # iteration. (With explicit values the py3.10 auto()-placement hazard that
    # motivated #4897 is gone, but aliases stay last by convention.)
    CB_REGISTER_ROPE_V3 = CB_REGISTER_ROPE
    CB_UNREGISTER_ROPE_V3 = CB_UNREGISTER_ROPE
    CB_RETRIEVE_PRE_COMPUTED_V3 = CB_RETRIEVE_PRE_COMPUTED


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
