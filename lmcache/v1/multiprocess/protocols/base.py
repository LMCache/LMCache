# SPDX-License-Identifier: Apache-2.0
"""
Base types and classes for the multiprocess protocol system.
"""

# Standard
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
