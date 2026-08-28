# SPDX-License-Identifier: Apache-2.0
"""Shared execution-mode enum for multiprocess gRPC handlers."""

# Standard
import enum


class HandlerType(enum.Enum):
    """
    Defines how an RPC handler should be executed.

    - SYNC: Handler runs directly in the gRPC worker thread.
    - BLOCKING: Handler runs in a dedicated thread pool.
    - NON_BLOCKING: Reserved for future async handlers.
    """

    SYNC = enum.auto()
    BLOCKING = enum.auto()
    NON_BLOCKING = enum.auto()


__all__ = ["HandlerType"]
