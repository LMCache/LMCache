# SPDX-License-Identifier: Apache-2.0

"""
Definition of errors for class APIs.
"""

# Standard
from typing import Union
import enum


class L1MemoryManagerError(enum.Enum):
    """Errors *Returned* by L1MemoryManager class APIs."""

    SUCCESS = enum.auto()
    """ Operation succeeded. """

    OUT_OF_MEMORY = enum.auto()
    """ Operation failed due to insufficient memory. """


class L1ObjectManagerError(enum.Enum):
    """Errors *Returned* by L1ObjectManager class APIs."""

    SUCCESS = enum.auto()
    """ Operation succeeded. """

    KEYS_NOT_FOUND = enum.auto()
    """ Expected existing keys but found keys not found. """

    KEYS_ALREADY_EXIST = enum.auto()
    """ Expected non-exist keys but found keys existed. """

    KEYS_NOT_RESERVED = enum.auto()
    """ Expected non-reserved keys but found keys reserved. """

    KEYS_NOT_COMMITTED = enum.auto()
    """ Expected committed keys but found keys not committed. """


ErrorType = Union[L1MemoryManagerError, L1ObjectManagerError]


def strerror(error: ErrorType) -> str:
    """Convert error code to human-readable string.

    Args:
        error (ErrorType): The error code.

    Returns:
        str: The human-readable string.
    """
    if isinstance(error, L1MemoryManagerError):
        if error == L1MemoryManagerError.SUCCESS:
            return "Operation succeeded."
        elif error == L1MemoryManagerError.OUT_OF_MEMORY:
            return "Operation failed due to insufficient memory."

    elif isinstance(error, L1ObjectManagerError):
        if error == L1ObjectManagerError.SUCCESS:
            return "Operation succeeded."
        elif error == L1ObjectManagerError.KEYS_NOT_FOUND:
            return "Expected existing keys but found keys not found."
        elif error == L1ObjectManagerError.KEYS_ALREADY_EXIST:
            return "Expected non-exist keys but found keys existed."
        elif error == L1ObjectManagerError.KEYS_NOT_RESERVED:
            return "Expected non-reserved keys but found keys reserved."
        elif error == L1ObjectManagerError.KEYS_NOT_COMMITTED:
            return "Expected committed keys but found keys not committed."

    return "Unknown error."
