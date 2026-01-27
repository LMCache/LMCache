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


class L1ObjectManagerError(enum.IntFlag):
    """Errors *Returned* by L1ObjectManager class APIs.

    Support mixing using bitwise OR operation. Uses IntFlag to allow
    combining multiple error flags into a single value.
    """

    SUCCESS = 0x00
    """ Operation succeeded. """

    KEYS_NOT_FOUND = 0x01
    """ Expected existing keys but found keys not found. """

    KEYS_ALREADY_EXIST = 0x02
    """ Expected non-exist keys but found keys existed. """

    KEYS_NOT_RESERVED = 0x04
    """ Expected non-reserved keys but found keys reserved. """

    KEYS_NOT_COMMITTED = 0x08
    """ Expected committed keys but found keys not committed. """

    KEYS_ALREADY_LOCKED = 0x10
    """ Expected unlocked keys but found keys already locked. """

    def has_error(self, error: "L1ObjectManagerError") -> bool:
        """Check if the error code has the specific error in it.

        Returns:
            bool: True if there is any error, False otherwise.
        """
        return self & error != 0

    def mix_error(self, error: "L1ObjectManagerError") -> "L1ObjectManagerError":
        """Mix the current error code with another error code using bitwise OR.

        Returns:
            L1ObjectManagerError: The mixed error code.
        """
        return self | error


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

        # Handle multiple errors combined with bitwise OR
        error_messages = []
        if error.has_error(L1ObjectManagerError.KEYS_NOT_FOUND):
            error_messages.append("Expected existing keys but found keys not found.")
        if error.has_error(L1ObjectManagerError.KEYS_ALREADY_EXIST):
            error_messages.append("Expected non-exist keys but found keys existed.")
        if error.has_error(L1ObjectManagerError.KEYS_NOT_RESERVED):
            error_messages.append("Expected non-reserved keys but found keys reserved.")
        if error.has_error(L1ObjectManagerError.KEYS_NOT_COMMITTED):
            error_messages.append(
                "Expected committed keys but found keys not committed."
            )
        if error.has_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED):
            error_messages.append(
                "Expected unlocked keys but found keys already locked."
            )

        if error_messages:
            return " ".join(error_messages)

    return "Unknown error."
