# SPDX-License-Identifier: Apache-2.0
"""
Class for distributed storage manager internal API data structures
"""

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.v1.multiprocess.distributed.api import ObjectKey
from lmcache.v1.multiprocess.distributed.error import L1ObjectManagerError


@dataclass
class L1OperationResult:
    """
    The result of an operation on the L1 object manager
    """

    error: L1ObjectManagerError = L1ObjectManagerError.SUCCESS
    """ The error code of the operation """

    success_keys: list[ObjectKey] = field(default_factory=list)
    """ The list of keys that were successfully processed """

    failed_keys: list[ObjectKey] = field(default_factory=list)
    """ The list of keys that failed to be processed """

    failed_reasons: list[L1ObjectManagerError] = field(default_factory=list)
    """ The list of reasons for the failures """

    skipped_keys: list[ObjectKey] = field(default_factory=list)
    """ The list of keys that were skipped """

    def add_success(self, key: ObjectKey) -> None:
        """Adds a key to the list of successfully processed keys."""
        self.success_keys.append(key)

    def add_error(self, key: ObjectKey, error: L1ObjectManagerError) -> None:
        """Adds a key to the list of failed keys with the corresponding error."""
        self.failed_keys.append(key)
        self.failed_reasons.append(error)
        self.error = self.error.mix_error(error)

    def add_skipped(self, key: ObjectKey) -> None:
        """Adds a key to the list of skipped keys."""
        self.skipped_keys.append(key)

    def mark_success_as_skipped(self) -> None:
        """Moves all successfully processed keys to the skipped list."""
        self.skipped_keys.extend(self.success_keys)
        self.success_keys.clear()

    def is_successful(self) -> bool:
        """Returns True if the operation was successful, False otherwise."""
        return self.error == L1ObjectManagerError.SUCCESS
