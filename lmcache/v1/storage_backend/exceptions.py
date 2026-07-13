# SPDX-License-Identifier: Apache-2.0
"""Storage backend error types.

Exceptions raised by the storage backends when a retrieve or store
operation cannot complete. Callers that can degrade gracefully (for
example, by treating a failed retrieve as a cache miss and recomputing)
should catch these rather than a bare ``Exception``.
"""


class StorageBackendError(Exception):
    """Base class for all storage backend errors."""


class StagingAllocationError(StorageBackendError):
    """A CPU staging buffer could not be allocated for a retrieve.

    Raised by the disk/GDS retrieve path when the CPU staging pool is
    exhausted and ``busy_loop=False`` allocation returns ``None``. The
    keys being retrieved were already committed (pinned) by a preceding
    lookup, so a partial result cannot be represented safely; the backend
    rolls back every resource it acquired for the aborted call before
    raising this so the caller can treat the whole retrieve as a miss.
    """
