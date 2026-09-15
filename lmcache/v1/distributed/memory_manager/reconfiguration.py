# SPDX-License-Identifier: Apache-2.0
"""HTTP-mappable error type for runtime L1 memory reconfiguration.

This module provides the L1 counterpart to
:class:`~lmcache.v1.distributed.l2_adapters.reconfiguration.L2ReconfigureError`.
L1 currently uses concrete memory-manager and arena-status types. A generic L1
reconfiguration status and manager protocol can be added here if additional L1
backends gain runtime reconfiguration support. The L1 memory manager translates
allocator failures into this error while the allocator remains free of HTTP
concerns.
"""


class L1ReconfigureError(RuntimeError):
    """HTTP-mappable runtime L1 reconfiguration error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Create a runtime reconfiguration error.

        Args:
            status_code: HTTP status code the API should return.
            message: Human-readable error message.
            payload: Optional response body. When omitted, ``{"error": message}``
                is used.
        """
        super().__init__(message)
        self.status_code = status_code
        self.payload: dict[str, object] = (
            payload if payload is not None else {"error": message}
        )
