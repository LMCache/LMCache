# SPDX-License-Identifier: Apache-2.0
"""Generic runtime reconfiguration protocol for L2 adapters."""

# Future
from __future__ import annotations

# Standard
from typing import Callable, Optional, Protocol, TypedDict, runtime_checkable


class L2ReconfigureError(RuntimeError):
    """HTTP-mappable runtime L2 reconfiguration error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        payload: Optional[dict[str, object]] = None,
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
        self.payload = payload if payload is not None else {"error": message}


class L2ReconfigureStatus(TypedDict):
    """Standard status envelope for runtime-reconfigurable L2 adapters."""

    backend: str
    supported_operations: list[str]
    status: dict[str, object]


@runtime_checkable
class L2DeviceOwner(Protocol):
    """Protocol for L2 adapters that own a local physical device."""

    def owns_range(
        self, device_path: str, offset_in_bytes: int, size_in_bytes: int
    ) -> bool:
        """Return whether the adapter maps a range overlapping the given one.

        Args:
            device_path: Path whose physical backing identity should be checked.
            offset_in_bytes: Start of the candidate range on that device.
            size_in_bytes: Length of the candidate range.

        Returns:
            ``True`` when a mapping on the same physical device, even one opened
            through another path, overlaps the range; otherwise ``False``.
        """
        ...


@runtime_checkable
class L2ReconfigurableAdapter(Protocol):
    """Protocol implemented by L2 adapters with runtime reconfiguration."""

    def reconfigure_status(self) -> L2ReconfigureStatus:
        """Return JSON-serializable runtime reconfiguration status."""
        ...

    def reconfigure(
        self,
        operation: str,
        payload: dict[str, object],
        *,
        device_owners: Callable[[str, int, int], list[str]],
    ) -> dict:
        """Apply an adapter-specific runtime reconfiguration operation.

        Args:
            operation: Adapter-specific operation name.
            payload: Adapter-specific operation payload.
            device_owners: Query other L1 or L2 owners of a device byte range
                ``(path, offset, size)`` on each add or resize, before
                acquiring the adapter's device lock. The caller holds the
                lifecycle lock through add completion. The callback must not
                reacquire it; it may acquire other owners' locks one at a time.

        Returns:
            JSON-serializable operation result.
        """
        ...
