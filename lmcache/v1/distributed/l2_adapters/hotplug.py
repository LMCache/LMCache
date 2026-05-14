# SPDX-License-Identifier: Apache-2.0
"""Generic runtime hotplug protocol for L2 adapters."""

# Future
from __future__ import annotations

# Standard
from typing import Literal, Optional, Protocol, runtime_checkable

HotplugRemoveMode = Literal["migrate", "evict", "drain"]
HotplugResizeMode = Literal["migrate", "evict"]


class L2HotplugError(RuntimeError):
    """HTTP-mappable runtime hotplug operation error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        payload: Optional[dict[str, object]] = None,
    ) -> None:
        """Create a runtime hotplug error.

        Args:
            status_code: HTTP status code the API should return.
            message: Human-readable error message.
            payload: Optional response body. When omitted, ``{"error": message}``
                is used.
        """
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload if payload is not None else {"error": message}


@runtime_checkable
class L2HotplugAdapter(Protocol):
    """Protocol implemented by L2 adapters with runtime hotplug support."""

    def hotplug_status(self) -> dict: ...

    def hotplug_add_device(
        self,
        device_path: str,
        size_bytes: int,
    ) -> dict: ...

    def hotplug_remove_device(
        self,
        device_path: str,
        mode: HotplugRemoveMode,
        force: bool = False,
    ) -> dict: ...

    def hotplug_resize_device(
        self,
        device_path: str,
        size_bytes: int,
        mode: HotplugResizeMode,
        force: bool = False,
    ) -> dict: ...
