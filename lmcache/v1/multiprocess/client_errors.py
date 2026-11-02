# SPDX-License-Identifier: Apache-2.0
"""Timeout identity shared by client and server; observer registered by server."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable

_observer: Callable[[LMCacheTimeoutError, str], None] | None = None


def register_timeout_observer(
    observer: Callable[[LMCacheTimeoutError, str], None],
) -> None:
    """Set the callback every ``LMCacheTimeoutError`` reports itself to.

    Args:
        observer: Called with the new error and its session id; replaces any
            previously registered observer.
    """
    global _observer
    _observer = observer


class LMCacheTimeoutError(TimeoutError):
    """``TimeoutError`` that reports itself to the registered observer.

    Subclasses the builtin, so existing ``except TimeoutError`` handlers still
    catch it. The server registers an observer that publishes a
    ``TIMEOUT_RAISED`` event; an observer failure never propagates out of
    ``__init__``.

    Args:
        message: Timeout description.
        session_id: Request id for correlating the timeout with its originating
            request; empty when unavailable.
    """

    def __init__(self, message: str, *, session_id: str = "") -> None:
        super().__init__(message)
        if _observer is not None:
            try:
                _observer(self, session_id)
            except Exception:
                # A telemetry observer must never replace the timeout.
                pass
