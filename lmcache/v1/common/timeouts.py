# SPDX-License-Identifier: Apache-2.0

"""Timeout error raised by client and server code.

Raise :class:`LMCacheTimeoutError` instead of the builtin ``TimeoutError``. The
``ban-raw-timeout-error`` pre-commit hook enforces this across ``lmcache/``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable

_observer: Callable[[LMCacheTimeoutError, str], None] | None = None


def register_timeout_observer(
    observer: Callable[[LMCacheTimeoutError, str], None],
) -> None:
    """Set the callback that every new ``LMCacheTimeoutError`` is passed to.

    MP observability's ``init_event_bus`` registers its ``TIMEOUT_RAISED``
    publisher here. The observer must not raise.

    Args:
        observer: Called with the new error and its session id; replaces any
            previously registered observer.
    """
    global _observer
    _observer = observer


class LMCacheTimeoutError(TimeoutError):
    """``TimeoutError`` that is passed to the registered observer on construction.

    Subclasses the builtin, so existing ``except TimeoutError`` handlers still
    catch it.

    Args:
        message: Timeout description.
        session_id: Request id for correlating the timeout with its originating
            request; empty when unavailable.
    """

    def __init__(self, message: str, *, session_id: str = "") -> None:
        super().__init__(message)
        if _observer is not None:
            _observer(self, session_id)
