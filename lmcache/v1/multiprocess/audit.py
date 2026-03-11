# SPDX-License-Identifier: Apache-2.0

"""Audit logging for the MP cache server.

Provides :class:`AuditContext`, a context-manager that collects
key-value metadata throughout a request's lifetime and emits a
single structured audit log line on exit (success, failure, or
exception).

Controlled by the ``LMCACHE_AUDIT_LOG`` environment variable
(``0``/``1``, default ``0``) or programmatically via
:func:`set_audit_enabled`.
"""

# Future
from __future__ import annotations

# Standard
from contextvars import ContextVar
from typing import Any
import logging
import os
import time

# First Party
from lmcache.logging import init_logger

_audit_logger: logging.Logger = init_logger("lmcache.audit")

_audit_enabled: bool = os.environ.get("LMCACHE_AUDIT_LOG", "0") == "1"

# Populated after AuditContext is defined below.
_current_audit_ctx: ContextVar["AuditContext | None"]


def is_audit_enabled() -> bool:
    """Return whether audit logging is currently enabled."""
    return _audit_enabled


def set_audit_enabled(enabled: bool) -> None:
    """Programmatically enable or disable audit logging."""
    global _audit_enabled
    _audit_enabled = enabled


def get_current_audit_context() -> AuditContext | None:
    """Return the active :class:`AuditContext`, if any.

    Useful for callees that want to attach extra metadata
    without receiving the context as an explicit parameter.
    """
    return _current_audit_ctx.get()


class AuditContext:
    """Accumulate audit metadata and log on exit.

    Usage::

        with AuditContext(
            request_type="STORE",
            params={"key": "...", "instance_id": 1},
        ) as ctx:
            ctx.add(stored_count=5)
            # ... business logic ...

    On ``__exit__`` a single INFO-level line is written to the
    ``lmcache.audit`` logger containing all collected fields.

    The context is also available to nested calls via
    :func:`get_current_audit_context`.
    """

    def __init__(
        self,
        *,
        request_type: str = "",
        source: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.request_type = request_type
        self.source = source
        self.params: dict[str, Any] = dict(params) if params else {}
        self.extras: dict[str, Any] = {}
        self.result: Any = None
        self.success: bool | None = None
        self.error: str | None = None
        self._start: float = 0.0
        self._token: Any = None  # ContextVar token

    # -- public helpers ------------------------------------------------

    def add(self, **kwargs: Any) -> None:
        """Attach arbitrary key-value pairs to this context.

        Can be called from any layer that has access to the
        context (directly or via
        :func:`get_current_audit_context`).
        """
        self.extras.update(kwargs)

    def set_result(
        self,
        result: Any,
        *,
        success: bool = True,
    ) -> None:
        """Record the handler's return value and status."""
        self.result = result
        self.success = success

    def set_error(self, error: str) -> None:
        """Record an error message (called on exception)."""
        self.error = error
        self.success = False

    # -- context-manager protocol --------------------------------------

    def __enter__(self) -> AuditContext:
        self._start = time.perf_counter()
        self._token = _current_audit_ctx.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[type-arg]
        elapsed = time.perf_counter() - self._start

        if exc_type is not None:
            self.success = False
            self.error = "%s: %s" % (
                exc_type.__name__,
                exc_val,
            )

        if self.success is None:
            self.success = True

        record: dict[str, Any] = {
            "request_type": self.request_type,
            "source": self.source,
            "success": self.success,
            "elapsed_ms": round(elapsed * 1000, 3),
        }
        if self.params:
            record["params"] = self.params
        if self.result is not None:
            record["result"] = _safe_repr(self.result)
        if self.error:
            record["error"] = self.error
        if self.extras:
            record["extras"] = self.extras

        _audit_logger.info("AUDIT %s", record)

        # Restore previous context (supports nesting)
        _current_audit_ctx.reset(self._token)

        # Do NOT suppress the exception
        return None


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    """Best-effort short repr, never raises."""
    try:
        s = repr(obj)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return s
    except Exception:
        return "<unrepresentable>"


# ContextVar so that async / threaded handlers each get
# their own current AuditContext automatically.
# Defined after AuditContext so the name is resolved.
_current_audit_ctx = ContextVar["AuditContext | None"](
    "_current_audit_ctx", default=None
)
