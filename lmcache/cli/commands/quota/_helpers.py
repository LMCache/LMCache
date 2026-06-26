# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for ``lmcache quota`` subcommands."""

# Standard
from typing import Any, Optional
import json
import sys

# First Party
# ``normalize_url`` is imported (and thus still importable from this module
# for backward compatibility); it now lives in the shared CLI HTTP module.
from lmcache.cli.http import CliHttpError, normalize_url, request_json
from lmcache.logging import init_logger

logger = init_logger(__name__)

__all__ = [
    "DEFAULT_SALT_SENTINEL",
    "escape_salt",
    "http_request",
    "normalize_url",
    "unescape_salt",
]

# The MP HTTP server uses "_default" as a sentinel for the empty-string
# cache_salt (anonymous / un-salted traffic).
DEFAULT_SALT_SENTINEL = "_default"


def escape_salt(salt: str) -> str:
    """Translate the empty-string salt to the URL sentinel."""
    return DEFAULT_SALT_SENTINEL if salt == "" else salt


def unescape_salt(salt: str) -> str:
    """Translate the URL sentinel back to the empty-string salt."""
    return "" if salt == DEFAULT_SALT_SENTINEL else salt


def http_request(
    method: str,
    url: str,
    data: Optional[dict[str, Any]] = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Send an HTTP request and return the parsed JSON response.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE).
        url: Full URL to request.
        data: Optional JSON body to send.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        SystemExit: On connection error or non-2xx HTTP response.
    """
    try:
        return request_json(method, url, data=data, timeout=timeout)
    except CliHttpError as exc:
        if exc.status is not None:
            msg = f"HTTP Error {exc.status}: {exc.reason}"
            if exc.body is not None:
                try:
                    error_body = json.loads(exc.body)
                    msg = error_body.get("error") or error_body.get("message") or msg
                except (json.JSONDecodeError, ValueError, OSError):
                    pass
            logger.error("Server error: %s", msg)
            sys.exit(1)
        logger.error("Cannot reach %s — is the server running? (%s)", url, exc.reason)
        sys.exit(1)
