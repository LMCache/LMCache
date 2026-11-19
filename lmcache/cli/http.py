# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP helpers for CLI commands."""

# Standard
from http.client import IncompleteRead
from typing import Any
import json
import urllib.error
import urllib.request

DEFAULT_URLS: dict[str, str] = {
    "kvcache": "http://localhost:8080",
    "engine": "http://localhost:8000",
}


class CLIHTTPError(Exception):
    """Raised when a CLI HTTP request fails."""


def normalize_url(url: str) -> str:
    """Ensure *url* has an ``http://`` or ``https://`` scheme.

    Existing HTTP(S) schemes are recognized case-insensitively.

    Args:
        url: The URL string to normalize.

    Returns:
        The normalized URL string with a scheme and without trailing slashes.
    """
    if not url.lower().startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Extract a human-readable reason from an HTTP error response body.

    Prefers FastAPI-style ``{"detail": ...}`` bodies, then the ``error`` and
    ``message`` keys used by some endpoints. Bodies that are not a JSON
    object, or that cannot be read, fall back to the status-line reason.

    Args:
        exc: The HTTP error whose response body should be inspected.

    Returns:
        The explanation to show alongside the status code.
    """
    try:
        raw = exc.read().decode()
    except (IncompleteRead, OSError, UnicodeDecodeError):
        return str(exc.reason)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return str(exc.reason)
    if not isinstance(payload, dict):
        return str(exc.reason)
    detail = payload.get("detail") or payload.get("error") or payload.get("message")
    if not detail:
        return str(exc.reason)
    return detail if isinstance(detail, str) else json.dumps(detail)


def fetch_json(
    url: str,
    timeout: int = 10,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an HTTP request and return the parsed JSON body.

    Args:
        url: The URL to request.
        timeout: Request timeout in seconds.
        method: HTTP method (e.g. ``"GET"``, ``"POST"``).
        data: Optional JSON body to send with the request.

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        CLIHTTPError: On network or HTTP errors — including the server's
            ``detail``/``error`` explanation from JSON error bodies when
            present — and on JSON decoding errors.
    """
    body = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise CLIHTTPError(
            f"HTTP {exc.code} from {url}: {_error_detail(exc)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CLIHTTPError(f"Cannot connect to {url}: {exc.reason}") from exc
    except IncompleteRead as exc:
        raise CLIHTTPError(f"Incomplete HTTP response from {url}: {exc}") from exc
    except ValueError as exc:
        raise CLIHTTPError(f"Invalid JSON response from {url}: {exc}") from exc
    except OSError as exc:
        raise CLIHTTPError(f"Cannot connect to {url}: {exc}") from exc
