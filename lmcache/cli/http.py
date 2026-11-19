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

# Upper bound on error-body bytes read for a diagnostic; the rest of an
# oversized body is not buffered, and the response is closed either way.
_MAX_ERROR_BODY_BYTES = 4096


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


def _error_body(url: str, exc: urllib.error.HTTPError) -> tuple[str, bool]:
    """Read a bounded prefix of an HTTP error body for diagnostics.

    Args:
        url: The URL the failed request targeted (used in error messages).
        exc: The HTTP error whose response body should be read.

    Returns:
        ``(body, truncated)`` — the decoded body text, cut at the last
        complete character when the bound splits one, and whether the
        bound truncated the body at all.

    Raises:
        CLIHTTPError: When the body ends before its declared length or
            cannot be read or decoded; the original failure is chained
            as the cause.
    """
    try:
        try:
            raw = exc.read(_MAX_ERROR_BODY_BYTES + 1)
        except IncompleteRead as read_exc:
            raise CLIHTTPError(
                f"Incomplete HTTP response from {url}: {read_exc}"
            ) from read_exc
        except OSError as read_exc:
            raise CLIHTTPError(
                f"Cannot read HTTP {exc.code} response from {url}: {read_exc}"
            ) from read_exc

        # A bounded read returns a short body instead of raising when the
        # server declares Content-Length and closes early; detect that.
        declared = exc.headers.get("Content-Length")
        try:
            expected = int(declared) if declared is not None else None
        except ValueError:
            expected = None
        if expected is not None and len(raw) < min(expected, _MAX_ERROR_BODY_BYTES + 1):
            incomplete = IncompleteRead(raw, expected)
            raise CLIHTTPError(
                f"Incomplete HTTP response from {url}: {incomplete}"
            ) from incomplete

        truncated = len(raw) > _MAX_ERROR_BODY_BYTES
        try:
            return raw[:_MAX_ERROR_BODY_BYTES].decode(), truncated
        except UnicodeDecodeError as read_exc:
            if truncated:
                # The bound split a multi-byte character, not the server.
                return raw[:_MAX_ERROR_BODY_BYTES].decode("utf-8", "ignore"), True
            raise CLIHTTPError(
                f"Cannot read HTTP {exc.code} response from {url}: {read_exc}"
            ) from read_exc
    finally:
        exc.close()


def _detail_from_body(body: str) -> str | None:
    """Extract a human-readable reason from a JSON error body.

    Prefers FastAPI-style ``{"detail": ...}`` bodies, then the ``error``
    and ``message`` keys used by some endpoints.

    Args:
        body: The decoded error body text.

    Returns:
        The explanation, or ``None`` when the body is not a JSON object
        carrying one of the known fields.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail") or payload.get("error") or payload.get("message")
    if not detail:
        return None
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
            ``detail``/``error``/``message`` explanation from JSON error
            bodies when present — and on JSON decoding errors. HTTP 503
            keeps the legacy ``Server unhealthy: ...`` wording.
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
        error_body, truncated = _error_body(url, exc)
        detail = _detail_from_body(error_body)
        if exc.code == 503:
            # Legacy contract kept for existing consumers (see the
            # describe design doc); non-JSON bodies are shown verbatim.
            reason = detail if detail is not None else error_body
            if truncated and detail is None:
                reason += " [truncated]"
            raise CLIHTTPError(f"Server unhealthy: {reason}") from exc
        reason = detail if detail is not None else str(exc.reason)
        raise CLIHTTPError(f"HTTP {exc.code} from {url}: {reason}") from exc
    except urllib.error.URLError as exc:
        raise CLIHTTPError(f"Cannot connect to {url}: {exc.reason}") from exc
    except IncompleteRead as exc:
        raise CLIHTTPError(f"Incomplete HTTP response from {url}: {exc}") from exc
    except ValueError as exc:
        raise CLIHTTPError(f"Invalid JSON response from {url}: {exc}") from exc
    except OSError as exc:
        raise CLIHTTPError(f"Cannot connect to {url}: {exc}") from exc
