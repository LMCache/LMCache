# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP helpers for CLI commands."""

# Standard
from http.client import IncompleteRead
from typing import Any
import codecs
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

    Preserves the diagnostics ``describe`` already guarantees: a body
    that ends before the response's effective length is reported as an
    incomplete response, and undecodable bytes as a read failure. A
    multi-byte character split by the bound is the one tolerated defect:
    the prefix is all that is shown either way.

    Args:
        url: The URL the failed request targeted (used in error messages).
        exc: The HTTP error whose response body should be read.

    Returns:
        ``(body, truncated)`` — the decoded body text and whether the
        bound truncated the body at all.

    Raises:
        CLIHTTPError: When the body cannot be read or decoded; the
            original failure is chained as the cause.
    """
    try:
        # The length the client expects for this response, read before
        # read() consumes it: 0 for HEAD and 204/304 (which carry no body
        # regardless of Content-Length, RFC 9110 section 8.6), None when
        # chunked — an interrupted chunk makes read() raise below.
        expected = getattr(getattr(exc, "fp", None), "length", None)
        raw = exc.read(_MAX_ERROR_BODY_BYTES + 1)
    except IncompleteRead as read_exc:
        raise CLIHTTPError(
            f"Incomplete HTTP response from {url}: {read_exc}"
        ) from read_exc
    except OSError as read_exc:
        raise CLIHTTPError(
            f"Cannot read HTTP {exc.code} response from {url}: {read_exc}"
        ) from read_exc
    finally:
        exc.close()

    # A bounded read returns a short body instead of raising when the
    # server declares Content-Length and closes early; detect that.
    if isinstance(expected, int) and len(raw) < min(
        expected, _MAX_ERROR_BODY_BYTES + 1
    ):
        incomplete = IncompleteRead(raw, expected - len(raw))
        raise CLIHTTPError(
            f"Incomplete HTTP response from {url}: {incomplete}"
        ) from incomplete

    truncated = len(raw) > _MAX_ERROR_BODY_BYTES
    try:
        # Incremental strict decode: a character split by the bound is
        # tolerated, invalid bytes anywhere in the prefix are not.
        decoder = codecs.getincrementaldecoder("utf-8")()
        body = decoder.decode(raw[:_MAX_ERROR_BODY_BYTES], final=not truncated)
    except UnicodeDecodeError as read_exc:
        raise CLIHTTPError(
            f"Cannot read HTTP {exc.code} response from {url}: {read_exc}"
        ) from read_exc
    return body, truncated


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
    report_503_as_unhealthy: bool = True,
) -> dict[str, Any]:
    """Send an HTTP request and return the parsed JSON body.

    Args:
        url: The URL to request.
        timeout: Request timeout in seconds.
        method: HTTP method (e.g. ``"GET"``, ``"POST"``).
        data: Optional JSON body to send with the request.
        report_503_as_unhealthy: Use the legacy ``Server unhealthy`` prefix for
            HTTP 503. If false, include the HTTP status and URL.

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        CLIHTTPError: On network or HTTP errors — including the server's
            ``detail``/``error``/``message`` explanation from JSON error
            bodies when present — and on JSON decoding errors.
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
            # A 503 can carry its diagnostic in the raw response body.
            reason = detail if detail is not None else error_body
            if truncated and detail is None:
                reason += " [truncated]"
            if report_503_as_unhealthy:
                raise CLIHTTPError(f"Server unhealthy: {reason}") from exc
        else:
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
