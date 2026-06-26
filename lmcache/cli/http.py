# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP helpers for the ``lmcache`` CLI client commands.

Centralizes URL normalization, the default server URLs, and a small
JSON-over-HTTP request helper so individual commands don't re-implement (and
drift on) the same ``urllib`` boilerplate. See issue #3868.
"""

# Standard
from typing import Any, Optional
import json
import urllib.error
import urllib.request

# Default base URLs for the LMCache-side servers, keyed by CLI target.
DEFAULT_URLS: dict[str, str] = {
    "kvcache": "http://localhost:8080",
    "engine": "http://localhost:8000",
}


def normalize_url(url: str) -> str:
    """Ensure *url* has an ``http://`` or ``https://`` scheme.

    Args:
        url: A bare ``host[:port]`` or a full URL.

    Returns:
        The URL with a scheme and no trailing slashes.
    """
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


class CliHttpError(Exception):
    """Raised by :func:`request_json` on any HTTP or connection failure.

    Carries the structured failure details so each command can render its own
    user-facing message and exit behavior.

    Attributes:
        url: The request URL.
        status: The HTTP status code for an error response, or ``None`` for a
            connection-level failure (DNS, connection refused, timeout).
        reason: The HTTP reason phrase, or the connection error description.
        body: The decoded error response body for HTTP errors, else ``None``.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status: Optional[int] = None,
        reason: str = "",
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.reason = reason
        self.body = body


def request_json(
    method: str,
    url: str,
    *,
    data: Optional[dict[str, Any]] = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Send an HTTP request with an optional JSON body and parse the response.

    Args:
        method: HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``).
        url: Full request URL.
        data: Optional object serialized as a JSON request body. When set, a
            ``Content-Type: application/json`` header is added.
        timeout: Socket timeout in seconds.

    Returns:
        The parsed JSON response object.

    Raises:
        CliHttpError: On any HTTP error status or connection-level failure.
    """
    body: Optional[bytes] = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            err_body: Optional[str] = exc.read().decode()
        except OSError:
            err_body = None
        raise CliHttpError(
            f"HTTP {exc.code} from {url}: {exc.reason}",
            url=url,
            status=exc.code,
            reason=str(exc.reason),
            body=err_body,
        ) from exc
    except urllib.error.URLError as exc:
        raise CliHttpError(
            f"Cannot connect to {url}: {exc.reason}",
            url=url,
            reason=str(exc.reason),
        ) from exc
    except OSError as exc:
        raise CliHttpError(
            f"Cannot connect to {url}: {exc}",
            url=url,
            reason=str(exc),
        ) from exc
