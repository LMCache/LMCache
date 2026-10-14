# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP helpers for CLI commands."""

# Standard
from typing import Any
import json
import os
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

    Args:
        url: The URL string to normalize.

    Returns:
        The normalized URL string with a scheme and without trailing slashes.
    """
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def kvcache_auth_headers() -> dict[str, str]:
    """Return the headers that authenticate to the LMCache HTTP server.

    Reads ``LMCACHE_HTTP_API_KEY``, the variable the server takes its key
    from, so one exported value configures both sides.

    Returns:
        ``{"Authorization": "Bearer <key>"}`` when the variable is set,
        otherwise an empty dict.
    """
    api_key = os.getenv("LMCACHE_HTTP_API_KEY", "")
    return {"Authorization": "Bearer " + api_key} if api_key else {}


def fetch_json(
    url: str, timeout: int = 10, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """GET *url* and return the parsed JSON body.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
        headers: Extra request headers, e.g. :func:`kvcache_auth_headers`.

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        CLIHTTPError: On network, HTTP, or JSON decoding errors.
    """
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            body = exc.read().decode()
            try:
                detail = json.loads(body).get("error", body)
            except (json.JSONDecodeError, AttributeError):
                detail = body
            raise CLIHTTPError(f"Server unhealthy: {detail}") from exc
        raise CLIHTTPError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise CLIHTTPError(f"Cannot connect to {url}: {exc.reason}") from exc
    except ValueError as exc:
        raise CLIHTTPError(f"Invalid JSON response from {url}: {exc}") from exc
    except OSError as exc:
        raise CLIHTTPError(f"Cannot connect to {url}: {exc}") from exc
