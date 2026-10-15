# SPDX-License-Identifier: Apache-2.0
"""Authentication helpers for the multiprocess HTTP frontend."""

# Standard
from collections.abc import Sequence
from pathlib import Path
import hmac
import secrets

# Third Party
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

PUBLIC_HTTP_PATHS = frozenset({"/", "/healthcheck"})


class HTTPAuthenticationConfig:
    """Pre-hashed API keys used by the HTTP authentication middleware.

    Plaintext keys are converted to keyed HMAC digests as soon as this object
    is created. A per-process random secret prevents those digests from being
    useful for offline key guessing. The object deliberately hides the secret
    and digests from its representation so they are not exposed by logs or the
    ``/config`` endpoint.

    Args:
        api_keys: Plaintext API keys accepted by the HTTP frontend.

    Raises:
        ValueError: If a configured API key is empty.
    """

    def __init__(self, api_keys: Sequence[str] = ()) -> None:
        if any(not api_key for api_key in api_keys):
            raise ValueError("HTTP API keys must not be empty")
        self._comparison_secret = secrets.token_bytes(32)
        self._api_key_hashes = tuple(self._digest(api_key) for api_key in api_keys)

    def _digest(self, api_key: str) -> bytes:
        """Return a fixed-length, process-bound comparison value."""
        return hmac.digest(
            self._comparison_secret,
            api_key.encode("utf-8"),
            "sha256",
        )

    @classmethod
    def from_key_files(
        cls, key_files: Sequence[str | Path]
    ) -> "HTTPAuthenticationConfig":
        """Load one API key from each file.

        Leading and trailing whitespace is removed. Each file must contain a
        non-empty key on a single line.

        Args:
            key_files: Paths containing plaintext API keys.

        Returns:
            Authentication configuration containing the hashed keys.

        Raises:
            OSError: If a key file cannot be read.
            ValueError: If a file is empty or contains more than one line.
        """
        api_keys: list[str] = []
        for key_file in key_files:
            path = Path(key_file)
            value = path.read_text(encoding="utf-8").strip()
            if not value:
                raise ValueError(f"HTTP API key file is empty: {path}")
            if len(value.splitlines()) != 1:
                raise ValueError(
                    f"HTTP API key file must contain exactly one line: {path}"
                )
            api_keys.append(value)
        return cls(api_keys)

    @property
    def enabled(self) -> bool:
        """Return whether at least one API key is configured."""
        return bool(self._api_key_hashes)

    def accepts_authorization_header(self, authorization: str | None) -> bool:
        """Validate one HTTP ``Authorization`` header.

        Args:
            authorization: Header value in ``Bearer <api-key>`` form.

        Returns:
            ``True`` when the bearer token matches a configured key.
        """
        if authorization is None:
            return False
        scheme, separator, token = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not token:
            return False

        candidate_hash = self._digest(token)
        matched = False
        for expected_hash in self._api_key_hashes:
            matched |= secrets.compare_digest(candidate_hash, expected_hash)
        return matched

    def __repr__(self) -> str:
        """Return a representation that never includes keys or key hashes."""
        return (
            f"{type(self).__name__}(enabled={self.enabled}, "
            f"key_count={len(self._api_key_hashes)})"
        )


class HTTPAuthenticationMiddleware:
    """Require a bearer API key for non-public LMCache HTTP routes.

    The active :class:`HTTPAuthenticationConfig` is read from
    ``scope["app"].state.http_authentication``. Authentication is disabled
    when no keys are configured. The root liveness route and healthcheck are
    always public so probes do not need credentials.

    Args:
        app: Downstream ASGI application.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate an HTTP request before forwarding it downstream."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        auth_config: HTTPAuthenticationConfig = scope["app"].state.http_authentication
        if not auth_config.enabled or scope["path"] in PUBLIC_HTTP_PATHS:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if auth_config.accepts_authorization_header(headers.get("authorization")):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)
