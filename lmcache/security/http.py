# SPDX-License-Identifier: Apache-2.0
"""API-key authentication adapter for ASGI HTTP services."""

# Standard
from collections.abc import Collection

# Third Party
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# First Party
from lmcache.security.api_key import APIKeyVerifier


def bearer_token(authorization: str | None) -> str | None:
    """Extract a token from an HTTP Bearer authorization value.

    Args:
        authorization: Complete ``Authorization`` header value.

    Returns:
        The Bearer token, or ``None`` for a missing or malformed value.
    """
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        return None
    return token


def bearer_authorization_header(api_key: str) -> str:
    """Build an HTTP Bearer authorization value from one API key.

    Args:
        api_key: Plaintext API key to send to an HTTP service.

    Returns:
        A value suitable for the HTTP ``Authorization`` header.
    """
    return f"Bearer {api_key}"


class HTTPBearerAuthenticationMiddleware:
    """Require a configured API key for non-public HTTP routes.

    The verifier is read from ``scope["app"].state.http_authentication``.
    Authentication is disabled when it contains no keys.
    Accepts ``Authorization: Bearer <key>`` or ``X-LMCache-API-Key: <key>``
    using the same verifier. When both are present, either valid key suffices.

    Args:
        app: Downstream ASGI application.
        public_paths: Exact paths that remain public when authentication is on.
        public_path_prefixes: Path prefixes that remain public, such as static
            asset directories.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        public_paths: Collection[str] = (),
        public_path_prefixes: Collection[str] = (),
    ) -> None:
        self.app = app
        self.public_paths = frozenset(public_paths)
        self.public_path_prefixes = tuple(public_path_prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate one HTTP request before forwarding it downstream."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        verifier: APIKeyVerifier = scope["app"].state.http_authentication
        path = scope["path"]
        if (
            not verifier.enabled
            or path in self.public_paths
            or path.startswith(self.public_path_prefixes)
        ):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if verifier.verify_token(
            bearer_token(headers.get("authorization"))
        ) or verifier.verify_token(headers.get("x-lmcache-api-key")):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)
