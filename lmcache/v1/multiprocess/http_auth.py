# SPDX-License-Identifier: Apache-2.0
"""Bearer-token authentication for the MP HTTP server.

Every HTTP request must carry ``Authorization: Bearer <key>`` except
``/healthcheck``, which Kubernetes probes call without credentials. The
middleware is pure ASGI so it wraps every route, including ones mounted later
by router auto-discovery, without buffering request or response bodies.
"""

# Standard
import hmac

# Third Party
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class APIKeyAuthMiddleware:
    """Reject HTTP requests that do not present the configured API key."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        """Wrap ``app`` so its HTTP routes require ``api_key``.

        Args:
            app: The ASGI application to protect.
            api_key: The bearer token callers must present. Must be non-empty;
                when no key is configured the middleware is not installed.

        Raises:
            ValueError: If ``api_key`` is empty.
        """
        if not api_key:
            raise ValueError("APIKeyAuthMiddleware requires a non-empty api_key")
        self.app = app
        self.expected_header = ("Bearer " + api_key).encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward authorized HTTP requests and answer the rest with ``401``.

        Non-HTTP scopes (``lifespan``) pass through untouched.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.
        """
        if (
            scope["type"] != "http"
            or scope["path"] == "/healthcheck"
            or self._is_authorized(scope)
        ):
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)

    def _is_authorized(self, scope: Scope) -> bool:
        """Return whether the request's ``Authorization`` header matches.

        Args:
            scope: The ASGI HTTP scope carrying the raw request headers.

        Returns:
            ``True`` if exactly the expected ``Bearer`` header was sent.
        """
        for name, value in scope["headers"]:
            if name == b"authorization":
                # Constant-time so response timing does not leak the key.
                return hmac.compare_digest(value, self.expected_header)
        return False
