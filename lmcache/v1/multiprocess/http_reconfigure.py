# SPDX-License-Identifier: Apache-2.0
"""Bind resource-pool hotplug requests to one worker lifetime."""

# Standard
import uuid

# Third Party
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ReconfigureInstanceMiddleware:
    """Advertise a worker incarnation and reject stale reconfiguration requests.

    Existing callers may omit the expected-instance header. Pool controllers
    must send it so a replacement worker's empty status cannot authorize release
    of memory still mapped by the previous worker.

    Args:
        app: The downstream ASGI application.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._instance = uuid.uuid4().hex

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Guard hotplug requests; attach the incarnation to every hotplug reply."""
        if scope["type"] == "lifespan":
            self._instance = uuid.uuid4().hex
        if scope["type"] != "http" or not scope["path"].startswith("/reconfigure/"):
            await self.app(scope, receive, send)
            return
        expected = Headers(scope=scope).get("X-LMCache-Expected-Instance")
        if expected is not None and expected != self._instance:
            await JSONResponse(
                {"error": "worker changed; reconcile pooled memory offline"},
                status_code=409,
                headers={"X-LMCache-Instance": self._instance},
            )(scope, receive, send)
            return

        async def with_instance(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-LMCache-Instance"] = self._instance
            await send(message)

        await self.app(scope, receive, with_instance)
