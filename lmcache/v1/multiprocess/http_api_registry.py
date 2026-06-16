# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter, FastAPI

# First Party
from lmcache.v1.utils.router_discovery import discover_api_routers


class HTTPAPIRegistry:
    """
    Automatically discovers and registers HTTP API routes
    from the ``http_apis`` sub-package.

    Any module whose name ends with ``_api`` and exposes a
    module-level ``router`` (:class:`~fastapi.APIRouter`) will
    be picked up automatically.
    """

    def __init__(self, app: FastAPI):
        self.app = app
        self.router = APIRouter()

    def register_all_apis(self) -> None:
        """
        Discover and register all ``*_api`` modules under
        the ``http_apis`` directory.
        """
        apis_package = f"{__package__}.http_apis"

        for r in discover_api_routers(apis_package):
            self.router.include_router(r)

        self.app.include_router(self.router)
