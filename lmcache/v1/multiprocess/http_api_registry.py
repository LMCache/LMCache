# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import importlib.util
import pkgutil

# Third Party
from fastapi import APIRouter, FastAPI

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


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
        apis_path = Path(__file__).parent / "http_apis"
        if not apis_path.exists():
            logger.warning("http_apis directory not found")
            return

        apis_package = f"{__package__}.http_apis"

        for _, module_name, _ in pkgutil.iter_modules([str(apis_path)]):
            if not module_name.endswith("_api"):
                continue
            full_name = f"{apis_package}.{module_name}"
            module = importlib.import_module(full_name)
            if hasattr(module, "router") and isinstance(module.router, APIRouter):
                self.router.include_router(module.router)
                logger.info(
                    "Registered HTTP API module: %s",
                    module_name,
                )

        self.app.include_router(self.router)
