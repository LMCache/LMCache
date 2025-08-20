# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import importlib.util
import pkgutil

# Third Party
from fastapi import APIRouter


class APIRegistry:
    """
    Automatically discovers and registers API routes
    """

    def __init__(self, app):
        self.app = app
        self.router = APIRouter()

    def register_all_apis(self):
        """
        Discover and register all API modules in this package
        """
        package_path = Path(__file__).parent
        package_name = __package__

        for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
            if module_name.endswith("_api"):
                full_module_name = f"{package_name}.{module_name}"
                module = importlib.import_module(full_module_name)
                # Include the router if it exists
                if hasattr(module, "router") and isinstance(module.router, APIRouter):
                    self.router.include_router(module.router)

        self.app.include_router(self.router)
