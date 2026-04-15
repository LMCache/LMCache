# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import List, Literal, Optional
import importlib.util
import pkgutil

# Third Party
from fastapi import APIRouter

APICategory = Literal["common", "vllm", "controller"]


class APIRegistry:
    """
    Automatically discovers and registers API routes by category

    Categories:
    - common: APIs that work for all components (metrics, logs, etc.)
    - vllm: APIs specific to vLLM scheduler/worker
    - controller: APIs specific to LMCache controller
    """

    def __init__(self, app):
        self.app = app
        self.router = APIRouter()

    def register_all_apis(self, categories: Optional[List[APICategory]] = None):
        """
        Discover and register API modules from specified categories

        Args:
            categories: List of categories to register.
                       If None, registers all categories.
        """
        if categories is None:
            categories = ["common", "vllm", "controller"]

        package_path = Path(__file__).parent
        package_name = __package__

        for category in categories:
            category_path = package_path / category
            if not category_path.exists():
                continue

            category_package = f"{package_name}.{category}"

            for _, module_name, _ in pkgutil.iter_modules([str(category_path)]):
                if module_name.endswith("_api"):
                    full_module_name = f"{category_package}.{module_name}"
                    module = importlib.import_module(full_module_name)
                    # Include the router if it exists
                    if hasattr(module, "router") and isinstance(
                        module.router, APIRouter
                    ):
                        self.router.include_router(module.router)

        self.app.include_router(self.router)
