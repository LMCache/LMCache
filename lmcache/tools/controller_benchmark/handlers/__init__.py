"""Operation handlers with dynamic discovery and registration"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict
import importlib
import inspect
import pkgutil

# Local
from .base import OperationHandler

# Operation handler registry
OPERATION_HANDLERS: Dict[str, OperationHandler] = {}


def _discover_and_register_handlers():
    """Dynamically discover and register all operation handlers"""
    # Get the current package
    package = __package__
    package_path = __path__

    # Iterate through all modules in the handlers package
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        # Skip base module
        if module_name == "base":
            continue

        # Import the module
        module = importlib.import_module("." + module_name, package=package)

        # Find all classes that inherit from OperationHandler
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, OperationHandler)
                and obj is not OperationHandler
                and not inspect.isabstract(obj)
            ):
                # Instantiate and register the handler
                handler = obj()
                OPERATION_HANDLERS[handler.operation_name] = handler


# Auto-discover and register handlers on import
_discover_and_register_handlers()
