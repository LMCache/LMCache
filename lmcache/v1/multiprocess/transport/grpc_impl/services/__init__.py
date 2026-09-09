# SPDX-License-Identifier: Apache-2.0
"""Concrete implementations for generated multiprocess gRPC services."""

# Standard
from typing import Any

# Local
from .blend import BlendServiceImpl
from .controller import ControllerServiceImpl
from .debug import DebugServiceImpl
from .engine_driven import EngineDrivenServiceImpl
from .lmcache_driven import LMCacheDrivenServiceImpl
from .lookup import LookupServiceImpl
from .observability import ObservabilityServiceImpl
from .p2p import P2PServiceImpl
from .qstore import QStoreServiceImpl

_SERVICE_IMPLEMENTATION_CLASSES: dict[str, type[Any]] = {
    "BlendService": BlendServiceImpl,
    "ControllerService": ControllerServiceImpl,
    "DebugService": DebugServiceImpl,
    "EngineDrivenService": EngineDrivenServiceImpl,
    "LMCacheDrivenService": LMCacheDrivenServiceImpl,
    "LookupService": LookupServiceImpl,
    "ObservabilityService": ObservabilityServiceImpl,
    "P2PService": P2PServiceImpl,
    "QStoreService": QStoreServiceImpl,
}


def get_service_implementation_class(service_name: str) -> type[Any]:
    """Return the annotated implementation class for a gRPC service.

    Args:
        service_name: Name declared by the generated protobuf service.

    Returns:
        The class whose method annotations define the Python-side gRPC
        request and response types.

    Raises:
        RuntimeError: If no implementation contract exists for the service.
    """
    implementation_class = _SERVICE_IMPLEMENTATION_CLASSES.get(service_name)
    if implementation_class is None:
        raise RuntimeError(
            f"Generated gRPC service {service_name!r} has no implementation contract"
        )
    return implementation_class


__all__ = [
    "BlendServiceImpl",
    "ControllerServiceImpl",
    "DebugServiceImpl",
    "EngineDrivenServiceImpl",
    "LMCacheDrivenServiceImpl",
    "LookupServiceImpl",
    "ObservabilityServiceImpl",
    "P2PServiceImpl",
    "QStoreServiceImpl",
    "get_service_implementation_class",
]
