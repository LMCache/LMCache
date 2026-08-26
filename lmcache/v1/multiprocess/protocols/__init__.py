# SPDX-License-Identifier: Apache-2.0
"""
Protocol initialization and registration system.

This module provides the initialize_protocols() function that:
1. Collects protocol definitions from all protocol modules
2. Validates that the gRPC service descriptor matches protocol definitions
3. Ensures all protocol definitions have matching gRPC methods and vice versa
"""

# First Party
from lmcache.v1.multiprocess.protocols import (
    blend,
    blend_v2,
    blend_v3,
    controller,
    debug,
    engine,
    observability,
    p2p,
)
from lmcache.v1.multiprocess.protocols.base import (
    HandlerType,
    ProtocolDefinition,
    request_name_to_method_name,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)

lmcache_mq_pb2 = _pb2_typed


class ProtocolInitializationError(Exception):
    """Raised when there's an error during protocol initialization."""

    pass


_PROTOCOL_MODULES = [
    ("engine", engine),
    ("controller", controller),
    ("debug", debug),
    ("blend", blend),
    ("blend_v2", blend_v2),
    ("blend_v3", blend_v3),
    ("observability", observability),
    ("p2p", p2p),
]


def initialize_protocols() -> dict[str, ProtocolDefinition]:
    """
    Initialize the protocol system by collecting all protocol definitions
    and validating them against the gRPC service descriptor.

    This function:
    1. Collects protocol definitions from all protocol modules
    2. Validates that each service method has a definition
    3. Validates that each definition has a corresponding service method
    4. Ensures no duplicate or orphaned definitions

    Returns:
        protocol_definitions: Dict mapping legacy request names
        (for example ``"PING"``) to ProtocolDefinition.

    Raises:
        ProtocolInitializationError: If there are mismatches between the proto
        service and the Python protocol definitions.
    """
    # Protocol modules to load
    global _PROTOCOL_MODULES

    # Step 1: Collect protocol definitions from all modules
    protocol_definitions: dict[str, ProtocolDefinition] = {}
    defined_names = set()
    name_to_module: dict[str, str] = {}

    for module_name, module in _PROTOCOL_MODULES:
        module_defs = module.get_protocol_definitions()

        # Check for duplicates across modules
        for name in module_defs.keys():
            if name in name_to_module:
                raise ProtocolInitializationError(
                    f"Duplicate protocol definition '{name}' found in modules "
                    f"'{name_to_module[name]}' and '{module_name}'"
                )
            name_to_module[name] = module_name

        # Validate that all names in REQUEST_NAMES have definitions
        for name in module.REQUEST_NAMES:
            if name not in module_defs:
                raise ProtocolInitializationError(
                    f"Request name '{name}' in module '{module_name}' "
                    f"is listed in REQUEST_NAMES but has no protocol definition"
                )
            defined_names.add(name)

        # Keep the legacy request names as the registry keys so other layers
        # can decide whether they want ``PING`` or ``Ping`` style spellings.
        for name, definition in module_defs.items():
            if name not in module.REQUEST_NAMES:
                raise ProtocolInitializationError(
                    f"Protocol definition '{name}' in module '{module_name}' "
                    "is not listed in REQUEST_NAMES"
                )
            protocol_definitions[name] = definition

    missing_declared_definitions = {
        name for name in defined_names if name not in protocol_definitions
    }
    if missing_declared_definitions:
        raise ProtocolInitializationError(
            "REQUEST_NAMES entries without definitions: "
            f"{sorted(missing_declared_definitions)}"
        )

    actual_methods = {
        method.name
        for service in lmcache_mq_pb2.DESCRIPTOR.services_by_name.values()
        for method in service.methods
    }
    expected_methods = {
        request_name_to_method_name(request_name)
        for request_name in protocol_definitions
    }

    missing_methods = sorted(expected_methods - actual_methods)
    missing_definitions = sorted(actual_methods - expected_methods)
    if missing_methods or missing_definitions:
        raise ProtocolInitializationError(
            "gRPC services / protocol definition mismatch: "
            f"missing_methods={missing_methods}, "
            f"missing_definitions={missing_definitions}"
        )

    return protocol_definitions


# Export the base types for convenience
__all__ = [
    "initialize_protocols",
    "ProtocolDefinition",
    "HandlerType",
    "ProtocolInitializationError",
]
