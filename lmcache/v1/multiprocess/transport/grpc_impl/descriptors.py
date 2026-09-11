# SPDX-License-Identifier: Apache-2.0
"""Discover generated gRPC services from protobuf descriptors."""

# Standard
from dataclasses import dataclass
from functools import lru_cache
import importlib
import pkgutil
import re

# Third Party
from google.protobuf.descriptor import MethodDescriptor, ServiceDescriptor

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl import _proto_gen


@dataclass(frozen=True)
class ServiceBinding:
    """Generated descriptor for one protobuf service."""

    descriptor: ServiceDescriptor


def client_method_name(method_name: str) -> str:
    """Convert a protobuf RPC method name to the public client method name."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", method_name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower().replace("p2_p", "p2p")


@lru_cache(maxsize=1)
def get_service_bindings() -> dict[str, ServiceBinding]:
    """Load every generated ``*_service_pb2`` module in ``_proto_gen``."""
    package_prefix = f"{_proto_gen.__name__}."
    bindings: dict[str, ServiceBinding] = {}
    for module_info in pkgutil.iter_modules(_proto_gen.__path__, package_prefix):
        if not module_info.name.endswith("_service_pb2"):
            continue
        proto_module = importlib.import_module(module_info.name)
        for descriptor in proto_module.DESCRIPTOR.services_by_name.values():
            if descriptor.name in bindings:
                raise RuntimeError(f"Duplicate gRPC service: {descriptor.name}")
            bindings[descriptor.name] = ServiceBinding(descriptor)
    if not bindings:
        raise RuntimeError(
            "No generated gRPC services found. Run "
            "`python -m lmcache.v1.multiprocess.transport.grpc_impl."
            "_proto_gen._generate`."
        )
    return bindings


def iter_methods() -> list[tuple[ServiceBinding, MethodDescriptor]]:
    """Return all generated unary RPC methods in descriptor order."""
    return [
        (binding, method)
        for binding in get_service_bindings().values()
        for method in binding.descriptor.methods
    ]
