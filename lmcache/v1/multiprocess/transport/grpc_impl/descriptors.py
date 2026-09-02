# SPDX-License-Identifier: Apache-2.0
"""Discover generated gRPC services from protobuf descriptors."""

# Standard
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
import importlib
import pkgutil
import re

# Third Party
from google.protobuf.descriptor import MethodDescriptor, ServiceDescriptor
from google.protobuf.message import Message
from google.protobuf.message_factory import GetMessageClass

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl import protos


@dataclass(frozen=True)
class ServiceBinding:
    """Generated descriptor and gRPC module for one protobuf service."""

    descriptor: ServiceDescriptor
    grpc_module: ModuleType


def client_method_name(method_name: str) -> str:
    """Convert a protobuf RPC method name to the public client method name."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", method_name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower().replace("p2_p", "p2p")


def message_class(descriptor: object) -> type[Message]:
    """Return the generated message class for a protobuf descriptor."""
    return GetMessageClass(descriptor)  # type: ignore[arg-type, no-any-return]


@lru_cache(maxsize=1)
def get_service_bindings() -> dict[str, ServiceBinding]:
    """Load every generated ``*_service_pb2`` module in the proto package."""
    package_prefix = f"{protos.__name__}."
    bindings: dict[str, ServiceBinding] = {}
    for module_info in pkgutil.iter_modules(protos.__path__, package_prefix):
        if not module_info.name.endswith("_service_pb2"):
            continue
        proto_module = importlib.import_module(module_info.name)
        grpc_module = importlib.import_module(f"{module_info.name}_grpc")
        for descriptor in proto_module.DESCRIPTOR.services_by_name.values():
            if descriptor.name in bindings:
                raise RuntimeError(f"Duplicate gRPC service: {descriptor.name}")
            bindings[descriptor.name] = ServiceBinding(descriptor, grpc_module)
    if not bindings:
        raise RuntimeError(
            "No generated gRPC services found. Run "
            "`python -m lmcache.v1.multiprocess.transport.grpc_impl.protos.generate`."
        )
    return bindings


def iter_methods() -> list[tuple[ServiceBinding, MethodDescriptor]]:
    """Return all generated unary RPC methods in descriptor order."""
    return [
        (binding, method)
        for binding in get_service_bindings().values()
        for method in binding.descriptor.methods
    ]
