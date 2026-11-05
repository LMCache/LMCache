# SPDX-License-Identifier: Apache-2.0
"""Compatibility aliases for generated shared-transport gRPC modules."""

# Standard
from importlib import import_module
import sys

_GENERATED_MODULES = (
    "blend_service_pb2",
    "blend_service_pb2_grpc",
    "common_pb2",
    "common_pb2_grpc",
    "engine_driven_service_pb2",
    "engine_driven_service_pb2_grpc",
    "extension_service_pb2",
    "extension_service_pb2_grpc",
    "kv_event_service_pb2",
    "kv_event_service_pb2_grpc",
    "lmcache_driven_service_pb2",
    "lmcache_driven_service_pb2_grpc",
    "lookup_service_pb2",
    "lookup_service_pb2_grpc",
    "management_service_pb2",
    "management_service_pb2_grpc",
    "p2p_service_pb2",
    "p2p_service_pb2_grpc",
    "qstore_service_pb2",
    "qstore_service_pb2_grpc",
)

for _module_name in _GENERATED_MODULES:
    _module = import_module(f"lmcache.v1.transport.grpc._proto_gen.{_module_name}")
    globals()[_module_name] = _module
    sys.modules[f"{__name__}.{_module_name}"] = _module

__all__ = list(_GENERATED_MODULES)
