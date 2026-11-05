# SPDX-License-Identifier: Apache-2.0
"""Compatibility aliases for memory-orchestrator generated bindings."""

# Standard
from importlib import import_module
import sys

_GENERATED_MODULES = (
    "memory_orchestrator_pb2",
    "memory_orchestrator_pb2_grpc",
)

for _module_name in _GENERATED_MODULES:
    _module = import_module(
        f"lmcache.v1.transport.grpc.memory_orchestrator._proto_gen.{_module_name}"
    )
    globals()[_module_name] = _module
    sys.modules[f"{__name__}.{_module_name}"] = _module

__all__ = list(_GENERATED_MODULES)
