# SPDX-License-Identifier: Apache-2.0
"""Generated protobuf and gRPC stubs for the multiprocess message queue.

The stub modules are created during package builds and test setup. They are
never checked into Git; ``lmcache_mq.proto`` is the single source of truth.
Regenerate them manually after changing the schema with::

    pip install -r requirements/proto.txt
    python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

Stub modules are imported independently so loading protobuf message types does
not also initialize the gRPC native runtime.
"""

# Standard
from typing import Any
import importlib

_STUB_MODULE_NAMES = frozenset(("lmcache_mq_pb2", "lmcache_mq_pb2_grpc"))


def __getattr__(name: str) -> Any:
    """Import a generated stub module on first access."""
    if name not in _STUB_MODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ImportError(
            "LMCache gRPC stubs have not been generated. Install "
            "requirements/proto.txt and run "
            "'python -m lmcache.v1.multiprocess.transport.grpc_impl."
            "_proto_gen._generate'."
        ) from exc
    globals()[name] = module
    return module


__all__ = ["lmcache_mq_pb2", "lmcache_mq_pb2_grpc"]
