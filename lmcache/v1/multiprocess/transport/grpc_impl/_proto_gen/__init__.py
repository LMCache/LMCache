# SPDX-License-Identifier: Apache-2.0

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
