# SPDX-License-Identifier: Apache-2.0
"""Generated protobuf / gRPC stubs.

The ``*_pb2.py`` / ``*_pb2_grpc.py`` modules are produced from
``../proto/lmcache_mq.proto`` by :mod:`._generate`.  They are **not**
checked into git (see ``.gitignore``); the ``.proto`` file is the single
source of truth. On first access to either stub this package generates it
lazily, so neither developers nor CI have to run a manual codegen step --
they only need ``grpcio-tools`` available at generation time (it ships in
``requirements/test.txt``).

Regenerate manually after editing the ``.proto`` source::

    python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

``grpcio-tools`` is only needed at generation time (dev / CI), never on
the pure-runtime install path once the stubs exist.
"""

# Standard
from pathlib import Path
from typing import Any
import importlib
import os
import sys
import threading

HERE = Path(__file__).resolve().parent
_STUB_MODULE_NAMES = (
    f"{__name__}.lmcache_mq_pb2",
    f"{__name__}.lmcache_mq_pb2_grpc",
)
_STUB_FILES = (
    HERE / "lmcache_mq_pb2.py",
    HERE / "lmcache_mq_pb2_grpc.py",
)
_generation_lock = threading.Lock()


def _import_stub(name: str) -> Any:
    """Import one generated stub module by its short name."""
    return importlib.import_module(f"{__name__}.{name}")


def _clear_stub_modules() -> None:
    """Drop partially imported generated stub modules before a retry."""
    for module_name in _STUB_MODULE_NAMES:
        sys.modules.pop(module_name, None)


def _drop_stub_files() -> None:
    """Remove stale generated files so regeneration starts from a clean slate."""
    for path in _STUB_FILES:
        path.unlink(missing_ok=True)


def _generate_stubs_once():
    """Generate the stubs by shelling out to :mod:`._generate`.

    Raises ``ImportError`` with an actionable message if generation
    fails (e.g. ``grpcio-tools`` missing).
    """
    # First Party
    from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import _generate

    if _generate.main() != 0:
        raise ImportError(
            "gRPC stubs for the mp transport are missing and could not be "
            "generated. Install the dev/test extras (which include "
            "grpcio-tools), or run: python -m lmcache.v1.multiprocess."
            "transport.grpc_impl._proto_gen._generate"
        )


# ``_generate`` re-imports this package inside its health-check
# subprocess; the env flag breaks that recursion so a genuinely broken
# generation surfaces as the original ImportError instead of looping.
_IN_HEALTHCHECK = os.environ.get("LMCACHE_MQ_PROTO_GEN_HEALTHCHECK") == "1"


def __getattr__(name: str) -> Any:
    """Load a generated stub only when callers request that module."""
    if name not in ("lmcache_mq_pb2", "lmcache_mq_pb2_grpc"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = _import_stub(name)
    except Exception:
        if _IN_HEALTHCHECK:
            raise
        with _generation_lock:
            try:
                module = _import_stub(name)
            except Exception:
                _clear_stub_modules()
                _drop_stub_files()
                _generate_stubs_once()
                _clear_stub_modules()
                module = _import_stub(name)

    globals()[name] = module
    return module


__all__ = ["lmcache_mq_pb2", "lmcache_mq_pb2_grpc"]
