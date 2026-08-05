# SPDX-License-Identifier: Apache-2.0
"""Generated protobuf / gRPC stubs.

The ``*_pb2.py`` / ``*_pb2_grpc.py`` modules are produced from
``../proto/lmcache_mq.proto`` by :mod:`._generate`.  They are **not**
checked into git (see ``.gitignore``); the ``.proto`` file is the single
source of truth.  On first import this package generates them lazily, so
neither developers nor CI have to run a manual codegen step -- they only
need ``grpcio-tools`` available at generation time (it ships in
``requirements/test.txt``).

Regenerate manually after editing the ``.proto`` source::

    python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

``grpcio-tools`` is only needed at generation time (dev / CI), never on
the pure-runtime install path once the stubs exist.
"""

# Standard
import importlib
import os


def _import_stubs():
    """Import and return the ``(pb2, pb2_grpc)`` module pair.

    Raises ``ImportError`` if the stubs have not been generated yet.
    """
    base = __name__
    pb2 = importlib.import_module(f"{base}.lmcache_mq_pb2")
    pb2_grpc = importlib.import_module(f"{base}.lmcache_mq_pb2_grpc")
    return pb2, pb2_grpc


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

try:
    lmcache_mq_pb2, lmcache_mq_pb2_grpc = _import_stubs()
except ImportError:
    if _IN_HEALTHCHECK:
        lmcache_mq_pb2 = None  # type: ignore[assignment]
        lmcache_mq_pb2_grpc = None  # type: ignore[assignment]
    else:
        _generate_stubs_once()
        lmcache_mq_pb2, lmcache_mq_pb2_grpc = _import_stubs()

__all__ = ["lmcache_mq_pb2", "lmcache_mq_pb2_grpc"]
