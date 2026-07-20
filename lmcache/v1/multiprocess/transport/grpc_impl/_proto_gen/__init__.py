# SPDX-License-Identifier: Apache-2.0
"""Generated protobuf / gRPC stubs (kept out of git).

The ``*_pb2.py`` / ``*_pb2_grpc.py`` modules live here on disk but are
never checked in.  They are produced from the ``.proto`` sources under
``../proto/`` by :mod:`._generate`, which we invoke automatically on
first import (and again if the previously generated stubs turn out to
be incompatible with the local ``protobuf`` runtime -- typical after a
python-environment change).

Regenerate manually after editing the ``.proto`` source::

    python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

Requires ``grpcio-tools`` at generation time; a missing toolchain or
an unrecoverable version mismatch raises ``ImportError``, which the
parent ``transport/__init__.py`` swallows just like a missing
``grpcio``.
"""

# Standard
from pathlib import Path
import importlib
import os

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import _generate

_HERE = Path(__file__).resolve().parent
_PB2 = _HERE / "lmcache_mq_pb2.py"
_PB2_GRPC = _HERE / "lmcache_mq_pb2_grpc.py"
_PKG = "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen"
_HEALTHCHECK_ENV = "LMCACHE_MQ_PROTO_GEN_HEALTHCHECK"


def _try_import():
    """Import the two stub modules; propagate any failure to the caller.

    A version-mismatched ``*_pb2.py`` raises
    :class:`google.protobuf.runtime_version.VersionError` at import time,
    which subclasses ``Exception`` (not ``ImportError``); catch broadly
    so we can wipe and regenerate for any kind of stale-stub failure.
    """
    pb2 = importlib.import_module(_PKG + ".lmcache_mq_pb2")
    pb2_grpc = importlib.import_module(_PKG + ".lmcache_mq_pb2_grpc")
    return pb2, pb2_grpc


def _wipe_stubs() -> None:
    for path in (_PB2, _PB2_GRPC):
        path.unlink(missing_ok=True)


def _load_or_generate():
    # Inside _generate's health-check subprocess we must never
    # regenerate: propagate whatever the import raises so the parent
    # sees a non-zero rc and can wipe the stubs itself.
    if os.environ.get(_HEALTHCHECK_ENV) == "1":
        return _try_import()

    if _PB2.exists() and _PB2_GRPC.exists():
        try:
            return _try_import()
        except Exception:
            # Stubs on disk are stale (e.g. protobuf gencode/runtime
            # mismatch after a venv switch); regenerate below.
            _wipe_stubs()

    rc = _generate.main()
    if rc != 0 or not (_PB2.exists() and _PB2_GRPC.exists()):
        raise ImportError(
            "failed to generate gRPC stubs for LMCache mp transport; "
            "install 'grpcio-tools' (matching your grpc runtime) or run "
            "'python -m lmcache.v1.multiprocess.transport."
            "grpc_impl._proto_gen._generate' manually"
        )
    return _try_import()


lmcache_mq_pb2, lmcache_mq_pb2_grpc = _load_or_generate()

__all__ = ["lmcache_mq_pb2", "lmcache_mq_pb2_grpc"]
