# SPDX-License-Identifier: Apache-2.0
"""Regenerate the gRPC Python stubs for ``lmcache_mq.proto``.

The ``.proto`` source lives under ``../proto/`` (checked into git);
the emitted ``*_pb2.py`` / ``*_pb2_grpc.py`` files land right next to
this module and are checked into git as well so runtime users do not
need ``grpcio-tools`` on the install path.

Run as a module so relative paths resolve correctly::

    python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

Requires the ``grpcio-tools`` package (dev-only).  The generated
files are then patched so their internal import uses the full
package path (grpc's generator emits a flat
``import lmcache_mq_pb2`` by default) and so mypy skips them
(message classes come out of the descriptor pool at runtime).
"""

# Standard
from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROTO_DIR = HERE.parent / "proto"
PROTO = PROTO_DIR / "lmcache_mq.proto"
PB2 = HERE / "lmcache_mq_pb2.py"
PB2_GRPC = HERE / "lmcache_mq_pb2_grpc.py"

FLAT_IMPORT = "import lmcache_mq_pb2 as lmcache__mq__pb2"
FQ_IMPORT = (
    "from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen "
    "import lmcache_mq_pb2 as lmcache__mq__pb2"
)
SPDX_HEADER = "# SPDX-License-Identifier: Apache-2.0\n"
MYPY_IGNORE = "# mypy: ignore-errors\n"


def _ensure_headers(path: Path) -> None:
    """Prepend SPDX + mypy markers if they are not already present."""
    text = path.read_text()
    prefix = ""
    if not text.startswith(SPDX_HEADER):
        prefix += SPDX_HEADER
    if MYPY_IGNORE not in text.splitlines()[:5]:
        prefix += MYPY_IGNORE
    if prefix:
        path.write_text(prefix + text)


def main() -> int:
    try:
        # Third Party
        import grpc_tools.protoc  # noqa: F401
    except ImportError:
        print(
            "grpcio-tools is not installed; cannot generate gRPC stubs "
            "for the mp transport. Install it with 'pip install "
            "grpcio-tools' or run this generator manually.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(PROTO_DIR),
        "--python_out",
        str(HERE),
        "--grpc_python_out",
        str(HERE),
        str(PROTO),
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    # Patch the flat import so ``lmcache_mq_pb2_grpc`` can be loaded
    # as a proper submodule.
    text = PB2_GRPC.read_text()
    if FLAT_IMPORT in text:
        text = text.replace(FLAT_IMPORT, FQ_IMPORT)
        PB2_GRPC.write_text(text)

    # Keep the SPDX license header pre-commit expects at the top of
    # every source file, plus a mypy marker so static analysis skips
    # the dynamically generated message classes.
    _ensure_headers(PB2)
    _ensure_headers(PB2_GRPC)

    # Health-check the freshly generated stubs in a subprocess so we do
    # not pollute the caller's sys.modules and can cleanly detect the
    # classic protobuf gencode-vs-runtime version mismatch. Kill the
    # stubs on failure so the next import retries generation rather
    # than reusing a broken pair.  The env flag prevents ``_proto_gen``
    # from recursing back into us if the import still fails inside.
    env = dict(os.environ)
    env["LMCACHE_MQ_PROTO_GEN_HEALTHCHECK"] = "1"
    rc = subprocess.call(
        [
            sys.executable,
            "-c",
            (
                "from lmcache.v1.multiprocess.transport.grpc_impl."
                "_proto_gen import lmcache_mq_pb2, lmcache_mq_pb2_grpc"
            ),
        ],
        env=env,
    )
    if rc != 0:
        for path in (PB2, PB2_GRPC):
            path.unlink(missing_ok=True)
        print(
            "generated stubs failed health check (likely a protobuf "
            "gencode/runtime version mismatch); removed and giving up. "
            "Install a grpcio-tools version compatible with your local "
            "protobuf runtime, e.g. 'pip install \"grpcio-tools==%s.*\"' "
            "matching the grpc runtime already installed." % _grpc_major_minor(),
            file=sys.stderr,
        )
        return rc

    print("regenerated:", PB2.name, "+", PB2_GRPC.name)
    return 0


def _grpc_major_minor() -> str:
    try:
        # Third Party
        import grpc
    except ImportError:
        return "1.x"
    return ".".join(grpc.__version__.split(".")[:2])


if __name__ == "__main__":
    raise SystemExit(main())
