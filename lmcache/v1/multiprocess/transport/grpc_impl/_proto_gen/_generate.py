# SPDX-License-Identifier: Apache-2.0
"""Generate protobuf descriptors for multiprocess gRPC service discovery."""

# Standard
from pathlib import Path
import re
import subprocess
import sys

# Third Party
from grpc_tools import protoc

GENERATED_DIR = Path(__file__).resolve().parent
PROTO_DIR = GENERATED_DIR.parent / "protos"
PROJECT_ROOT = GENERATED_DIR.parents[5]
GENERATED_PACKAGE = "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen"

SPDX_HEADER = "# SPDX-License-Identifier: Apache-2.0\n"
MYPY_IGNORE = "# mypy: ignore-errors\n"
RUFF_IGNORE = "# ruff: noqa\n"
FORMAT_OFF = "# fmt: off\n"
ISORT_SKIP = "# isort: skip_file\n"
FLAT_PB2_IMPORT_RE = re.compile(
    r"^import ([A-Za-z_][A-Za-z0-9_]*_pb2) as ([A-Za-z_][A-Za-z0-9_]*)$",
    re.MULTILINE,
)


def _generated_files(directory: Path) -> tuple[Path, ...]:
    """Return generated protobuf modules and obsolete generated artifacts."""
    return (
        tuple(directory.glob("*_pb2.py"))
        + tuple(directory.glob("*_pb2.pyi"))
        + tuple(directory.glob("*_pb2_grpc.py"))
    )


def _cleanup_generated_files() -> None:
    """Remove current output and legacy output next to the proto sources."""
    for path in _generated_files(GENERATED_DIR) + _generated_files(PROTO_DIR):
        path.unlink(missing_ok=True)


def _patch_generated_file(path: Path) -> None:
    """Make generated imports package-safe and add repository headers."""
    text = path.read_text()
    text = FLAT_PB2_IMPORT_RE.sub(
        rf"from {GENERATED_PACKAGE} import \1 as \2",
        text,
    )
    prefix = ""
    if not text.startswith(SPDX_HEADER):
        prefix += SPDX_HEADER
    if path.suffix == ".py":
        if MYPY_IGNORE not in text.splitlines()[:5]:
            prefix += MYPY_IGNORE
    else:
        if RUFF_IGNORE not in text.splitlines()[:5]:
            prefix += RUFF_IGNORE
        if FORMAT_OFF not in text.splitlines()[:5]:
            prefix += FORMAT_OFF
        if ISORT_SKIP not in text.splitlines()[:5]:
            prefix += ISORT_SKIP
    path.write_text(prefix + text)


def generate() -> None:
    """Generate protobuf descriptor modules under ``_proto_gen``.

    Returns:
        None.

    Raises:
        RuntimeError: If no schemas exist, compilation fails, or a generated
            module cannot be imported.
    """
    proto_files = tuple(sorted(PROTO_DIR.glob("*.proto")))
    if not proto_files:
        raise RuntimeError(f"No proto sources found under {PROTO_DIR}")

    _cleanup_generated_files()
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={GENERATED_DIR}",
            *(str(path) for path in proto_files),
        ]
    )
    if result != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {result}")

    generated_files = tuple(sorted(GENERATED_DIR.glob("*_pb2.py")))
    for path in generated_files:
        _patch_generated_file(path)

    modules = "; ".join(
        f"import {GENERATED_PACKAGE}.{path.stem}"
        for path in generated_files
        if path.suffix == ".py"
    )
    result = subprocess.call(
        [sys.executable, "-c", modules],
        cwd=PROJECT_ROOT,
    )
    if result != 0:
        _cleanup_generated_files()
        raise RuntimeError("Generated gRPC modules failed their import check")


if __name__ == "__main__":
    generate()
