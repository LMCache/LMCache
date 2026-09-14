# SPDX-License-Identifier: Apache-2.0
"""Generate Python gRPC bindings from the multiprocess service schemas."""

# Standard
from pathlib import Path
import re
import subprocess
import sys

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
    """Return generated protobuf modules, stubs, and gRPC modules."""
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


def _check_generated_imports(generated_files: tuple[Path, ...]) -> bool:
    """Import generated modules without importing ``lmcache.__init__``."""
    module_names = tuple(path.stem for path in generated_files if path.suffix == ".py")
    script = f"""
from pathlib import Path
import importlib
import sys
import types

package = {GENERATED_PACKAGE!r}
project_root = Path({str(PROJECT_ROOT)!r})
generated_dir = Path({str(GENERATED_DIR)!r})
parts = package.split(".")

for index in range(1, len(parts) + 1):
    name = ".".join(parts[:index])
    package_path = (
        generated_dir
        if index == len(parts)
        else project_root.joinpath(*parts[:index])
    )
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(package_path)]
        sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)

for stem in {module_names!r}:
    importlib.import_module(f"{{package}}.{{stem}}")
"""
    return subprocess.call([sys.executable, "-c", script], cwd=PROJECT_ROOT) == 0


def generate() -> None:
    """Generate all protobuf and gRPC modules under ``_proto_gen``.

    Returns:
        None.

    Raises:
        RuntimeError: If no schemas exist, compilation fails, or a generated
            module cannot be imported.
    """
    proto_files = tuple(sorted(PROTO_DIR.glob("*.proto")))
    if not proto_files:
        raise RuntimeError(f"No proto sources found under {PROTO_DIR}")

    # Third Party
    from grpc_tools import protoc

    _cleanup_generated_files()
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={GENERATED_DIR}",
            f"--pyi_out={GENERATED_DIR}",
            f"--grpc_python_out={GENERATED_DIR}",
            *(str(path) for path in proto_files),
        ]
    )
    if result != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {result}")

    generated_files = _generated_files(GENERATED_DIR)
    for path in generated_files:
        _patch_generated_file(path)

    if not _check_generated_imports(generated_files):
        _cleanup_generated_files()
        raise RuntimeError("Generated gRPC modules failed their import check")


if __name__ == "__main__":
    generate()
