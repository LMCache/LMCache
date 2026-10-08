# SPDX-License-Identifier: Apache-2.0
"""Generate Python gRPC bindings from LMCache service schemas.

Each ``_proto_gen`` package is generated from the ``protos`` directory next to
it. Without arguments this builds every package in ``GENERATED_PACKAGES``; pass
package names to build only those.
"""

# Standard
from pathlib import Path
import argparse
import re
import subprocess
import sys

GENERATED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GENERATED_DIR.parents[5]
GENERATED_PACKAGE = "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen"
GENERATED_PACKAGES = (
    GENERATED_PACKAGE,
    "lmcache.v1.memory_orchestrator._proto_gen",
)

SPDX_HEADER = "# SPDX-License-Identifier: Apache-2.0\n"
MYPY_IGNORE = "# mypy: ignore-errors\n"
RUFF_IGNORE = "# ruff: noqa\n"
FORMAT_OFF = "# fmt: off\n"
ISORT_SKIP = "# isort: skip_file\n"
FLAT_PB2_IMPORT_RE = re.compile(
    r"^import ([A-Za-z_][A-Za-z0-9_]*_pb2) as ([A-Za-z_][A-Za-z0-9_]*)$",
    re.MULTILINE,
)


def _package_dir(generated_package: str) -> Path:
    """Return the source directory of a package under ``PROJECT_ROOT``."""
    return PROJECT_ROOT.joinpath(*generated_package.split("."))


def _generated_files(directory: Path) -> tuple[Path, ...]:
    """Return generated protobuf modules, stubs, and gRPC modules."""
    return (
        tuple(directory.glob("*_pb2.py"))
        + tuple(directory.glob("*_pb2.pyi"))
        + tuple(directory.glob("*_pb2_grpc.py"))
    )


def _cleanup_generated_files(generated_dir: Path, proto_dir: Path) -> None:
    """Remove current output and legacy output next to the proto sources."""
    for path in _generated_files(generated_dir) + _generated_files(proto_dir):
        path.unlink(missing_ok=True)


def _patch_generated_file(path: Path, generated_package: str) -> None:
    """Make generated imports package-safe and add repository headers."""
    text = path.read_text()
    text = FLAT_PB2_IMPORT_RE.sub(
        rf"from {generated_package} import \1 as \2",
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


def _check_generated_imports(
    generated_files: tuple[Path, ...],
    generated_package: str = GENERATED_PACKAGE,
) -> bool:
    """Import generated modules without importing ``lmcache.__init__``."""
    module_names = tuple(path.stem for path in generated_files if path.suffix == ".py")
    script = f"""
from pathlib import Path
import importlib
import sys
import types

package = {generated_package!r}
project_root = Path({str(PROJECT_ROOT)!r})
generated_dir = Path({str(_package_dir(generated_package))!r})
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


def generate(generated_package: str = GENERATED_PACKAGE) -> None:
    """Generate all protobuf and gRPC modules of one ``_proto_gen`` package.

    Args:
        generated_package: Dotted name of the package that receives the
            generated modules; its directory must exist under the project
            root. Defaults to the multiprocess transport bindings.

    Raises:
        RuntimeError: If no schemas exist, compilation fails, or a generated
            module cannot be imported.
    """
    generated_dir = _package_dir(generated_package)
    proto_dir = generated_dir.parent / "protos"
    proto_files = tuple(sorted(proto_dir.glob("*.proto")))
    if not proto_files:
        raise RuntimeError(f"No proto sources found under {proto_dir}")

    # Third Party
    from grpc_tools import protoc

    _cleanup_generated_files(generated_dir, proto_dir)
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={generated_dir}",
            f"--pyi_out={generated_dir}",
            f"--grpc_python_out={generated_dir}",
            *(str(path) for path in proto_files),
        ]
    )
    if result != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {result}")

    generated_files = _generated_files(generated_dir)
    for path in generated_files:
        _patch_generated_file(path, generated_package)

    if not _check_generated_imports(generated_files, generated_package):
        _cleanup_generated_files(generated_dir, proto_dir)
        raise RuntimeError("Generated gRPC modules failed their import check")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "generated_packages",
        nargs="*",
        default=GENERATED_PACKAGES,
        help="Dotted names of the _proto_gen packages to build (default: all).",
    )
    for package in parser.parse_args().generated_packages:
        generate(package)
