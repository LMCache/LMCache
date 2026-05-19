# SPDX-License-Identifier: Apache-2.0
"""Build helper for the standalone native MP shared library."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import shutil
import sys
import subprocess


def shared_library_name() -> str:
    if sys.platform == "darwin":
        return "liblmcache_mp_cpp.dylib"
    if sys.platform == "win32":
        return "lmcache_mp_cpp.dll"
    return "liblmcache_mp_cpp.so"


def packaged_library_path() -> Path:
    return Path(__file__).resolve().parent / "lib" / shared_library_name()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_dir() -> Path:
    return project_root() / ".build" / "python"


def library_path() -> Path:
    return build_dir() / shared_library_name()


def _source_is_newer(binary: Path, root: Path) -> bool:
    if not binary.exists():
        return True
    binary_mtime = binary.stat().st_mtime
    suffixes = {".cpp", ".cu", ".cuh", ".h", ".hpp", ".txt"}
    for path in root.rglob("*"):
        if ".build" in path.parts or not path.is_file():
            continue
        if (path.name == "CMakeLists.txt" or path.suffix in suffixes) and (
            path.stat().st_mtime > binary_mtime
        ):
            return True
    return False


def _cmake_cache_source_dir(path: Path) -> Path | None:
    cache = path / "CMakeCache.txt"
    if not cache.exists():
        return None
    for raw in cache.read_text(errors="ignore").splitlines():
        if raw.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            return Path(raw.split("=", 1)[1])
    return None


def _prepare_build_dir(path: Path, root: Path) -> None:
    cached_source = _cmake_cache_source_dir(path)
    if cached_source is not None and cached_source.resolve() != root.resolve():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_library(force: bool = False) -> Path:
    packaged = packaged_library_path()
    if packaged.exists() and not force:
        return packaged

    root = project_root()
    if not (root / "CMakeLists.txt").exists():
        raise FileNotFoundError(
            "packaged native MP shared library is missing and native source "
            f"tree is unavailable: {packaged}"
        )
    out = library_path()
    if not force and not _source_is_newer(out, root):
        return out

    _prepare_build_dir(build_dir(), root)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_dir()),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLMCACHE_BUILD_NATIVE_MP=OFF",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir()), "--target", "lmcache_mp_cpp"],
        cwd=root,
        check=True,
    )
    return out
