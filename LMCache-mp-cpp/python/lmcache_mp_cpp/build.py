# SPDX-License-Identifier: Apache-2.0
"""Build helper for the standalone C++ tiered-cache shared library."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import subprocess


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def library_path() -> Path:
    suffix = ".dylib" if __import__("sys").platform == "darwin" else ".so"
    return project_root() / ".build" / f"liblmcache_mp_cpp{suffix}"


def build_library(force: bool = False) -> Path:
    root = project_root()
    out = library_path()
    sources = [
        root / "src" / "tiered_cache.cpp",
        root / "src" / "key_compat.cpp",
        root / "src" / "ipc_key.cpp",
        root / "src" / "l2_adapter.cpp",
        root / "src" / "protocol_compat.cpp",
    ]
    headers = [
        root / "include" / "lmcache_mp_cpp" / "tiered_cache.h",
        root / "include" / "lmcache_mp_cpp" / "key_compat.h",
        root / "include" / "lmcache_mp_cpp" / "ipc_key.h",
        root / "include" / "lmcache_mp_cpp" / "l2_adapter.h",
        root / "include" / "lmcache_mp_cpp" / "protocol.h",
    ]

    if (
        not force
        and out.exists()
        and out.stat().st_mtime
        >= max(path.stat().st_mtime for path in sources + headers)
    ):
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        *(str(src) for src in sources),
        "-I",
        str(root / "include"),
        "-I",
        "/usr/include",
        "/usr/lib/llvm-18/lib/libLLVM-18.so",
        "-o",
        str(out),
    ]
    subprocess.run(cmd, cwd=root, check=True)
    return out
