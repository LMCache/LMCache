# SPDX-License-Identifier: Apache-2.0
"""Optional native NIXL storage backend build profile."""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import ctypes.util
import os

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile


class NixlStorageBackend(StorageBackendProfile):
    """Build the native NIXL connector against a NIXL 1.3+ SDK."""

    name = "nixl"
    env_var = "BUILD_WITH_NIXL"

    def detect(self) -> bool:
        """Return whether both the NIXL header and library are available."""
        return self._find_include_dir() is not None and self._has_library()

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the isolated C++20 ``lmcache_nixl`` extension.

        Args:
            extra_cxx_flags: Platform-specific compiler flags selected by the
                main build profile.

        Returns:
            The NIXL extension definition.

        Raises:
            RuntimeError: If a NIXL development header or library is missing.
        """
        # Third Party
        from torch.utils import cpp_extension

        include_dir = self._find_include_dir()
        if include_dir is None:
            raise RuntimeError(
                "BUILD_WITH_NIXL=1 requires NIXL >= 1.3 development headers. "
                "Set NIXL_INCLUDE_DIR to the directory containing nixl.h."
            )

        library_dirs = self._library_dirs()
        if not self._has_library():
            raise RuntimeError(
                "BUILD_WITH_NIXL=1 requires libnixl from NIXL >= 1.3. Set "
                "NIXL_LIBRARY_DIR to the directory containing libnixl.so."
            )

        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_nixl",
                sources=[
                    "csrc/storage_backends/nixl/pybind.cpp",
                    "csrc/storage_backends/nixl/connector.cpp",
                    "csrc/storage_backends/nixl/storage.cpp",
                ],
                include_dirs=[
                    "csrc/storage_backends",
                    "csrc/storage_backends/nixl",
                    str(include_dir),
                ],
                library_dirs=library_dirs,
                libraries=["nixl"],
                runtime_library_dirs=list(library_dirs),
                extra_compile_args={
                    "cxx": extra_cxx_flags + ["-O3", "-std=c++20"],
                },
            )
        ]

    @staticmethod
    def _candidate_include_dirs() -> list[Path]:
        configured = os.environ.get("NIXL_INCLUDE_DIR", "")
        prefix = os.environ.get("NIXL_PREFIX", "")
        candidates: list[Path] = []
        if configured:
            candidates.extend(Path(item) for item in configured.split(os.pathsep))
        if prefix:
            candidates.append(Path(prefix) / "include")
        candidates.extend(
            [
                Path("/usr/local/nixl/include"),
                Path("/usr/local/include"),
                Path("/usr/include"),
            ]
        )
        return candidates

    @classmethod
    def _find_include_dir(cls) -> Path | None:
        return next(
            (
                path
                for path in cls._candidate_include_dirs()
                if (path / "nixl.h").is_file()
            ),
            None,
        )

    @staticmethod
    def _library_dirs() -> list[str]:
        configured = os.environ.get("NIXL_LIBRARY_DIR", "")
        prefix = os.environ.get("NIXL_PREFIX", "")
        paths = configured.split(os.pathsep) if configured else []
        if prefix:
            prefix_path = Path(prefix)
            paths.extend(
                str(path)
                for path in (
                    prefix_path / "lib",
                    prefix_path / "lib64",
                    prefix_path / "lib" / "x86_64-linux-gnu",
                )
                if path.is_dir()
            )
        return paths

    @classmethod
    def _has_library(cls) -> bool:
        if any((Path(path) / "libnixl.so").is_file() for path in cls._library_dirs()):
            return True
        return ctypes.util.find_library("nixl") is not None
