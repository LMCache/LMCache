# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

"""HF3FS native L2 storage backend profile.

Builds the ``lmcache.lmcache_hf3fs`` extension only when explicitly
requested: ``BUILD_HF3FS=1`` builds after checking that the 3FS SDK and
Abseil are present, and ``BUILD_WITH_HF3FS=1`` builds without checks.
``HF3FS_INCLUDE_DIR`` / ``HF3FS_LIB_DIR`` only locate a 3FS SDK installed
outside the system paths; they do not enable the build.

The HF3FS backend depends on the system Abseil library (libabsl-dev)
for hash set implementations (``sharded_flat_hash_set``, ``flat_hash_set``).
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import ctypes.util
import os

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile

ROOT_DIR = Path(__file__).parent.parent.parent

# Default compiler search paths where system packages or source installs
# place their headers.
SYSTEM_INCLUDE_DIRS = ("/usr/include", "/usr/local/include")

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension


def _env_dirs(name: str) -> list[str]:
    """Return the non-empty ``;``-separated directories in env var *name*."""
    return [d for d in os.environ.get(name, "").split(";") if d]


def _has_file(name: str, dirs: list[str]) -> bool:
    """Return whether file *name* exists under any directory in *dirs*."""
    return any(os.path.isfile(os.path.join(d, name)) for d in dirs)


class Hf3fsStorageBackend(StorageBackendProfile):
    """HF3FS native 3FS L2 storage backend."""

    name = "hf3fs"
    env_var = "BUILD_WITH_HF3FS"

    def detect(self) -> bool:
        """Return whether ``BUILD_HF3FS=1`` requests the HF3FS extension.

        The extension is never built by default. When requested, the
        3FS usrbio and Abseil headers are searched in
        ``HF3FS_INCLUDE_DIR`` (``;``-separated) and then the system
        include paths, matching the compiler's include path; the 3FS
        and Abseil libraries are searched in ``HF3FS_LIB_DIR`` and then
        the system library paths, matching the linker's search path.

        Returns:
            ``True`` if ``BUILD_HF3FS=1`` and all dependencies are
            found, ``False`` if ``BUILD_HF3FS`` is not ``1``.

        Raises:
            RuntimeError: If ``BUILD_HF3FS=1`` but a dependency is
                missing, so an explicit request never silently produces
                an install without ``lmcache.lmcache_hf3fs``.
        """
        if os.environ.get("BUILD_HF3FS") != "1":
            return False

        missing = self._missing_dependencies()
        if missing:
            raise RuntimeError(
                "BUILD_HF3FS=1 is set but HF3FS build dependencies are missing:\n  "
                + "\n  ".join(missing)
                + "\nInstall them, or unset BUILD_HF3FS to skip the HF3FS extension."
            )
        return True

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the ``lmcache.lmcache_hf3fs`` CppExtension.

        The extension uses system Abseil (libabsl-dev) for header-only
        inline symbols.

        Args:
            extra_cxx_flags: Additional C++ compiler flags from the
                selected GPU backend profile.

        Returns:
            List with the ``lmcache.lmcache_hf3fs`` Extension.
        """
        # Third Party
        from torch.utils import cpp_extension

        include_dirs = [
            str(ROOT_DIR / "csrc/storage_backends"),
            str(ROOT_DIR / "csrc/storage_backends/hf3fs"),
            *_env_dirs("HF3FS_INCLUDE_DIR"),
        ]
        library_dirs = _env_dirs("HF3FS_LIB_DIR")
        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_hf3fs",
                sources=[
                    "csrc/storage_backends/hf3fs/pybind.cpp",
                    "csrc/storage_backends/hf3fs/connector.cpp",
                    "csrc/storage_backends/hf3fs/hf3fs_absl_init.cpp",
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=[
                    "hf3fs_api_shared",
                    "absl_hash",
                    "absl_city",
                    "absl_int128",
                    "absl_raw_hash_set",
                ],
                runtime_library_dirs=list(library_dirs),
                extra_compile_args={
                    "cxx": extra_cxx_flags + ["-O3", "-std=c++17"],
                },
            ),
        ]

    @staticmethod
    def _missing_dependencies() -> list[str]:
        """Describe each missing build dependency, with where it was searched."""
        missing = []
        include_dirs = _env_dirs("HF3FS_INCLUDE_DIR") + list(SYSTEM_INCLUDE_DIRS)
        if not _has_file("hf3fs_usrbio.h", include_dirs):
            missing.append(
                "3FS header hf3fs_usrbio.h not found in "
                + ", ".join(include_dirs)
                + " (set HF3FS_INCLUDE_DIR)"
            )

        lib_dirs = _env_dirs("HF3FS_LIB_DIR")
        if not (
            _has_file("libhf3fs_api_shared.so", lib_dirs)
            or ctypes.util.find_library("hf3fs_api_shared")
        ):
            missing.append(
                "3FS library libhf3fs_api_shared.so not found in HF3FS_LIB_DIR"
                " or the system library paths (set HF3FS_LIB_DIR)"
            )

        # build() passes every HF3FS_INCLUDE_DIR / HF3FS_LIB_DIR entry to the
        # compiler / linker, so Abseil there is as usable as system Abseil.
        absl_header = os.path.join("absl", "container", "flat_hash_set.h")
        if not _has_file(absl_header, include_dirs):
            missing.append(
                f"Abseil header {absl_header} not found in "
                + ", ".join(include_dirs)
                + " (install libabsl-dev)"
            )
        if not (
            _has_file("libabsl_hash.so", lib_dirs)
            or ctypes.util.find_library("absl_hash")
        ):
            missing.append(
                "Abseil library libabsl_hash.so not found in HF3FS_LIB_DIR"
                " or the system library paths (install libabsl-dev)"
            )

        return missing
