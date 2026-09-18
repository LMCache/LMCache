# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
# Authors: Wenwen Chen <wenwen.chen@samsung.com>

"""HF3FS native L2 storage backend profile.

Builds the ``lmcache.lmcache_hf3fs`` extension when the HF3FS SDK
is available.  Enabled via ``BUILD_WITH_HF3FS=1`` or auto-detected
through ``HF3FS_INCLUDE_DIR`` / ``HF3FS_LIB_DIR`` env vars.

The HF3FS backend depends on the system Abseil library (libabsl-dev)
for hash set implementations (``sharded_flat_hash_set``, ``flat_hash_set``).
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import os

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile

ROOT_DIR = Path(__file__).parent.parent.parent

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension


class Hf3fsStorageBackend(StorageBackendProfile):
    """HF3FS native 3FS L2 storage backend."""

    name = "hf3fs"
    env_var = "BUILD_WITH_HF3FS"

    def detect(self) -> bool:
        """Detect HF3FS SDK via ``HF3FS_INCLUDE_DIR`` / ``HF3FS_LIB_DIR``.

        Verifies both the Abseil header and shared library are present.
        """
        hf3fs_env = os.environ.get("BUILD_HF3FS")
        if hf3fs_env == "0":
            return False

        if hf3fs_env == "1":
            if self._check_absl_available():
                return True

            # Abseil missing
            print("[HF3FS] WARNING: BUILD_HF3FS=1 set but Abseil not available !!!")
            print(
                "[HF3FS] WARNING: unset BUILD_HF3FS or install libabsl-dev:"
                " sudo apt-get install -y libabsl-dev "
            )
            return False

        # Not explicitly enabled - auto-detect skips
        return False

    @staticmethod
    def _check_absl_available() -> bool:
        """Check if Abseil headers and shared libraries are available.

        Returns:
            ``True`` if ``absl_hash`` shared library is findable via
            :func:`ctypes.util.find_library` AND the
            ``flat_hash_set.h`` header is present under
            ``HF3FS_INCLUDE_DIR`` (default ``/usr/include``).
        """
        # Standard
        import ctypes.util

        # 1. Check Abseil shared library via ctypes
        if not ctypes.util.find_library("absl_hash"):
            print("[HF3FS] WARNING: Abseil library (libabsl-dev) not found !!!")
            return False

        # 2. Check header file
        include_dir = os.environ.get("HF3FS_INCLUDE_DIR", "/usr/include")
        header_path = os.path.join(include_dir, "absl", "container", "flat_hash_set.h")
        if not os.path.isfile(header_path):
            print(f"[HF3FS] WARNING: Abseil header not found at {include_dir}. !!!")
            return False

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

        hf3fs_include = os.environ.get("HF3FS_INCLUDE_DIR", "")
        hf3fs_lib = os.environ.get("HF3FS_LIB_DIR", "")
        include_dirs = [
            str(ROOT_DIR / "csrc/storage_backends"),
            str(ROOT_DIR / "csrc/storage_backends/hf3fs"),
        ]
        if hf3fs_include:
            include_dirs.extend(hf3fs_include.split(";"))
        library_dirs: list[str] = []
        if hf3fs_lib:
            library_dirs.extend(hf3fs_lib.split(";"))
        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_hf3fs",
                sources=[
                    "csrc/storage_backends/hf3fs/pybind.cpp",
                    "csrc/storage_backends/hf3fs/connector.cpp",
                    "csrc/storage_backends/hf3fs/hf3fs_absl_init.cpp",
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs + ["/usr/lib/x86_64-linux-gnu"],
                libraries=[
                    "hf3fs_api_shared",
                    "absl_hash",
                    "absl_city",
                    "absl_int128",
                    "absl_raw_hash_set",
                ],
                runtime_library_dirs=library_dirs + ["/usr/lib/x86_64-linux-gnu"],
                extra_compile_args={
                    "cxx": extra_cxx_flags + ["-O3", "-std=c++17"],
                },
            ),
        ]
