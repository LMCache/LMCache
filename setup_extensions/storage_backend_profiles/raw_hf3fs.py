# SPDX-License-Identifier: Apache-2.0
"""
Build profile for LMCache 3FS connector extension.

This module defines the build configuration for the 3FS connector C++
extension, including library directories, link libraries, and compiler
flags needed to compile against the 3FS library.
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import os

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension


def _threefs_root() -> Path:
    """Locate the 3FS source tree.

    REQUIRED: set HF3FS_3FS_DIR before building. No hardcoded fallback, so
    no machine-specific path can leak into the built artifact.
    """
    override = os.environ.get("HF3FS_3FS_DIR")
    if not override:
        raise RuntimeError(
            "HF3FS_3FS_DIR is not set. Point it at the 3FS source tree "
            "(containing build/) before building, e.g. "
            "HF3FS_3FS_DIR=/path/to/3FS BUILD_WITH_RAW_HF3FS=1 pip install -e ."
        )
    return Path(override)


class RawHf3fsStorageBackend(StorageBackendProfile):
    """Build profile for the raw 3FS storage backend (lmcache_raw3fs).

    Coexists with upstream's separate HF3FS backend (``hf3fs.py`` /
    ``BUILD_WITH_HF3FS``). Enable this one with
    ``BUILD_WITH_RAW_HF3FS=1`` on top of ``HF3FS_3FS_DIR``.
    """

    name: str = "raw_hf3fs"
    env_var: str = "BUILD_WITH_RAW_HF3FS"

    def is_explicitly_requested(self) -> bool:
        """Return True when 3FS backend was selected via env var."""
        if not self.env_var:
            return False
        return os.environ.get(self.env_var, "0") == "1"

    def detect(self) -> bool:
        """Auto-detect if 3FS SDK is available.

        Returns False when ``HF3FS_3FS_DIR`` is unset or does not point at
        a built 3FS tree, so machines without 3FS skip this backend
        instead of failing the build. The explicit-request path
        (``BUILD_WITH_RAW_HF3FS=1``) still raises via ``_threefs_root()``
        with an actionable message.
        """
        override = os.environ.get("HF3FS_3FS_DIR")
        if not override:
            return False
        return (Path(override) / "build" / "src" / "client" / "meta").exists()

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the 3FS connector extension module.

        Args:
            extra_cxx_flags: Additional C++ compiler flags (unused for 3FS).

        Returns:
            List of Extension objects for the 3FS backend.
        """
        # Third Party
        from torch.utils import cpp_extension

        threefs_root = _threefs_root()
        third_party = threefs_root / "third_party"
        build_tp = threefs_root / "build" / "third_party"
        ch_contrib = build_tp / "clickhouse-cpp" / "contrib"

        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_raw3fs",
                sources=[
                    "csrc/storage_backends/raw_hf3fs/pybind.cpp",
                    "csrc/storage_backends/raw_hf3fs/hf3fs_connector.cc",
                    "csrc/storage_backends/raw_hf3fs/data_store.cc",
                ],
                include_dirs=[
                    # 3FS bundled third-party libs MUST come first
                    # to override torch/system headers (e.g. fmt v10 vs v11)
                    str(third_party / "fmt" / "include"),
                    str(third_party / "scnlib" / "include"),
                    str(third_party / "robin_hood" / "include"),
                    str(third_party / "folly"),
                    str(third_party),
                    # For "third_party/..." relative includes from 3FS src
                    str(threefs_root),
                    str(build_tp),
                    str(build_tp / "folly"),
                    str(build_tp / "folly" / "include"),
                    str(build_tp / "fmt"),
                    str(build_tp / "robin_hood"),
                    # 3FS source
                    str(threefs_root / "src"),
                    str(threefs_root / "build" / "src"),
                    # LMCache source
                    "csrc/storage_backends",
                    "csrc/storage_backends/raw_hf3fs",
                    # Torch (must come last to avoid overriding 3FS libs)
                    str(
                        third_party
                        / "scnlib"
                        / "src"
                        / "deps"
                        / "fast_float"
                        / "single_include"
                    ),
                ],
                library_dirs=[
                    # 3FS client / stubs / fbs libraries actually linked
                    # (see `libraries`)
                    str(threefs_root / "build" / "src" / "client" / "meta"),
                    str(threefs_root / "build" / "src" / "client" / "storage"),
                    str(threefs_root / "build" / "src" / "client" / "mgmtd"),
                    str(threefs_root / "build" / "src" / "common"),
                    str(threefs_root / "build" / "src" / "stubs" / "mgmtd"),
                    str(threefs_root / "build" / "src" / "stubs" / "MetaService"),
                    str(threefs_root / "build" / "src" / "fbs" / "meta"),
                    str(threefs_root / "build" / "src" / "fbs" / "mgmtd"),
                    str(threefs_root / "build" / "src" / "fbs" / "storage"),
                    str(threefs_root / "build" / "src" / "fbs" / "core" / "user"),
                    str(threefs_root / "build" / "src" / "fbs" / "monitor_collector"),
                    # Bundled third-party libraries actually linked
                    str(build_tp / "folly"),
                    str(build_tp / "fmt"),
                    str(build_tp / "scnlib"),
                    str(build_tp / "liburing-cmake"),
                    str(build_tp / "clickhouse-cpp" / "clickhouse"),
                    str(ch_contrib / "cityhash"),
                    str(ch_contrib / "lz4"),
                    str(ch_contrib / "absl"),
                    # glog: correct ABI (0.5.0) from /usr/local/lib
                    "/usr/local/lib",
                ],
                libraries=[
                    # Core 3FS libraries
                    "meta-client",
                    "storage-client",
                    "mgmtd-client",
                    "common",
                    "mgmtd-stub",
                    "meta-stub",
                    "meta-fbs",
                    "mgmtd-fbs",
                    "storage-fbs",
                    # Logging
                    "glog",
                    # Folly (static - must come before other libs that depend on it)
                    "folly",
                    # Utility libraries
                    "fmt",
                    "scn",
                    # NOTE: glog is intentionally NOT listed here as a static library.
                    # We use the system shared glog (libglog.so) to avoid conflicts
                    # with glog symbols embedded in the static folly/common libraries.
                    "gflags",
                    "ibverbs",
                    "event",
                    "boost_context",
                    "boost_thread",
                    "boost_system",
                    "boost_regex",
                    "boost_filesystem",
                    "boost_atomic",
                    "boost_program_options",
                    "double-conversion",
                    "aio",
                    "atomic",
                    "bz2",
                    "lz4",
                    "lzma",
                    "numa",
                    "snappy",
                    "unwind",
                    "uring",
                    "z",
                    "zstd",
                    "dwarf",
                    "clickhouse-cpp-lib-static",
                    "cityhash-lib",
                    "lz4-lib",
                    "absl-lib",
                    # 3fs_liburing (io_uring for RDMA buffer registration)
                    "3fs_liburing",
                    # MonitorCollector RPC types (pulled by client monitor reporting)
                    "MonitorCollectorService-fbs",
                    # core-user-fbs defines flat::UserInfo (construct/isRoot/inGroup),
                    # referenced by the retained client libs; removing it breaks
                    # dlopen with an undefined symbol at load time.
                    "core-user-fbs",
                ],
                extra_compile_args=[
                    "-std=c++20",
                    "-fPIC",
                    # Optimized release build. The previous "-O0 -g" pulled the
                    # static 3FS/folly object files' DWARF debug sections into
                    # lmcache_raw3fs.so (~184 MB of the 225 MB total). We now
                    # compile our own objects with -O2 -g0 and strip debug at
                    # link time (--strip-debug) while keeping the compile-time
                    # DBG() macro fully functional via -DLMCACHE_DBG.
                    "-O2",
                    "-g0",
                    # Enable linker dead-code elimination (--gc-sections).
                    "-ffunction-sections",
                    "-fdata-sections",
                    # Hide our own symbols from the dynamic table; pybind11
                    # re-exports PyInit_lmcache_raw3fs with default visibility.
                    "-fvisibility=hidden",
                    # Keep the DBG() macro available: set
                    # BUILD_WITH_HF3FS_DBG=1 to include it.
                ]
                + (
                    ["-DLMCACHE_DBG"]
                    if os.environ.get("BUILD_WITH_HF3FS_DBG", "0") == "1"
                    else []
                ),
                extra_link_args=[
                    "-Wl,--no-as-needed",
                    "-Wl,--as-needed",
                    # Discard DWARF .debug_* sections that the static 3FS/folly
                    # libraries embed (built with -g). This removes ~184 MB.
                    "-Wl,--strip-debug",
                    # Drop unused functions/data compiled into
                    # -ffunction-sections/-fdata-sections.
                    "-Wl,--gc-sections",
                    # Optimize the link (shrinks code, dedupes strings).
                    "-Wl,-O1",
                ],
            )
        ]


# Module-level instance for auto-discovery
raw_hf3fs_backend = RawHf3fsStorageBackend()


def get_3fs_build_profile():
    """Get the build profile for 3FS connector.

    Returns:
        dict: Build configuration including library dirs, link libraries,
              and compiler flags.
    """
    # 3FS build directories
    threefs_root = _threefs_root()

    # Library directories
    lib_dirs = [
        str(threefs_root / "build" / "src" / "client" / "meta"),
        str(threefs_root / "build" / "src" / "client" / "storage"),
        str(threefs_root / "build" / "src" / "common"),
        str(threefs_root / "build" / "third_party" / "folly"),
        str(threefs_root / "build" / "third_party" / "fmt"),
        str(threefs_root / "build" / "third_party" / "robin_hood"),
    ]

    # Link libraries
    link_libs = [
        "hf3fs_meta_client",
        "hf3fs_storage_client",
        "hf3fs_common",
        "hf3fs_folly",
        "hf3fs_fmt",
        "hf3fs_robin_hood",
    ]

    # Compiler flags
    compiler_flags = [
        "-std=c++17",
        "-fPIC",
        "-O2",
        "-DNDEBUG",
        "-I" + str(threefs_root / "src"),
        "-I" + str(threefs_root / "build" / "src"),
        "-I" + str(threefs_root / "third_party"),
    ]

    return {
        "lib_dirs": lib_dirs,
        "link_libs": link_libs,
        "compiler_flags": compiler_flags,
        "threefs_root": str(threefs_root),
    }


# For direct use
if __name__ == "__main__":
    profile = get_3fs_build_profile()
    print(f"3FS Root: {profile['threefs_root']}")
    print(f"Library Dirs: {len(profile['lib_dirs'])}")
    print(f"Link Libraries: {len(profile['link_libs'])}")
    print(f"Compiler Flags: {len(profile['compiler_flags'])}")
