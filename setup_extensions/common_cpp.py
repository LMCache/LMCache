# SPDX-License-Identifier: Apache-2.0
"""Common C++ extension builders shared by all backends.

These extensions (storage manager, Redis, filesystem, optional Mooncake)
are always compiled regardless of which backend is selected.
"""

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension


def _mooncake_extension(
    cpp_extension,
    mooncake_sources: list[str],
    extra_cxx_flags: list[str],
) -> list["Extension"]:
    """Build mooncake CppExtension if enabled via env vars."""
    # Standard
    import os

    mc_env = os.environ.get("BUILD_MOONCAKE")
    if mc_env is not None:
        build_mc = mc_env == "1"
    else:
        build_mc = os.environ.get("MOONCAKE_INCLUDE_DIR", "") != ""
    if not build_mc:
        return []

    mc_include = os.environ.get("MOONCAKE_INCLUDE_DIR", "")
    mc_lib = os.environ.get("MOONCAKE_LIB_DIR", "")
    mc_include_dirs = [
        "csrc/storage_backends",
        "csrc/storage_backends/mooncake",
    ]
    if mc_include:
        mc_include_dirs.extend(mc_include.split(";"))
    mc_library_dirs: list[str] = []
    if mc_lib:
        mc_library_dirs.extend(mc_lib.split(";"))
    return [
        cpp_extension.CppExtension(
            "lmcache.lmcache_mooncake",
            sources=mooncake_sources,
            include_dirs=mc_include_dirs,
            library_dirs=mc_library_dirs,
            libraries=["mooncake_store"],
            runtime_library_dirs=mc_library_dirs,
            extra_compile_args={
                "cxx": extra_cxx_flags + ["-O3", "-std=c++20", "-DYLT_ENABLE_IBV"],
            },
        ),
    ]


def build_common_cpp(
    extra_cxx_flags: list[str],
    fs_extra_cxx_flags: list[str] | None = None,
) -> tuple[list["Extension"], dict]:
    """Build pure C++ extensions that do not depend on any backend.

    Args:
        extra_cxx_flags: Additional C++ compiler flags applied to
            ``native_storage_ops``, ``lmcache_redis``, and mooncake.
        fs_extra_cxx_flags: Additional C++ flags for ``lmcache_fs``.
            Defaults to ``extra_cxx_flags`` when not set.

    Notes:
        ``fs_extra_cxx_flags`` preserves pre-refactor SYCL behaviour where
        ``lmcache_fs`` intentionally omitted the ABI define.

    Returns:
        ``(ext_modules, cmdclass)`` tuple.
    """
    # Third Party
    from torch.utils import cpp_extension

    if fs_extra_cxx_flags is None:
        fs_extra_cxx_flags = extra_cxx_flags

    storage_manager_sources = [
        "csrc/storage_manager/bitmap.cpp",
        "csrc/storage_manager/periodic_event_notifier.cpp",
        "csrc/storage_manager/pybind.cpp",
        "csrc/storage_manager/ttl_lock.cpp",
        "csrc/storage_manager/utils.cpp",
    ]
    redis_sources = [
        "csrc/storage_backends/redis/pybind.cpp",
        "csrc/storage_backends/redis/connector.cpp",
    ]
    fs_sources = [
        "csrc/storage_backends/fs/pybind.cpp",
        "csrc/storage_backends/fs/connector.cpp",
    ]
    mooncake_sources = [
        "csrc/storage_backends/mooncake/pybind.cpp",
        "csrc/storage_backends/mooncake/connector.cpp",
    ]
    ext_modules = [
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": extra_cxx_flags + ["-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_redis",
            sources=redis_sources,
            include_dirs=[
                "csrc/storage_backends",
                "csrc/storage_backends/redis",
            ],
            extra_compile_args={
                "cxx": extra_cxx_flags + ["-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_fs",
            sources=fs_sources,
            include_dirs=[
                "csrc/storage_backends",
                "csrc/storage_backends/fs",
            ],
            extra_compile_args={
                "cxx": fs_extra_cxx_flags + ["-O3", "-std=c++17"],
            },
        ),
    ]
    # Mooncake extension is optional.
    ext_modules.extend(
        _mooncake_extension(cpp_extension, mooncake_sources, extra_cxx_flags)
    )
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass
