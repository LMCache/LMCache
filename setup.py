# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os
import sys

# Third Party
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent
HIPIFY_DIR = os.path.join(ROOT_DIR, "csrc/")
HIPIFY_OUT_DIR = os.path.join(ROOT_DIR, "csrc_hip/")

# python -m build --sdist
# will run python setup.py sdist --dist-dir dist
BUILDING_SDIST = "sdist" in sys.argv or os.environ.get("NO_CUDA_EXT", "0") == "1"

# New environment variable to choose between CUDA, HIP, and SYCL
BUILD_WITH_HIP = os.environ.get("BUILD_WITH_HIP", "0") == "1"
BUILD_WITH_SYCL = os.environ.get("BUILD_WITH_SYCL", "0") == "1"

ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"


def _read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []

    reqs: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


def hipify_wrapper() -> None:
    # Third Party
    from torch.utils.hipify.hipify_python import hipify

    print("Hipifying sources ")

    # Get absolute path for all source files.
    extra_files = [
        os.path.abspath(os.path.join(HIPIFY_DIR, item))
        for item in os.listdir(HIPIFY_DIR)
        if os.path.isfile(os.path.join(HIPIFY_DIR, item))
    ]

    hipify_result = hipify(
        project_directory=HIPIFY_DIR,
        output_directory=HIPIFY_OUT_DIR,
        header_include_dirs=[],
        includes=[],
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    hipified_sources = []
    for source in extra_files:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (
                s_abs in hipify_result
                and hipify_result[s_abs].hipified_path is not None
            )
            else s_abs
        )
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(extra_files)


def _mooncake_extension(
    cpp_extension,
    mooncake_sources: list[str],
    extra_cxx_flags: list[str],
) -> list:
    """Build mooncake CppExtension if enabled via env vars.

    Returns a list with zero or one Extension objects.
    """
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


def cuda_extension() -> tuple[list, dict]:
    # Third Party
    from torch.utils import cpp_extension  # Import here

    print("Building CUDA extensions")
    global ENABLE_CXX11_ABI
    if ENABLE_CXX11_ABI:
        flag_cxx_abi = "-D_GLIBCXX_USE_CXX11_ABI=1"
    else:
        flag_cxx_abi = "-D_GLIBCXX_USE_CXX11_ABI=0"

    cuda_sources = [
        "csrc/pybind.cpp",
        "csrc/mem_kernels.cu",
        "csrc/mp_mem_kernels.cu",
        "csrc/cal_cdf.cu",
        "csrc/ac_enc.cu",
        "csrc/ac_dec.cu",
        "csrc/pos_kernels.cu",
        "csrc/mem_alloc.cpp",
        "csrc/utils.cpp",
        "csrc/event_recorder.cpp",
    ]
    storage_manager_sources = [
        "csrc/storage_manager/bitmap.cpp",
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
        cpp_extension.CUDAExtension(
            "lmcache.c_ops",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-std=c++17"],
                "nvcc": [flag_cxx_abi],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_redis",
            sources=redis_sources,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/redis"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_fs",
            sources=fs_sources,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/fs"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
        ),
    ]
    # Mooncake extension is optional.
    ext_modules.extend(
        _mooncake_extension(cpp_extension, mooncake_sources, [flag_cxx_abi])
    )
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def rocm_extension() -> tuple[list, dict]:
    # Third Party
    from torch.utils import cpp_extension  # Import here

    print("Building ROCM extensions")
    hipify_wrapper()
    hip_sources = [
        "csrc/pybind_hip.cpp",  # Use the hipified pybind
        "csrc/mem_kernels.hip",
        "csrc/mp_mem_kernels.hip",
        "csrc/cal_cdf.hip",
        "csrc/ac_enc.hip",
        "csrc/ac_dec.hip",
        "csrc/pos_kernels.hip",
        "csrc/mem_alloc_hip.cpp",
        "csrc/utils_hip.cpp",
        "csrc/event_recorder.cpp",
    ]
    storage_manager_sources = [
        "csrc/storage_manager/bitmap.cpp",
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
    # For HIP, we generally use CppExtension and let hipcc handle things.
    # Ensure CXX environment variable is set to hipcc when running this build.
    # e.g., CXX=hipcc python setup.py install
    define_macros = [("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1")]
    ext_modules = [
        cpp_extension.CppExtension(
            "lmcache.c_ops",
            sources=hip_sources,
            extra_compile_args={
                "cxx": [  # hipcc is typically invoked as a C++ compiler
                    # '-D_GLIBCXX_USE_CXX11_ABI=0',
                    "-O3",
                    "-std=c++17",
                    # Add any HIP specific flags if needed.
                    # For example, if you need to specify ROCm architecture:
                    # '--offload-arch=gfx942' # (replace with your target arch)
                    # '-x hip' # Sometimes needed to explicitly treat files as HIP
                ],
                # No 'nvcc' key for hipcc with CppExtension
            },
            # You might need to specify include paths for ROCm if not found
            # automatically
            include_dirs=[
                os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include")
            ],
            library_dirs=[
                os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
            ],
            # libraries=['amdhip64'] # Or other relevant HIP libs if needed
            define_macros=define_macros,
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_redis",
            sources=redis_sources,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/redis"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_fs",
            sources=fs_sources,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/fs"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
    ]
    # Mooncake extension is optional.
    ext_modules.extend(_mooncake_extension(cpp_extension, mooncake_sources, []))
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def sycl_extension() -> tuple[list, dict]:
    # Third Party
    from torch.utils import cpp_extension  # Import here

    print("Building SYCL/XPU extensions")

    # Standard
    import shutil

    if shutil.which("icpx") is None:
        sys.exit("icpx not found. Please source oneAPI setvars.sh at first")
    os.environ["CXX"] = "icpx"
    oneapi_root = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
    include_dirs = [f"{oneapi_root}/include"]
    library_dirs = [f"{oneapi_root}/lib"]

    sycl_sources = [
        "csrc/sycl/pybind_sycl.cpp",
        "csrc/sycl/mem_kernels_sycl.cpp",
    ]
    storage_manager_sources = [
        "csrc/storage_manager/bitmap.cpp",
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
    # Use CppExtension with DPC++ compiler (set CXX=icpx before invoking).
    # The -fsycl flag enables SYCL compilation and linking.
    # Intel XPU optimizations:
    #   -ftarget-register-alloc-mode=pvc:auto
    #       Register allocation tuned for Ponte Vecchio / Data Center GPU Max.
    #   -fno-sycl-id-queries-fit-in-int
    #       Allow 64-bit index arithmetic in SYCL kernels.
    #   -ffast-math
    #       Aggressive FP opts (safe for current implementation: kernels
    #       only copy data, no FP arithmetic.  Review if FP math is added).
    #   -funroll-loops
    #       Unroll inner copy loops for better instruction packing.
    ext_modules = [
        cpp_extension.SyclExtension(
            "lmcache.xpu_ops",
            sources=sycl_sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={
                "cxx": [
                    "-std=c++17",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    "-O3",
                    "-fsycl",
                    "-fno-sycl-id-queries-fit-in-int",
                    "-ffast-math",
                    "-funroll-loops",
                    # Suppress deprecation warnings from SYCL standard
                    # headers (sycl/accessor.hpp references the deprecated
                    # 'host_buffer' internally; our code uses USM only).
                    "-Wno-deprecated-declarations",
                    "-Wno-nan-infinity-disabled",
                ],
            },
            extra_link_args=["-fsycl"],
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_redis",
            sources=redis_sources,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/redis"],
            extra_compile_args={
                "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_fs",
            sources=fs_sources,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/fs"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def source_dist_extension() -> tuple[list, dict]:
    print("Not building CUDA/HIP/SYCL extensions for sdist")
    return [], {}


if __name__ == "__main__":
    if BUILDING_SDIST:
        get_extension = source_dist_extension
    elif BUILD_WITH_SYCL:
        get_extension = sycl_extension
    elif BUILD_WITH_HIP:
        get_extension = rocm_extension
    else:
        get_extension = cuda_extension

    ext_modules, cmdclass = get_extension()

    install_requires = _read_requirements(ROOT_DIR / "requirements" / "common.txt")
    if BUILD_WITH_HIP:
        core_file = "rocm_core.txt"
    elif BUILD_WITH_SYCL:
        core_file = "xpu_core.txt"
    else:
        # CUDA major selects between cu12 and cu13 vendor pins (cupy, nixl).
        # Defaults to cu13 (the PyPI build); cu12.9 wheel builds set
        # LMCACHE_CUDA_MAJOR=12 to pull cu12 wheels of those deps.
        cuda_major = os.environ.get("LMCACHE_CUDA_MAJOR", "13")
        if cuda_major not in ("12", "13"):
            raise ValueError(
                f"LMCACHE_CUDA_MAJOR must be '12' or '13', got '{cuda_major}'"
            )
        core_file = f"cuda{cuda_major}_core.txt"
    install_requires += _read_requirements(ROOT_DIR / "requirements" / core_file)

    setup(
        packages=find_packages(
            exclude=("csrc",)
        ),  # Ensure csrc is excluded if it only contains sources
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
        install_requires=install_requires,
    )
