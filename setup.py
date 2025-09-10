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

# New environment variable to choose between CUDA and HIP
BUILD_WITH_HIP = os.environ.get("BUILD_WITH_HIP", "0") == "1"

ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"


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
        "csrc/cal_cdf.cu",
        "csrc/ac_enc.cu",
        "csrc/ac_dec.cu",
        "csrc/pos_kernels.cu",
        "csrc/mem_alloc.cpp",
        "csrc/utils.cpp",
    ]
    ext_modules = [
        cpp_extension.CUDAExtension(
            "lmcache.c_ops",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": [flag_cxx_abi],
                "nvcc": [flag_cxx_abi],
            },
        ),
    ]
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
        "csrc/cal_cdf.hip",
        "csrc/ac_enc.hip",
        "csrc/ac_dec.hip",
        "csrc/pos_kernels.hip",
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
                    "-O3"
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
        )
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def source_dist_extension() -> tuple[list, dict]:
    print("Not building CUDA/HIP extensions for sdist")
    return [], {}


if __name__ == "__main__":
    if BUILDING_SDIST:
        get_extension = source_dist_extension
    elif BUILD_WITH_HIP:
        get_extension = rocm_extension
    else:
        get_extension = cuda_extension

    ext_modules, cmdclass = get_extension()

    setup(
        packages=find_packages(
            exclude=("csrc",)
        ),  # Ensure csrc is excluded if it only contains sources
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
    )
