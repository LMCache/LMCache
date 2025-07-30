# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os
import sys

# Third Party
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

import logging
import sysconfig
import subprocess
import platform
import shutil


ROOT_DIR = Path(__file__).parent
HIPIFY_DIR = os.path.join(ROOT_DIR, "csrc/")
HIPIFY_OUT_DIR = os.path.join(ROOT_DIR, "csrc_hip/")

# python -m build --sdist
# will run python setup.py sdist --dist-dir dist
BUILDING_SDIST = "sdist" in sys.argv or os.environ.get("NO_CUDA_EXT", "0") == "1"

# Environment variable to choose between CUDA, HIP, Ascend
TARGET_DEVICE = os.environ.get("LMCACHE_TARGET_DEVICE", "CUDA") 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_ascend_home_path():
    # NOTE: standard Ascend CANN toolkit path
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

def _get_ascend_driver_path():
    # NOTE: standard Ascend path
    return os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver")

def _get_ascend_env_path():
    # NOTE: standard Ascend Environment variable setup path
    env_script_path = os.path.realpath(os.path.join(_get_ascend_home_path(), "..", "set_env.sh"))
    if not os.path.exists(env_script_path):
        raise ValueError(f"The file '{env_script_path}' is not found, "
                            "please make sure environment variable 'ASCEND_HOME_PATH' is set correctly.")
    return env_script_path

def _get_npu_soc():
    _soc_version = os.getenv("SOC_VERSION", None)
    if _soc_version is None:
        npu_smi_cmd = [
            "bash",
            "-c",
            "npu-smi info | grep OK | awk '{print $3}' | head -n 1",
        ]
        try:
            _soc_version = subprocess.check_output(npu_smi_cmd,
                                                   text=True).strip()
            _soc_version = _soc_version.split("-")[0]
            _soc_version = "Ascend"+_soc_version
            return _soc_version
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Retrieve SoC version failed: {e}")
    return _soc_version   

class CMakeExtension(Extension):

    def __init__(self,
                 name: str,
                 cmake_lists_dir: str = ".",
                 **kwargs) -> None:
        super().__init__(name, sources=[], py_limited_api=False, **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class custom_install(install):
    def run(self):
        self.run_command("build_ext")
        install.run(self)
        
class CustomAscendCmakeBuildExt(build_ext):
    
    def build_extension(self, ext):
        # build the so as c_ops
        ext_name = ext.name.split(".")[-1]
        so_name = ext_name + ".so"
        logger.info(f"Building {so_name} ...")
        OPS_DIR = os.path.join(ROOT_DIR, "csrc", "ascend")
        BUILD_OPS_DIR = os.path.join(ROOT_DIR, "build", "ascend")
        os.makedirs(BUILD_OPS_DIR, exist_ok=True)
       
        ascend_home_path = _get_ascend_home_path()
        ascend_driver_path = _get_ascend_driver_path()
        env_path = _get_ascend_env_path()
        _soc_version = _get_npu_soc()
        _cxx_compiler = os.getenv("CXX")
        _cc_compiler = os.getenv("CC")
        python_executable = sys.executable

        try:
            # if pybind11 is installed via pip
            pybind11_cmake_path = (subprocess.check_output(
                [python_executable, "-m", "pybind11",
                 "--cmakedir"]).decode().strip())
        except subprocess.CalledProcessError as e:
            # else specify pybind11 path installed from source code on CI container
            raise RuntimeError(f"CMake configuration failed: {e}")
        
        import torch_npu
        torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        import torch
        torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        
        # python include
        python_include_path = sysconfig.get_path('include', scheme='posix_prefix')
        
        arch = platform.machine()
        install_path = os.path.join(BUILD_OPS_DIR, 'install')
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path=BUILD_OPS_DIR
        
        cmake_cmd = [
            f"source {env_path} && "
            f"cmake -S {OPS_DIR} -B {BUILD_OPS_DIR}"
            f"  -DSOC_VERSION={_soc_version}"
            f"  -DARCH={arch}"
            "  -DUSE_ASCEND=1"
            f"  -DPYTHON_EXECUTABLE={python_executable}"
            f"  -DCMAKE_PREFIX_PATH={pybind11_cmake_path}"
            f"  -DCMAKE_BUILD_TYPE=Release"
            f"  -DCMAKE_INSTALL_PREFIX={install_path}"
            f"  -DPYTHON_INCLUDE_PATH={python_include_path}"
            f"  -DTORCH_NPU_PATH={torch_npu_path}"
            f"  -DTORCH_PATH={torch_path}"
            f"  -DASCEND_CANN_PACKAGE_PATH={ascend_home_path}"
            f"  -DASCEND_DRIVER_PATH={ascend_driver_path}"
            "  -DCMAKE_VERBOSE_MAKEFILE=ON"
        ]
        
        if _cxx_compiler is not None:
            cmake_cmd += [f"  -DCMAKE_CXX_COMPILER={_cxx_compiler}"]
        
        if _cc_compiler is not None:
            cmake_cmd += [f"  -DCMAKE_C_COMPILER={_cc_compiler}"]
        
        cmake_cmd += [f" && cmake --build {BUILD_OPS_DIR} -j --verbose"]
        cmake_cmd += [f" && cmake --install {BUILD_OPS_DIR}"]
        cmake_cmd = "".join(cmake_cmd)
        
        logger.info(f"Start running CMake commands:\n{cmake_cmd}")
        try:
            result = subprocess.run(cmake_cmd, cwd=ROOT_DIR, text=True, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build {so_name}: {e}")
        
        build_lib_dir = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(build_lib_dir), exist_ok=True)
        
        package_name = ext.name.split('.')[0] # e.g., 'lmcache'
        src_dir = os.path.join(ROOT_DIR, package_name)
        
        for root, _, files in os.walk(install_path):
            for file in files:
                if file.endswith(".so"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(os.path.dirname(build_lib_dir), file)
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    
                    if isinstance(self.distribution.get_command_obj("develop"), develop):
                        # For the ascend kernels
                        src_dir_file = os.path.join(src_dir, file)
                        shutil.copy(src_path, src_dir_file)
                    shutil.copy(src_path, dst_path)

                    logger.info(f"Copied {file} to {dst_path}")
        
        
        
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

    cuda_sources = [
        "csrc/pybind.cpp",
        "csrc/mem_kernels.cu",
        "csrc/cal_cdf.cu",
        "csrc/ac_enc.cu",
        "csrc/ac_dec.cu",
        "csrc/pos_kernels.cu",
    ]
    ext_modules = [
        cpp_extension.CUDAExtension(
            "lmcache.c_ops",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=0"],
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


def ascend_extension():
    print("Building Ascend extensions")
    return [CMakeExtension(name="lmcache.c_ops")], \
        {"build_ext": CustomAscendCmakeBuildExt}


def source_dist_extension() -> tuple[list, dict]:
    print("Not building CUDA/HIP extensions for sdist")
    return [], {}


if __name__ == "__main__":
    if BUILDING_SDIST:
        get_extension = source_dist_extension
        TARGET_DEVICE = "EMPTY"
    elif TARGET_DEVICE == "HIP":
        get_extension = rocm_extension
    elif TARGET_DEVICE == "CUDA":
        get_extension = cuda_extension
    elif TARGET_DEVICE == "ASCEND":
        get_extension = ascend_extension
        
    ext_modules, cmdclass = get_extension()

    setup(
        packages=find_packages(
            exclude=("csrc",)
        ),  # Ensure csrc is excluded if it only contains sources
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
    )
