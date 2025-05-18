import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
# It's good practice to import cpp_extension only when needed
# from torch.utils import cpp_extension

ROOT_DIR = Path(__file__).parent

# Taken from https://github.com/vllm-project/vllm/blob/main/setup.py
def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith("--") and not line.startswith(
                    "#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements

# python -m build --sdist
# will run python setup.py sdist --dist-dir dist
BUILDING_SDIST = "sdist" in sys.argv or \
                os.environ.get("NO_CUDA_EXT", "0") == "1" # Keep NO_CUDA_EXT for sdist logic

# New environment variable to choose between CUDA and HIP
BUILD_WITH_HIP = os.environ.get("BUILD_WITH_HIP", "0") == "1"

if not BUILDING_SDIST:
    from torch.utils import cpp_extension # Import here

    if BUILD_WITH_HIP:
        print("Building HIP extensions")
        hip_sources = [
            'csrc/pybind_hip.cpp', # Use the hipified pybind
            'csrc/mem_kernels.hip',
            'csrc/cal_cdf.hip',
            'csrc/ac_enc.hip',
            'csrc/ac_dec.hip',
            'csrc/pos_kernels.hip'
        ]
        # For HIP, we generally use CppExtension and let hipcc handle things.
        # Ensure CXX environment variable is set to hipcc when running this build.
        # e.g., CXX=hipcc python setup.py install
        define_macros = []
        define_macros.append(('__HIP_PLATFORM_HCC__', '1'))
        define_macros.append(('USE_ROCM', '1'))
        ext_modules = [
            cpp_extension.CppExtension(
                'lmcache.c_ops',
                sources=hip_sources,
                extra_compile_args={
                    'cxx': [ # hipcc is typically invoked as a C++ compiler
                        # '-D_GLIBCXX_USE_CXX11_ABI=0',
                        '-O3'
                        # Add any HIP specific flags if needed.
                        # For example, if you need to specify ROCm architecture:
                        # '--offload-arch=gfx942' # (replace with your target arch)
                        # '-x hip' # Sometimes needed to explicitly treat files as HIP
                    ],
                    # No 'nvcc' key for hipcc with CppExtension
                },
                # You might need to specify include paths for ROCm if not found automatically
                include_dirs=[os.path.join(os.environ.get('ROCM_PATH', '/opt/rocm'), 'include')],
                library_dirs=[os.path.join(os.environ.get('ROCM_PATH', '/opt/rocm'), 'lib')],
                # libraries=['amdhip64'] # Or other relevant HIP libs if needed
                define_macros=define_macros,
            )
        ]
        cmdclass = {'build_ext': cpp_extension.BuildExtension}

    else: # Build with CUDA (original logic)
        print("Building CUDA extensions")
        cuda_sources = [
            'csrc/pybind.cpp',
            'csrc/mem_kernels.cu',
            'csrc/cal_cdf.cu',
            'csrc/ac_enc.cu',
            'csrc/ac_dec.cu',
            'csrc/pos_kernels.cu'
        ]
        ext_modules = [
            cpp_extension.CUDAExtension(
                'lmcache.c_ops',
                sources=cuda_sources,
                extra_compile_args={
                    'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=0'],
                    'nvcc': ['-D_GLIBCXX_USE_CXX11_ABI=0']
                },
            ),
        ]
        cmdclass = {'build_ext': cpp_extension.BuildExtension}
else:
    # don't build CUDA/HIP extensions when building sdist
    print("Not building CUDA/HIP extensions for sdist")
    ext_modules = []
    cmdclass = {}

setup(
    packages=find_packages(exclude=("csrc",)), # Ensure csrc is excluded if it only contains sources
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
)
