# Standard
from pathlib import Path
import os
import sys

# Third Party
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent


# python -m build --sdist
# will run python setup.py sdist --dist-dir dist
BUILDING_SDIST = "sdist" in sys.argv or os.environ.get("NO_CUDA_EXT", "0") == "1"

if not BUILDING_SDIST:
    print("Building CUDA extensions")
    # Third Party
    from torch.utils import cpp_extension

    ext_modules = [
        cpp_extension.CUDAExtension(
            "lmcache.c_ops",
            [
                "csrc/pybind.cpp",
                "csrc/mem_kernels.cu",
                "csrc/cal_cdf.cu",
                "csrc/ac_enc.cu",
                "csrc/ac_dec.cu",
                "csrc/pos_kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=0"],
            },
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
else:
    # don't build CUDA extensions when building sdist
    print("Not building CUDA extensions")
    ext_modules = []
    cmdclass = {}

setup(
    packages=find_packages(exclude=("csrc")),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
)
