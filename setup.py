import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent


# Taken from https://github.com/vllm-project/vllm/blob/main/setup.py
def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

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

    requirements = _read_requirements("common.txt")
    return requirements


# python -m build --sdist
# will run python setup.py sdist --dist-dir dist
BUILDING_SDIST = "sdist" in sys.argv or \
                os.environ.get("NO_CUDA_EXT", "0") == "1"

if not BUILDING_SDIST:
    print("Building CUDA extensions")
    from torch.utils import cpp_extension
    ext_modules = [
        cpp_extension.CUDAExtension(
            'lmcache.c_ops',
            [
                'csrc/pybind.cpp', 'csrc/mem_kernels.cu', 'csrc/cal_cdf.cu',
                'csrc/ac_enc.cu', 'csrc/ac_dec.cu', 'csrc/pos_kernels.cu'
            ],
            extra_compile_args={
                'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-D_GLIBCXX_USE_CXX11_ABI=0']
            },
        ),
    ]
    cmdclass = {'build_ext': cpp_extension.BuildExtension}
else:
    # don't build CUDA extensions when building sdist
    print("Not building CUDA extensions")
    ext_modules = []
    cmdclass = {}

setup(
    packages=find_packages(exclude=("csrc")),
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
)
