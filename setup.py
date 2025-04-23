import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent


def get_version():
    version_file = ROOT_DIR / "lmcache" / "_version.py"
    with open(version_file) as f:
        version_ns = {}
        exec(f.read(), version_ns)
        return version_ns["__version__"]


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
                os.environ.get("NO_CUDA_EXT", "0") == "1"

if not BUILDING_SDIST:
    print("Building CUDA extensions")
    from torch.utils import cpp_extension
    ext_modules = [
        cpp_extension.CUDAExtension(
            'lmcache.c_ops',
            [
                'csrc/pybind.cpp',
                'csrc/mem_kernels.cu',
                'csrc/cal_cdf.cu',
                'csrc/ac_enc.cu',
                'csrc/ac_dec.cu',
            ],
        ),
    ]
    cmdclass = {'build_ext': cpp_extension.BuildExtension}
else:
    # don't build CUDA extensions when building sdist
    print("Not building CUDA extensions")
    ext_modules = []
    cmdclass = {}

setup(
    name="lmcache",
    version=get_version(),
    description="LMCache: prefill your long contexts only once",
    author="LMCache team",
    author_email="lmcacheteam@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("csrc")),
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        # Trove classifiers
        # Full list at https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="Apache-2.0",
    license_files=["LICENSE"],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            # Add command-line scripts here
            # e.g., "my_command=my_package.module:function"
            "lmcache_server=lmcache.server.__main__:main",
            "lmcache_experimental_server=lmcache.experimental.server.__main__:main",
            "lmcache_controller=lmcache.experimental.api_server.__main__:main",
        ],
    },
)
