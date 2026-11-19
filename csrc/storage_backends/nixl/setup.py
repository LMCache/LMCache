# SPDX-License-Identifier: Apache-2.0
"""Build the lmcache-nixl wheel: the native NIXL connector as ``lmcache_nixl``.

Needs the NIXL SDK: NIXL_INCLUDE_DIR (nixl.h) and NIXL_LIBRARY_DIR (libnixl.so).
"""

# Standard
import os

# Third Party
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup


def _sdk_dir(env_var: str, required_file: str) -> str:
    path = os.environ.get(env_var, "")
    if not os.path.isfile(os.path.join(path, required_file)):
        raise RuntimeError(
            f"{env_var} must name the directory containing {required_file}"
        )
    return path


setup(
    ext_modules=[
        Pybind11Extension(
            "lmcache_nixl",
            sources=["pybind.cpp", "connector.cpp", "storage.cpp"],
            include_dirs=[".", "..", _sdk_dir("NIXL_INCLUDE_DIR", "nixl.h")],
            library_dirs=[_sdk_dir("NIXL_LIBRARY_DIR", "libnixl.so")],
            libraries=["nixl"],
            cxx_std=20,
            extra_compile_args=["-O3"],
        )
    ],
)
