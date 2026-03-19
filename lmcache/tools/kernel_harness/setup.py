# SPDX-License-Identifier: Apache-2.0

"""
Standalone build script for the kernel harness CUDA extension.

Usage:
    cd lmcache/tools/kernel_harness
    python setup.py build_ext --inplace

This produces kernel_harness_ops.*.so in the current directory.
"""

# Third Party
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="kernel_harness_ops",
    ext_modules=[
        CUDAExtension(
            "kernel_harness_ops",
            sources=[
                "csrc/pybind.cpp",
                "csrc/multi_layer_block_kv_transfer.cu",
            ],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
