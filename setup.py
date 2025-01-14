from setuptools import find_packages, setup
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

setup(
    name="lmcache",
    version="0.1.4",
    description="LMCache: prefill your long contexts only once",
    author="LMCache team",
    author_email="lmcacheteam@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("csrc")),
    install_requires=[
        "torch >= 2.2.0",
        "numpy==1.26.4",
        "aiofiles",
        "pyyaml",
        "redis",
        "nvtx",
        "safetensors",
        "transformers",
        "torchac_cuda >= 0.2.5",
        "sortedcontainers",
    ],
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
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            # Add command-line scripts here
            # e.g., "my_command=my_package.module:function"
            "lmcache_server=lmcache.server.__main__:main",
        ],
    },
)
