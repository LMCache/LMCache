from setuptools import setup, find_packages

setup(
    name="lmcache",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "redis",
        "torchac",
        "pyyaml",
        # List any dependencies here
        # e.g., "numpy", "requests"
    ],
    entry_points={
        'console_scripts': [
            # Add command-line scripts here
            # e.g., "my_command=my_package.module:function"
        ],
    },
)

