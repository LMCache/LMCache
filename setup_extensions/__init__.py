# SPDX-License-Identifier: Apache-2.0
"""Extension build framework using strategy pattern.

Usage from setup.py::

    from setup_extensions import BuildPolicy
    policy = BuildPolicy()
    strategy = policy.resolve_strategy()
    ext_modules, cmdclass, req_file = policy.collect_extensions(strategy)
"""

# First Party
from setup_extensions.common_cpp import build_common_cpp  # noqa: F401
from setup_extensions.policy import BuildPolicy, discover_subclasses  # noqa: F401
from setup_extensions.storage_backends import StorageBackendStrategy  # noqa: F401
from setup_extensions.strategies import BuildStrategy  # noqa: F401

# Re-export build-mode flags from BuildStrategy for callers who prefer
# module-level access (e.g. setup.py).
BUILDING_SDIST = BuildStrategy.is_building_sdist()
NO_NATIVE_EXT = BuildStrategy.is_native_ext_disabled()
NO_GPU_EXT = BuildStrategy.is_gpu_ext_disabled()
