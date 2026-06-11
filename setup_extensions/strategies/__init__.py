# SPDX-License-Identifier: Apache-2.0
"""Base classes for the extension build strategy pattern.

Each backend (CUDA, ROCm, SYCL, MUSA, ...) implements :class:`BuildStrategy`.
The :class:`BuildPolicy` orchestrates auto-detection, fallback, and building.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension


class BuildStrategy(ABC):
    """Strategy for building platform-specific extensions.

    Subclasses must define:
        name     – unique identifier string.
        env_var  – ``BUILD_WITH_*`` environment variable name for explicit
                   selection.

    Subclasses must implement:
        detect() – auto-detect if this backend's hardware/compiler is present.
        build()  – return ``(ext_modules, cmdclass)`` for extensions.

    Subclasses may override:
        common_cpp_flags()    – C++ flags for common storage extensions.
        fs_cpp_flags()        – C++ flags for the ``lmcache_fs`` extension
                                (defaults to ``common_cpp_flags()``).
        requirements_file()   – core requirements file name.
    """

    name: str = ""
    env_var: str = ""

    # ------------------------------------------------------------------
    # Build-mode flags (owned by the strategy, not the policy)
    # ------------------------------------------------------------------

    @classmethod
    def is_building_sdist(cls) -> bool:
        """Return True when building a source distribution."""
        # Standard
        import sys

        return "sdist" in sys.argv

    @classmethod
    def is_native_ext_disabled(cls) -> bool:
        """Return True when native extensions are disabled."""
        # Standard
        import os
        import sys

        if os.environ.get("NO_CUDA_EXT", "0") == "1":
            print(
                "warning: NO_CUDA_EXT is deprecated; use NO_NATIVE_EXT=1 instead.",
                file=sys.stderr,
            )
        return (
            os.environ.get("NO_NATIVE_EXT", "0") == "1"
            or os.environ.get("NO_CUDA_EXT", "0") == "1"
        )

    @classmethod
    def is_gpu_ext_disabled(cls) -> bool:
        """Return True when GPU extensions are disabled."""
        # Standard
        import os

        return os.environ.get("NO_GPU_EXT", "0") == "1"

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def is_explicitly_requested(self) -> bool:
        """Return True when this backend was selected via env var."""
        if not self.env_var:
            return False
        # Standard
        import os

        return os.environ.get(self.env_var, "0") == "1"

    @abstractmethod
    def detect(self) -> bool:
        """Auto-detect if this backend's toolchain / hardware is available."""
        ...

    @abstractmethod
    def build(self) -> tuple[list["Extension"], dict]:
        """Build backend-specific extension modules.

        Returns:
            ``(ext_modules, cmdclass)`` tuple.
        """
        ...

    def common_cpp_flags(self) -> list[str]:
        """Additional C++ compile flags for common extensions."""
        return []

    def fs_cpp_flags(self) -> list[str]:
        """Additional C++ compile flags for ``lmcache_fs``.

        Defaults to ``common_cpp_flags()``; override when flags differ.
        """
        return self.common_cpp_flags()

    def requirements_file(self) -> Optional[str]:
        """Core requirements file name, relative to ``requirements/``.

        Return ``None`` when this backend has no extra deps.
        """
        return None
