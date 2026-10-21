# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU build profile.

The Ascend native kernels live in the separately-distributed
LMCache-Ascend plugin (``lmcache_ascend.c_ops``) for now, and loaded at runtime.

"""

# Standard
from typing import TYPE_CHECKING, Optional
import os

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile

ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"

DEFAULT_ASCEND_HOME = "/usr/local/Ascend/ascend-toolkit/latest"


class AscendProfile(BuildProfile):
    """Ascend NPU build profile (detection + ABI; no in-repo kernels)."""

    name = "ascend"
    env_var = "BUILD_WITH_ASCEND"

    def detect(self) -> bool:
        """Detect the Ascend CANN toolchain via ``ASCEND_HOME_PATH``.

        Resolves ``ASCEND_HOME_PATH`` (falling back to the standard
        ``/usr/local/Ascend/ascend-toolkit/latest``) and treats an existing
        directory as the toolchain-present signal.  This probes only the
        installed toolkit directory, never the runtime NPU device -- mirroring
        how the CUDA profile checks for ``nvcc`` rather than
        ``torch.cuda.is_available()``.
        """
        ascend_home = os.environ.get("ASCEND_HOME_PATH", DEFAULT_ASCEND_HOME)
        return os.path.isdir(ascend_home)

    def build(self) -> tuple[list["Extension"], dict]:
        """Build no in-repo Ascend extension.

        NPU fused kernels are compiled in the LMCache-Ascend plugin
        (``lmcache_ascend.c_ops``) and loaded at runtime; the upstream package
        ships only the common C++ extensions.
        """
        print(
            "Ascend NPU kernels live in the lmcache_ascend plugin; "
            "building only common C++ extensions"
        )
        return [], {}

    def extra_cxx_flags_for(self, spec) -> list[str]:
        """All common extensions share the same ABI flag under Ascend."""
        return self.default_cxx_flags()

    def default_cxx_flags(self) -> list[str]:
        """ABI-aware default flags, matching torch / torch-npu / the plugin."""
        if ENABLE_CXX11_ABI:
            return ["-D_GLIBCXX_USE_CXX11_ABI=1"]
        return ["-D_GLIBCXX_USE_CXX11_ABI=0"]

    def requirements_file(self) -> Optional[str]:
        """Ascend runtime deps (torch-npu / plugin) are installed separately."""
        return None
