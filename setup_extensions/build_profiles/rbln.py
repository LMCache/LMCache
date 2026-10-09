# SPDX-License-Identifier: Apache-2.0
"""Rebellions RBLN NPU build profile.

RBLN has no in-repo native kernels: ``RblnDeviceOps`` moves KV with torch ops
on ``torch.rbln``, so this profile builds only the common C++ extensions.
The RBLN runtime stack (``torch-rbln`` + ``rebel-compiler``) is installed
separately: ``rebel-compiler`` is served only from the RBLN Portal index, and
the compatible pair is dictated by the host's RBLN driver.

Detection keys off an installed ``torch_rbln`` package rather than the NPU
device nodes, mirroring how the CUDA profile checks for ``nvcc`` rather than
``torch.cuda.is_available()``. ``torch.utils.cpp_extension`` already derives
the C++ ABI from the installed torch, so no ABI flag is injected here.
"""

# Standard
from typing import TYPE_CHECKING
import importlib.util

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile


class RblnProfile(BuildProfile):
    """RBLN NPU build profile (detection only; no in-repo kernels)."""

    name = "rbln"
    env_var = "BUILD_WITH_RBLN"

    def detect(self) -> bool:
        """Detect RBLN via an installed ``torch_rbln`` package.

        Uses ``importlib.util.find_spec`` so detection never imports
        ``torch_rbln``, which would initialize the RBLN runtime at build time.
        """
        return importlib.util.find_spec("torch_rbln") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build no RBLN extension; KV transfer runs on torch ops."""
        print("RBLN has no native kernels; building only common C++ extensions")
        return [], {}

    def requirements_file(self) -> str | None:
        """RBLN runtime deps (torch-rbln / rebel-compiler) are installed separately."""
        return None
